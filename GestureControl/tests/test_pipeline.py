"""End-to-end checks of the capture loop with a stand-in camera.

Real landmark detection is MediaPipe's job; what matters here is that our
wiring around it holds up -- timestamps advance, missing hands are handled,
handedness is corrected for the mirrored frame, the overlay renders, and the
loop always shuts down cleanly.
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gesturectl import app, hud                                         # noqa: E402
from gesturectl.config import Config                                    # noqa: E402
from gesturectl.cursor import CursorPump                                # noqa: E402
from gesturectl.controller import HandStatus, Status                    # noqa: E402
from gesturectl.gestures import FingerTracker, Gesture, build_hand      # noqa: E402
from gesturectl.macos_input import MouseController, parse_shortcut      # noqa: E402
from tests import synthetic as syn                                      # noqa: E402

MODEL = Config().model_file
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="model not downloaded")


class FakeCapture:
    def __init__(self, frames=6, stop_with_interrupt=False):
        self.remaining = frames
        self.stop_with_interrupt = stop_with_interrupt
        self.released = False

    def isOpened(self):
        return True

    def set(self, *_):
        return True

    def read(self):
        if self.remaining <= 0 and self.stop_with_interrupt:
            raise KeyboardInterrupt
        self.remaining -= 1
        return True, np.random.randint(0, 40, (480, 640, 3), dtype=np.uint8)

    def release(self):
        self.released = True


@pytest.fixture
def headless(monkeypatch):
    shown = []
    monkeypatch.setattr(cv2, "imshow", lambda *a: shown.append(a))
    monkeypatch.setattr(cv2, "waitKey", lambda *_: 255)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    return shown


@needs_model
def test_loop_runs_and_quits_on_keypress(monkeypatch, headless):
    cap = FakeCapture(frames=100)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: cap)
    keys = iter([255] * 5 + [ord("q")])
    monkeypatch.setattr(cv2, "waitKey", lambda *_: next(keys, ord("q")))

    assert app.run(Config(), show_preview=True, dry_run=True) == 0
    assert cap.released
    assert len(headless) >= 5


@needs_model
def test_loop_runs_headless_and_survives_interrupt(monkeypatch):
    cap = FakeCapture(frames=5, stop_with_interrupt=True)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: cap)
    assert app.run(Config(), show_preview=False, dry_run=True) == 0
    assert cap.released


@needs_model
def test_landmarker_is_configured_for_two_hands():
    lm = app.build_landmarker(Config())
    assert lm is not None
    lm.close()


@needs_model
def test_space_toggles_arming_from_the_preview(monkeypatch, headless):
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: FakeCapture(frames=100))
    captured = {}
    real = app.GestureController

    def spy(*a, **kw):
        ctl = real(*a, **kw)
        captured["ctl"] = ctl
        return ctl

    monkeypatch.setattr(app, "GestureController", spy)
    keys = iter([ord(" "), 255, ord("q")])
    monkeypatch.setattr(cv2, "waitKey", lambda *_: next(keys, ord("q")))

    cfg = Config()
    cfg.start_armed = False
    app.run(cfg, show_preview=True, dry_run=True)
    assert captured["ctl"].armed


def test_camera_failure_explains_the_permission_fix(monkeypatch):
    class Closed(FakeCapture):
        def isOpened(self):
            return False

    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: Closed())
    with pytest.raises(app.CameraError) as exc:
        app.open_camera(Config())
    assert "Camera" in str(exc.value)


def test_refuses_to_run_without_accessibility(monkeypatch, capsys):
    monkeypatch.setattr(app, "accessibility_trusted", lambda: False)
    assert app.run(Config(), show_preview=False, dry_run=False) == 2
    assert "Accessibility" in capsys.readouterr().err


# -- handedness ------------------------------------------------------------

@dataclass
class FakeCategory:
    category_name: str
    score: float = 0.99


@dataclass
class FakeResult:
    handedness: list


def test_handedness_is_flipped_for_the_mirrored_frame():
    """The frame is mirrored before detection, so MediaPipe's labels arrive
    backwards and must be swapped to match the user's actual hands."""
    result = FakeResult([[FakeCategory("Left")], [FakeCategory("Right")]])
    assert app.assign_labels(result) == [("Right", 0), ("Left", 1)]


def test_a_single_hand_still_gets_a_label():
    assert app.assign_labels(FakeResult([[FakeCategory("Left")]])) == [("Right", 0)]


def test_two_hands_never_share_a_label():
    """Both detections occasionally come back with the same handedness; the
    weaker one is reassigned so the hands stay independently addressable."""
    result = FakeResult([[FakeCategory("Left", 0.9)], [FakeCategory("Left", 0.6)]])
    labels = [label for label, _ in app.assign_labels(result)]
    assert sorted(labels) == ["Left", "Right"]


def test_the_more_confident_detection_keeps_its_label():
    result = FakeResult([[FakeCategory("Left", 0.6)], [FakeCategory("Left", 0.95)]])
    assigned = dict((idx, label) for label, idx in app.assign_labels(result))
    assert assigned[1] == "Right", "the confident one keeps the swapped label"
    assert assigned[0] == "Left"


def test_no_hands_yields_no_labels():
    assert app.assign_labels(FakeResult([])) == []


# -- shortcuts -------------------------------------------------------------

def test_shortcut_parsing():
    keycode, flags = parse_shortcut("cmd+w")
    assert keycode == 13 and flags != 0
    assert parse_shortcut("w")[1] == 0
    assert parse_shortcut("CMD+Shift+W")[1] != parse_shortcut("cmd+w")[1]


def test_bad_shortcuts_are_rejected():
    for combo in ("", "cmd+", "meta+w", "cmd+notakey"):
        with pytest.raises(ValueError):
            parse_shortcut(combo)


# -- overlay ---------------------------------------------------------------

def _hand(pose, label="Right"):
    h = build_hand(pose.image, syn.ASPECT, pose.world, label)
    return FingerTracker().apply(h)


def test_overlay_draws_without_error():
    frame = np.zeros((480, 640, 3), np.uint8)
    status = Status(armed=True, hands=[HandStatus("Right", Gesture.POINT, True)],
                    cursor=(100.0, 100.0), arm_progress=0.5, events=["click"])
    hud.draw_hand(frame, _hand(syn.point()), status)
    hud.draw_region(frame, Config(), True)
    hud.draw_status(frame, status, 29.7, dry_run=False)
    assert frame.any()


def test_overlay_draws_two_hands_and_the_resize_link():
    frame = np.zeros((480, 640, 3), np.uint8)
    status = Status(armed=True, resizing=True,
                    hands=[HandStatus("Left", Gesture.POINT), HandStatus("Right", Gesture.POINT)])
    left = _hand(syn.translate(syn.point(), dx=-0.15), "Left")
    right = _hand(syn.translate(syn.point(), dx=0.15), "Right")
    hud.draw_hand(frame, left, status)
    hud.draw_hand(frame, right, status)
    hud.draw_resize_link(frame, left, right)
    hud.draw_status(frame, status, 30.0, dry_run=False)
    assert frame.any()


def test_overlay_handles_every_gesture_label():
    frame = np.zeros((480, 640, 3), np.uint8)
    for g in Gesture:
        hud.draw_status(frame, Status(hands=[HandStatus("Right", g, True)]), 30.0, False)


def test_overlay_handles_no_hands():
    frame = np.zeros((480, 640, 3), np.uint8)
    hud.draw_status(frame, Status(), 30.0, False)


def test_debug_overlay_stays_inside_the_frame():
    """It anchors to the wrist, which can sit anywhere including the edges."""
    frame = np.zeros((480, 640, 3), np.uint8)
    for dx, dy in ((-0.6, -0.6), (0.6, 0.6), (0.0, 0.0), (0.45, -0.3)):
        hud.draw_debug(frame, _hand(syn.translate(syn.point(), dx, dy)), Config())


@needs_model
def test_debug_mode_runs(monkeypatch, headless):
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: FakeCapture(frames=100))
    keys = iter([255, 255, ord("q")])
    monkeypatch.setattr(cv2, "waitKey", lambda *_: next(keys, ord("q")))
    assert app.run(Config(), show_preview=True, dry_run=True, debug=True) == 0


# -- config portability ----------------------------------------------------

def test_saved_config_has_no_machine_specific_paths(tmp_path):
    """Regression: init-config used to serialise the model's absolute path,
    so config.json only worked in the directory it was written in -- moving or
    cloning the project left it pointing at a file that was no longer there."""
    out = tmp_path / "config.json"
    Config().save(out)
    assert "/Users/" not in out.read_text()
    assert json.loads(out.read_text())["model_path"] == ""


def test_a_blank_model_path_resolves_to_the_bundled_model():
    cfg = Config()
    assert cfg.model_path == ""
    assert cfg.model_file.name == "hand_landmarker.task"
    assert cfg.model_file.is_absolute()


def test_an_explicit_model_path_is_still_honoured(tmp_path):
    elsewhere = tmp_path / "custom.task"
    cfg = Config()
    cfg.model_path = str(elsewhere)
    assert cfg.model_file == elsewhere


def test_config_survives_a_save_load_round_trip(tmp_path):
    out = tmp_path / "config.json"
    cfg = Config()
    cfg.gesture.toggle_pose = "OPEN_PALM"
    cfg.save(out)
    assert Config.load(out).gesture.toggle_pose == "OPEN_PALM"


# -- cursor pump -----------------------------------------------------------

def _pump(**cursor):
    cfg = Config()
    for k, v in cursor.items():
        setattr(cfg.cursor, k, v)
    mouse = MouseController(dry_run=True)
    return CursorPump(cfg, mouse), mouse


def test_the_pump_does_nothing_without_a_target():
    pump, _ = _pump(threaded=False)
    assert pump.step(0.0) is None


def test_the_pump_moves_the_cursor_toward_an_active_target():
    pump, mouse = _pump(threaded=False)
    start = mouse.position
    t = 0.0
    for _ in range(40):
        t += 1 / 500
        pump.set_target(900.0, 500.0, active=True)
        pump.step(t)
    assert mouse.position != start
    assert pump.updates == 40


def test_an_inactive_target_is_tracked_but_not_applied():
    """Scrolling reads the smoothed position while the cursor stays parked, so
    the filter has to keep running even when it is not steering."""
    pump, mouse = _pump(threaded=False)
    parked = mouse.position
    t = 0.0
    for _ in range(20):
        t += 1 / 500
        pump.set_target(900.0, 500.0, active=False)
        pump.step(t)
    assert mouse.position == parked
    assert pump.value is not None
    assert pump.updates == 0


def test_recentring_forgets_the_target():
    pump, _ = _pump(threaded=False)
    pump.set_target(400.0, 400.0, active=True)
    pump.step(0.002)
    pump.recentre()
    assert pump.value is None
    assert pump.step(0.004) is None


def test_the_threaded_pump_runs_near_its_configured_rate():
    pump, mouse = _pump(threaded=True, poll_hz=200.0)
    pump.set_target(700.0, 400.0, active=True)
    pump.start()
    try:
        assert pump.running
        time.sleep(0.4)
    finally:
        pump.stop()
    assert not pump.running
    # Generous bounds: this asserts the loop is paced, not that the machine is
    # quiet enough to hit the rate exactly.
    assert 40 < pump.updates < 160, pump.updates


def test_stopping_a_pump_that_never_started_is_harmless():
    pump, _ = _pump(threaded=True)
    pump.stop()
    assert not pump.running


def test_threading_can_be_turned_off():
    pump, _ = _pump(threaded=False)
    pump.start()
    assert not pump.running
