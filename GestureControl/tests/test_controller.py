import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gesturectl.config import Config                                    # noqa: E402
from gesturectl.controller import GestureController, TrackedHand        # noqa: E402
from gesturectl.gestures import (                                       # noqa: E402
    INDEX_MCP, INDEX_TIP, FingerTracker, Gesture, build_hand,
)
from gesturectl.macos_input import (                                    # noqa: E402
    KeyboardController, MouseController, WindowController,
)
from tests import synthetic as syn                                      # noqa: E402

ASPECT = syn.ASPECT


class Driver:
    """Feeds poses through the controller against a clock we control."""

    def __init__(self, armed=True, **overrides):
        self.cfg = Config()
        for key, value in overrides.items():
            section, _, field = key.partition(".")
            setattr(getattr(self.cfg, section), field, value)
        # Drive the pump synchronously so cursor motion is deterministic;
        # the background thread has its own tests.
        self.cfg.cursor.threaded = overrides.get("cursor.threaded", False)
        self.mouse = MouseController(dry_run=True)
        self.keyboard = KeyboardController(dry_run=True)
        self.windows = WindowController(dry_run=True)
        self.ctl = GestureController(self.cfg, self.mouse, self.keyboard, self.windows)
        self.ctl.armed = armed
        self.t = 0.0

    def _tracked(self, label, pose, gesture):
        hand = build_hand(pose.image, ASPECT, pose.world, label)
        FingerTracker(self.cfg.gesture.extend_below, self.cfg.gesture.curl_above).apply(hand)
        return TrackedHand(hand, gesture)

    def feed(self, hands, dt=1 / 30):
        """hands: list of (label, pose, gesture)."""
        self.t += dt
        return self.ctl.update([self._tracked(*h) for h in hands], self.t)

    def one(self, pose, gesture, label="Right", dt=1 / 30):
        return self.feed([(label, pose, gesture)], dt)

    def idle(self, seconds, pose=None, gesture=Gesture.POINT, label="Right"):
        end = self.t + seconds
        while self.t < end:
            self.one(pose or syn.point(), gesture, label)

    def hold_two(self, seconds, left, right, gesture=Gesture.POINT):
        end = self.t + seconds
        while self.t < end:
            self.feed([("Left", left, gesture), ("Right", right, gesture)])

    @property
    def log(self):
        return self.mouse.log

    @property
    def keys(self):
        return self.keyboard.log

    @property
    def resizes(self):
        return self.windows.log


def _sizes(log):
    out = []
    for entry in log:
        dims = entry.split()[1].split("@")[0].strip()
        w, h = dims.split("x")
        out.append((int(w), int(h)))
    return out


# -- clicking --------------------------------------------------------------

def test_quick_pinch_produces_a_single_click():
    d = Driver()
    d.one(syn.pinch(), Gesture.PINCH)
    d.one(syn.point(), Gesture.POINT, dt=0.15)
    assert d.log == []
    d.idle(0.6)
    assert d.log == ["click"]


def test_two_quick_taps_produce_one_double_click():
    d = Driver()
    d.one(syn.pinch(), Gesture.PINCH)
    d.one(syn.point(), Gesture.POINT, dt=0.12)
    d.one(syn.pinch(), Gesture.PINCH, dt=0.10)
    d.one(syn.point(), Gesture.POINT, dt=0.12)
    d.idle(0.6)
    assert d.log == ["double click"]


def test_slow_taps_stay_two_separate_clicks():
    d = Driver()
    for _ in range(2):
        d.one(syn.pinch(), Gesture.PINCH)
        d.one(syn.point(), Gesture.POINT, dt=0.15)
        d.idle(0.7)
    assert d.log == ["click", "click"]


# -- dragging --------------------------------------------------------------

def test_held_pinch_becomes_a_drag():
    d = Driver()
    d.one(syn.pinch(), Gesture.PINCH)
    d.idle(0.5, syn.pinch(), Gesture.PINCH)
    assert "left down" in d.log
    d.one(syn.point(), Gesture.POINT)
    assert d.log[-1] == "left up"


def test_a_drag_does_not_also_emit_a_click():
    d = Driver()
    d.one(syn.pinch(), Gesture.PINCH)
    d.idle(0.5, syn.pinch(), Gesture.PINCH)
    d.one(syn.point(), Gesture.POINT)
    d.idle(0.8)
    assert d.log == ["left down", "left up"]


def test_tap_then_drag_keeps_the_earlier_click():
    d = Driver()
    d.one(syn.pinch(), Gesture.PINCH)
    d.one(syn.point(), Gesture.POINT, dt=0.12)
    d.one(syn.pinch(), Gesture.PINCH, dt=0.10)
    d.idle(0.5, syn.pinch(), Gesture.PINCH)
    d.one(syn.point(), Gesture.POINT)
    assert d.log == ["click", "left down", "left up"]


def test_losing_the_hands_mid_drag_releases_the_button():
    d = Driver()
    d.one(syn.pinch(), Gesture.PINCH)
    d.idle(0.5, syn.pinch(), Gesture.PINCH)
    assert d.log[-1] == "left down"
    d.ctl.hands_lost(d.t + 0.1)
    assert d.log[-1] == "left up"


def test_disarming_mid_drag_releases_the_button():
    d = Driver()
    d.one(syn.pinch(), Gesture.PINCH)
    d.idle(0.5, syn.pinch(), Gesture.PINCH)
    d.ctl.toggle_armed()
    assert d.log[-1] == "left up"


# -- scroll and right click ------------------------------------------------

def _scroll_deltas(log):
    return [int(e.split("dy=")[1].split()[0]) for e in log if e.startswith("scroll")]


def test_moving_the_hand_up_while_scrolling_emits_wheel_events():
    d = Driver()
    for i in range(14):
        d.one(syn.translate(syn.peace(), dy=-0.02 * i), Gesture.SCROLL)
    deltas = _scroll_deltas(d.log)
    assert deltas and all(v > 0 for v in deltas), deltas


def test_moving_the_hand_down_scrolls_the_other_way():
    d = Driver()
    for i in range(14):
        d.one(syn.translate(syn.peace(), dy=0.02 * i), Gesture.SCROLL)
    deltas = _scroll_deltas(d.log)
    assert deltas and all(v < 0 for v in deltas), deltas


def test_scrolling_works_with_fingers_angled_at_the_camera():
    """The pose that used to break detection must still scroll normally."""
    angled = syn.peace(syn.ANGLED_DOWN)
    d = Driver()
    for i in range(14):
        d.one(syn.translate(angled, dy=-0.02 * i), Gesture.SCROLL)
    assert _scroll_deltas(d.log)


def test_three_fingers_right_clicks_once_per_entry():
    d = Driver()
    for _ in range(11):
        d.one(syn.three(), Gesture.THREE)
    assert d.log == ["right click"]


# -- exit gesture ----------------------------------------------------------

def test_dropping_a_raised_pinky_closes_the_window():
    d = Driver()
    d.idle(0.6, syn.pinky_up(), Gesture.PINKY_UP)
    assert d.keys == [], "raising the pinky alone must not close anything"
    d.one(syn.fist(), Gesture.FIST)
    assert d.keys == ["key cmd+w"]


def test_the_exit_gesture_fires_only_once():
    d = Driver()
    d.idle(0.6, syn.pinky_up(), Gesture.PINKY_UP)
    d.idle(1.0, syn.fist(), Gesture.FIST)
    assert d.keys == ["key cmd+w"]


def test_a_brief_pinky_does_not_close_the_window():
    """A pinky passing through on the way to another pose is not a flick."""
    d = Driver()
    d.one(syn.pinky_up(), Gesture.PINKY_UP)
    d.one(syn.fist(), Gesture.FIST, dt=0.05)
    assert d.keys == []


def test_opening_the_hand_from_a_raised_pinky_does_not_close_the_window():
    """The pinky must actually go down; it stays up when you open the hand."""
    d = Driver()
    d.idle(0.6, syn.pinky_up(), Gesture.PINKY_UP)
    d.one(syn.open_palm(), Gesture.OPEN_PALM)
    assert d.keys == []


def test_losing_the_hand_with_a_raised_pinky_does_not_close_the_window():
    d = Driver()
    d.idle(0.6, syn.pinky_up(), Gesture.PINKY_UP)
    d.ctl.hands_lost(d.t + 0.1)
    assert d.keys == []


def test_the_exit_gesture_respects_its_cooldown():
    d = Driver()
    for _ in range(3):
        d.idle(0.5, syn.pinky_up(), Gesture.PINKY_UP)
        d.idle(0.2, syn.fist(), Gesture.FIST)
    assert d.keys == ["key cmd+w"]


def test_the_exit_gesture_can_be_disabled():
    d = Driver(**{"exit_gesture.enabled": False})
    d.idle(0.6, syn.pinky_up(), Gesture.PINKY_UP)
    d.one(syn.fist(), Gesture.FIST)
    assert d.keys == []


def test_a_disarmed_controller_never_closes_a_window():
    d = Driver(armed=False)
    d.idle(0.6, syn.pinky_up(), Gesture.PINKY_UP)
    d.one(syn.fist(), Gesture.FIST)
    assert d.keys == []


# -- two hands -------------------------------------------------------------

def test_either_hand_can_drive_on_its_own():
    for label in ("Left", "Right"):
        d = Driver()
        d.one(syn.pinch(), Gesture.PINCH, label=label)
        d.one(syn.point(), Gesture.POINT, label=label)
        d.idle(0.6, gesture=Gesture.POINT, label=label)
        assert d.log == ["click"], label


def test_two_raised_hands_do_not_double_click():
    """Only the acting hand may fire, or every action would happen twice."""
    d = Driver()
    d.feed([("Left", syn.pinch(), Gesture.PINCH), ("Right", syn.pinch(), Gesture.PINCH)])
    # Right keeps steering so the pinch is seen to end; left drops out of play.
    d.feed([("Left", syn.fist(), Gesture.FIST), ("Right", syn.point(), Gesture.POINT)], dt=0.15)
    d.idle(0.7)
    assert d.log == ["click"]


def test_the_primary_hand_wins_when_both_are_active():
    """Both hands steering: the primary one gets it, and only it."""
    d = Driver()
    status = d.feed([("Left", syn.pinch(), Gesture.PINCH),
                     ("Right", syn.pinch(), Gesture.PINCH)])
    acting = [h.label for h in status.hands if h.acting]
    assert acting == [d.cfg.hands.primary]


def test_two_raised_hands_that_are_not_steering_do_nothing():
    """Two hands up with neither pointing or pinching is ambiguous about which
    hand means what, so nothing fires until one of them steers."""
    d = Driver()
    for i in range(14):
        moved = syn.translate(syn.peace(), dy=-0.02 * i)
        d.feed([("Left", moved, Gesture.SCROLL), ("Right", moved, Gesture.SCROLL)])
    assert _scroll_deltas(d.log) == []


def test_one_hand_scrolling_alone_still_works():
    """The same motion with only one hand in frame is unambiguous."""
    d = Driver()
    for i in range(14):
        d.one(syn.translate(syn.peace(), dy=-0.02 * i), Gesture.SCROLL)
    assert _scroll_deltas(d.log)


def test_a_steering_hand_still_acts_with_the_other_hand_in_frame():
    """The rule only suppresses when *neither* hand is steering."""
    d = Driver()
    start = d.mouse.position
    for i in range(12):
        d.feed([("Left", syn.peace(), Gesture.SCROLL),
                ("Right", syn.translate(syn.point(), dx=0.015 * i), Gesture.POINT)])
    assert d.mouse.position != start
    assert _scroll_deltas(d.log) == [], "the idle hand scrolled anyway"


def test_a_completed_click_survives_both_hands_going_idle():
    """Nothing new starts, but input that already finished is not thrown away.

    A tap is held back briefly in case it turns into a double click, so it is
    still pending at the moment both hands relax. It must not be discarded.
    """
    d = Driver()
    d.one(syn.pinch(), Gesture.PINCH)                 # pinch down
    d.one(syn.point(), Gesture.POINT)                 # and up: that is the tap
    d.feed([("Left", syn.fist(), Gesture.FIST),
            ("Right", syn.fist(), Gesture.FIST)], dt=0.05)
    d.idle(0.7)
    assert d.log == ["click"]


def test_a_resting_primary_hand_yields_to_the_other():
    d = Driver()
    status = d.feed([("Left", syn.point(), Gesture.POINT),
                     ("Right", syn.fist(), Gesture.FIST)])
    acting = [h.label for h in status.hands if h.acting]
    assert acting == ["Left"]


def test_handing_over_mid_drag_releases_the_button():
    d = Driver()
    d.one(syn.pinch(), Gesture.PINCH, label="Right")
    d.idle(0.5, syn.pinch(), Gesture.PINCH, label="Right")
    assert d.log[-1] == "left down"
    d.feed([("Left", syn.point(), Gesture.POINT), ("Right", syn.fist(), Gesture.FIST)])
    assert d.log[-1] == "left up"


# -- two-handed resize -----------------------------------------------------

def _spread(gap):
    """Two pointing hands whose index tips sit `gap` apart across the frame."""
    return syn.translate(syn.point(), dx=-gap / 2), syn.translate(syn.point(), dx=gap / 2)


def test_moving_pointing_hands_apart_grows_the_window():
    d = Driver()
    left, right = _spread(0.25)
    d.hold_two(0.2, left, right)
    for i in range(1, 12):
        left, right = _spread(0.25 + 0.02 * i)
        d.feed([("Left", left, Gesture.POINT), ("Right", right, Gesture.POINT)])
    sizes = _sizes(d.resizes)
    assert sizes, "no resize was applied"
    assert sizes[-1][0] > 900 and sizes[-1][1] > 600, sizes[-1]


def test_moving_pointing_hands_together_shrinks_the_window():
    d = Driver()
    left, right = _spread(0.55)
    d.hold_two(0.2, left, right)
    for i in range(1, 12):
        left, right = _spread(0.55 - 0.02 * i)
        d.feed([("Left", left, Gesture.POINT), ("Right", right, Gesture.POINT)])
    sizes = _sizes(d.resizes)
    assert sizes and sizes[-1][0] < 900 and sizes[-1][1] < 600, sizes


def test_resizing_keeps_the_aspect_ratio():
    d = Driver()
    left, right = _spread(0.25)
    d.hold_two(0.2, left, right)
    for i in range(1, 12):
        left, right = _spread(0.25 + 0.02 * i)
        d.feed([("Left", left, Gesture.POINT), ("Right", right, Gesture.POINT)])
    w, h = _sizes(d.resizes)[-1]
    assert abs(w / h - 900 / 600) < 0.05, (w, h)


def test_holding_two_hands_still_does_not_resize():
    d = Driver()
    left, right = _spread(0.35)
    d.hold_two(1.0, left, right)
    assert _sizes(d.resizes) == []


def test_resizing_does_not_also_drag_the_cursor_around():
    """A two-handed gesture owns both hands: the cursor stays put and nothing
    clicks while the window is being resized."""
    d = Driver()
    left, right = _spread(0.25)
    d.hold_two(0.2, left, right)
    parked = d.mouse.position
    for i in range(1, 12):
        left, right = _spread(0.25 + 0.02 * i)
        d.feed([("Left", left, Gesture.POINT), ("Right", right, Gesture.POINT)])
    assert d.log == [], d.log
    assert d.mouse.position == parked


def test_single_hand_control_resumes_after_a_resize():
    d = Driver()
    left, right = _spread(0.25)
    d.hold_two(0.3, left, right)
    assert _sizes(d.resizes) == [] or d.ctl.resizing
    d.one(syn.pinch(), Gesture.PINCH)
    d.one(syn.point(), Gesture.POINT, dt=0.15)
    d.idle(0.6)
    assert d.log == ["click"]


def test_resize_needs_both_hands_pointing():
    d = Driver()
    left, right = _spread(0.25)
    for i in range(12):
        d.feed([("Left", left, Gesture.POINT), ("Right", right, Gesture.FIST)])
    assert _sizes(d.resizes) == []


def test_resize_can_be_disabled():
    d = Driver(**{"resize.enabled": False})
    left, right = _spread(0.25)
    d.hold_two(0.2, left, right)
    for i in range(1, 12):
        left, right = _spread(0.25 + 0.02 * i)
        d.feed([("Left", left, Gesture.POINT), ("Right", right, Gesture.POINT)])
    assert _sizes(d.resizes) == []


def test_a_disarmed_controller_never_resizes():
    d = Driver(armed=False)
    left, right = _spread(0.25)
    d.hold_two(0.2, left, right)
    for i in range(1, 12):
        left, right = _spread(0.25 + 0.02 * i)
        d.feed([("Left", left, Gesture.POINT), ("Right", right, Gesture.POINT)])
    assert _sizes(d.resizes) == []


def test_resizing_is_scale_invariant_to_camera_distance():
    """Leaning toward the camera enlarges both the gap and the hands, so the
    window must not drift while you hold a steady spread."""
    d = Driver()
    left, right = _spread(0.30)
    d.hold_two(0.3, left, right)
    for scale in (1.1, 1.25, 1.4):
        big_l = syn.make_pose(index=syn.STRAIGHT)
        big_r = syn.make_pose(index=syn.STRAIGHT)
        gap = 0.30 * scale
        # Hands and gap grow together, as they do when you lean in.
        left = syn.translate(_scaled(big_l, scale), dx=-gap / 2)
        right = syn.translate(_scaled(big_r, scale), dx=gap / 2)
        for _ in range(4):
            d.feed([("Left", left, Gesture.POINT), ("Right", right, Gesture.POINT)])
    sizes = _sizes(d.resizes)
    if sizes:
        assert abs(sizes[-1][0] - 900) < 90, sizes[-1]


def _scaled(pose, factor):
    cx = sum(p.x for p in pose.image) / 21
    cy = sum(p.y for p in pose.image) / 21
    image = [syn.LM(cx + (p.x - cx) * factor, cy + (p.y - cy) * factor, p.z) for p in pose.image]
    return syn.Pose(image, pose.world)


# -- arming ----------------------------------------------------------------

def test_disarmed_controller_sends_nothing():
    d = Driver(armed=False)
    d.one(syn.pinch(), Gesture.PINCH)
    d.one(syn.point(), Gesture.POINT, dt=0.15)
    d.idle(0.8)
    d.one(syn.three(), Gesture.THREE)
    assert d.log == []


def test_disarming_cancels_a_click_that_was_still_pending():
    d = Driver()
    d.one(syn.pinch(), Gesture.PINCH)
    d.one(syn.point(), Gesture.POINT, dt=0.15)
    assert d.log == []
    d.ctl.toggle_armed()
    d.idle(1.0)
    assert d.log == []


def test_holding_a_fist_toggles_control():
    d = Driver(armed=False)
    d.idle(1.3, syn.fist(), Gesture.FIST)
    assert d.ctl.armed


def test_a_fist_toggles_only_once_per_hold():
    d = Driver(armed=False)
    d.idle(3.0, syn.fist(), Gesture.FIST)
    assert d.ctl.armed


def test_a_fist_can_toggle_again_after_a_break():
    d = Driver(armed=False)
    d.idle(1.3, syn.fist(), Gesture.FIST)
    d.idle(0.3, syn.point(), Gesture.POINT)
    d.idle(1.3, syn.fist(), Gesture.FIST)
    assert not d.ctl.armed


def test_a_brief_fist_does_not_toggle_control():
    d = Driver(armed=False)
    d.idle(0.5, syn.fist(), Gesture.FIST)
    assert not d.ctl.armed


def test_either_hand_can_toggle_control():
    d = Driver(armed=False)
    d.idle(1.3, syn.fist(), Gesture.FIST, label="Left")
    assert d.ctl.armed


def test_an_open_palm_no_longer_toggles_control():
    """It is inert now: too easy to drift into from the action gestures."""
    d = Driver(armed=False)
    d.idle(3.0, syn.open_palm(), Gesture.OPEN_PALM)
    assert not d.ctl.armed


def test_an_open_palm_does_nothing_at_all():
    d = Driver()
    d.idle(2.0, syn.open_palm(), Gesture.OPEN_PALM)
    assert d.log == [] and d.keys == []


def test_moving_between_action_gestures_never_toggles_control():
    """The reported failure: switching poses flashed through open palm and
    could hand control over. No action gesture curls the index, so none of
    these transitions can pass through the toggle pose either."""
    sequence = [
        (syn.point(), Gesture.POINT),
        (syn.pinch(), Gesture.PINCH),
        (syn.open_palm(), Gesture.OPEN_PALM),      # the accidental read
        (syn.point(), Gesture.POINT),
        (syn.peace(), Gesture.SCROLL),
        (syn.open_palm(), Gesture.OPEN_PALM),
        (syn.three(), Gesture.THREE),
        (syn.pinch(), Gesture.PINCH),
        (syn.point(), Gesture.POINT),
    ]
    d = Driver(armed=True)
    for _ in range(6):
        for pose, gesture in sequence:
            d.idle(0.25, pose, gesture)
    assert d.ctl.armed, "control was handed over mid-session"


def test_the_toggle_pose_is_configurable():
    d = Driver(armed=False, **{"gesture.toggle_pose": "OPEN_PALM"})
    d.idle(1.3, syn.open_palm(), Gesture.OPEN_PALM)
    assert d.ctl.armed


def test_an_unknown_toggle_pose_falls_back_to_a_fist():
    d = Driver(armed=False, **{"gesture.toggle_pose": "NONSENSE"})
    assert d.ctl.toggle_pose == Gesture.FIST
    d.idle(1.3, syn.fist(), Gesture.FIST)
    assert d.ctl.armed


# -- coordinate mapping ----------------------------------------------------

def test_active_region_maps_onto_the_whole_screen():
    d = Driver()
    xm, ym = d.cfg.region.x_margin, d.cfg.region.y_margin
    assert d.ctl.map(xm, ym) == (0.0, 0.0)
    br = d.ctl.map(1 - xm, 1 - ym)
    assert abs(br[0] - (d.mouse.width - 1)) < 1e-6
    assert abs(br[1] - (d.mouse.height - 1)) < 1e-6


def test_positions_outside_the_region_clamp_to_the_edge():
    d = Driver()
    assert d.ctl.map(-0.5, -0.5) == (0.0, 0.0)
    assert d.ctl.map(1.5, 1.5) == (d.mouse.width - 1, d.mouse.height - 1)


# -- cursor anchor ---------------------------------------------------------

def _hand_of(pose, label="Right"):
    return build_hand(pose.image, ASPECT, pose.world, label)


def test_the_cursor_rides_the_middle_of_the_index_finger():
    h = _hand_of(syn.point())
    mid, mcp, tip = h.anchor("index_mid"), h.raw[INDEX_MCP], h.raw[INDEX_TIP]
    assert np.allclose(mid, (mcp + tip) / 2.0)
    for lo, hi, m in zip(mcp, tip, mid):
        assert min(lo, hi) <= m <= max(lo, hi)


def test_the_midpoint_travels_less_than_the_fingertip_when_the_finger_bends():
    """Why the cursor sits mid-finger rather than on the tip: bending the index
    swings the tip a long way while the knuckle stays put, so the midpoint moves
    half as far and the cursor does not lurch as the finger folds."""
    straight = _hand_of(syn.point(syn.STRAIGHT))
    bent = _hand_of(syn.point((0.0, 45.0, 30.0)))

    def shift(name):
        return float(np.linalg.norm(bent.anchor(name) - straight.anchor(name)))

    assert shift("index_tip") > 0.0, "precondition: bending moved the tip"
    assert shift("index_mid") < shift("index_tip")


def test_an_unknown_anchor_name_falls_back_to_the_midpoint():
    h = _hand_of(syn.point())
    assert np.allclose(h.anchor("nonsense"), h.anchor("index_mid"))


def test_the_anchor_is_configurable():
    h = _hand_of(syn.point())
    assert np.allclose(h.anchor("index_tip"), h.raw[INDEX_TIP])
    assert np.allclose(h.anchor("index_mcp"), h.raw[INDEX_MCP])


# -- cursor movement -------------------------------------------------------

def test_pointing_moves_the_cursor():
    d = Driver()
    start = d.mouse.position
    for i in range(10):
        d.one(syn.translate(syn.point(), dx=0.015 * i), Gesture.POINT)
    assert d.mouse.position != start


def test_a_pinch_keeps_steering_so_a_drag_can_go_somewhere():
    d = Driver()
    d.idle(0.3, gesture=Gesture.POINT)
    before = d.mouse.position
    for i in range(10):
        d.one(syn.translate(syn.pinch(), dx=0.015 * i), Gesture.PINCH)
    assert d.mouse.position != before


def test_other_poses_leave_the_cursor_where_it_is():
    """Scrolling and right-clicking move the hand a long way; the cursor must
    stay put or every scroll would fling it across the screen."""
    d = Driver()
    d.idle(0.3, gesture=Gesture.POINT)
    parked = d.mouse.position
    for i in range(10):
        d.one(syn.translate(syn.three(), dx=0.03 * i), Gesture.THREE)
    assert d.mouse.position == parked


def test_the_cursor_does_not_move_while_disarmed():
    d = Driver(armed=False)
    start = d.mouse.position
    for i in range(10):
        d.one(syn.translate(syn.point(), dx=0.02 * i), Gesture.POINT)
    assert d.mouse.position == start


# -- which hand has the cursor --------------------------------------------

def test_the_right_hand_takes_the_cursor_when_both_arrive_together():
    d = Driver()
    st = d.feed([("Left", syn.peace(), Gesture.SCROLL),
                 ("Right", syn.point(), Gesture.POINT)])
    assert st.acting == "Right"


def test_the_right_hand_alone_gets_the_cursor():
    assert Driver().one(syn.point(), Gesture.POINT, label="Right").acting == "Right"


def test_the_left_hand_alone_gets_the_cursor():
    assert Driver().one(syn.point(), Gesture.POINT, label="Left").acting == "Left"


def test_a_second_hand_entering_does_not_steal_the_cursor():
    """The rule that matters in use: the cursor must not jump hands mid-motion
    just because the other hand wandered into frame."""
    d = Driver()
    assert d.one(syn.point(), Gesture.POINT, label="Left").acting == "Left"
    st = None
    for _ in range(12):
        st = d.feed([("Left", syn.point(), Gesture.POINT),
                     ("Right", syn.point(), Gesture.SCROLL)])
    assert st.acting == "Left", "the right hand stole the cursor"


def test_the_primary_hand_only_wins_when_the_cursor_is_unclaimed():
    d = Driver()
    d.one(syn.point(), Gesture.POINT, label="Left")
    d.ctl.toggle_armed()                       # drop the claim
    d.ctl.toggle_armed()
    st = d.feed([("Left", syn.point(), Gesture.POINT),
                 ("Right", syn.point(), Gesture.SCROLL)])
    assert st.acting == "Right"


# -- losing the acting hand ------------------------------------------------

def test_control_disarms_when_the_acting_hand_leaves():
    d = Driver()
    d.idle(0.3, gesture=Gesture.POINT)
    assert d.ctl.armed
    for _ in range(int(0.9 / (1 / 30))):
        d.feed([])
    assert not d.ctl.armed


def test_a_dropped_frame_does_not_disarm():
    """Detection misses the odd frame; control that fell out on every blink
    would be unusable."""
    d = Driver()
    d.idle(0.3, gesture=Gesture.POINT)
    for _ in range(3):
        d.feed([])
    assert d.ctl.armed
    d.one(syn.point(), Gesture.POINT)
    assert d.ctl.armed


def test_the_other_hand_does_not_inherit_control_when_the_first_leaves():
    """Control belongs to the hand that took it. If that hand goes, control ends
    rather than quietly changing owner -- even with the other hand right there
    and pointing."""
    d = Driver()
    d.one(syn.point(), Gesture.POINT, label="Left")
    parked = d.mouse.position
    for _ in range(int(0.9 / (1 / 30))):
        d.feed([("Right", syn.point(), Gesture.POINT)])
    assert not d.ctl.armed
    assert "Left hand left the frame" in d.ctl.events
    assert d.mouse.position == parked, "the right hand drove the cursor anyway"


def test_the_cursor_freezes_while_the_acting_hand_is_missing():
    d = Driver()
    d.idle(0.3, gesture=Gesture.POINT)
    parked = d.mouse.position
    for _ in range(3):
        d.feed([])
    assert d.mouse.position == parked


def test_a_held_button_is_released_the_moment_the_hand_vanishes():
    d = Driver()
    for _ in range(20):
        d.one(syn.pinch(), Gesture.PINCH)
    assert "left down" in d.log
    d.feed([])
    assert d.log[-1] == "left up"
