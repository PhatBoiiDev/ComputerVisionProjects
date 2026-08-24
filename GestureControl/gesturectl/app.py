"""Capture loop: camera -> landmarks -> gestures -> cursor and windows."""

from __future__ import annotations

import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from . import hud
from .config import Config
from .controller import GestureController, TrackedHand
from .gestures import FingerTracker, PinchDetector, Stabiliser, build_hand, classify
from .macos_input import KeyboardController, MouseController, WindowController, accessibility_trusted

WINDOW = "Gesture Control"
LABELS = ("Left", "Right")


class CameraError(RuntimeError):
    pass


class HandState:
    """Per-hand detection state, kept across frames so hysteresis works."""

    def __init__(self, cfg: Config) -> None:
        g = cfg.gesture
        self.fingers = FingerTracker(g.extend_below, g.curl_above)
        self.stabiliser = Stabiliser(g.stable_frames)
        self.pinch = PinchDetector(cfg.pinch.engage, cfg.pinch.release)

    def reset(self) -> None:
        self.fingers.reset()
        self.stabiliser.reset()
        self.pinch.reset()


def open_camera(cfg: Config) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(cfg.camera.index)
    if not cap.isOpened():
        raise CameraError(
            f"Could not open camera index {cfg.camera.index}.\n"
            "On macOS this is almost always a missing Camera permission for the app\n"
            "running this script (Terminal, iTerm, VS Code...).\n"
            "Grant it in System Settings > Privacy & Security > Camera, then relaunch\n"
            "that app completely."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
    cap.set(cv2.CAP_PROP_FPS, cfg.camera.fps)
    return cap


def build_landmarker(cfg: Config) -> vision.HandLandmarker:
    g = cfg.gesture
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(cfg.model_file)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=max(1, min(2, cfg.hands.count)),
        min_hand_detection_confidence=g.min_detection_confidence,
        min_hand_presence_confidence=g.min_presence_confidence,
        min_tracking_confidence=g.min_tracking_confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


def assign_labels(result) -> list[tuple[str, int]]:
    """Pair each detection with a hand label, as (label, detection index).

    MediaPipe works out handedness assuming an unmirrored image, but the frame
    is flipped for selfie view before detection, so every label arrives
    backwards and has to be swapped back.
    """
    entries = []
    for i, handedness in enumerate(result.handedness):
        raw = handedness[0].category_name if handedness else "Right"
        score = handedness[0].score if handedness else 0.0
        entries.append(("Right" if raw == "Left" else "Left", score, i))

    # Two detections occasionally come back with the same label; the weaker one
    # is reassigned so both hands stay independently addressable.
    if len(entries) == 2 and entries[0][0] == entries[1][0]:
        keep, other = sorted(entries, key=lambda e: -e[1])
        flipped = "Left" if keep[0] == "Right" else "Right"
        entries = [keep, (flipped, other[1], other[2])]

    return [(label, idx) for label, _, idx in entries]


def run(cfg: Config, show_preview: bool = True, dry_run: bool = False,
        debug: bool = False) -> int:
    if not dry_run and not accessibility_trusted():
        print(
            "Accessibility permission is not granted, so the cursor cannot be moved.\n"
            "Grant it in System Settings > Privacy & Security > Accessibility for the\n"
            "app running this script, then relaunch that app completely.\n"
            "Run with --dry-run to test gesture detection without controlling the cursor.",
            file=sys.stderr,
        )
        return 2

    cap = open_camera(cfg)
    landmarker = build_landmarker(cfg)
    mouse = MouseController(dry_run=dry_run)
    controller = GestureController(
        cfg, mouse,
        keyboard=KeyboardController(dry_run=dry_run),
        windows=WindowController(dry_run=dry_run),
    )
    states = {label: HandState(cfg) for label in LABELS}
    controller.pump.start()

    rate = f"{cfg.cursor.poll_hz:.0f} Hz" if cfg.cursor.threaded else "camera rate"
    print(f"Screen: {mouse.width}x{mouse.height}   preview: {show_preview}   dry run: {dry_run}")
    print(f"Cursor: {rate}   camera: {cfg.camera.fps:.0f} fps")
    print(f"Hold a {cfg.gesture.toggle_pose} for ~{cfg.gesture.arm_toggle_hold:.0f}s "
          "to arm or disarm control.")
    if show_preview:
        print("In the preview window: q quits, space arms/disarms, r recentres.")
    else:
        print("Press Ctrl+C to quit.")

    fps = 0.0
    last_t = time.perf_counter()
    start = last_t
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)  # mirror, so moving right moves the cursor right
            h, w = frame.shape[:2]
            aspect = w / h
            now = time.perf_counter()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((now - start) * 1000)
            # MediaPipe rejects a timestamp that does not advance.
            timestamp_ms = max(timestamp_ms, frame_index)
            frame_index = timestamp_ms + 1

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            tracked: list[TrackedHand] = []
            seen: set[str] = set()
            for label, idx in assign_labels(result):
                world = None
                if result.hand_world_landmarks and idx < len(result.hand_world_landmarks):
                    world = result.hand_world_landmarks[idx]
                hand = build_hand(result.hand_landmarks[idx], aspect, world, label)

                state = states[label]
                state.fingers.apply(hand)
                engaged = state.pinch.update(hand.pinch_distance)
                gesture = state.stabiliser.update(
                    classify(hand, engaged, cfg.pinch.max_index_curl)
                )
                tracked.append(TrackedHand(hand, gesture))
                seen.add(label)

            for label in LABELS:
                if label not in seen:
                    states[label].reset()

            if tracked:
                status = controller.update(tracked, now)
            else:
                status = controller.hands_lost(now)

            dt = now - last_t
            last_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            if show_preview:
                for t in tracked:
                    hud.draw_hand(frame, t.hand, status)
                    if debug:
                        hud.draw_debug(frame, t.hand, cfg)
                hud.draw_region(frame, cfg, status.armed)
                if status.resizing and len(tracked) == 2:
                    hud.draw_resize_link(frame, tracked[0].hand, tracked[1].hand)
                hud.draw_status(frame, status, fps, dry_run)
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    controller.toggle_armed()
                if key == ord("r"):
                    controller.recentre()
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        # Stop the pump before releasing buttons, so nothing is still posting
        # movement while the mouse is being put back into a clean state.
        controller.pump.stop()
        mouse.release_all()
        cap.release()
        landmarker.close()
        if show_preview:
            cv2.destroyAllWindows()
    return 0
