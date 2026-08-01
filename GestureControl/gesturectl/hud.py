"""Preview window overlay: skeletons, active region and status readout."""

from __future__ import annotations

import cv2
import numpy as np

from .config import Config
from .controller import Status
from .gestures import INDEX_TIP, CONNECTIONS, Gesture, Hand

FONT = cv2.FONT_HERSHEY_SIMPLEX

GREEN = (120, 230, 120)
RED = (90, 90, 240)
AMBER = (60, 190, 245)
BLUE = (235, 180, 90)
WHITE = (240, 240, 240)
GREY = (150, 150, 150)
DARK = (35, 35, 35)

GESTURE_HELP = {
    Gesture.POINT: "move cursor",
    Gesture.PINCH: "click / drag",
    Gesture.SCROLL: "scroll",
    Gesture.THREE: "right click",
    Gesture.FIST: "rest (no action)",
    Gesture.OPEN_PALM: "rest (no action)",
    Gesture.PINKY_UP: "drop pinky to close window",
    Gesture.UNKNOWN: "-",
    Gesture.NONE: "-",
}


def gesture_help(gesture: Gesture, toggle_pose: Gesture) -> str:
    """The toggle pose is configurable, so its label is resolved at draw time."""
    if gesture == toggle_pose:
        return "hold to arm/disarm"
    return GESTURE_HELP.get(gesture, "-")


def draw_hand(frame: np.ndarray, hand: Hand, status: Status) -> None:
    h, w = frame.shape[:2]
    acting = any(s.label == hand.label and s.acting for s in status.hands)
    if not status.armed:
        colour = GREY
    elif status.resizing:
        colour = BLUE
    else:
        colour = GREEN if acting else (110, 140, 110)

    pts = [(int(x * w), int(y * h)) for x, y in hand.raw]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], colour, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        r = 5 if i in (4, INDEX_TIP) else 3
        cv2.circle(frame, p, r, WHITE, -1, cv2.LINE_AA)

    cv2.putText(frame, hand.label, (pts[0][0] - 18, pts[0][1] + 26),
                FONT, 0.44, colour, 1, cv2.LINE_AA)


def draw_resize_link(frame: np.ndarray, a: Hand, b: Hand) -> None:
    """Show the span the resize gesture is measuring."""
    h, w = frame.shape[:2]
    pa = (int(a.raw[INDEX_TIP][0] * w), int(a.raw[INDEX_TIP][1] * h))
    pb = (int(b.raw[INDEX_TIP][0] * w), int(b.raw[INDEX_TIP][1] * h))
    cv2.line(frame, pa, pb, BLUE, 2, cv2.LINE_AA)
    for p in (pa, pb):
        cv2.circle(frame, p, 9, BLUE, 2, cv2.LINE_AA)
    mid = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)
    cv2.putText(frame, "RESIZE", (mid[0] - 34, mid[1] - 14), FONT, 0.52, BLUE, 2, cv2.LINE_AA)


def draw_debug(frame: np.ndarray, hand: Hand, cfg: Config) -> None:
    """Live per-finger bend angles, for tuning the extension thresholds.

    A finger counts as extended below `extend_below` and curled above
    `curl_above`; between the two it holds whatever it was. Watching the real
    numbers is the quickest way to see why a pose is not being recognised.
    """
    h, w = frame.shape[:2]
    x = int(hand.raw[0][0] * w) - 46
    y = min(int(hand.raw[0][1] * h) + 46, h - 92)
    x = max(4, min(x, w - 116))

    cv2.putText(frame, f"{hand.label} curl", (x, y - 6), FONT, 0.38, BLUE, 1, cv2.LINE_AA)
    for i, name in enumerate(("thumb", "index", "middle", "ring", "pinky")):
        curl = hand.curls.get(name, 0.0)
        extended = hand.extended.get(name, False)
        tone = GREEN if extended else GREY
        row = f"{name[:3]:<4}{curl:5.0f}"
        cv2.putText(frame, row, (x, y + 14 + i * 15), FONT, 0.38, tone, 1, cv2.LINE_AA)
    cv2.putText(frame, f"pinch {hand.pinch_distance:.2f}", (x, y + 14 + 5 * 15),
                FONT, 0.38, GREY, 1, cv2.LINE_AA)


def draw_region(frame: np.ndarray, cfg: Config, active: bool) -> None:
    h, w = frame.shape[:2]
    x0 = int(cfg.region.x_margin * w)
    y0 = int(cfg.region.y_margin * h)
    cv2.rectangle(frame, (x0, y0), (w - x0, h - y0), GREEN if active else GREY, 1, cv2.LINE_AA)


def draw_status(frame: np.ndarray, status: Status, fps: float, dry_run: bool) -> None:
    h, w = frame.shape[:2]
    panel = frame[0:104, 0:w].copy()
    cv2.rectangle(panel, (0, 0), (w, 104), DARK, -1)
    cv2.addWeighted(panel, 0.62, frame[0:104, 0:w], 0.38, 0, frame[0:104, 0:w])

    state, colour = ("ARMED", GREEN) if status.armed else ("DISARMED", RED)
    if dry_run:
        state += "  (dry run)"
    cv2.circle(frame, (22, 28), 9, colour, -1, cv2.LINE_AA)
    cv2.putText(frame, state, (42, 34), FONT, 0.62, colour, 2, cv2.LINE_AA)

    if status.resizing:
        cv2.putText(frame, "two-handed resize", (42, 62), FONT, 0.55, BLUE, 2, cv2.LINE_AA)
        cv2.putText(frame, "move hands apart or together", (42, 86),
                    FONT, 0.45, GREY, 1, cv2.LINE_AA)
    elif status.hands:
        y = 62
        for hs in status.hands:
            mark = ">" if hs.acting else " "
            tone = WHITE if hs.acting else GREY
            cv2.putText(frame, f"{mark} {hs.label:<5} {hs.gesture.value}", (42, y),
                        FONT, 0.5, tone, 1, cv2.LINE_AA)
            if hs.acting:
                cv2.putText(frame, gesture_help(hs.gesture, status.toggle_pose), (250, y),
                            FONT, 0.44, GREY, 1, cv2.LINE_AA)
            y += 22
    else:
        cv2.putText(frame, "no hands", (42, 62), FONT, 0.55, WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, "show your hand", (42, 86), FONT, 0.45, GREY, 1, cv2.LINE_AA)

    cv2.putText(frame, f"{fps:4.1f} fps", (w - 100, 30), FONT, 0.5, GREY, 1, cv2.LINE_AA)
    if status.dragging:
        cv2.putText(frame, "DRAG", (w - 100, 56), FONT, 0.55, AMBER, 2, cv2.LINE_AA)

    if status.arm_progress > 0:
        bar_w = int(220 * status.arm_progress)
        cv2.rectangle(frame, (w // 2 - 110, h - 34), (w // 2 + 110, h - 18), GREY, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (w // 2 - 110, h - 34), (w // 2 - 110 + bar_w, h - 18), AMBER, -1)
        cv2.putText(frame, "hold to toggle", (w // 2 - 62, h - 42), FONT, 0.42, AMBER, 1, cv2.LINE_AA)

    for i, ev in enumerate(reversed(status.events[-4:])):
        cv2.putText(frame, ev, (12, h - 70 - i * 18), FONT, 0.42, GREY, 1, cv2.LINE_AA)

    cv2.putText(frame, "q quit   space arm/disarm   r recentre",
                (12, h - 8), FONT, 0.42, GREY, 1, cv2.LINE_AA)
