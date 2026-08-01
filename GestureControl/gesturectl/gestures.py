"""Turn 21 hand landmarks into a stable, named gesture.

Whether a finger counts as extended is decided from the *angles* at its middle
and end joints, not from how far the fingertip sits from the wrist. Distance
looks fine when the hand is flat to the camera but collapses as soon as you
angle your fingers toward the lens, because the finger projects shorter -- a
straight finger tilted about 70 degrees measures the same as a curled one. Joint
angles do not care which way the hand is pointing.

The angles are taken from MediaPipe's world landmarks, which are metric 3D and
so are unaffected by perspective; the 2D landmarks are still used for position
and for pinch distance, where image space is what matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Joint chains used for the curl measurement, base -> tip.
FINGER_JOINTS = {
    "thumb": (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}
FINGERS = ("index", "middle", "ring", "pinky")


class Gesture(str, Enum):
    NONE = "NONE"
    POINT = "POINT"
    PINCH = "PINCH"
    SCROLL = "SCROLL"
    THREE = "THREE"
    FIST = "FIST"
    OPEN_PALM = "OPEN_PALM"
    PINKY_UP = "PINKY_UP"
    UNKNOWN = "UNKNOWN"


@dataclass
class Hand:
    """Landmarks plus the derived features the classifier reads."""

    pts: np.ndarray              # (21, 2) aspect-corrected normalised units
    raw: np.ndarray              # (21, 2) original normalised units, for drawing
    palm: float                  # wrist -> middle knuckle, the hand's scale
    curls: dict[str, float]      # per finger, total bend in degrees
    pinch_distance: float        # thumb-to-index gap, in palm widths
    label: str = "Right"         # which hand, after mirroring
    extended: dict[str, bool] = field(default_factory=dict)

    def point(self, idx: int) -> np.ndarray:
        return self.raw[idx]

    @property
    def index_tip(self) -> np.ndarray:
        return self.pts[INDEX_TIP]


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(min(1.0, max(-1.0, cos)))))


def _curl(pts3: np.ndarray, chain: tuple[int, int, int, int]) -> float:
    """Total bend across a finger's two outer joints, in degrees.

    Only the middle and end joints are counted. The knuckle is deliberately
    left out: angling a straight finger down at the knuckle is how you aim your
    hand, not how you fold a finger away, and counting it is what made the
    scroll pose collapse into a fist.
    """
    base, mid, dist, tip = chain
    v1 = pts3[mid] - pts3[base]
    v2 = pts3[dist] - pts3[mid]
    v3 = pts3[tip] - pts3[dist]
    return _angle_between(v1, v2) + _angle_between(v2, v3)


def build_hand(
    landmarks,
    aspect: float,
    world_landmarks=None,
    label: str = "Right",
) -> Hand:
    raw = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float64)
    pts = raw.copy()
    pts[:, 0] *= aspect

    if world_landmarks is not None:
        pts3 = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks], dtype=np.float64)
    else:
        # Fallback: normalised landmarks carry a relative z on roughly the same
        # scale as x, so they still give usable angles if world points are absent.
        pts3 = np.array([[lm.x * aspect, lm.y, lm.z] for lm in landmarks], dtype=np.float64)

    palm = float(np.linalg.norm(pts[MIDDLE_MCP] - pts[WRIST]))
    if palm < 1e-6:
        palm = 1e-6

    curls = {name: _curl(pts3, chain) for name, chain in FINGER_JOINTS.items()}
    pinch = float(np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP]) / palm)

    return Hand(pts=pts, raw=raw, palm=palm, curls=curls, pinch_distance=pinch, label=label)


class FingerTracker:
    """Decide which fingers are extended, with hysteresis.

    One tracker per hand. A finger held near the threshold would otherwise flip
    state frame to frame, which surfaces as the gesture flickering between
    neighbouring poses.
    """

    def __init__(self, extend_below: float = 60.0, curl_above: float = 95.0) -> None:
        self.extend_below = extend_below
        self.curl_above = curl_above
        self.state = {name: False for name in FINGER_JOINTS}

    def apply(self, hand: Hand) -> Hand:
        for name, curl in hand.curls.items():
            if self.state[name]:
                if curl > self.curl_above:
                    self.state[name] = False
            elif curl < self.extend_below:
                self.state[name] = True
        hand.extended = dict(self.state)
        return hand

    def reset(self) -> None:
        for name in self.state:
            self.state[name] = False


def classify(hand: Hand, pinch_engaged: bool, max_index_curl: float = 110.0) -> Gesture:
    e = hand.extended
    index, middle, ring, pinky = e["index"], e["middle"], e["ring"], e["pinky"]

    # A closed fist parks the thumb tip beside the curled index tip, which reads
    # as a pinch on distance alone. Requiring the index to be reasonably straight
    # separates a real pinch from a resting fist, while still allowing the
    # partly-bent index most people actually pinch with.
    if pinch_engaged and hand.curls["index"] < max_index_curl and not (middle and ring and pinky):
        return Gesture.PINCH

    up = sum((index, middle, ring, pinky))
    if up == 0:
        return Gesture.FIST
    if pinky and not index and not middle and not ring:
        return Gesture.PINKY_UP
    if index and not middle and not ring and not pinky:
        return Gesture.POINT
    if index and middle and not ring and not pinky:
        return Gesture.SCROLL
    if index and middle and ring and not pinky:
        return Gesture.THREE
    if up == 4:
        # An open hand is inert, so it does not need a splayed thumb to qualify.
        # Being forgiving here is the point: transitions between the action
        # gestures often flash through an open hand, and landing on a pose that
        # does nothing is exactly what should happen when they do.
        return Gesture.OPEN_PALM
    return Gesture.UNKNOWN


class Stabiliser:
    """Suppress single-frame flickers by requiring N agreeing frames."""

    def __init__(self, stable_frames: int = 3) -> None:
        self.stable_frames = max(1, stable_frames)
        self.current = Gesture.NONE
        self._candidate = Gesture.NONE
        self._count = 0

    def update(self, gesture: Gesture) -> Gesture:
        if gesture == self.current:
            self._candidate, self._count = gesture, 0
            return self.current

        if gesture == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = gesture, 1

        # PINCH must take effect immediately or clicks feel unresponsive.
        threshold = 1 if gesture == Gesture.PINCH else self.stable_frames
        if self._count >= threshold:
            self.current = gesture
            self._count = 0
        return self.current

    def reset(self) -> None:
        self.current = Gesture.NONE
        self._candidate = Gesture.NONE
        self._count = 0


class PinchDetector:
    """Hysteresis latch so a hand hovering at the threshold does not chatter."""

    def __init__(self, engage: float, release: float) -> None:
        self.engage = engage
        self.release = release
        self.engaged = False

    def update(self, distance: float) -> bool:
        if self.engaged:
            if distance > self.release:
                self.engaged = False
        elif distance < self.engage:
            self.engaged = True
        return self.engaged

    def reset(self) -> None:
        self.engaged = False
