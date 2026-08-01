"""Build synthetic hands so gesture logic can be tested without a camera.

Each finger is posed by forward kinematics from three joint bends, which lets a
test describe a hand the way a hand actually moves: "straight, but angled 75
degrees down at the knuckle" is a pose, not a pile of coordinates.

The model is an upright right hand -- wrist low, knuckles in a row above it,
fingers extending further up -- built in isotropic units and projected to
normalised image coordinates on the way out. Bending a joint rotates the finger
toward the camera, which is what shortens its projection and what used to fool
distance-based finger detection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

ASPECT = 640 / 480

WRIST = (0.52, 0.90)
MCP_Y = 0.62
MCP_X = {"index": 0.44, "middle": 0.50, "ring": 0.555, "pinky": 0.61}
SEGMENTS = (0.080, 0.060, 0.050)   # proximal, middle, distal

# (knuckle, middle joint, end joint) bends in degrees.
STRAIGHT = (0.0, 0.0, 0.0)
CURLED = (45.0, 100.0, 65.0)
ANGLED_DOWN = (75.0, 5.0, 5.0)     # straight fingers aimed at the lens
ANGLED_HARD = (90.0, 5.0, 5.0)
RELAXED = (25.0, 30.0, 20.0)

THUMB_OUT = [(0.46, 0.82), (0.41, 0.76), (0.36, 0.71), (0.31, 0.67)]
THUMB_IN = [(0.48, 0.83), (0.46, 0.78), (0.45, 0.74), (0.54, 0.71)]


@dataclass
class LM:
    x: float
    y: float
    z: float = 0.0


@dataclass
class Pose:
    """A hand in both the forms MediaPipe hands us."""

    image: list[LM]    # normalised to the frame, x compressed by aspect
    world: list[LM]    # metric 3D, centred on the hand

    def __post_init__(self) -> None:
        assert len(self.image) == 21 and len(self.world) == 21


def _direction(theta_deg: float) -> tuple[float, float, float]:
    """Finger direction after bending `theta` from straight-up toward the lens."""
    t = math.radians(theta_deg)
    return (0.0, -math.cos(t), -math.sin(t))


def _finger(name: str, bends: tuple[float, float, float]) -> list[tuple[float, float, float]]:
    x = MCP_X[name]
    joints = [(x, MCP_Y, 0.0)]
    theta = 0.0
    for length, bend in zip(SEGMENTS, bends):
        theta += bend
        dx, dy, dz = _direction(theta)
        px, py, pz = joints[-1]
        joints.append((px + length * dx, py + length * dy, pz + length * dz))
    return joints


def make_pose(
    index: tuple[float, float, float] = CURLED,
    middle: tuple[float, float, float] = CURLED,
    ring: tuple[float, float, float] = CURLED,
    pinky: tuple[float, float, float] = CURLED,
    thumb_out: bool = False,
    thumb_tip: tuple[float, float] | None = None,
) -> Pose:
    pts: list[tuple[float, float, float]] = [(WRIST[0], WRIST[1], 0.0)]

    thumb = [(x, y, 0.0) for x, y in (THUMB_OUT if thumb_out else THUMB_IN)]
    if thumb_tip is not None:
        thumb[3] = (thumb_tip[0], thumb_tip[1], 0.0)
    pts += thumb

    for name, bends in (("index", index), ("middle", middle),
                        ("ring", ring), ("pinky", pinky)):
        pts += _finger(name, bends)

    image = [LM(x / ASPECT, y, z) for x, y, z in pts]
    cx = sum(p[0] for p in pts) / 21
    cy = sum(p[1] for p in pts) / 21
    cz = sum(p[2] for p in pts) / 21
    world = [LM(x - cx, y - cy, z - cz) for x, y, z in pts]
    return Pose(image, world)


def index_tip_of(bends: tuple[float, float, float] = STRAIGHT) -> tuple[float, float]:
    j = _finger("index", bends)[3]
    return j[0], j[1]


# -- transforms ------------------------------------------------------------

def translate(pose: Pose, dx: float = 0.0, dy: float = 0.0) -> Pose:
    """Move the hand across the frame. World landmarks are centred on the hand,
    so they are unchanged by definition."""
    return Pose([LM(p.x + dx, p.y + dy, p.z) for p in pose.image], list(pose.world))


def mirror(pose: Pose) -> Pose:
    """Flip left-to-right, turning a right hand into a left one."""
    return Pose(
        [LM(1.0 - p.x, p.y, p.z) for p in pose.image],
        [LM(-p.x, p.y, p.z) for p in pose.world],
    )


def rotate(pose: Pose, degrees: float) -> Pose:
    """Tilt the hand in the image plane, about the wrist."""
    rad = math.radians(degrees)
    cos, sin = math.cos(rad), math.sin(rad)
    ox, oy = pose.image[0].x * ASPECT, pose.image[0].y

    image = []
    for p in pose.image:
        x, y = p.x * ASPECT - ox, p.y - oy
        image.append(LM((x * cos - y * sin + ox) / ASPECT, x * sin + y * cos + oy, p.z))
    world = [LM(p.x * cos - p.y * sin, p.x * sin + p.y * cos, p.z) for p in pose.world]
    return Pose(image, world)


# -- named poses -----------------------------------------------------------

def fist() -> Pose:
    return make_pose()


def point(bends: tuple[float, float, float] = STRAIGHT) -> Pose:
    return make_pose(index=bends)


def peace(bends: tuple[float, float, float] = STRAIGHT) -> Pose:
    return make_pose(index=bends, middle=bends)


def three() -> Pose:
    return make_pose(index=STRAIGHT, middle=STRAIGHT, ring=STRAIGHT)


def open_palm() -> Pose:
    return make_pose(STRAIGHT, STRAIGHT, STRAIGHT, STRAIGHT, thumb_out=True)


def pinky_up() -> Pose:
    return make_pose(pinky=STRAIGHT)


def pinch() -> Pose:
    tx, ty = index_tip_of(STRAIGHT)
    return make_pose(index=STRAIGHT, thumb_tip=(tx + 0.004, ty + 0.006))


def pinch_open() -> Pose:
    return make_pose(index=STRAIGHT, thumb_out=True)
