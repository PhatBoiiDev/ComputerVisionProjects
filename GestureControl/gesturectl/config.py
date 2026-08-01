"""Tunable settings, loaded from an optional JSON file."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "hand_landmarker.task"
DEFAULT_CONFIG = PROJECT_ROOT / "config.json"


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class RegionConfig:
    """Fraction of the frame trimmed off each side to form the active box.

    The active box maps onto the whole screen, so you reach every corner with
    small hand movements near the centre of the camera's view.
    """

    x_margin: float = 0.22
    y_margin: float = 0.16


@dataclass
class SmoothingConfig:
    min_cutoff: float = 1.3
    beta: float = 0.045
    d_cutoff: float = 1.0


@dataclass
class PinchConfig:
    """Thumb-to-index distance as a fraction of palm size (hysteresis pair)."""

    engage: float = 0.42
    release: float = 0.60
    # Rejects a resting fist, whose thumb and index tips also sit close together.
    max_index_curl: float = 110.0


@dataclass
class ClickConfig:
    max_duration: float = 0.45
    drag_delay: float = 0.32
    drag_move_px: float = 22.0
    double_gap: float = 0.42
    cooldown: float = 0.22


@dataclass
class ScrollConfig:
    gain: float = 2.6
    deadzone_px: float = 2.0
    invert: bool = False
    horizontal: bool = True


@dataclass
class GestureConfig:
    stable_frames: int = 3
    # A fist is the arm/disarm switch because it is the one pose you cannot
    # drift into: every action gesture keeps the index finger extended, so
    # moving between them never passes through a closed hand.
    toggle_pose: str = "FIST"
    arm_toggle_hold: float = 1.0
    # Total bend across a finger's two outer joints, in degrees. The gap between
    # the two is hysteresis: a finger resting near one threshold would otherwise
    # flip state frame to frame and make the gesture flicker.
    extend_below: float = 55.0
    curl_above: float = 100.0
    min_detection_confidence: float = 0.6
    min_presence_confidence: float = 0.6
    min_tracking_confidence: float = 0.6


@dataclass
class HandsConfig:
    """Two hands are tracked; one drives the cursor at a time."""

    count: int = 2
    # Which hand owns the cursor when both are making an actionable gesture.
    primary: str = "Right"


@dataclass
class ResizeConfig:
    """Both index fingers out: their separation scales the focused window."""

    enabled: bool = True
    min_scale: float = 0.3
    max_scale: float = 3.5
    min_width: int = 200
    min_height: int = 150
    # Ignore separation changes below this fraction, so a steady hold is steady.
    deadzone: float = 0.02
    update_interval: float = 0.05


@dataclass
class ExitConfig:
    """Raise the pinky alone, then drop it, to close the focused window."""

    enabled: bool = True
    # The pose must be held this long before dropping it does anything, so a
    # pinky passing through on the way to another gesture cannot trigger it.
    arm_hold: float = 0.35
    cooldown: float = 1.5
    shortcut: str = "cmd+w"


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    region: RegionConfig = field(default_factory=RegionConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    pinch: PinchConfig = field(default_factory=PinchConfig)
    click: ClickConfig = field(default_factory=ClickConfig)
    scroll: ScrollConfig = field(default_factory=ScrollConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    hands: HandsConfig = field(default_factory=HandsConfig)
    resize: ResizeConfig = field(default_factory=ResizeConfig)
    exit_gesture: ExitConfig = field(default_factory=ExitConfig)

    # "index_mcp" is steadier and does not shift when you pinch;
    # "index_tip" feels more like pointing but jitters more.
    cursor_anchor: str = "index_mcp"
    model_path: str = str(DEFAULT_MODEL)
    start_armed: bool = False

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        cfg = cls()
        p = Path(path) if path else DEFAULT_CONFIG
        if p.exists():
            with open(p) as fh:
                _merge(cfg, json.load(fh))
        elif path:
            raise FileNotFoundError(f"config file not found: {p}")
        return cfg

    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path) if path else DEFAULT_CONFIG
        p.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return p


def _merge(target, data: dict) -> None:
    """Overlay a nested dict onto a dataclass instance, ignoring unknown keys."""
    known = {f.name: f for f in fields(target)}
    for key, value in data.items():
        if key not in known:
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(target, key, value)
