"""Drive the cursor and windows from a stream of classified gestures.

Both hands are tracked. Single-hand actions run for one hand at a time -- the
"acting" hand -- so two raised hands can never fire two clicks at once, while
two-handed gestures are checked first and suppress single-hand actions entirely
for as long as they are held.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .filters import Point2DFilter
from .gestures import INDEX_MCP, INDEX_TIP, Gesture, Hand
from .macos_input import KeyboardController, MouseController, WindowController

# Poses that mean "this hand wants to do something".
ACTIONABLE = (Gesture.POINT, Gesture.PINCH, Gesture.SCROLL,
              Gesture.THREE, Gesture.PINKY_UP)


@dataclass
class TrackedHand:
    hand: Hand
    gesture: Gesture

    @property
    def label(self) -> str:
        return self.hand.label


@dataclass
class HandStatus:
    label: str
    gesture: Gesture
    acting: bool = False


@dataclass
class Status:
    armed: bool = False
    hands: list[HandStatus] = field(default_factory=list)
    resizing: bool = False
    dragging: bool = False
    arm_progress: float = 0.0
    cursor: tuple[float, float] = (0.0, 0.0)
    events: list[str] = field(default_factory=list)
    toggle_pose: Gesture = Gesture.FIST

    @property
    def hand_present(self) -> bool:
        return bool(self.hands)


class ScreenMapper:
    """Maps the active region of the frame onto the whole screen."""

    def __init__(self, cfg: Config, width: int, height: int) -> None:
        self.cfg = cfg
        self.width = width
        self.height = height

    def __call__(self, nx: float, ny: float) -> tuple[float, float]:
        xm, ym = self.cfg.region.x_margin, self.cfg.region.y_margin
        span_x = max(1e-6, 1.0 - 2 * xm)
        span_y = max(1e-6, 1.0 - 2 * ym)
        u = min(1.0, max(0.0, (nx - xm) / span_x))
        v = min(1.0, max(0.0, (ny - ym) / span_y))
        return u * (self.width - 1), v * (self.height - 1)


class HandActions:
    """Everything one hand can do on its own."""

    def __init__(self, cfg: Config, mouse: MouseController, keyboard: KeyboardController,
                 mapper: ScreenMapper, log) -> None:
        self.cfg = cfg
        self.mouse = mouse
        self.keyboard = keyboard
        self.map = mapper
        self.log = log

        s = cfg.smoothing
        self._filter = Point2DFilter(s.min_cutoff, s.beta, s.d_cutoff)
        self._prev_gesture = Gesture.NONE

        self._pinch_active = False
        self._pinch_start_t = 0.0
        self._pinch_start_pos = (0.0, 0.0)
        self.dragging = False
        self._pending_click_t: float | None = None
        self._last_action_t = -10.0

        self._scroll_prev: tuple[float, float] | None = None
        self._scroll_accum = [0.0, 0.0]

        self._pinky_since: float | None = None
        self._last_exit_t = -10.0

    # -- lifecycle ----------------------------------------------------

    def activate(self) -> None:
        """Called when this hand takes over; forget stale motion history."""
        self._filter.reset()
        self._scroll_prev = None

    def release(self, now: float, fire_pending: bool = True) -> None:
        """Give up control cleanly, never leaving a button held."""
        if self.dragging:
            self.mouse.left_up()
            self.dragging = False
            self.log("drag released")
        if fire_pending:
            self._flush_pending_click(now, force=True)
        else:
            self._pending_click_t = None
        self._pinch_active = False
        self._scroll_prev = None
        self._pinky_since = None
        self._prev_gesture = Gesture.NONE

    def _cooling_down(self, now: float) -> bool:
        return (now - self._last_action_t) < self.cfg.click.cooldown

    # -- main update --------------------------------------------------

    def update(self, hand: Hand, gesture: Gesture, now: float) -> tuple[float, float]:
        anchor_idx = INDEX_TIP if self.cfg.cursor_anchor == "index_tip" else INDEX_MCP
        anchor = hand.point(anchor_idx)
        sx, sy = self.map(float(anchor[0]), float(anchor[1]))
        fx, fy = self._filter(sx, sy, now)

        if gesture in (Gesture.POINT, Gesture.PINCH):
            self.mouse.move_to(fx, fy)

        self._handle_pinch(gesture, now, fx, fy)
        self._handle_scroll(gesture, fx, fy)
        self._handle_right_click(gesture, now)
        self._handle_exit(hand, gesture, now)
        self._flush_pending_click(now)

        if gesture != Gesture.SCROLL:
            self._scroll_prev = None
            self._scroll_accum = [0.0, 0.0]

        self._prev_gesture = gesture
        return fx, fy

    # -- click and drag -----------------------------------------------

    def _handle_pinch(self, gesture: Gesture, now: float, fx: float, fy: float) -> None:
        pinching = gesture == Gesture.PINCH

        if pinching and not self._pinch_active:
            self._pinch_active = True
            self._pinch_start_t = now
            self._pinch_start_pos = (fx, fy)
            return

        if pinching:
            if not self.dragging:
                held = now - self._pinch_start_t
                dx = fx - self._pinch_start_pos[0]
                dy = fy - self._pinch_start_pos[1]
                moved = (dx * dx + dy * dy) ** 0.5
                if held >= self.cfg.click.drag_delay or moved >= self.cfg.click.drag_move_px:
                    # An earlier tap may still be held back waiting to see if it
                    # becomes a double click. This pinch is a drag, not a second
                    # tap, so settle that tap as a plain click before dragging.
                    self._flush_pending_click(now, force=True)
                    self.mouse.left_down()
                    self.dragging = True
                    self.log("drag start")
            return

        if self._pinch_active:
            self._pinch_active = False
            if self.dragging:
                self.mouse.left_up()
                self.dragging = False
                self.log("drag end")
                self._last_action_t = now
            elif (now - self._pinch_start_t) <= self.cfg.click.max_duration:
                if not self._cooling_down(now):
                    self._register_tap(now)

    def _register_tap(self, now: float) -> None:
        """Hold the first tap briefly so a second one can upgrade it to a double."""
        if self._pending_click_t is not None:
            self._pending_click_t = None
            self.mouse.click(clicks=2)
            self.log("double click")
            self._last_action_t = now
        else:
            self._pending_click_t = now

    def _flush_pending_click(self, now: float, force: bool = False) -> None:
        if self._pending_click_t is None:
            return
        if force or now - self._pending_click_t >= self.cfg.click.double_gap:
            self._pending_click_t = None
            self.mouse.click()
            self.log("click")
            self._last_action_t = now

    # -- scroll and right click ---------------------------------------

    def _handle_scroll(self, gesture: Gesture, fx: float, fy: float) -> None:
        if gesture != Gesture.SCROLL:
            return
        if self._scroll_prev is None:
            self._scroll_prev = (fx, fy)
            return

        dx = fx - self._scroll_prev[0]
        dy = fy - self._scroll_prev[1]
        self._scroll_prev = (fx, fy)

        cfg = self.cfg.scroll
        if abs(dy) < cfg.deadzone_px:
            dy = 0.0
        if abs(dx) < cfg.deadzone_px or not cfg.horizontal:
            dx = 0.0

        sign = 1.0 if cfg.invert else -1.0
        self._scroll_accum[0] += sign * dy * cfg.gain
        self._scroll_accum[1] += sign * dx * cfg.gain

        step_y = int(self._scroll_accum[0])
        step_x = int(self._scroll_accum[1])
        if step_y or step_x:
            self._scroll_accum[0] -= step_y
            self._scroll_accum[1] -= step_x
            self.mouse.scroll(step_y, step_x)

    def _handle_right_click(self, gesture: Gesture, now: float) -> None:
        if gesture == Gesture.THREE and self._prev_gesture != Gesture.THREE:
            if not self._cooling_down(now):
                self.mouse.right_click()
                self.log("right click")
                self._last_action_t = now

    # -- exit gesture --------------------------------------------------

    def _handle_exit(self, hand: Hand, gesture: Gesture, now: float) -> None:
        """Raise the pinky alone, then flick it down, to close the window.

        It fires on the way *down*, not the way up, so the window closes only
        once you have committed to it -- and only if the pinky itself went down,
        so opening into another pose leaves the window alone.
        """
        cfg = self.cfg.exit_gesture
        if not cfg.enabled:
            return

        if gesture == Gesture.PINKY_UP:
            if self._pinky_since is None:
                self._pinky_since = now
            return

        if self._pinky_since is None:
            return

        held = now - self._pinky_since
        self._pinky_since = None
        pinky_down = not hand.extended.get("pinky", False)
        if not pinky_down or held < cfg.arm_hold:
            return
        if now - self._last_exit_t < cfg.cooldown:
            return

        self._last_exit_t = now
        self._last_action_t = now
        self.keyboard.press(cfg.shortcut)
        self.log(f"exit window ({cfg.shortcut})")


class GestureController:
    def __init__(self, config: Config, mouse: MouseController,
                 keyboard: KeyboardController | None = None,
                 windows: WindowController | None = None) -> None:
        self.cfg = config
        self.mouse = mouse
        self.keyboard = keyboard or KeyboardController(dry_run=mouse.dry_run)
        self.windows = windows or WindowController(dry_run=mouse.dry_run)
        self.armed = config.start_armed

        self.map = ScreenMapper(config, mouse.width, mouse.height)
        self.events: list[str] = []
        self.actions = {
            label: HandActions(config, mouse, self.keyboard, self.map, self._log)
            for label in ("Left", "Right")
        }
        self._acting: str | None = None

        try:
            self.toggle_pose = Gesture(config.gesture.toggle_pose)
        except ValueError:
            self.toggle_pose = Gesture.FIST
            self._log(f"unknown toggle_pose {config.gesture.toggle_pose!r}, using FIST")

        self._toggle_hold_start: float | None = None
        self._arm_latched = False

        self._resize_base_sep: float | None = None
        self._resize_base_frame: tuple[float, float, float, float] | None = None
        self._resize_last_apply = -10.0
        self.resizing = False

    def _log(self, msg: str) -> None:
        self.events.append(msg)
        if len(self.events) > 6:
            del self.events[:-6]

    def recentre(self) -> None:
        for act in self.actions.values():
            act.activate()

    def toggle_armed(self) -> None:
        self.armed = not self.armed
        if not self.armed:
            self._release_all(0.0, fire_pending=False)
        self._log(f"control {'ARMED' if self.armed else 'DISARMED'}")

    def _release_all(self, now: float, fire_pending: bool = True) -> None:
        for act in self.actions.values():
            act.release(now, fire_pending=fire_pending)
        self._acting = None
        self._end_resize()

    # -- main update ---------------------------------------------------

    def hands_lost(self, now: float) -> Status:
        """Fail safe: never leave a button held when tracking drops out."""
        # A pinky raised as the hand leaves the frame is not a flick, so pending
        # exit state is dropped rather than fired.
        self._release_all(now, fire_pending=self.armed)
        self._toggle_hold_start = None
        self._arm_latched = False
        return Status(armed=self.armed, cursor=self.mouse.position,
                      events=list(self.events), toggle_pose=self.toggle_pose)

    def update(self, tracked: list[TrackedHand], now: float) -> Status:
        by_label = {t.label: t for t in tracked}
        gestures = [t.gesture for t in tracked]
        self._handle_arm_toggle(gestures, now)

        resizing = False
        if self.armed and self.cfg.resize.enabled and len(tracked) == 2:
            if all(g == Gesture.POINT for g in gestures):
                resizing = self._handle_resize(tracked, now)

        if resizing:
            # A two-handed gesture owns both hands; nothing else may fire.
            for label, act in self.actions.items():
                if label in by_label:
                    act.release(now, fire_pending=False)
            self._acting = None
        else:
            self._end_resize()
            self._dispatch_single_hand(by_label, now)

        self.resizing = resizing
        progress = 0.0
        if self._toggle_hold_start is not None and not self._arm_latched:
            progress = min(1.0, (now - self._toggle_hold_start) / self.cfg.gesture.arm_toggle_hold)

        return Status(
            armed=self.armed,
            hands=[HandStatus(t.label, t.gesture, t.label == self._acting) for t in tracked],
            resizing=resizing,
            dragging=any(a.dragging for a in self.actions.values()),
            arm_progress=progress,
            cursor=self.mouse.position,
            events=list(self.events),
            toggle_pose=self.toggle_pose,
        )

    def _dispatch_single_hand(self, by_label: dict[str, TrackedHand], now: float) -> None:
        acting = self._choose_acting(by_label)

        if self._acting is not None and self._acting != acting:
            self.actions[self._acting].release(now)
        if acting is None:
            self._acting = None
            return
        if self._acting != acting:
            self.actions[acting].activate()
            self._acting = acting

        if not self.armed:
            return
        t = by_label[acting]
        self.actions[acting].update(t.hand, t.gesture, now)

    def _choose_acting(self, by_label: dict[str, TrackedHand]) -> str | None:
        """Prefer the primary hand, but let the other one work on its own."""
        if not by_label:
            return None
        primary = self.cfg.hands.primary
        order = [primary] + [k for k in by_label if k != primary]
        for label in order:
            t = by_label.get(label)
            if t is not None and t.gesture in ACTIONABLE:
                return label
        return order[0] if order[0] in by_label else next(iter(by_label))

    # -- arm toggle ----------------------------------------------------

    def _handle_arm_toggle(self, gestures: list[Gesture], now: float) -> None:
        if self.toggle_pose not in gestures:
            self._toggle_hold_start = None
            self._arm_latched = False
            return
        if self._toggle_hold_start is None:
            self._toggle_hold_start = now
            return
        held = now - self._toggle_hold_start
        if held >= self.cfg.gesture.arm_toggle_hold and not self._arm_latched:
            self._arm_latched = True
            self.toggle_armed()

    # -- two-handed resize ----------------------------------------------

    def _separation(self, tracked: list[TrackedHand]) -> float:
        """Gap between the two index fingertips, in palm widths.

        Dividing by palm size keeps the measurement stable when you lean toward
        or away from the camera, which changes both distances together.
        """
        a, b = tracked[0].hand, tracked[1].hand
        gap = float(np.linalg.norm(a.index_tip - b.index_tip))
        palm = (a.palm + b.palm) / 2.0
        return gap / max(palm, 1e-6)

    def _handle_resize(self, tracked: list[TrackedHand], now: float) -> bool:
        sep = self._separation(tracked)

        if self._resize_base_sep is None:
            frame = self.windows.get_frame()
            if frame is None:
                self._log("resize: no resizable window")
                return False
            self._resize_base_sep = sep
            self._resize_base_frame = frame
            self._log("resize start")
            return True

        assert self._resize_base_frame is not None
        scale = sep / max(self._resize_base_sep, 1e-6)
        cfg = self.cfg.resize
        scale = min(cfg.max_scale, max(cfg.min_scale, scale))

        if abs(scale - 1.0) < cfg.deadzone:
            return True
        if now - self._resize_last_apply < cfg.update_interval:
            return True

        x, y, w, h = self._resize_base_frame
        new_w = max(cfg.min_width, w * scale)
        new_h = max(cfg.min_height, h * scale)
        # Grow and shrink about the window's centre so it stays where you put it.
        new_x = x + (w - new_w) / 2.0
        new_y = y + (h - new_h) / 2.0
        new_x = max(0.0, min(new_x, self.mouse.width - 60.0))
        new_y = max(0.0, min(new_y, self.mouse.height - 60.0))

        self._resize_last_apply = now
        self.windows.set_frame(new_x, new_y, new_w, new_h)
        return True

    def _end_resize(self) -> None:
        if self._resize_base_sep is not None:
            self._log("resize end")
        self._resize_base_sep = None
        self._resize_base_frame = None
