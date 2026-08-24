"""Screen control via CoreGraphics event injection.

Every synthetic event is posted to the HID tap, which is what the window server
treats as a real device, so applications cannot tell these apart from a trackpad.
Requires the host terminal to hold Accessibility permission.
"""

from __future__ import annotations

import ApplicationServices as AX
import Quartz
from ApplicationServices import AXIsProcessTrusted

_LEFT = Quartz.kCGMouseButtonLeft
_RIGHT = Quartz.kCGMouseButtonRight
_TAP = Quartz.kCGHIDEventTap

# Virtual keycodes for the few keys shortcuts need.
_KEYCODES = {
    "w": 13, "q": 12, "t": 17, "n": 45, "tab": 48, "escape": 53, "space": 49,
    "h": 4, "m": 46, "delete": 51, "left": 123, "right": 124, "up": 126, "down": 125,
}
_MODIFIERS = {
    "cmd": Quartz.kCGEventFlagMaskCommand,
    "command": Quartz.kCGEventFlagMaskCommand,
    "shift": Quartz.kCGEventFlagMaskShift,
    "alt": Quartz.kCGEventFlagMaskAlternate,
    "option": Quartz.kCGEventFlagMaskAlternate,
    "ctrl": Quartz.kCGEventFlagMaskControl,
    "control": Quartz.kCGEventFlagMaskControl,
}


def accessibility_trusted() -> bool:
    return bool(AXIsProcessTrusted())


def screen_size() -> tuple[int, int]:
    bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    return int(bounds.size.width), int(bounds.size.height)


class MouseController:
    """Cursor movement, buttons and scrolling.

    In dry_run nothing is posted to the OS; the intended actions are still
    reported so the pipeline can be exercised without Accessibility access.
    """

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.width, self.height = screen_size()
        self._pos = (self.width / 2.0, self.height / 2.0)
        self._left_down = False
        self.log: list[str] = []

    @property
    def position(self) -> tuple[float, float]:
        return self._pos

    def _note(self, msg: str) -> None:
        self.log.append(msg)
        if len(self.log) > 40:
            del self.log[:-40]

    def _post(self, event) -> None:
        if not self.dry_run:
            Quartz.CGEventPost(_TAP, event)

    def _mouse_event(self, kind, x: float, y: float, button, clicks: int = 1):
        ev = Quartz.CGEventCreateMouseEvent(None, kind, (x, y), button)
        if clicks > 1:
            Quartz.CGEventSetIntegerValueField(ev, Quartz.kCGMouseEventClickState, clicks)
        return ev

    def move_to(self, x: float, y: float) -> None:
        x = max(0.0, min(self.width - 1.0, x))
        y = max(0.0, min(self.height - 1.0, y))
        self._pos = (x, y)
        # While a button is held the window server expects Dragged, not Moved;
        # sending Moved would silently break drag gestures in most apps.
        kind = Quartz.kCGEventLeftMouseDragged if self._left_down else Quartz.kCGEventMouseMoved
        self._post(self._mouse_event(kind, x, y, _LEFT))

    def left_down(self) -> None:
        if self._left_down:
            return
        self._left_down = True
        x, y = self._pos
        self._post(self._mouse_event(Quartz.kCGEventLeftMouseDown, x, y, _LEFT))
        self._note("left down")

    def left_up(self) -> None:
        if not self._left_down:
            return
        self._left_down = False
        x, y = self._pos
        self._post(self._mouse_event(Quartz.kCGEventLeftMouseUp, x, y, _LEFT))
        self._note("left up")

    def click(self, clicks: int = 1) -> None:
        x, y = self._pos
        for n in range(1, clicks + 1):
            self._post(self._mouse_event(Quartz.kCGEventLeftMouseDown, x, y, _LEFT, n))
            self._post(self._mouse_event(Quartz.kCGEventLeftMouseUp, x, y, _LEFT, n))
        self._note("double click" if clicks > 1 else "click")

    def right_click(self) -> None:
        x, y = self._pos
        self._post(self._mouse_event(Quartz.kCGEventRightMouseDown, x, y, _RIGHT))
        self._post(self._mouse_event(Quartz.kCGEventRightMouseUp, x, y, _RIGHT))
        self._note("right click")

    def scroll(self, dy: int, dx: int = 0) -> None:
        if dy == 0 and dx == 0:
            return
        ev = Quartz.CGEventCreateScrollWheelEvent(
            None, Quartz.kCGScrollEventUnitPixel, 2, int(dy), int(dx)
        )
        self._post(ev)
        self._note(f"scroll dy={dy} dx={dx}")

    def release_all(self) -> None:
        self.left_up()


def parse_shortcut(combo: str) -> tuple[int, int]:
    """"cmd+w" -> (keycode, modifier mask). Raises ValueError on nonsense."""
    parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
    if not parts:
        raise ValueError("empty shortcut")
    key = parts[-1]
    if key not in _KEYCODES:
        raise ValueError(f"unsupported key {key!r} in shortcut {combo!r}")
    flags = 0
    for mod in parts[:-1]:
        if mod not in _MODIFIERS:
            raise ValueError(f"unsupported modifier {mod!r} in shortcut {combo!r}")
        flags |= _MODIFIERS[mod]
    return _KEYCODES[key], flags


class KeyboardController:
    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.log: list[str] = []

    def press(self, combo: str) -> None:
        keycode, flags = parse_shortcut(combo)
        self.log.append(f"key {combo}")
        if self.dry_run:
            return
        for down in (True, False):
            ev = Quartz.CGEventCreateKeyboardEvent(None, keycode, down)
            Quartz.CGEventSetFlags(ev, flags)
            Quartz.CGEventPost(_TAP, ev)


class WindowController:
    """Resize the focused window through the Accessibility API.

    Not every window can be resized -- system dialogs and some non-native apps
    refuse -- so every call reports success rather than raising, and the caller
    simply does nothing when a window will not cooperate.
    """

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.log: list[str] = []
        self._fake_frame = [200.0, 200.0, 900.0, 600.0]   # used only in dry runs

    def _focused_window(self):
        from AppKit import NSWorkspace

        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return None
        element = AX.AXUIElementCreateApplication(app.processIdentifier())
        err, window = AX.AXUIElementCopyAttributeValue(
            element, AX.kAXFocusedWindowAttribute, None
        )
        return window if err == 0 else None

    def _read(self, window, attribute, value_type):
        err, raw = AX.AXUIElementCopyAttributeValue(window, attribute, None)
        if err != 0 or raw is None:
            return None
        ok, value = AX.AXValueGetValue(raw, value_type, None)
        return value if ok else None

    def get_frame(self) -> tuple[float, float, float, float] | None:
        """(x, y, width, height) of the focused window, or None."""
        if self.dry_run:
            return tuple(self._fake_frame)
        window = self._focused_window()
        if window is None:
            return None
        pos = self._read(window, AX.kAXPositionAttribute, AX.kAXValueCGPointType)
        size = self._read(window, AX.kAXSizeAttribute, AX.kAXValueCGSizeType)
        if pos is None or size is None:
            return None
        return float(pos.x), float(pos.y), float(size.width), float(size.height)

    def set_frame(self, x: float, y: float, width: float, height: float) -> bool:
        self.log.append(f"resize {int(width)}x{int(height)} @ {int(x)},{int(y)}")
        if self.dry_run:
            self._fake_frame = [x, y, width, height]
            return True

        window = self._focused_window()
        if window is None:
            return False
        # Position first: shrinking a window that is already against the screen
        # edge can otherwise be clamped by the window server before it moves.
        point = AX.AXValueCreate(AX.kAXValueCGPointType, Quartz.CGPoint(x, y))
        size = AX.AXValueCreate(AX.kAXValueCGSizeType, Quartz.CGSize(width, height))
        e1 = AX.AXUIElementSetAttributeValue(window, AX.kAXPositionAttribute, point)
        e2 = AX.AXUIElementSetAttributeValue(window, AX.kAXSizeAttribute, size)
        return e1 == 0 and e2 == 0
