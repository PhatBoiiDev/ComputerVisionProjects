"""Cursor movement on its own clock, decoupled from the camera.

The capture loop can only produce a new hand position every camera frame -- 33 ms
at 30 fps -- and that is the floor on how fresh the cursor's information can be.
Nothing here invents motion that the camera did not see. What it does buy is two
measurable things:

  * a new sample is picked up within 1/poll_hz instead of waiting for the next
    trip around the capture loop, and
  * the smoothing filter is evaluated on a fine clock, so a hand that jumped
    33 ms worth of distance arrives as a ramp rather than a single hop.

Measured on this machine against a 30 fps source, 500 Hz sits at the knee: below
it both added lag and per-update step size climb steeply, above it lag stops
improving while CPU keeps climbing.
"""

from __future__ import annotations

import threading
import time

from .config import Config
from .filters import Point2DFilter
from .macos_input import MouseController


class CursorPump:
    """Holds the latest cursor target and applies it at a fixed rate."""

    def __init__(self, cfg: Config, mouse: MouseController) -> None:
        self.cfg = cfg
        self.mouse = mouse
        s = cfg.smoothing
        self._filter = Point2DFilter(s.min_cutoff, s.beta, s.d_cutoff)

        self._lock = threading.Lock()
        self._target: tuple[float, float] | None = None
        self._value: tuple[float, float] | None = None
        self._active = False

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.updates = 0

    # -- input from the capture loop -----------------------------------

    def set_target(self, x: float, y: float, active: bool) -> None:
        with self._lock:
            self._target = (x, y)
            self._active = active

    def deactivate(self) -> None:
        """Stop driving the cursor, but keep the filter's history."""
        with self._lock:
            self._active = False

    def recentre(self) -> None:
        """Drop motion history so the cursor snaps to the new target.

        Used when control passes to a different hand: easing across the gap
        would send the cursor sliding through everything in between.
        """
        with self._lock:
            self._filter.reset()
            self._target = None
            self._value = None
            self._active = False

    @property
    def value(self) -> tuple[float, float] | None:
        """The smoothed hand position, whether or not it is steering."""
        with self._lock:
            return self._value

    # -- output --------------------------------------------------------

    def step(self, now: float | None = None) -> tuple[float, float] | None:
        """Apply one update. Called by the thread, or directly when not threaded.

        The filter runs whenever there is a hand, even when that hand is not
        steering. Scrolling reads the same smoothed position to work out how far
        the hand travelled, and letting the filter go cold between poses would
        both break that and put a jump in the cursor on the way back to POINT.
        """
        now = time.perf_counter() if now is None else now
        with self._lock:
            target, active = self._target, self._active
        if target is None:
            return None
        x, y = self._filter(target[0], target[1], now)
        with self._lock:
            self._value = (x, y)
        if active:
            self.mouse.move_to(x, y)
            self.updates += 1
        return x, y

    # -- lifecycle -----------------------------------------------------

    def start(self) -> None:
        if self._thread is not None or not self.cfg.cursor.threaded:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="cursor-pump", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        period = 1.0 / max(1.0, self.cfg.cursor.poll_hz)
        nxt = time.perf_counter()
        while not self._stop.is_set():
            self.step()
            nxt += period
            delay = nxt - time.perf_counter()
            if delay > 0:
                self._stop.wait(delay)
            else:
                # Fell behind -- give up the missed ticks rather than trying to
                # catch up in a burst, which would spike the event rate.
                nxt = time.perf_counter()
