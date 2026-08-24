"""Signal filters for pointer smoothing."""

from __future__ import annotations

import math


class _LowPass:
    def __init__(self) -> None:
        self.y: float | None = None

    def __call__(self, x: float, alpha: float) -> float:
        self.y = x if self.y is None else alpha * x + (1.0 - alpha) * self.y
        return self.y

    def reset(self) -> None:
        self.y = None


class OneEuroFilter:
    """Adaptive low-pass filter: heavy smoothing when still, low lag when moving fast.

    A plain moving average forces a choice between jitter and lag; this trades
    between them per-sample based on the observed speed of the signal.
    """

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.035, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = _LowPass()
        self._dx = _LowPass()
        self._t_prev: float | None = None
        self._x_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self._t_prev is None:
            self._t_prev, self._x_prev = t, x
            return self._x(x, 1.0)

        dt = t - self._t_prev
        if dt <= 0:
            dt = 1e-3
        self._t_prev = t

        dx = (x - self._x_prev) / dt
        self._x_prev = x
        dx_hat = self._dx(dx, self._alpha(self.d_cutoff, dt))

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        return self._x(x, self._alpha(cutoff, dt))

    def reset(self) -> None:
        self._x.reset()
        self._dx.reset()
        self._t_prev = None
        self._x_prev = None


class Point2DFilter:
    """One Euro filter applied independently to x and y."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.035, d_cutoff: float = 1.0) -> None:
        self.fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def __call__(self, x: float, y: float, t: float) -> tuple[float, float]:
        return self.fx(x, t), self.fy(y, t)

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()
