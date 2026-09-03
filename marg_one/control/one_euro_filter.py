"""
MARG-One Control Subsystem: 1 Euro Filter Implementation.
High-precision, adaptive signal filter for zero-lag and jitter-free human-computer interaction.
Reference: Casiez, G., Roussel, N., & Vogel, D. (2012). 1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems. ACM CHI 2012.
"""

import math
import time
from typing import Tuple, Optional


class LowPassFilter:
    """Standard exponential smoothing low-pass filter."""

    def __init__(self, alpha: float = 0.5):
        self._alpha = alpha
        self._s: Optional[float] = None

    def filter(self, value: float, alpha: Optional[float] = None) -> float:
        if alpha is not None:
            self._alpha = alpha
        if self._s is None:
            self._s = value
        else:
            self._s = self._alpha * value + (1.0 - self._alpha) * self._s
        return self._s

    def last_value(self) -> Optional[float]:
        return self._s

    def reset(self):
        self._s = None


class OneEuroFilter:
    """
    1 Euro Filter for a 1D scalar signal.
    Dynamically adjusts cutoff frequency based on input speed:
    - Low speeds: low cutoff frequency to eliminate noise/jitter.
    - High speeds: high cutoff frequency to reduce lag to zero.
    """

    def __init__(
        self,
        min_cutoff: float = 0.5,
        beta: float = 0.005,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time: Optional[float] = None

    def _compute_alpha(self, rate: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / max(1e-4, rate)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        if timestamp is None:
            timestamp = time.time()

        if self.last_time is None:
            self.last_time = timestamp
            self.x_filter.filter(x, 1.0)
            self.dx_filter.filter(0.0, 1.0)
            return x

        dt = max(1e-4, timestamp - self.last_time)
        self.last_time = timestamp
        rate = 1.0 / dt

        # Estimate speed of change (derivative)
        prev_x = self.x_filter.last_value()
        dx = (x - prev_x) / dt if prev_x is not None else 0.0
        edx = self.dx_filter.filter(dx, self._compute_alpha(rate, self.d_cutoff))

        # Dynamically scale cutoff frequency
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self.x_filter.filter(x, self._compute_alpha(rate, cutoff))

    def reset(self):
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None


class Point2DOneEuroFilter:
    """
    2D spatial coordinate filter utilizing dual 1 Euro filters for X and Y axes.
    """

    def __init__(
        self,
        min_cutoff: float = 0.5,
        beta: float = 0.005,
        d_cutoff: float = 1.0,
    ):
        self.filter_x = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.filter_y = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

    def filter(self, x: float, y: float, timestamp: Optional[float] = None) -> Tuple[float, float]:
        if timestamp is None:
            timestamp = time.time()
        fx = self.filter_x.filter(x, timestamp)
        fy = self.filter_y.filter(y, timestamp)
        return (fx, fy)

    def reset(self):
        self.filter_x.reset()
        self.filter_y.reset()
