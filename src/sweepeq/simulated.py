"""A simulated device under test, for checking the measurement without hardware.

A Hammerstein-like chain: a static polynomial nonlinearity, then a linear filter (peaking
biquads, Bristow-Johnson's "Audio EQ Cookbook"), a pure delay, and additive noise. Its exact
linear response and, for a flat filter, its exact harmonic levels are known in closed form.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import signal

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def peaking_biquad(f0: float, gain_db: float, q: float, fs: float) -> FloatArray:
    """Second-order section ``[b0, b1, b2, 1, a1, a2]`` of a peaking EQ (RBJ cookbook)."""
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2.0 * q)
    b = np.array([1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a])
    den = np.array([1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a])
    return np.concatenate([b / den[0], den / den[0]])


def shelf_biquad(kind: str, f0: float, gain_db: float, q: float, fs: float) -> FloatArray:
    """Low (``kind="low"``) or high (``"high"``) shelving section with a Q-defined slope
    (RBJ cookbook)."""
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    c, alpha = np.cos(w0), np.sin(w0) / (2.0 * q)
    r = 2.0 * np.sqrt(a) * alpha
    if kind == "low":
        b = [
            a * ((a + 1) - (a - 1) * c + r),
            2 * a * ((a - 1) - (a + 1) * c),
            a * ((a + 1) - (a - 1) * c - r),
        ]
        den = [(a + 1) + (a - 1) * c + r, -2 * ((a - 1) + (a + 1) * c), (a + 1) + (a - 1) * c - r]
    elif kind == "high":
        b = [
            a * ((a + 1) + (a - 1) * c + r),
            -2 * a * ((a - 1) + (a + 1) * c),
            a * ((a + 1) + (a - 1) * c - r),
        ]
        den = [(a + 1) - (a - 1) * c + r, 2 * ((a - 1) - (a + 1) * c), (a + 1) - (a - 1) * c - r]
    else:
        raise ValueError(f"unknown shelf kind {kind!r}")
    bb, dd = np.asarray(b), np.asarray(den)
    return np.concatenate([bb / dd[0], dd / dd[0]])


@dataclass(frozen=True)
class SimulatedDevice:
    fs: int = 48_000
    sos: FloatArray = field(default_factory=lambda: np.empty((0, 6)))
    """Linear filter as second-order sections (none: flat)."""
    gain: float = 1.0
    delay_samples: int = 0
    poly: tuple[float, ...] = ()
    """Coefficients a2, a3, … of ``u = x + a2·x² + a3·x³ + …``, applied before the filter."""
    noise_rms: float = 0.0

    def respond(self, x: FloatArray, rng: np.random.Generator | None = None) -> FloatArray:
        u = x.copy()
        for k, a in enumerate(self.poly, start=2):
            u = u + a * x**k
        y = self.gain * (signal.sosfilt(self.sos, u) if len(self.sos) else u)
        y = np.concatenate([np.zeros(self.delay_samples), y])
        if self.noise_rms > 0.0:
            gen = rng if rng is not None else np.random.default_rng(0)
            y = y + self.noise_rms * gen.standard_normal(len(y))
        return np.asarray(y, dtype=np.float64)

    def linear_response(self, freqs: FloatArray) -> ComplexArray:
        """Exact response of the linear part, without the delay."""
        h = np.full(len(freqs), self.gain, dtype=np.complex128)
        if len(self.sos):
            _, hs = signal.sosfreqz(self.sos, worN=freqs, fs=self.fs)
            h = h * hs
        return h
