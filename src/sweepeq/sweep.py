"""Synchronized exponential sine sweep (Farina 2000; Novak, Lotton and Simon 2015).

An exponential sweep spends equal time per octave, so its spectrum falls 3 dB per octave and
every harmonic a nonlinear system produces appears, after deconvolution, as a separate
impulse response *before* the linear one: the n-th harmonic at ``-L·ln(n)`` seconds. The
synchronized variant rounds the sweep rate so that ``f1·L`` is an integer; the harmonics then
start in phase with the fundamental and their impulse responses are exact, not just their
magnitudes (Novak et al. 2015).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SweepSpec:
    """Parameters of a sweep. ``duration_s`` is a target: the synchronized sweep's actual
    length is ``rate_s · ln(f2/f1)``, within ``1/f1`` of it."""

    f1: float = 20.0
    """Start frequency (Hz)."""
    f2: float = 20_000.0
    """End frequency (Hz)."""
    duration_s: float = 5.0
    fs: int = 48_000
    """Sample rate (Hz)."""
    fade_in_oct: float = 0.25
    """Fade-in length, in octaves of the sweep (the time it takes to sweep that far). Long
    enough that a device with even-order distortion, whose DC offset switches on with the
    sweep, produces a slow transient: a short fade puts tens of hertz of switch-on transient
    into the recording, which deconvolution maps between the harmonic responses."""
    fade_out_s: float = 0.005
    amplitude: float = 0.5
    """Peak amplitude (full scale = 1)."""

    def __post_init__(self) -> None:
        if not 0.0 < self.f1 < self.f2 <= self.fs / 2:
            raise ValueError("need 0 < f1 < f2 <= fs/2")
        if self.duration_s <= 0.0 or self.amplitude <= 0.0:
            raise ValueError("duration_s and amplitude must be positive")
        if self.fade_in_oct < 0.0 or self.fade_out_s < 0.0:
            raise ValueError("fades must not be negative")
        if self.rate_s * self.f1 < 1.0:
            raise ValueError("sweep too short for f1: f1·L must be at least 1")

    @property
    def rate_s(self) -> float:
        """Sweep rate ``L`` (s): the time to sweep by a factor of e, rounded so ``f1·L`` is an
        integer (Novak et al. 2015)."""
        return round(self.f1 * self.duration_s / float(np.log(self.f2 / self.f1))) / self.f1

    @property
    def length_s(self) -> float:
        return self.rate_s * float(np.log(self.f2 / self.f1))

    def harmonic_delay_s(self, n: int) -> float:
        """How far before the linear impulse response the n-th harmonic one appears (s)."""
        return self.rate_s * float(np.log(n))


def synchronized_sweep(spec: SweepSpec) -> FloatArray:
    """The sweep ``a·sin(2π·f1·L·exp(t/L))`` with half-Hann fades (Novak et al. 2015)."""
    n = round(spec.length_s * spec.fs)
    t = np.arange(n) / spec.fs
    lr = spec.rate_s
    x = spec.amplitude * np.sin(2.0 * np.pi * spec.f1 * lr * np.exp(t / lr))
    n_in = round(spec.rate_s * np.log(2.0) * spec.fade_in_oct * spec.fs)
    return x * _fades(n, n_in, round(spec.fade_out_s * spec.fs))


def _fades(n: int, n_in: int, n_out: int) -> FloatArray:
    w = np.ones(n)
    if n_in > 0:
        w[:n_in] = 0.5 - 0.5 * np.cos(np.pi * np.arange(n_in) / n_in)
    if n_out > 0:
        w[n - n_out :] = 0.5 + 0.5 * np.cos(np.pi * (np.arange(n_out) + 1) / n_out)
    return w
