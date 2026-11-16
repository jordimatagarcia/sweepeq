"""Correction filter design: from a measured response and a target to an FIR filter.

The correction is computed in magnitude, from the measured response after fractional-octave
smoothing (inverting unsmoothed, position-dependent detail makes things worse a few
centimetres away; Norcross et al. 2004):

1. **Deviation.** ``d(f) = M(f) / T(f)``, the measured response over the target, aligned so
   the deviation averages 0 dB over a reference band.
2. **Peaks are inverted exactly**, ``C = 1/d`` where ``d ≥ 1``: cutting a resonance is safe.
3. **Dips are inverted with regularisation**, ``C = (d + β) / (d² + β)`` where ``d < 1``.
   This is the Tikhonov-regularised inverse that minimises ``|d·C - 1|² + β·|C - 1|²``,
   the approach of Kirkeby et al. (1998) with the prior "no correction" (C = 1) instead of
   "no output" (C = 0), so that a regularised band is left alone rather than muted. For
   dips much shallower than ``sqrt(β)`` it is the exact inverse; the boost it asks for
   peaks for a dip of about ``sqrt(β)`` (the notch depth, ``β = d_notch²``) and *falls*
   for deeper dips (``C ≈ 1 + d/β`` once ``d ≪ β``), reaching 0 dB only for extreme nulls.
   So the deeper a notch, the less it is chased: a deep notch is usually a cancellation
   that moves with position, and boosting into it wastes headroom. At d = 1 both branches
   give C = 1.
4. **Boost limit.** Boosts are then limited with a soft knee, ``min`` replaced by its
   smooth version ``-k·ln(e^(-b/k) + e^(-B/k))``: exact for boosts well below the limit B,
   never above it. (Setting β from the boost limit alone, as a first version did,
   under-corrects moderate dips: a 6 dB dip got 4.4 dB.)
5. **Band.** Outside the correction band the correction fades to 0 dB over a raised-cosine
   taper in log frequency.

The FIR filter is then designed by frequency sampling on a dense grid, with minimum phase
(from the real cepstrum, no added latency) or linear phase (symmetric, latency of half the
filter length), and windowed to the requested length.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from sweepeq.analysis import Curve, smooth
from sweepeq.deconv import minimum_phase

FloatArray = NDArray[np.float64]
Phase = Literal["minimum", "linear"]


@dataclass(frozen=True)
class DesignSpec:
    f_lo: float = 20.0
    """Correction band (Hz): outside it the correction fades to 0 dB."""
    f_hi: float = 16_000.0
    taper_oct: float = 0.5
    """Width of the fade at each band edge (octaves, outside the band)."""
    max_boost_db: float = 6.0
    max_cut_db: float = 24.0
    notch_depth_db: float = 20.0
    """Dips deeper than this are progressively left unfilled (sets β)."""
    knee_db: float = 0.5
    """Softness of the boost limit."""
    smoothing: float = 6.0
    """1/N-octave smoothing of the measurement before inversion (0: none)."""
    align_band: tuple[float, float] = (300.0, 3000.0)
    """The deviation is shifted to average 0 dB (in power) over this band."""
    fs: int = 48_000
    n_taps: int = 4096
    phase: Phase = "minimum"

    def __post_init__(self) -> None:
        if not 0.0 < self.f_lo < self.f_hi <= self.fs / 2:
            raise ValueError("need 0 < f_lo < f_hi <= fs/2")
        if min(self.max_boost_db, self.max_cut_db, self.notch_depth_db, self.knee_db) <= 0.0:
            raise ValueError(
                "max_boost_db, max_cut_db, notch_depth_db and knee_db must be positive"
            )
        if self.taper_oct < 0.0 or self.smoothing < 0.0:
            raise ValueError("taper_oct and smoothing must not be negative")
        if self.n_taps < 16:
            raise ValueError("n_taps must be at least 16")
        if self.phase not in ("minimum", "linear"):
            raise ValueError(f"unknown phase {self.phase!r}")


# --- targets -----------------------------------------------------------------------------


def target_curve(
    freqs: FloatArray,
    tilt_db_per_oct: float = 0.0,
    bass_db: float = 0.0,
    bass_hz: float = 105.0,
) -> Curve:
    """A simple parametric target: flat, with an optional tilt (dB per octave, about
    1 kHz) and an optional bass shelf (a smooth step of ``bass_db`` below ``bass_hz``).

    Published targets (e.g. the Harman curves) can be loaded as text files instead
    (``analysis.read_text_curve``).
    """
    f = np.asarray(freqs, dtype=np.float64)
    db = tilt_db_per_oct * np.log2(f / 1000.0)
    if bass_db != 0.0:
        # logistic step in log frequency, one octave wide
        db = db + bass_db / (1.0 + (f / bass_hz) ** 4)
    return Curve(f, np.asarray(db, dtype=np.float64))


# --- correction magnitude ------------------------------------------------------------------


def beta_for_notch(notch_depth_db: float) -> float:
    """β for which dips start being left unfilled around ``notch_depth_db``: the
    regularised inverse peaks at ``d* = √(β² + β) - β ≈ √β`` for small β."""
    return float(10.0 ** (-notch_depth_db / 10.0))


def soft_limit_boost(c_db: FloatArray, max_boost_db: float, knee_db: float) -> FloatArray:
    """Limit positive values to ``max_boost_db`` with a soft knee; cuts pass unchanged."""
    b = np.maximum(c_db, 0.0)
    limited = -knee_db * np.logaddexp(-b / knee_db, -max_boost_db / knee_db)
    return np.asarray(np.where(c_db > 0.0, limited, c_db), dtype=np.float64)


def regularised_inverse(d: FloatArray, beta: float, max_cut_db: float) -> FloatArray:
    """Correction gain (linear) for deviation ``d``: exact inverse of peaks (limited to
    ``max_cut_db``), Tikhonov-regularised inverse of dips."""
    d = np.maximum(np.asarray(d, dtype=np.float64), 1e-12)
    c = np.where(d >= 1.0, 1.0 / d, (d + beta) / (d * d + beta))
    return np.asarray(np.maximum(c, 10.0 ** (-max_cut_db / 20.0)), dtype=np.float64)


def band_fade(freqs: FloatArray, spec: DesignSpec) -> FloatArray:
    """1 inside [f_lo, f_hi], falling to 0 over ``taper_oct`` octaves outside it."""
    lf = np.log2(np.maximum(freqs, 1e-30))
    if spec.taper_oct == 0.0:
        return ((freqs >= spec.f_lo) & (freqs <= spec.f_hi)).astype(np.float64)
    lo = np.clip((lf - np.log2(spec.f_lo)) / spec.taper_oct + 1.0, 0.0, 1.0)
    hi = np.clip((np.log2(spec.f_hi) - lf) / spec.taper_oct + 1.0, 0.0, 1.0)
    return np.asarray((0.5 - 0.5 * np.cos(np.pi * lo)) * (0.5 - 0.5 * np.cos(np.pi * hi)))


def deviation(measured: Curve, target: Curve, spec: DesignSpec) -> Curve:
    """Smoothed measured response minus target (dB), aligned over ``align_band``."""
    m = smooth(measured, spec.smoothing) if spec.smoothing > 0 else measured
    dev = m.db - target.at(m.freqs)
    lo, hi = spec.align_band
    k = (m.freqs >= lo) & (m.freqs <= hi)
    if not np.any(k):
        raise ValueError("no measurement points in the alignment band")
    ref = 10.0 * np.log10(np.mean(10.0 ** (dev[k] / 10.0)))
    return Curve(m.freqs, dev - ref)


def correction_db(measured: Curve, target: Curve, spec: DesignSpec) -> Curve:
    """The correction to apply (dB) on the measurement's frequency grid."""
    dev = deviation(measured, target, spec)
    c = regularised_inverse(
        10.0 ** (dev.db / 20.0), beta_for_notch(spec.notch_depth_db), spec.max_cut_db
    )
    c_db = soft_limit_boost(20.0 * np.log10(c), spec.max_boost_db, spec.knee_db)
    return Curve(dev.freqs, c_db * band_fade(dev.freqs, spec))


# --- FIR ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Design:
    spec: DesignSpec
    desired: Curve
    """Correction asked for (dB)."""
    taps: FloatArray

    @property
    def latency_samples(self) -> float:
        return 0.0 if self.spec.phase == "minimum" else (self.spec.n_taps - 1) / 2.0

    def response_db(self, freqs: FloatArray) -> FloatArray:
        return fir_response_db(self.taps, freqs, self.spec.fs)

    @property
    def peak_gain_db(self) -> float:
        """Largest gain for a sine wave: the headroom a preamp must give."""
        grid = np.fft.rfft(self.taps, 1 << 18)
        return float(20.0 * np.log10(np.max(np.abs(grid))))

    @property
    def worst_case_gain_db(self) -> float:
        """Largest possible sample-peak gain for any input (the L1 norm of the taps):
        the strict bound, usually far above the sine-wave figure."""
        return float(20.0 * np.log10(np.sum(np.abs(self.taps))))


def design_fir(correction: Curve, spec: DesignSpec) -> Design:
    """FIR filter realising ``correction`` (dB), by frequency sampling."""
    n_fft = max(1 << 16, 8 * spec.n_taps)
    f = np.fft.rfftfreq(n_fft, 1.0 / spec.fs)
    # Below and above the curve's grid, hold its edge values (both are 0 dB after the
    # band fade when the grid spans the faded region).
    f_safe = np.clip(f, correction.freqs[0], correction.freqs[-1])
    mag = 10.0 ** (correction.at(f_safe) / 20.0)
    n = spec.n_taps
    if spec.phase == "minimum":
        h_full = np.fft.irfft(mag * np.exp(1j * minimum_phase(mag, n_fft)), n_fft)
        taps = h_full[:n] * _half_hann_tail(n, max(n // 8, 1))
    else:
        # Zero phase, then delayed by (n - 1)/2 samples (half a sample off the grid for
        # an even length), so the taps are symmetric about the filter's centre.
        k = np.arange(len(mag))
        h_full = np.fft.irfft(mag * np.exp(-1j * np.pi * k * (n - 1) / n_fft), n_fft)
        taps = h_full[:n] * np.hanning(n + 2)[1:-1]
    return Design(spec, correction, np.asarray(taps, dtype=np.float64))


def _half_hann_tail(n: int, n_tail: int) -> FloatArray:
    w = np.ones(n)
    w[n - n_tail :] = 0.5 + 0.5 * np.cos(np.pi * (np.arange(n_tail) + 1) / n_tail)
    return w


def fir_response_db(taps: FloatArray, freqs: FloatArray, fs: float) -> FloatArray:
    from scipy import signal

    _, h = signal.freqz(taps, worN=np.asarray(freqs), fs=fs)
    return np.asarray(20.0 * np.log10(np.maximum(np.abs(h), 1e-300)), dtype=np.float64)
