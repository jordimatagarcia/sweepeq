"""Fitting a parametric EQ (peaking and shelving biquads) to a correction curve.

The FIR filter realises the correction exactly; a parametric EQ approximates it with a few
filters that any EQ program accepts (Equalizer APO, EasyEffects, phone apps), at the cost of
some error. The fit is in two stages:

1. **Greedy initialisation.** Low and high shelves are placed from the average residual
   below 100 Hz and above 8 kHz, where it is at least 1 dB; then, while bands remain, a
   peaking filter is placed at the largest remaining error, with its gain, and a bandwidth
   estimated from where the error falls to half that value.
2. **Joint refinement.** All parameters are refined together by nonlinear least squares
   (``scipy.optimize.least_squares``, trust-region reflective) on the dB error over the
   correction band, within bounds on frequency, gain and Q. A small penalty on each
   filter's gain (``GAIN_PENALTY``) breaks ties between equivalent solutions in favour of
   the one with smaller gains, instead of pairs of large opposing filters that cancel.

The preamp is set to the negative of the EQ's largest gain, so it cannot clip.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from sweepeq.analysis import Curve
from sweepeq.apo import Band, Preset

FloatArray = NDArray[np.float64]

Q_MIN, Q_MAX = 0.3, 8.0
SHELF_Q = 0.7071


@dataclass(frozen=True)
class PeqSpec:
    n_bands: int = 10
    """Total number of filters, shelves included."""
    shelves: bool = True
    f_lo: float = 20.0
    f_hi: float = 16_000.0
    boost_limit_db: float = 6.0
    cut_limit_db: float = 24.0
    fs: float = 48_000.0
    n_points: int = 400
    """Frequencies (log-spaced over the band) the error is evaluated on."""

    def __post_init__(self) -> None:
        if self.n_bands < 1:
            raise ValueError("n_bands must be at least 1")
        if self.shelves and self.n_bands < 2:
            raise ValueError("with shelves, n_bands must be at least 2")
        if not 0.0 < self.f_lo < self.f_hi < self.fs / 2:
            raise ValueError("need 0 < f_lo < f_hi < fs/2")
        if self.boost_limit_db <= 0.0 or self.cut_limit_db <= 0.0:
            raise ValueError("boost_limit_db and cut_limit_db must be positive")


GAIN_PENALTY = 0.001
"""Weight of the gain penalty relative to the dB error at one frequency point."""
SHELF_MIN_DB = 1.0
"""A shelf is only used where the correction averages at least this much at that end."""


def freq_range(kind: str, spec: "PeqSpec") -> tuple[float, float]:
    """Allowed centre or corner frequency of each filter type: peaks inside the band, the
    low shelf in the bass, the high shelf in the treble, so each keeps its role."""
    if kind == "LS":
        return max(20.0, spec.f_lo), 500.0
    if kind == "HS":
        return 1000.0, spec.fs / 2 * 0.95
    return spec.f_lo, spec.f_hi


# --- vectorised biquad magnitudes ----------------------------------------------------------


def _band_db(kind: str, fc: float, gain: float, q: float, w: FloatArray, fs: float) -> FloatArray:
    """|H| in dB of one RBJ biquad at angular frequencies ``w`` (rad/sample)."""
    from sweepeq.simulated import peaking_biquad, shelf_biquad

    if kind == "PK":
        c = peaking_biquad(fc, gain, q, fs)
    else:
        c = shelf_biquad("low" if kind == "LS" else "high", fc, gain, q, fs)
    z = np.exp(-1j * w)
    num = c[0] + c[1] * z + c[2] * z * z
    den = c[3] + c[4] * z + c[5] * z * z
    return np.asarray(20.0 * np.log10(np.abs(num / den)), dtype=np.float64)


def preset_db(bands: list[Band], freqs: FloatArray, fs: float) -> FloatArray:
    w = 2.0 * np.pi * np.asarray(freqs) / fs
    total = np.zeros(len(freqs))
    for b in bands:
        total = total + _band_db(b.kind, b.fc, b.gain_db, b.q, w, fs)
    return total


# --- fit --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PeqFit:
    preset: Preset
    freqs: FloatArray
    wanted_db: FloatArray
    fitted_db: FloatArray
    """EQ response without the preamp."""

    @property
    def rms_error_db(self) -> float:
        return float(np.sqrt(np.mean((self.fitted_db - self.wanted_db) ** 2)))

    @property
    def max_error_db(self) -> float:
        return float(np.max(np.abs(self.fitted_db - self.wanted_db)))


def fit_peq(correction: Curve, spec: PeqSpec) -> PeqFit:
    freqs = np.geomspace(spec.f_lo, spec.f_hi, spec.n_points)
    target = correction.at(freqs)
    bands = _initial_bands(freqs, target, spec)
    bands = _refine(bands, freqs, target, spec)
    fitted = preset_db(bands, freqs, spec.fs)
    # Headroom: the largest gain anywhere up to Nyquist, not just inside the band.
    dense = np.geomspace(10.0, spec.fs / 2 * 0.999, 4000)
    peak = float(np.max(preset_db(bands, dense, spec.fs)))
    preamp = -max(peak, 0.0)
    preset = Preset(round(preamp - 0.05, 1) if preamp < 0 else 0.0, tuple(bands))
    return PeqFit(preset, freqs, target, fitted)


def _initial_bands(freqs: FloatArray, target: FloatArray, spec: PeqSpec) -> list[Band]:
    bands: list[Band] = []
    residual = target.copy()
    if spec.shelves:
        lo = freqs < 100.0
        hi = freqs > 8000.0
        g_lo = _clip_gain(float(np.mean(residual[lo])) if np.any(lo) else 0.0, spec)
        g_hi = _clip_gain(float(np.mean(residual[hi])) if np.any(hi) else 0.0, spec)
        if abs(g_lo) >= SHELF_MIN_DB:
            bands.append(Band("LS", 105.0, g_lo, SHELF_Q))
        if abs(g_hi) >= SHELF_MIN_DB:
            bands.append(Band("HS", 10_000.0, g_hi, SHELF_Q))
        residual = target - preset_db(bands, freqs, spec.fs)
    lf = np.log2(freqs)
    while len(bands) < spec.n_bands:
        k = int(np.argmax(np.abs(residual)))
        g = _clip_gain(float(residual[k]), spec)
        # bandwidth: where the residual falls to half its peak, on either side
        half = np.abs(residual) < abs(residual[k]) / 2
        left = np.nonzero(half[:k])[0]
        right = np.nonzero(half[k:])[0]
        lo_oct = lf[k] - (lf[left[-1]] if len(left) else lf[0])
        hi_oct = (lf[k + right[0]] if len(right) else lf[-1]) - lf[k]
        bw = max(lo_oct + hi_oct, 0.05)
        q = float(np.clip(np.sqrt(2.0**bw) / (2.0**bw - 1.0), Q_MIN, Q_MAX))
        bands.append(Band("PK", float(freqs[k]), g, q))
        residual = target - preset_db(bands, freqs, spec.fs)
    return bands


def _clip_gain(g: float, spec: PeqSpec) -> float:
    return float(np.clip(g, -spec.cut_limit_db, spec.boost_limit_db))


def _refine(bands: list[Band], freqs: FloatArray, target: FloatArray, spec: PeqSpec) -> list[Band]:
    kinds = [b.kind for b in bands]
    x0, lo, hi = [], [], []
    for b in bands:
        x0 += [np.log(b.fc), b.gain_db, np.log(b.q)]
        f_min, f_max = freq_range(b.kind, spec)
        lo += [np.log(f_min), -spec.cut_limit_db, np.log(Q_MIN)]
        hi += [np.log(f_max), spec.boost_limit_db, np.log(Q_MAX if b.kind == "PK" else 2.0)]
    x0 = np.clip(x0, np.asarray(lo) + 1e-9, np.asarray(hi) - 1e-9)

    def unpack(x: FloatArray) -> list[Band]:
        return [
            Band(kind, float(np.exp(x[3 * i])), float(x[3 * i + 1]), float(np.exp(x[3 * i + 2])))
            for i, kind in enumerate(kinds)
        ]

    weight = np.sqrt(GAIN_PENALTY * len(freqs) / max(len(kinds), 1)) / np.sqrt(len(freqs))

    def residual(x: FloatArray) -> FloatArray:
        err = (preset_db(unpack(x), freqs, spec.fs) - target) / np.sqrt(len(freqs))
        return np.concatenate([err, weight * np.asarray(x[1::3])])

    sol = optimize.least_squares(residual, x0, bounds=(lo, hi), method="trf", x_scale="jac")
    return unpack(sol.x)
