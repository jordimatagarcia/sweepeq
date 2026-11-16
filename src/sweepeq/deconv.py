"""Impulse responses from a sweep recording (Farina 2000; Müller and Massarani 2001).

Deconvolution is a regularised spectral division ``H = Y·X* / (|X|² + ε)``. Inside the
sweep's band ε is negligible; outside it ε dominates, so the band the sweep never excited is
attenuated instead of amplifying noise (Müller and Massarani 2001). ε changes between
the two over a raised-cosine taper in log frequency, not a step: a step would make the
impulse response ring at the band edges, before as well as after time zero, and the ringing
before it would leak into the harmonic windows.

The regularised division is still zero-phase in its band limiting: ``|X|²/(|X|² + ε)`` is
real, so the impulse response it gives is symmetric, and its pre-ringing would be cut off by
the window and spill into the harmonic windows. The band limit is therefore made minimum
phase (same magnitude, causal), computed from the real cepstrum (Oppenheim and Schafer,
2010): the measured response then behaves like the true one passed through a causal
band-pass, as a real measurement chain would.

The FFT is long enough that the result is linear, not circular, deconvolution: the harmonic
impulse responses, which belong at negative times, wrap to the end of the buffer.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sweepeq.sweep import SweepSpec

FloatArray = NDArray[np.float64]

OUT_OF_BAND_EPS = 1e3
"""Regularisation outside the sweep band, relative to the in-band ``|X|²`` level."""
IN_BAND_EPS = 1e-12
LOW_TAPER_OCT = 0.5
"""Width of the regularisation taper above f1 (octaves)."""
HIGH_TAPER_OCT = 1.0 / 6.0
"""Width of the taper below f2 (octaves)."""


def deconvolve(recorded: FloatArray, sweep: FloatArray, spec: SweepSpec) -> FloatArray:
    """Full deconvolution result: the linear impulse response from index 0, the harmonic
    ones wrapped around to the end. Length is the FFT size."""
    n_fft = _next_fast_len(len(recorded) + len(sweep))
    x = np.fft.rfft(sweep, n_fft)
    y = np.fft.rfft(recorded, n_fft)
    f = np.fft.rfftfreq(n_fft, 1.0 / spec.fs)
    level = float(np.median(np.abs(x[(f > spec.f1 * 2) & (f < spec.f2 / 2)]) ** 2))
    eps = (IN_BAND_EPS + OUT_OF_BAND_EPS * (1.0 - band_weight(f, spec))) * level
    x2 = np.abs(x) ** 2
    band = x2 / (x2 + eps)
    hz = y * np.conj(x) / (x2 + eps)  # true response times the zero-phase band limit
    h = np.fft.irfft(hz * np.exp(1j * minimum_phase(band, n_fft)), n_fft)
    return np.asarray(h, dtype=np.float64)


def valid_range(f1: float, f2: float) -> tuple[float, float]:
    """Frequencies over which a measurement with sweep band ``[f1, f2]`` is accurate: inside
    the regularisation tapers (``LOW_TAPER_OCT`` above f1, ``HIGH_TAPER_OCT`` below f2).
    Outside it the measurement rolls off steeply (about 9 dB at 25 Hz and 59 dB at 18.5 kHz
    for a 20 Hz-20 kHz sweep), so nothing should be corrected from data there."""
    return f1 * 2.0**LOW_TAPER_OCT, f2 * 2.0**-HIGH_TAPER_OCT


def minimum_phase(mag: FloatArray, n_fft: int) -> FloatArray:
    """Phase (rad) of the minimum-phase response with magnitude ``mag`` on the rfft grid of
    an ``n_fft``-point transform: fold the real cepstrum of ``ln|H|`` onto positive
    quefrencies."""
    c = np.fft.irfft(np.log(np.maximum(mag, 1e-300)), n_fft)
    fold = np.zeros(n_fft)
    fold[0] = c[0]
    half = (n_fft + 1) // 2
    fold[1:half] = 2.0 * c[1:half]
    if n_fft % 2 == 0:
        fold[n_fft // 2] = c[n_fft // 2]
    return np.asarray(np.angle(np.exp(np.fft.rfft(fold))), dtype=np.float64)


def band_weight(f: FloatArray, spec: SweepSpec) -> FloatArray:
    """1 inside the sweep band, 0 outside it, with raised-cosine tapers in log frequency
    from f1 up ``LOW_TAPER_OCT`` octaves and from f2 down ``HIGH_TAPER_OCT`` octaves."""
    with np.errstate(divide="ignore"):
        lf = np.log2(np.maximum(f, 1e-30))
    lo = np.clip((lf - np.log2(spec.f1)) / LOW_TAPER_OCT, 0.0, 1.0)
    hi = np.clip((np.log2(spec.f2) - lf) / HIGH_TAPER_OCT, 0.0, 1.0)
    return np.asarray((0.5 - 0.5 * np.cos(np.pi * lo)) * (0.5 - 0.5 * np.cos(np.pi * hi)))


@dataclass(frozen=True)
class HarmonicIr:
    """One separated impulse response."""

    order: int
    """1 for the linear response, n for the n-th harmonic."""
    ir: FloatArray
    offset_s: float
    """Time of ``ir[0]`` relative to this response's own time zero (slightly negative: the
    window starts ``pre_s`` early, and the harmonic delays are not whole samples)."""


def separate(
    h: FloatArray,
    spec: SweepSpec,
    n_max: int = 5,
    pre_s: float = 0.002,
    max_ir_s: float = 0.5,
) -> list[HarmonicIr]:
    """Cut the deconvolution result into the linear response and harmonics 2…n_max.

    The n-th harmonic's window runs from ``pre_s`` before its start to ``pre_s`` before the
    (n-1)-th one's, so it captures as much decay as the spacing allows. The linear window is
    ``max_ir_s`` long, or shorter if it would reach the wrapped-around harmonics. Edges get
    short half-Hann tapers.
    """
    fs = spec.fs
    n_fft = len(h)
    pre = round(pre_s * fs)
    out: list[HarmonicIr] = []
    for order in range(1, n_max + 1):
        zero = -spec.harmonic_delay_s(order) * fs  # fractional sample index of time zero
        i0 = round(zero) - pre
        if order == 1:
            room = n_fft - round(spec.harmonic_delay_s(n_max) * fs) - 2 * pre
            i1 = min(round(max_ir_s * fs), room)
        else:
            i1 = round(-spec.harmonic_delay_s(order - 1) * fs) - pre
        if i1 - i0 < 4 * pre:
            raise ValueError(f"harmonic {order} is too close to its neighbour: use a longer sweep")
        idx = np.arange(i0, i1) % n_fft
        out.append(HarmonicIr(order, h[idx] * _taper(i1 - i0, pre), (i0 - zero) / fs))
    return out


def _taper(n: int, n_edge: int) -> FloatArray:
    w = np.ones(n)
    if n_edge > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(n_edge) / n_edge)
        w[:n_edge] = ramp
        w[n - n_edge :] = ramp[::-1]
    return w


def _next_fast_len(n: int) -> int:
    from scipy.fft import next_fast_len

    return int(next_fast_len(n, real=True))
