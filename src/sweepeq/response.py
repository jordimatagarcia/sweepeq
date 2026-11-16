"""Frequency responses, delay and harmonic distortion from separated impulse responses."""

import numpy as np
from numpy.typing import NDArray

from sweepeq.deconv import HarmonicIr

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def frequency_response(hir: HarmonicIr, freqs: FloatArray, fs: float) -> ComplexArray:
    """Complex response at ``freqs`` (Hz), with the window offset removed from the phase so
    the result is relative to the response's own time zero."""
    n = np.arange(len(hir.ir))
    t = n / fs + hir.offset_s
    # Direct DFT at arbitrary frequencies: exact, and cheap for the few hundred points a
    # plot or a test needs.
    kernel = np.exp(-2j * np.pi * np.outer(freqs, t))
    return np.asarray(kernel @ hir.ir, dtype=np.complex128)


def delay_s(hir: HarmonicIr, fs: float, reference: HarmonicIr | None = None) -> float:
    """Arrival time of the impulse response: its peak, refined by parabolic interpolation.

    The measurement's own causal band limit (see ``deconv``) delays the peak by a few
    samples. Pass ``reference``, the same measurement of a direct loopback (no device), and
    that delay is subtracted, as with a hardware loopback reference.
    """
    if reference is not None:
        return delay_s(hir, fs) - delay_s(reference, fs)
    a = np.abs(hir.ir)
    k = int(np.argmax(a))
    frac = 0.0
    if 0 < k < len(a) - 1:
        y0, y1, y2 = a[k - 1], a[k], a[k + 1]
        denom = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    return (k + frac) / fs + hir.offset_s


def harmonic_distortion(
    linear: HarmonicIr, harmonic: HarmonicIr, freqs: FloatArray, fs: float
) -> FloatArray:
    """Level of the n-th harmonic relative to the fundamental, for an input tone at each of
    ``freqs``: ``|H_n(n·f)| / |H_1(f)|``.

    For an exponential sweep this needs no further scaling: the n-th harmonic sweeps n times
    faster through its band, which exactly cancels the sweep spectrum's 1/√f slope (by the
    stationary-phase approximation; see docs/measurement.md).
    """
    n = harmonic.order
    h_n = frequency_response(harmonic, n * freqs, fs)
    h_1 = frequency_response(linear, freqs, fs)
    return np.asarray(np.abs(h_n) / np.abs(h_1), dtype=np.float64)


HD_LOW_OCT = 2.0
"""Harmonic distortion is reported from this many octaves above the sweep start."""


def hd_range(f1: float, f2: float, order: int) -> tuple[float, float]:
    """Fundamental frequencies over which the n-th harmonic level is valid.

    The n-th harmonic sweep only starts at ``n·f1``, so its response has a band edge there
    whose ringing the window truncates; two octaves above ``f1`` that is negligible. At the
    top, ``n·f`` must stay at or below ``f2/2`` (the analysis passband). For distortion at
    lower frequencies, start the sweep lower.
    """
    return f1 * 2.0**HD_LOW_OCT, f2 / 2.0 / order


def to_db(x: FloatArray | ComplexArray) -> FloatArray:
    return np.asarray(20.0 * np.log10(np.maximum(np.abs(x), 1e-300)), dtype=np.float64)
