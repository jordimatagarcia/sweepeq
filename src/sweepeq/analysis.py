"""Post-processing of measured responses: microphone calibration, fractional-octave
smoothing, averaging over positions, and comparison with a reference measurement."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Curve:
    """A magnitude response on a frequency grid."""

    freqs: FloatArray
    db: FloatArray

    def at(self, freqs: FloatArray) -> FloatArray:
        """Level at ``freqs``, interpolated linearly in log frequency (edges held)."""
        return np.asarray(np.interp(np.log(freqs), np.log(self.freqs), self.db), dtype=np.float64)


def response_curve(freqs: FloatArray, response: NDArray[np.complex128]) -> Curve:
    return Curve(np.asarray(freqs, dtype=np.float64), 20.0 * np.log10(np.abs(response)))


def write_text_curve(curve: Curve, path: Path) -> None:
    """Write ``frequency level`` lines, the format REW imports."""
    lines = ["* sweepeq frequency response: frequency (Hz), level (dB)"]
    lines += [f"{f:.4f} {d:.4f}" for f, d in zip(curve.freqs, curve.db, strict=True)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_text_curve(path: Path) -> Curve:
    """Read a frequency/level text file: a microphone calibration file (miniDSP, Dayton,
    and most others) or a REW "Export measurement as text" file.

    Lines are ``frequency level [phase]``, separated by spaces, tabs or commas. Lines that
    do not start with a number (headers, ``*`` comments, ``"Sens Factor"`` lines) are
    skipped.
    """
    freqs: list[float] = []
    levels: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.replace(",", " ").replace("\t", " ").split()
        if len(parts) < 2:
            continue
        try:
            f, level = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if f > 0.0:
            freqs.append(f)
            levels.append(level)
    if len(freqs) < 2:
        raise ValueError(f"{path}: no frequency/level data found")
    order = np.argsort(freqs)
    return Curve(np.asarray(freqs)[order], np.asarray(levels)[order])


def apply_calibration(curve: Curve, calibration: Curve) -> Curve:
    """Remove the microphone's own response: a calibration file lists the microphone's
    deviation, which is subtracted from the measured level."""
    return Curve(curve.freqs, curve.db - calibration.at(curve.freqs))


def smooth(curve: Curve, fraction: float) -> Curve:
    """1/``fraction``-octave smoothing: each point becomes the power average over a
    window of that width centred on it (log-frequency, rectangular, as REW's
    "1/N octave" smoothing)."""
    if fraction <= 0.0:
        raise ValueError("fraction must be positive (e.g. 6 for 1/6 octave)")
    half = 0.5 / fraction
    lf = np.log2(curve.freqs)
    power = 10.0 ** (curve.db / 10.0)
    # Cumulative sum over log frequency, so every window is two lookups.
    dlf = np.gradient(lf)
    csum = np.concatenate([[0.0], np.cumsum(power * dlf)])
    cw = np.concatenate([[0.0], np.cumsum(dlf)])
    lo = np.searchsorted(lf, lf - half, side="left")
    hi = np.searchsorted(lf, lf + half, side="right")
    avg = (csum[hi] - csum[lo]) / (cw[hi] - cw[lo])
    return Curve(curve.freqs, 10.0 * np.log10(avg))


def average(curves: list[Curve]) -> Curve:
    """Power average of several measurements (e.g. microphone positions) on the first
    curve's grid."""
    if not curves:
        raise ValueError("nothing to average")
    grid = curves[0].freqs
    power = np.mean([10.0 ** (c.at(grid) / 10.0) for c in curves], axis=0)
    return Curve(grid, 10.0 * np.log10(power))


def normalise(curve: Curve, at_hz: float = 1000.0) -> Curve:
    """Shift so the level at ``at_hz`` is 0 dB."""
    return Curve(curve.freqs, curve.db - float(curve.at(np.array([at_hz]))[0]))


@dataclass(frozen=True)
class Comparison:
    freqs: FloatArray
    difference_db: FloatArray

    @property
    def max_abs_db(self) -> float:
        return float(np.max(np.abs(self.difference_db)))


def compare(
    measured: Curve, reference: Curve, f_lo: float, f_hi: float, n: int = 300
) -> Comparison:
    """Level difference over ``[f_lo, f_hi]`` after normalising both at 1 kHz (sweepeq and
    REW use different absolute references)."""
    grid = np.geomspace(f_lo, f_hi, n)
    a = normalise(measured)
    b = normalise(reference)
    return Comparison(grid, a.at(grid) - b.at(grid))
