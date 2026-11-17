"""Reading AutoEq results and comparing sweepeq's correction with AutoEq's.

The AutoEq results CSV holds a measurement (``raw``), the target, and AutoEq's correction
(``equalization``). sweepeq designs its own correction from ``raw`` and ``target`` with the
same +6 dB boost limit; the two are compared on the corrections themselves (after removing
their level difference) and on how close each brings the headphone to the target.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sweepeq.analysis import Curve, smooth
from sweepeq.design import DesignSpec, correction_db

FloatArray = NDArray[np.float64]
BANDS = (
    (20.0, 200.0),
    (200.0, 2000.0),
    (2000.0, 6000.0),
    (6000.0, 10_000.0),
    (10_000.0, 20_000.0),
)
ALIGN = (300.0, 3000.0)


@dataclass(frozen=True)
class AutoEqResult:
    freqs: FloatArray
    raw: Curve
    target: Curve
    equalization: FloatArray


def read_autoeq_csv(path: Path) -> AutoEqResult:
    d = np.genfromtxt(path, delimiter=",", names=True)
    f = np.asarray(d["frequency"], dtype=np.float64)
    return AutoEqResult(f, Curve(f, d["raw"]), Curve(f, d["target"]), d["equalization"])


def sweepeq_correction(r: AutoEqResult, boost_limit_db: float = 6.0) -> FloatArray:
    spec = DesignSpec(f_lo=20.0, f_hi=20_000.0, smoothing=6.0, boost_limit_db=boost_limit_db)
    return correction_db(r.raw, r.target, spec).db


def level_free(diff: FloatArray, freqs: FloatArray) -> FloatArray:
    """Remove the mean over ``ALIGN``: the two tools normalise the level differently."""
    k = (freqs >= ALIGN[0]) & (freqs <= ALIGN[1])
    return diff - float(np.mean(diff[k]))


def residual(r: AutoEqResult, eq: FloatArray) -> FloatArray:
    """Equalised response minus target, 1/6-octave smoothed, level-aligned."""
    e = smooth(Curve(r.freqs, r.raw.db + eq - r.target.db), 6.0).db
    return level_free(e, r.freqs)


def rms(x: FloatArray) -> float:
    return float(np.sqrt(np.mean(x**2)))
