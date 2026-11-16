"""Comparison with AutoEq on a published measurement (testdata/autoeq, MIT-licensed).

Given the same measurement and target, sweepeq's correction must match AutoEq's to within
0.6 dB below 6 kHz (above it, AutoEq follows the treble less closely by design), and bring
the headphone as close to the target as AutoEq does.
"""

from pathlib import Path

import numpy as np

from sweepeq.autoeq import level_free, read_autoeq_csv, residual, rms, sweepeq_correction

REPO = Path(__file__).resolve().parents[1]

CSV = REPO / "testdata/autoeq/hd600_oratory1990.csv"


def test_correction_matches_autoeq_below_6_khz() -> None:
    r = read_autoeq_csv(CSV)
    diff = level_free(sweepeq_correction(r) - r.equalization, r.freqs)
    k = r.freqs < 6000.0
    assert np.max(np.abs(diff[k])) < 0.6
    mid = (r.freqs >= 200.0) & (r.freqs < 6000.0)
    assert rms(diff[mid]) < 0.1


def test_corrected_headphone_as_close_to_target_as_autoeq() -> None:
    r = read_autoeq_csv(CSV)
    k = r.freqs <= 10_000.0
    ours = rms(residual(r, sweepeq_correction(r))[k])
    theirs = rms(residual(r, r.equalization)[k])
    uncorrected = rms(residual(r, np.zeros_like(r.freqs))[k])
    assert ours < theirs + 0.1
    assert ours < uncorrected / 3


def test_boost_limit_matches_autoeq() -> None:
    r = read_autoeq_csv(CSV)
    assert np.max(sweepeq_correction(r)) <= 6.0
    assert np.max(r.equalization) <= 6.01
