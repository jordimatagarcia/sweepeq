"""M2 analysis: calibration files, smoothing, averaging, comparison with REW exports."""

from pathlib import Path

import numpy as np
import pytest

from sweepeq.analysis import (
    Curve,
    apply_calibration,
    average,
    compare,
    read_text_curve,
    smooth,
    write_text_curve,
)
from sweepeq.cli import main
from sweepeq.simulated import SimulatedDevice, peaking_biquad

GRID = np.geomspace(20.0, 20_000.0, 2000)


def test_reads_calibration_and_rew_formats(tmp_path: Path) -> None:
    cal = tmp_path / "cal.txt"
    cal.write_text('"Sens Factor =-1.23dB, SERNO: 000"\n20 -2.5\n1000\t0.0\n20000, 3.0, 12.5\n')
    curve = read_text_curve(cal)
    np.testing.assert_allclose(curve.freqs, [20, 1000, 20000])
    np.testing.assert_allclose(curve.db, [-2.5, 0.0, 3.0])
    rew = tmp_path / "rew.txt"
    rew.write_text(
        "* Measurement data measured by REW\n* Freq(Hz) SPL(dB) Phase(degrees)\n"
        "100.0 80.0 10.0\n200.0 81.0 5.0\n"
    )
    assert read_text_curve(rew).db.tolist() == [80.0, 81.0]
    with pytest.raises(ValueError, match="no frequency"):
        (tmp_path / "empty.txt").write_text("nothing here\n")
        read_text_curve(tmp_path / "empty.txt")


def test_text_round_trip(tmp_path: Path) -> None:
    c = Curve(GRID, np.sin(np.log(GRID)))
    write_text_curve(c, tmp_path / "c.txt")
    back = read_text_curve(tmp_path / "c.txt")
    np.testing.assert_allclose(back.db, c.db, atol=1e-4)


def test_calibration_is_subtracted() -> None:
    cal = Curve(np.array([20.0, 20_000.0]), np.array([-3.0, 3.0]))
    out = apply_calibration(Curve(GRID, np.zeros_like(GRID)), cal)
    assert out.db[0] == pytest.approx(3.0)
    assert out.db[-1] == pytest.approx(-3.0)


def test_smoothing_keeps_flat_and_slopes_and_removes_ripple() -> None:
    flat = smooth(Curve(GRID, np.full_like(GRID, 5.0)), 6)
    np.testing.assert_allclose(flat.db, 5.0, atol=1e-9)
    # ripple faster than 1/6 octave is averaged out (in power)
    ripple = 3.0 * np.sin(2 * np.pi * np.log2(GRID) * 24)
    smoothed = smooth(Curve(GRID, ripple), 3)
    inner = (GRID > 100) & (GRID < 10_000)
    assert np.max(np.abs(smoothed.db[inner] - np.mean(smoothed.db[inner]))) < 0.3
    with pytest.raises(ValueError):
        smooth(flat, 0)


def test_power_average() -> None:
    a = Curve(GRID, np.zeros_like(GRID))
    b = Curve(GRID, np.full_like(GRID, -np.inf))
    np.testing.assert_allclose(average([a, b]).db, 10 * np.log10(0.5))


def test_compare_ignores_absolute_level() -> None:
    db = np.log2(GRID)
    cmp = compare(Curve(GRID, db + 7.0), Curve(GRID, db), 50, 10_000)
    assert cmp.max_abs_db < 1e-9


def test_analyse_cli_against_a_rew_style_reference(tmp_path: Path) -> None:
    # The simulated device's exact response, written as a REW export, must match the
    # measurement of it: the end-to-end check M2 repeats with real hardware.
    assert main(["simulate", "--duration", "2", "--out", str(tmp_path / "m.npz")]) == 0
    fs = 48_000
    dut = SimulatedDevice(
        fs=fs,
        sos=np.vstack([peaking_biquad(80.0, 6.0, 1.0, fs), peaking_biquad(3000.0, -6.0, 2.0, fs)]),
    )
    ref = Curve(GRID, 20 * np.log10(np.abs(dut.linear_response(GRID))) + 80.0)
    write_text_curve(ref, tmp_path / "rew.txt")
    out = tmp_path / "result.txt"
    assert (
        main(
            [
                "analyse",
                str(tmp_path / "m.npz"),
                "--compare",
                str(tmp_path / "rew.txt"),
                "--export",
                str(out),
            ]
        )
        == 0
    )
    result = read_text_curve(out)
    cmp = compare(result, smooth(ref, 6), 50, 10_000)
    assert cmp.max_abs_db < 0.2
