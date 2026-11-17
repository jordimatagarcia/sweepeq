"""M3 acceptance tests: correction filter design.

Targets (docs/design.md): the corrected response matches the target within ±1 dB in the
correction band, boosts never exceed the limit, the FIR filter
realises the designed correction, and the parametric EQ fit recovers known presets.
Dips deeper than the notch depth get progressively less boost.
"""

from pathlib import Path

import numpy as np
import pytest

from sweepeq.analysis import Curve
from sweepeq.apo import format_apo, parse_apo, read_apo
from sweepeq.design import (
    DesignSpec,
    band_fade,
    beta_for_notch,
    build_correction_fir,
    correction_db,
    regularised_inverse,
    soft_limit_boost,
    target_curve,
)
from sweepeq.peq import PeqSpec, fit_peq
from sweepeq.simulated import SimulatedDevice, peaking_biquad

FS = 48_000
F = np.geomspace(10.0, 24_000.0, 4000)
FLAT = Curve(F, np.zeros_like(F))
REPO = Path(__file__).resolve().parents[1]


def device(notch: bool = False) -> Curve:
    bands = [
        peaking_biquad(100.0, 6.0, 1.0, FS),
        peaking_biquad(1000.0, -6.0, 2.0, FS),
        peaking_biquad(3000.0, -8.0, 2.0, FS),
        peaking_biquad(8000.0, 5.0, 1.5, FS),
    ]
    if notch:
        bands.append(peaking_biquad(5000.0, -40.0, 30.0, FS))
    dut = SimulatedDevice(fs=FS, sos=np.vstack(bands), gain=0.5)
    return Curve(F, 20 * np.log10(np.abs(dut.linear_response(F))))


# --- building blocks -----------------------------------------------------------------------


def test_soft_limit_never_exceeds_and_is_exact_below() -> None:
    b = np.linspace(-20, 40, 2001)
    out = soft_limit_boost(b, 6.0, 0.5)
    assert np.max(out) <= 6.0
    small = b <= 3.0
    np.testing.assert_allclose(out[small], b[small], atol=0.002)
    np.testing.assert_array_equal(out[b < 0], b[b < 0])


def test_regularised_inverse() -> None:
    beta = beta_for_notch(20.0)
    d = np.array([4.0, 2.0, 1.0, 0.5, 0.25, 1e-4])
    c = regularised_inverse(d, beta, 24.0)
    np.testing.assert_allclose(c[:3], [0.25, 0.5, 1.0])  # peaks: exact inverse
    assert 20 * np.log10(c[3]) == pytest.approx(6.02, abs=0.2)  # a 6 dB dip: nearly exact
    assert 20 * np.log10(c[5]) < 0.1  # an 80 dB null: left (almost) alone
    deep_cut = regularised_inverse(np.array([1e3]), beta, 24.0)
    assert 20 * np.log10(deep_cut[0]) == pytest.approx(-24.0)


def test_band_fade() -> None:
    spec = DesignSpec(f_lo=100.0, f_hi=10_000.0, taper_oct=1.0)
    w = band_fade(np.array([25.0, 50.0, 100.0, 1000.0, 10_000.0, 20_000.0]), spec)
    np.testing.assert_allclose(w, [0, 0, 1, 1, 1, 0], atol=1e-12)


def test_target_tilt_and_bass() -> None:
    t = target_curve(np.array([125.0, 1000.0, 8000.0]), tilt_db_per_oct=-1.0)
    np.testing.assert_allclose(t.db, [3.0, 0.0, -3.0])
    b = target_curve(np.array([20.0, 105.0, 2000.0]), bass_db=6.0)
    assert b.db[0] == pytest.approx(6.0, abs=0.01)
    assert b.db[1] == pytest.approx(3.0)
    assert b.db[2] == pytest.approx(0.0, abs=0.001)


def test_invalid_specs_are_rejected() -> None:
    for kwargs in (
        dict(f_lo=0.0),
        dict(f_hi=30_000.0),
        dict(boost_limit_db=0.0),
        dict(n_taps=8),
        dict(phase="mixed"),
    ):
        with pytest.raises(ValueError):
            DesignSpec(**kwargs)


# --- design ---------------------------------------------------------------------------------


@pytest.mark.parametrize("phase", ["minimum", "linear"])
def test_corrected_response_within_1_db(phase: str) -> None:
    spec = DesignSpec(f_lo=40.0, f_hi=16_000.0, smoothing=0, phase=phase)  # type: ignore[arg-type]
    meas = device()
    design = build_correction_fir(correction_db(meas, FLAT, spec), spec)
    g = np.geomspace(60.0, 12_000.0, 500)
    corrected = meas.at(g) + design.magnitude_db(g)
    assert np.max(np.abs(corrected - np.median(corrected))) < 1.0


def test_boost_limit_caps_a_deep_notch() -> None:
    spec = DesignSpec(smoothing=0, boost_limit_db=6.0)
    corr = correction_db(device(notch=True), FLAT, spec)
    assert np.max(corr.db) <= 6.0  # the 40 dB notch gets at most the limit, not +40 dB
    design = build_correction_fir(corr, spec)
    assert design.peak_gain_db <= 6.2  # FIR ripple on top of the limit


def test_deeper_dips_get_less_boost_beyond_the_notch_depth() -> None:
    beta = beta_for_notch(20.0)
    depths_db = np.array([-25.0, -30.0, -40.0, -60.0, -80.0])
    boost = 20 * np.log10(regularised_inverse(10 ** (depths_db / 20), beta, 24.0))
    assert np.all(np.diff(boost) < 0)
    assert boost[-1] < 0.1


def test_no_correction_outside_the_band() -> None:
    spec = DesignSpec(f_lo=200.0, f_hi=5000.0, taper_oct=0.5, smoothing=0)
    design = build_correction_fir(correction_db(device(), FLAT, spec), spec)
    outside = np.array([40.0, 80.0, 12_000.0, 18_000.0])
    assert np.max(np.abs(design.magnitude_db(outside))) < 0.5


@pytest.mark.parametrize("phase", ["minimum", "linear"])
def test_fir_realises_the_correction(phase: str) -> None:
    spec = DesignSpec(smoothing=6, n_taps=4096, phase=phase)  # type: ignore[arg-type]
    corr = correction_db(device(), FLAT, spec)
    design = build_correction_fir(corr, spec)
    g = np.geomspace(100.0, 16_000.0, 400)
    assert np.max(np.abs(design.magnitude_db(g) - corr.at(g))) < 0.25


def test_phase_properties() -> None:
    corr = correction_db(device(), FLAT, DesignSpec(smoothing=0))
    lin = build_correction_fir(corr, DesignSpec(phase="linear", n_taps=1024))
    np.testing.assert_allclose(lin.taps, lin.taps[::-1], atol=1e-9)
    assert lin.delay_samples == pytest.approx(511.5)
    mp = build_correction_fir(corr, DesignSpec(phase="minimum", n_taps=1024))
    energy = np.cumsum(mp.taps**2) / np.sum(mp.taps**2)
    assert energy[50] > 0.9  # minimum phase: the energy is at the start
    assert mp.delay_samples == 0.0
    assert mp.worst_case_gain_db >= mp.peak_gain_db


def test_alignment_band_sets_the_level() -> None:
    # A device 10 dB louder gets the same correction: only the shape is corrected.
    spec = DesignSpec(smoothing=0)
    loud = Curve(F, device().db + 10.0)
    np.testing.assert_allclose(
        correction_db(loud, FLAT, spec).db, correction_db(device(), FLAT, spec).db, atol=1e-9
    )


# --- parametric EQ -------------------------------------------------------------------------


def test_peq_recovers_a_known_preset() -> None:
    preset = read_apo(REPO / "docs/data/headphone_preset_apo.txt")
    curve = Curve(F, preset.magnitude_db(F, FS) - preset.preamp_db)
    fit = fit_peq(curve, PeqSpec(n_bands=len(preset.bands)))
    assert fit.max_error_db < 0.1
    assert fit.preset.preamp_db <= -max(np.max(fit.fitted_db), 0.0) + 1e-9


def test_peq_fits_a_device_correction() -> None:
    corr = correction_db(device(), FLAT, DesignSpec(smoothing=0, f_lo=40.0))
    fit = fit_peq(corr, PeqSpec(n_bands=6, f_lo=40.0))
    assert fit.rms_error_db < 0.3
    assert max(b.gain_db for b in fit.preset.bands) <= 6.0


def test_apo_round_trip() -> None:
    preset = read_apo(REPO / "docs/data/headphone_preset_apo.txt")
    again = parse_apo(format_apo(preset))
    assert again.preamp_db == pytest.approx(preset.preamp_db, abs=0.05)
    for a, b in zip(again.bands, preset.bands, strict=True):
        assert a.kind == b.kind
        assert a.fc == pytest.approx(b.fc, abs=0.05)
        assert a.gain_db == pytest.approx(b.gain_db)


def test_design_cli(tmp_path: Path) -> None:
    from scipy.io import wavfile

    from sweepeq.cli import main

    m = tmp_path / "m.npz"
    assert main(["simulate", "--duration", "2", "--out", str(m)]) == 0
    fir, apo = tmp_path / "f.wav", tmp_path / "p.txt"
    assert (
        main(
            ["design", str(m), "--fir", str(fir), "--peq", "4", "--apo", str(apo), "--taps", "2048"]
        )
        == 0
    )
    fs, taps = wavfile.read(fir)
    assert fs == 48_000 and len(taps) == 2048 and taps.dtype == np.float32
    assert len(read_apo(apo).bands) == 4


def test_design_cli_corrects_the_treble(tmp_path: Path) -> None:
    # Review fix: the CLI limited correction to [2*f1, f2/2] (40 Hz-10 kHz for a 20 Hz-20 kHz
    # sweep), although the measurement is accurate to 17.8 kHz. A +6 dB peak at 14 kHz must
    # now be corrected, and nothing outside the valid range may be read.
    from scipy.io import wavfile

    from sweepeq.cli import main
    from sweepeq.design import taps_response_db
    from sweepeq.measure import SimulatedRecorder, measure
    from sweepeq.sweep import SweepSpec

    dut = SimulatedDevice(fs=FS, sos=np.vstack([peaking_biquad(14_000.0, 6.0, 3.0, FS)]))
    m = measure(SweepSpec(duration_s=2.0), SimulatedRecorder(dut))
    path, fir = tmp_path / "m.npz", tmp_path / "f.wav"
    m.save(path)
    assert main(["design", str(path), "--fir", str(fir), "--smooth", "0"]) == 0
    _, taps = wavfile.read(fir)
    g = np.array([1000.0, 14_000.0])
    corrected = taps_response_db(taps.astype(np.float64), g, FS) + 20 * np.log10(
        np.abs(dut.linear_response(g))
    )
    assert corrected[1] - corrected[0] == pytest.approx(0.0, abs=0.5)
    # below the valid range (28 Hz) the correction fades out instead of reading the roll-off
    low = taps_response_db(taps.astype(np.float64), np.array([12.0, 20.0]), FS)
    assert np.max(np.abs(low)) < 0.5
