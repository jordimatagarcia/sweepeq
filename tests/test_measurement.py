"""M1 acceptance tests: the measurement recovers known simulated systems.

Targets (docs/measurement.md): magnitude within 0.1 dB in the passband [2·f1, f2/2], delay
to the sample, second and third harmonic levels within 1 dB.
"""

import numpy as np
import pytest

from sweepeq.deconv import HarmonicIr, band_weight, deconvolve, minimum_phase, separate
from sweepeq.measure import SimulatedRecorder, measure
from sweepeq.plant import Plant, peaking_biquad
from sweepeq.response import delay_s, frequency_response, harmonic_distortion, hd_range, to_db
from sweepeq.sweep import SweepSpec, synchronized_sweep

FS = 48_000
SPEC = SweepSpec(duration_s=3.0, fs=FS)
PASSBAND = np.geomspace(2 * SPEC.f1, SPEC.f2 / 2, 400)


def eq_plant(**kw: object) -> Plant:
    sos = np.vstack(
        [
            peaking_biquad(100.0, 6.0, 1.0, FS),
            peaking_biquad(3000.0, -8.0, 2.0, FS),
            peaking_biquad(10_000.0, 4.0, 0.7, FS),
        ]
    )
    return Plant(fs=FS, sos=sos, **kw)  # type: ignore[arg-type]


def linear_ir(plant: Plant, seed: int = 0) -> HarmonicIr:
    return measure(SPEC, SimulatedRecorder(plant, seed)).linear


@pytest.fixture(scope="module")
def loopback() -> HarmonicIr:
    return linear_ir(Plant(fs=FS))


# --- sweep ------------------------------------------------------------------------------


def test_sweep_is_synchronized() -> None:
    # f1·L must be an integer for the harmonics to start in phase (Novak et al. 2015).
    for duration in (1.0, 2.7, 5.0, 10.0):
        spec = SweepSpec(duration_s=duration)
        assert spec.f1 * spec.rate_s == pytest.approx(round(spec.f1 * spec.rate_s), abs=1e-9)
        assert abs(spec.length_s - duration) <= 1.0 / spec.f1 * np.log(spec.f2 / spec.f1)


def test_sweep_amplitude_and_fades() -> None:
    x = synchronized_sweep(SPEC)
    assert np.max(np.abs(x)) == pytest.approx(SPEC.amplitude, rel=1e-3)
    assert x[0] == 0.0
    assert abs(x[-1]) < 1e-3


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(f1=0.0),
        dict(f2=30_000.0),
        dict(f1=100.0, f2=50.0),
        dict(duration_s=0.0),
        dict(duration_s=0.01),
    ],
)
def test_invalid_sweeps_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        SweepSpec(**kwargs)  # type: ignore[arg-type]


# --- linear response -----------------------------------------------------------------------


def test_flat_system_is_flat() -> None:
    err = to_db(frequency_response(linear_ir(Plant(fs=FS)), PASSBAND, FS))
    assert np.max(np.abs(err)) < 0.01


@pytest.mark.parametrize("delay", [0, 123, 4800])
def test_eq_magnitude_within_0_1_db(delay: int) -> None:
    plant = eq_plant(delay_samples=delay, noise_rms=1e-4, gain=0.7)
    ir = linear_ir(plant)
    err = to_db(frequency_response(ir, PASSBAND, FS)) - to_db(plant.linear_response(PASSBAND))
    assert np.max(np.abs(err)) < 0.1


@pytest.mark.parametrize("delay", [0, 1, 123, 4800])
def test_delay_to_the_sample(delay: int, loopback: HarmonicIr) -> None:
    measured = delay_s(linear_ir(Plant(fs=FS, delay_samples=delay)), FS, loopback) * FS
    assert measured == pytest.approx(delay, abs=0.01)
    # With a filter the peak also moves by the filter's own group delay, so the delay is
    # checked relative to the same filter measured without it.
    eq0 = linear_ir(eq_plant())
    with_eq = delay_s(linear_ir(eq_plant(delay_samples=delay, noise_rms=1e-4)), FS, eq0) * FS
    assert with_eq == pytest.approx(delay, abs=0.01)


# --- harmonic distortion -------------------------------------------------------------------


@pytest.mark.parametrize(("a2", "a3"), [(0.1, 0.0), (0.0, 0.05), (0.1, 0.05), (0.02, 0.01)])
def test_harmonic_levels_within_1_db(a2: float, a3: float) -> None:
    plant = Plant(fs=FS, poly=(a2, a3), noise_rms=1e-5)
    m = measure(SPEC, SimulatedRecorder(plant))
    a = SPEC.amplitude
    fundamental = a + 0.75 * a3 * a**3
    expected = {2: a2 * a**2 / 2 / fundamental, 3: a3 * a**3 / 4 / fundamental}
    for order in (2, 3):
        if expected[order] == 0.0:
            continue
        f = np.geomspace(*hd_range(SPEC.f1, SPEC.f2, order), 60)
        hd = to_db(harmonic_distortion(m.linear, m.harmonics[order - 1], f, FS))
        assert np.max(np.abs(hd - 20 * np.log10(expected[order]))) < 1.0, order


def test_linear_system_has_no_harmonics() -> None:
    m = measure(SPEC, SimulatedRecorder(eq_plant(noise_rms=1e-6)))
    f = np.geomspace(*hd_range(SPEC.f1, SPEC.f2, 2), 60)
    hd = to_db(harmonic_distortion(m.linear, m.harmonics[1], f, FS))
    assert np.max(hd) < -80.0


# --- building blocks -----------------------------------------------------------------------


def test_minimum_phase_of_a_known_filter() -> None:
    # A first-order low-pass is minimum phase: its phase must be recovered from |H| alone.
    n = 4096
    w = 2 * np.pi * np.fft.rfftfreq(n)
    h = 1 / (1 - 0.9 * np.exp(-1j * w))
    phase = minimum_phase(np.abs(h), n)
    assert np.max(np.abs(np.angle(np.exp(1j * (phase - np.angle(h)))))) < 1e-6


def test_band_weight_edges() -> None:
    f = np.array([0.0, SPEC.f1, 2 * SPEC.f1, 1000.0, SPEC.f2 / 2, SPEC.f2, FS / 2])
    np.testing.assert_allclose(band_weight(f, SPEC), [0, 0, 1, 1, 1, 0, 0], atol=1e-12)


def test_too_short_sweep_for_harmonic_separation() -> None:
    spec = SweepSpec(duration_s=0.3)
    x = synchronized_sweep(spec)
    with pytest.raises(ValueError, match="longer sweep"):
        separate(deconvolve(x, x, spec), spec, n_max=8, pre_s=0.01)


def test_save_and_cli(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from sweepeq.cli import main

    out = tmp_path / "m.npz"
    assert main(["simulate", "--duration", "2", "--out", str(out)]) == 0
    data = np.load(out)
    assert int(data["fs"]) == 48_000
    assert data["ir_1"].size > 0 and data["ir_3"].size > 0
