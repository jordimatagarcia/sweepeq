"""Equalizer APO presets and RBJ shelving filters."""

import numpy as np
import pytest
from scipy import signal

from sweepeq.apo import parse_apo
from sweepeq.plant import peaking_biquad, shelf_biquad

FS = 48_000


def mag_db(sos_row: np.ndarray, f: float) -> float:
    _, h = signal.sosfreqz(sos_row[None, :], worN=[f], fs=FS)
    return float(20 * np.log10(np.abs(h[0])))


@pytest.mark.parametrize("gain", [-6.0, 3.0, 9.0])
def test_shelves_reach_their_gain_and_unity(gain: float) -> None:
    low = shelf_biquad("low", 200.0, gain, 0.7071, FS)
    high = shelf_biquad("high", 4000.0, gain, 0.7071, FS)
    assert mag_db(low, 10.0) == pytest.approx(gain, abs=0.05)
    assert mag_db(low, 10_000.0) == pytest.approx(0.0, abs=0.05)
    assert mag_db(high, 20_000.0) == pytest.approx(gain, abs=0.3)
    assert mag_db(high, 50.0) == pytest.approx(0.0, abs=0.05)
    # half the gain (in dB) at the corner frequency, for both shelves
    assert mag_db(low, 200.0) == pytest.approx(gain / 2, abs=0.05)
    assert mag_db(high, 4000.0) == pytest.approx(gain / 2, abs=0.05)
    with pytest.raises(ValueError):
        shelf_biquad("mid", 1000.0, gain, 0.7, FS)


def test_parse_apo_preset() -> None:
    preset = parse_apo(
        "Preamp: -4.2 dB\n"
        "Filter 1: ON LSC Fc 105 Hz Gain -1.7 dB Q 0.70\n"
        "Filter 2: ON PK Fc 217.4 Hz Gain 4.4 dB Q 2.66\n"
        "Filter 3: OFF PK Fc 1000 Hz Gain 9 dB Q 1\n"
        "Filter 4: ON HS Fc 10000 Hz Gain -4 dB\n"
        "# a comment\n"
    )
    assert preset.preamp_db == -4.2
    assert [b.kind for b in preset.bands] == ["LS", "PK", "HS"]
    assert preset.bands[2].q == pytest.approx(0.7071)
    f = np.array([217.4, 1000.0])
    db = preset.response_db(f, FS)
    # at 217.4 Hz: preamp + the peak, plus the small tails of the shelves
    expected = -4.2 + mag_db(peaking_biquad(217.4, 4.4, 2.66, FS), 217.4)
    assert db[0] == pytest.approx(expected, abs=0.3)


def test_peaking_filter_needs_q() -> None:
    with pytest.raises(ValueError, match="without Q"):
        parse_apo("Filter 1: ON PK Fc 100 Hz Gain 3 dB\n")


def test_empty_preset_is_flat() -> None:
    preset = parse_apo("nothing")
    np.testing.assert_allclose(preset.response_db(np.array([100.0, 1000.0]), FS), 0.0)


@pytest.mark.parametrize(
    "line",
    ["Filter 1: ON HP Fc 80 Hz", "Filter 2: ON NO Fc 500 Hz", "Filter: ON LP Fc 15000 Hz Q 0.7"],
)
def test_unsupported_enabled_filters_are_errors(line: str) -> None:
    # Review fix: these were silently dropped, so the preset read as a different curve.
    with pytest.raises(ValueError, match="unsupported filter"):
        parse_apo(f"Preamp: -3 dB\n{line}\nFilter 3: ON PK Fc 1000 Hz Gain 3 dB Q 1\n")


def test_disabled_unsupported_filters_are_ignored() -> None:
    assert len(parse_apo("Filter 1: OFF HP Fc 80 Hz\n").bands) == 0
