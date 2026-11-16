"""Write the golden vectors the Rust engine is tested against.

    uv run python tools/make_golden.py

Each case under ``testdata/golden/<name>/`` holds a correction filter designed by the Python
reference (``sweepeq.design``), an input signal, and the expected output: the causal
convolution of the two, computed in float64. Files are raw little-endian arrays
(``taps.f32``, ``input.f32``, ``expected.f64``) plus ``meta.json``.
"""

import json
from pathlib import Path

import numpy as np
from scipy import signal

from sweepeq.analysis import Curve
from sweepeq.autoeq import read_autoeq_csv
from sweepeq.design import DesignSpec, correction_db, design_fir
from sweepeq.plant import Plant, peaking_biquad
from sweepeq.sweep import SweepSpec, synchronized_sweep

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "testdata" / "golden"
FS = 48_000
N_INPUT = 12_000


def device_correction(n_taps: int, phase: str) -> np.ndarray:
    f = np.geomspace(10.0, 24_000.0, 4000)
    sos = np.vstack(
        [
            peaking_biquad(100.0, 6.0, 1.0, FS),
            peaking_biquad(1000.0, -6.0, 2.0, FS),
            peaking_biquad(3000.0, -8.0, 2.0, FS),
            peaking_biquad(8000.0, 5.0, 1.5, FS),
        ]
    )
    measured = Curve(f, 20 * np.log10(np.abs(Plant(fs=FS, sos=sos).linear_response(f))))
    spec = DesignSpec(f_lo=40.0, smoothing=0, n_taps=n_taps, phase=phase)  # type: ignore[arg-type]
    return design_fir(correction_db(measured, Curve(f, np.zeros_like(f)), spec), spec).taps


def hd600_correction(n_taps: int) -> np.ndarray:
    r = read_autoeq_csv(ROOT / "testdata/autoeq/hd600_oratory1990.csv")
    spec = DesignSpec(f_lo=20.0, f_hi=20_000.0, n_taps=n_taps)
    return design_fir(correction_db(r.raw, r.target, spec), spec).taps


def inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(2026)
    impulse = np.zeros(N_INPUT)
    impulse[[0, 1000, 5000]] = [1.0, -0.5, 0.25]
    sweep = synchronized_sweep(SweepSpec(duration_s=0.25, fs=FS, f1=20.0, amplitude=0.5))
    return {
        "noise": rng.uniform(-0.5, 0.5, N_INPUT),
        "impulses": impulse,
        "sweep": np.pad(sweep, (0, max(0, N_INPUT - len(sweep))))[:N_INPUT],
    }


def write_case(name: str, taps: np.ndarray, x: np.ndarray, note: str) -> None:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    taps32 = taps.astype("<f4")
    x32 = x.astype("<f4")
    # the reference uses the f32-rounded values the engine sees, convolved in float64
    expected = signal.lfilter(taps32.astype(np.float64), [1.0], x32.astype(np.float64))
    taps32.tofile(d / "taps.f32")
    x32.tofile(d / "input.f32")
    expected.astype("<f8").tofile(d / "expected.f64")
    meta = {"taps": len(taps32), "samples": len(x32), "fs": FS, "note": note}
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"{name}: {len(taps32)} taps, {len(x32)} samples")


def main() -> None:
    sig = inputs()
    write_case(
        "device_min4096_noise",
        device_correction(4096, "minimum"),
        sig["noise"],
        "minimum-phase correction of a 4-band simulated device; white noise",
    )
    write_case(
        "device_lin1023_sweep",
        device_correction(1023, "linear"),
        sig["sweep"],
        "linear-phase correction, odd length; exponential sweep",
    )
    write_case(
        "hd600_min2048_impulses",
        hd600_correction(2048),
        sig["impulses"],
        "minimum-phase HD 600 correction (AutoEq data); three impulses",
    )
    write_case(
        "short_37_noise",
        device_correction(4096, "minimum")[:37],
        sig["noise"],
        "a 37-tap filter: shorter than any block size",
    )


if __name__ == "__main__":
    main()
