"""The README figure: a published headphone measurement before and after correction.

    uv run python tools/readme_figure.py docs/images/hd600_corrected.png

The Sennheiser HD 600 measurement (oratory1990, via AutoEq; testdata/autoeq) is corrected
towards its target with a 4096-tap minimum-phase filter from ``sweepeq design``'s method.
The "corrected" curve is the measurement plus the response of the filter as the Rust
engine runs it: an impulse is passed through ``sweepeq_engine`` and its spectrum taken.
"""

import sys
from pathlib import Path

import numpy as np

from sweepeq.analysis import Curve, smooth
from sweepeq.autoeq import read_autoeq_csv
from sweepeq.design import DesignSpec, build_correction_fir, correction_db

ROOT = Path(__file__).resolve().parents[1]
FS = 48_000


def engine_response_db(taps: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    import sweepeq_engine

    n = 1 << 16
    impulse = np.zeros(n, np.float32)
    impulse[0] = 1.0
    c = sweepeq_engine.Convolver(taps, method="partitioned", block=256)
    y = c.process(np.concatenate([impulse, np.zeros(c.latency, np.float32)]))[c.latency :]
    spec = np.fft.rfft(y.astype(np.float64))
    f = np.fft.rfftfreq(n, 1 / FS)
    return np.asarray(np.interp(freqs, f, 20 * np.log10(np.abs(spec))))


def main(out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = read_autoeq_csv(ROOT / "testdata/autoeq/hd600_oratory1990.csv")
    spec = DesignSpec(f_lo=20.0, f_hi=20_000.0, n_taps=4096, fs=FS)
    design = build_correction_fir(correction_db(r.raw, r.target, spec), spec)
    f = r.freqs
    corrected = r.raw.db + engine_response_db(design.taps.astype(np.float32), f)
    k = (f >= 300) & (f <= 3000)

    def aligned(db: np.ndarray) -> np.ndarray:
        return db - np.mean(db[k] - r.target.db[k])

    before = smooth(Curve(f, aligned(r.raw.db)), 6).db
    after = smooth(Curve(f, aligned(corrected)), 6).db
    band = f <= 10_000
    rms_before = np.sqrt(np.mean((before - r.target.db)[band] ** 2))
    rms_after = np.sqrt(np.mean((after - r.target.db)[band] ** 2))

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.semilogx(f, r.target.db, "k--", lw=1.2, label="target")
    ax.semilogx(f, before, color="C1", label=f"measured ({rms_before:.1f} dB rms from target)")
    ax.semilogx(f, after, color="C0", lw=2, label=f"corrected ({rms_after:.1f} dB rms)")
    ax.set(
        xlim=(20, 20_000),
        xlabel="Frequency (Hz)",
        ylabel="Level (dB)",
        title="Sennheiser HD 600: corrected by a 4096-tap filter run in the Rust engine",
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    ax.text(
        0.99,
        0.02,
        "measurement: oratory1990 via AutoEq · 1/6-octave smoothing · rms over 20 Hz-10 kHz",
        transform=ax.transAxes,
        ha="right",
        fontsize=7,
        color="0.4",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"saved {out}: {rms_before:.2f} -> {rms_after:.2f} dB rms (20 Hz-10 kHz)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/images/hd600_corrected.png")
