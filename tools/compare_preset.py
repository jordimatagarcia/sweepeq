"""Compare the measured effect of an EQ preset with the curve its settings predict.

    uv run python tools/compare_preset.py OFF.npz ON.npz PRESET_APO.txt \
        --fs-eq 96000 --plot docs/images/preset_vs_measured.png

OFF and ON are two sweepeq measurements of the same setup, without and with the EQ; their
difference is the EQ as measured. ``--fs-eq`` is the sample rate the EQ runs at (it shapes
the biquads near Nyquist). The level offset between measurement and prediction is reported
separately from the shape error.
"""

import argparse
from pathlib import Path

import numpy as np

from sweepeq.analysis import Curve, response_curve, smooth
from sweepeq.apo import read_apo
from sweepeq.measure import load_linear
from sweepeq.response import frequency_response

BANDS = ((100.0, 2000.0), (2000.0, 5000.0), (5000.0, 16000.0))


def measured_curve(path: Path, freqs: np.ndarray, fraction: float) -> Curve:
    hir, fs, _, _ = load_linear(path)
    return smooth(response_curve(freqs, frequency_response(hir, freqs, fs)), fraction)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("off", type=Path)
    p.add_argument("on", type=Path)
    p.add_argument("preset", type=Path, help="Equalizer APO text preset")
    p.add_argument("--fs-eq", type=float, default=48_000.0)
    p.add_argument("--smooth", type=float, default=6.0)
    p.add_argument("--plot", type=Path)
    args = p.parse_args()

    f = np.geomspace(20.0, 20_000.0, 2000)
    measured = (
        measured_curve(args.on, f, args.smooth).db - measured_curve(args.off, f, args.smooth).db
    )
    predicted = smooth(Curve(f, read_apo(args.preset).response_db(f, args.fs_eq)), args.smooth).db
    in_range = (f >= BANDS[0][0]) & (f <= BANDS[-1][1])
    offset = float(np.median((measured - predicted)[in_range]))
    print(f"level offset (measured - predicted): {offset:+.2f} dB")
    for lo, hi in BANDS:
        k = (f >= lo) & (f <= hi)
        e = measured[k] - predicted[k] - offset
        print(
            f"{lo:6.0f}-{hi:6.0f} Hz: shape error rms {np.sqrt(np.mean(e**2)):.2f} dB, "
            f"max {np.max(np.abs(e)):.1f} dB"
        )

    if args.plot is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.semilogx(f, measured, label="Measured: on - off", color="C2")
        ax.semilogx(f, predicted, label="Predicted from the preset", color="k", ls="--")
        ax.semilogx(
            f, predicted + offset, label=f"Predicted, shifted {offset:+.1f} dB", color="0.5", ls=":"
        )
        ax.set(
            xlabel="Frequency (Hz)",
            ylabel="dB",
            xlim=(40, 16_000),
            title=f"EQ preset: measured vs predicted (1/{args.smooth:g}-octave smoothed)",
        )
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"saved {args.plot}")


if __name__ == "__main__":
    main()
