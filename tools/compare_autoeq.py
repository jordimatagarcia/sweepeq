"""Compare sweepeq's correction with AutoEq's for the same measurement and target.

    uv run python tools/compare_autoeq.py testdata/autoeq/hd600_oratory1990.csv \
        --peq testdata/autoeq/hd600_oratory1990_ParametricEQ.txt \
        --plot docs/images/autoeq_hd600.png

See ``sweepeq.autoeq`` for what is compared.
"""

import argparse
from pathlib import Path

import numpy as np

from sweepeq.apo import read_apo
from sweepeq.autoeq import (
    ALIGN,
    BANDS,
    level_free,
    read_autoeq_csv,
    residual,
    rms,
    sweepeq_correction,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("csv", type=Path, help="AutoEq results CSV")
    p.add_argument("--peq", type=Path, help="AutoEq ParametricEQ.txt, compared as well")
    p.add_argument("--plot", type=Path)
    args = p.parse_args()

    r = read_autoeq_csv(args.csv)
    f = r.freqs
    ours = sweepeq_correction(r)
    diff = level_free(ours - r.equalization, f)
    print("correction, sweepeq - AutoEq (level removed):")
    for lo, hi in BANDS:
        k = (f >= lo) & (f < hi)
        worst = float(np.max(np.abs(diff[k])))
        print(f"  {lo:6.0f}-{hi:6.0f} Hz: rms {rms(diff[k]):.2f} dB, max {worst:.2f} dB")
    candidates = {"uncorrected": np.zeros_like(f), "AutoEq": r.equalization, "sweepeq": ours}
    if args.peq is not None:
        preset = read_apo(args.peq)
        candidates["AutoEq PEQ"] = preset.response_db(f, 48_000.0) - preset.preamp_db
    print("equalised response vs target (1/6 octave, level removed), 20 Hz-10 kHz:")
    k = f <= 10_000.0
    for name, eq in candidates.items():
        e = residual(r, eq)
        print(f"  {name:12s} rms {rms(e[k]):.2f} dB, max {np.max(np.abs(e[k])):.2f} dB")

    if args.plot is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
        ax[0].semilogx(f, r.equalization, label="AutoEq", color="k", ls="--")
        ax[0].semilogx(
            f,
            ours + float(np.mean((r.equalization - ours)[(f >= ALIGN[0]) & (f <= ALIGN[1])])),
            label="sweepeq",
            color="C0",
        )
        ax[0].set_ylabel("Correction (dB)")
        ax[0].legend()
        for name, eq in candidates.items():
            if name != "AutoEq PEQ":
                ax[1].semilogx(
                    f, residual(r, eq), label=name, alpha=0.4 if name == "uncorrected" else 1
                )
        ax[1].set(ylabel="Corrected - target (dB)", xlabel="Frequency (Hz)", xlim=(20, 20_000))
        ax[1].legend()
        for a in ax:
            a.grid(True, which="both", alpha=0.3)
        fig.suptitle(f"{args.csv.stem}: sweepeq vs AutoEq, same measurement and target")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"saved {args.plot}")


if __name__ == "__main__":
    main()
