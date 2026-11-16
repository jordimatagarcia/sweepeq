"""Command line: ``sweepeq measure`` and ``sweepeq simulate``."""

import argparse
import sys
from pathlib import Path

import numpy as np

from sweepeq.analysis import Curve
from sweepeq.measure import CLIP_LEVEL, Measurement, Recorder, SimulatedRecorder, measure
from sweepeq.plant import Plant, peaking_biquad
from sweepeq.response import frequency_response, harmonic_distortion, hd_range, to_db
from sweepeq.sweep import SweepSpec

REPORT_FREQS = (31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sweepeq", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_ in (
        ("measure", "measure a device through the sound card"),
        ("simulate", "measure a simulated device (no hardware)"),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--out", type=Path, help="save the measurement (.npz)")
        p.add_argument("--f1", type=float, default=20.0)
        p.add_argument("--f2", type=float, default=20_000.0)
        p.add_argument("--duration", type=float, default=5.0, help="sweep length (s)")
        p.add_argument("--fs", type=int, default=48_000)
        p.add_argument("--level", type=float, default=0.3, help="sweep peak (full scale 1)")
        if name == "measure":
            p.add_argument("--device", default=None, help="sound device (name or index)")
            p.add_argument("--channel", type=int, default=0, help="input channel (0-based)")
    sub.add_parser("devices", help="list sound devices")
    a = sub.add_parser("analyse", help="smooth, calibrate, average and compare measurements")
    a.add_argument("measurements", type=Path, nargs="+", help="saved .npz measurements")
    a.add_argument("--cal", type=Path, help="microphone calibration file")
    a.add_argument("--smooth", type=float, default=6.0, help="1/N octave (0: none)")
    a.add_argument("--compare", type=Path, help="REW text export to compare against")
    a.add_argument("--range", type=float, nargs=2, default=(50.0, 10_000.0), metavar=("LO", "HI"))
    a.add_argument("--export", type=Path, help="write the result as REW-importable text")
    a.add_argument("--plot", type=Path, help="save a plot (needs the plot extra)")
    args = parser.parse_args(argv)
    if args.command == "devices":
        import sounddevice as sd

        print(sd.query_devices())
        return 0
    if args.command == "analyse":
        return _analyse(args)

    try:
        spec = SweepSpec(args.f1, args.f2, args.duration, args.fs, amplitude=args.level)
    except ValueError as exc:
        parser.error(str(exc))
    recorder: Recorder
    if args.command == "measure":
        from sweepeq.measure import SoundDeviceRecorder

        device = int(args.device) if args.device and args.device.isdigit() else args.device
        recorder = SoundDeviceRecorder(device, args.channel)
    else:
        recorder = SimulatedRecorder(_demo_plant(args.fs))
    m = measure(spec, recorder)
    _report(m)
    if args.out is not None:
        m.save(args.out)
        print(f"saved {args.out}")
    return 0


def _analyse(args: argparse.Namespace) -> int:
    from sweepeq.analysis import (
        apply_calibration,
        average,
        compare,
        read_text_curve,
        response_curve,
        smooth,
        write_text_curve,
    )
    from sweepeq.measure import load_linear

    curves = []
    for path in args.measurements:
        hir, fs, f1, f2 = load_linear(path)
        freqs = np.geomspace(f1, f2, 2000)
        curves.append(response_curve(freqs, frequency_response(hir, freqs, fs)))
    result = average(curves)
    if args.cal is not None:
        result = apply_calibration(result, read_text_curve(args.cal))
    if args.smooth > 0:
        result = smooth(result, args.smooth)
    if args.export is not None:
        write_text_curve(result, args.export)
        print(f"wrote {args.export}")
    reference = None
    if args.compare is not None:
        reference = read_text_curve(args.compare)
        if args.smooth > 0:
            reference = smooth(reference, args.smooth)
        cmp = compare(result, reference, *args.range)
        lo, hi = args.range
        print(
            f"vs {args.compare.name}, {lo:g}-{hi:g} Hz, both normalised at 1 kHz: "
            f"max difference {cmp.max_abs_db:.2f} dB"
        )
    if args.plot is not None:
        _plot(result, reference, args.plot)
        print(f"saved {args.plot}")
    return 0


def _plot(result: Curve, reference: Curve | None, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sweepeq.analysis import normalise

    fig, ax = plt.subplots(figsize=(8, 4.5))
    r = normalise(result)
    ax.semilogx(r.freqs, r.db, label="sweepeq")
    if reference is not None:
        ref = normalise(reference)
        ax.semilogx(ref.freqs, ref.db, label="REW", linestyle="--")
    ax.set(xlabel="Frequency (Hz)", ylabel="Level re 1 kHz (dB)", xlim=(20, 20_000))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def _demo_plant(fs: int) -> Plant:
    sos = np.vstack([peaking_biquad(80.0, 6.0, 1.0, fs), peaking_biquad(3000.0, -6.0, 2.0, fs)])
    return Plant(fs=fs, sos=sos, delay_samples=240, poly=(0.05, 0.02), noise_rms=1e-4)


def _report(m: Measurement) -> None:
    fs = m.spec.fs
    if m.peak_level >= CLIP_LEVEL:
        print(
            f"warning: recording peaked at {m.peak_level:.3f}: it may have clipped", file=sys.stderr
        )
    freqs = np.array([f for f in REPORT_FREQS if 2 * m.spec.f1 <= f <= m.spec.f2 / 2])
    mag = to_db(frequency_response(m.linear, freqs, fs))
    print(f"{'f (Hz)':>8} {'level (dB)':>11} {'HD2 (dB)':>9} {'HD3 (dB)':>9}")
    hd = {}
    for h in m.harmonics[1:3]:
        lo, hi = hd_range(m.spec.f1, m.spec.f2, h.order)
        ok = (freqs >= lo) & (freqs <= hi)
        vals = np.full(len(freqs), np.nan)
        vals[ok] = to_db(harmonic_distortion(m.linear, h, freqs[ok], fs))
        hd[h.order] = vals
    for i, f in enumerate(freqs):
        print(f"{f:8.1f} {mag[i]:11.2f} {hd[2][i]:9.1f} {hd[3][i]:9.1f}")


if __name__ == "__main__":
    raise SystemExit(main())
