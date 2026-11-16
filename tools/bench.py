"""Benchmark the Rust engine, keep per-machine baselines, and chart the results.

    uv run python tools/bench.py                    # run, compare with this machine's baseline
    uv run python tools/bench.py --save-baseline    # run and store it as the baseline
    uv run python tools/bench.py --plot docs/images/bench.png

The engine's ``bench`` command times every method over filter lengths and block sizes.
On Linux the run is pinned to one performance core when the CPU reports them (hybrid
Intel CPUs mix fast and slow cores, and a run that migrates between them is noise).
Baselines live in ``bench/baselines/<machine>.json``; a case more than ``--tolerance``
slower than its baseline is reported as a regression (exit status 1 with ``--check``).
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine"
BIN = ENGINE / "target" / "release" / "sweepeq-engine"
BASELINES = ROOT / "bench" / "baselines"
TRADEOFF_TAPS = 4096
"""Filter length of the latency/cost chart: a typical correction filter."""


def performance_core() -> str | None:
    """First performance core on a hybrid Intel CPU, or core 0; None if not on Linux."""
    p = Path("/sys/devices/cpu_core/cpus")
    if p.exists():
        return p.read_text().strip().split("-")[0].split(",")[0]
    return "0" if sys.platform == "linux" else None


def run_bench(quick: bool) -> dict[str, Any]:
    subprocess.run(
        ["cargo", "build", "--quiet", "--release", "-p", "sweepeq-cli"], cwd=ENGINE, check=True
    )
    cmd = [str(BIN), "bench"] + (["--quick"] if quick else [])
    core = performance_core()
    if core is not None and shutil.which("taskset"):
        cmd = ["taskset", "-c", core, *cmd]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    result: dict[str, Any] = json.loads(out)
    result["pinned_core"] = core
    return result


def machine_slug(result: dict[str, Any]) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(result["cpu"]).lower()).strip("-")


def key(case: dict[str, Any]) -> tuple[str, int, int]:
    return (case["method"], case["block"], case["taps"])


def compare(result: dict[str, Any], baseline: dict[str, Any], tolerance: float) -> list[str]:
    base = {key(c): c["ns_per_sample"] for c in baseline["cases"]}
    regressions = []
    for c in result["cases"]:
        old = base.get(key(c))
        if old is None:
            continue
        ratio = c["ns_per_sample"] / old
        if ratio > 1.0 + tolerance:
            regressions.append(
                f"{c['method']} block {c['block']} taps {c['taps']}: "
                f"{c['ns_per_sample']:.1f} ns/sample vs {old:.1f} ({ratio:.2f}x)"
            )
    return regressions


def best_partitioned(cases: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    best: dict[int, dict[str, Any]] = {}
    for c in cases:
        if c["method"] == "partitioned":
            b = best.get(c["taps"])
            if b is None or c["ns_per_sample"] < b["ns_per_sample"]:
                best[c["taps"]] = c
    return best


def crossover(cases: list[dict[str, Any]], method: str) -> int | None:
    """Shortest filter for which some partitioned block size beats ``method``."""
    best = best_partitioned(cases)
    for c in sorted((c for c in cases if c["method"] == method), key=lambda c: c["taps"]):
        p = best.get(c["taps"])
        if p is not None and p["ns_per_sample"] < c["ns_per_sample"]:
            return int(c["taps"])
    return None


def plot(result: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = result["cases"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    methods = sorted({c["method"] for c in cases if c["method"] != "partitioned"})
    for m in methods:
        rows = sorted((c for c in cases if c["method"] == m), key=lambda c: c["taps"])
        ax[0].loglog([c["taps"] for c in rows], [c["ns_per_sample"] for c in rows], "o-", label=m)
    for b in sorted({c["block"] for c in cases if c["method"] == "partitioned"}):
        rows = sorted(
            (c for c in cases if c["method"] == "partitioned" and c["block"] == b),
            key=lambda c: c["taps"],
        )
        ax[0].loglog(
            [c["taps"] for c in rows],
            [c["ns_per_sample"] for c in rows],
            ".--",
            alpha=0.6,
            label=f"partitioned B={b}",
        )
    period = 1e9 / result["sample_rate"]
    ax[0].axhline(period, color="k", lw=0.8)
    ax[0].text(17, period * 0.6, "one sample period at 48 kHz = 100% of a core", fontsize=8)
    ax[0].set(
        xlabel="Filter length (taps)",
        ylabel="Time per sample (ns)",
        title="Cost per sample, one channel",
    )
    ax[0].legend(fontsize=7, ncol=2, loc="lower right")
    ax[0].grid(True, which="both", alpha=0.3)

    # the real decision: for a typical correction filter, cost and worst callback against
    # the latency each block size adds
    n = (
        TRADEOFF_TAPS
        if any(c["taps"] == TRADEOFF_TAPS for c in cases)
        else max(c["taps"] for c in cases)
    )
    fs = result["sample_rate"]
    rows = sorted(
        (c for c in cases if c["method"] == "partitioned" and c["taps"] == n),
        key=lambda c: c["block"],
    )
    lat = [c["block"] / fs * 1e3 for c in rows]
    ax[1].semilogy(lat, [c["cpu_percent_per_channel"] for c in rows], "o-", label="average CPU")
    ax[1].semilogy(
        lat,
        [100 * c["worst_call_deadline_fraction"] for c in rows],
        "s--",
        label=f"slowest call, % of a {result['chunk']}-sample deadline",
    )
    for c, x in zip(rows, lat, strict=True):
        ax[1].annotate(
            f"B={c['block']}",
            (x, c["cpu_percent_per_channel"]),
            fontsize=7,
            textcoords="offset points",
            xytext=(3, -10),
        )
    d = next(
        (
            c
            for c in cases
            if c["method"].startswith("direct-")
            and c["taps"] == n
            and c["method"] != "direct-Scalar"
        ),
        None,
    )
    if d is not None:
        ax[1].plot([0.0], [d["cpu_percent_per_channel"]], "k^", label="direct (no latency)")
    ax[1].set(
        xlabel="Added latency (ms)",
        ylabel="% of one core, per channel",
        title=f"Latency against cost, {n} taps",
    )
    ax[1].legend(fontsize=7)
    ax[1].grid(True, which="both", alpha=0.3)
    fig.suptitle(f"{result['cpu']}, {result['simd']}, {result['chunk']}-sample calls")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--save-baseline", action="store_true")
    p.add_argument("--check", action="store_true", help="exit 1 on regressions")
    p.add_argument("--tolerance", type=float, default=0.25)
    p.add_argument("--plot", type=Path)
    p.add_argument("--result", type=Path, help="use a saved result instead of running")
    args = p.parse_args()

    result = json.loads(args.result.read_text()) if args.result else run_bench(args.quick)
    path = BASELINES / f"{machine_slug(result)}.json"
    for m in sorted({c["method"] for c in result["cases"] if c["method"] != "partitioned"}):
        x = crossover(result["cases"], m)
        print(f"partitioned beats {m} from {x} taps" if x else f"{m} is never beaten")
    status = 0
    if args.save_baseline:
        BASELINES.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"saved baseline {path.relative_to(ROOT)}")
    elif path.exists():
        regressions = compare(result, json.loads(path.read_text()), args.tolerance)
        print(
            f"vs {path.relative_to(ROOT)}: {len(regressions)} regression(s) "
            f"beyond {args.tolerance:.0%}"
        )
        for r in regressions:
            print("  " + r)
        status = 1 if regressions and args.check else 0
    else:
        print(f"no baseline for this machine ({path.name}); use --save-baseline")
    if args.plot is not None:
        plot(result, args.plot)
        print(f"saved {args.plot}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
