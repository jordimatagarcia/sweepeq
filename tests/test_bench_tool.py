"""tools/bench.py's analysis: crossover, best block, regression check."""

import importlib.util
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("bench_tool", REPO / "tools" / "bench.py")
assert _spec is not None and _spec.loader is not None
bench: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def case(method: str, block: int, taps: int, ns: float) -> dict[str, Any]:
    return {"method": method, "block": block, "taps": taps, "ns_per_sample": ns}


CASES = [
    case("direct-Avx2Fma", 0, 64, 3.0),
    case("partitioned", 64, 64, 5.0),
    case("partitioned", 256, 64, 4.0),
    case("direct-Avx2Fma", 0, 1024, 30.0),
    case("partitioned", 64, 1024, 20.0),
    case("partitioned", 256, 1024, 8.0),
]


def test_crossover_and_best_block() -> None:
    assert bench.crossover(CASES, "direct-Avx2Fma") == 1024
    best = bench.best_partitioned(CASES)
    assert best[64]["block"] == 256 and best[1024]["block"] == 256
    assert bench.crossover([CASES[0], CASES[1]], "direct-Avx2Fma") is None


def test_regressions_beyond_tolerance_only() -> None:
    baseline = {"cases": CASES}
    slower = [dict(c) for c in CASES]
    slower[3]["ns_per_sample"] = 30.0 * 1.2  # within 25%
    slower[5]["ns_per_sample"] = 8.0 * 1.5  # beyond
    found = bench.compare({"cases": slower}, baseline, 0.25)
    assert len(found) == 1 and "taps 1024" in found[0] and "1.50x" in found[0]


def test_machine_slug() -> None:
    assert bench.machine_slug({"cpu": "13th Gen Intel(R) Core(TM) i7"}) == (
        "13th-gen-intel-r-core-tm-i7"
    )
