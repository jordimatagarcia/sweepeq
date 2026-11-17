"""The Rust engine through its Python bindings (engine/sweepeq-py), against the Python
reference. Skipped unless the bindings are installed (``uv pip install ./engine/sweepeq-py``);
CI sets SWEEPEQ_REQUIRE_ENGINE=1 so that a missing build fails instead of skipping."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import signal

if os.environ.get("SWEEPEQ_REQUIRE_ENGINE"):
    import sweepeq_engine
else:
    sweepeq_engine = pytest.importorskip("sweepeq_engine")

from sweepeq.autoeq import read_autoeq_csv
from sweepeq.design import DesignSpec, build_correction_fir, correction_db

REPO = Path(__file__).resolve().parents[1]


def hd600_taps() -> np.ndarray:
    r = read_autoeq_csv(REPO / "testdata/autoeq/hd600_oratory1990.csv")
    spec = DesignSpec(f_lo=20.0, f_hi=20_000.0, n_taps=2048)
    return build_correction_fir(correction_db(r.raw, r.target, spec), spec).taps.astype(np.float32)


def rel_error(y: np.ndarray, ref: np.ndarray, h: np.ndarray, x: np.ndarray) -> float:
    return float(np.max(np.abs(y - ref)) / (np.sum(np.abs(h)) * np.max(np.abs(x))))


@pytest.mark.parametrize(
    ("method", "block"), [("direct", 256), ("partitioned", 64), ("partitioned", 512)]
)
def test_engine_matches_scipy(method: str, block: int) -> None:
    h = hd600_taps()
    x = np.random.default_rng(0).uniform(-0.5, 0.5, 20_000).astype(np.float32)
    c = sweepeq_engine.Convolver(h, method=method, block=block)
    # in uneven chunks, to exercise the streaming
    y = np.concatenate([c.process(part) for part in np.array_split(x, [100, 1100, 1117, 9000])])
    lat = c.latency
    ref = signal.lfilter(h.astype(np.float64), [1.0], x.astype(np.float64))
    assert rel_error(y[lat:], ref[: len(x) - lat], h, x) < 5e-7
    assert np.all(y[:lat] == 0.0)
    assert lat == (0 if method == "direct" else block)


def test_reset_and_repr() -> None:
    h = np.full(8, 1 / 8, dtype=np.float32)
    c = sweepeq_engine.Convolver(h, method="direct")
    x = np.ones(32, np.float32)
    first = c.process(x)
    c.reset()
    np.testing.assert_array_equal(c.process(x), first)
    assert "8 taps" in repr(c) and c.taps == 8


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"method": "fast"}, "unknown method"), ({"block": 100}, "power of two")],
)
def test_invalid_arguments(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        sweepeq_engine.Convolver(np.ones(4, np.float32), **kwargs)
    with pytest.raises(ValueError, match="empty"):
        sweepeq_engine.Convolver(np.zeros(0, np.float32))


def test_accepts_common_array_types() -> None:
    # Review fix: float64 (NumPy's default, and what sweepeq.design returns), lists and
    # strided views were rejected with an unhelpful TypeError.
    h64 = hd600_taps().astype(np.float64)
    c = sweepeq_engine.Convolver(h64, method="direct")
    x = np.random.default_rng(1).uniform(-0.5, 0.5, 4000)
    y = c.process(x)
    assert y.dtype == np.float32 and len(y) == len(x)
    ref = signal.lfilter(h64.astype(np.float32).astype(np.float64), [1.0], x.astype(np.float32))
    assert rel_error(y, ref, h64, x) < 5e-7
    c.reset()
    np.testing.assert_array_equal(c.process(x.tolist()), y)
    wide = np.repeat(x.astype(np.float32), 2)
    c.reset()
    np.testing.assert_array_equal(c.process(wide[::2]), y)
