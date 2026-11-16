# sweepeq

Measure a loudspeaker's or headphone's frequency response, design a correction filter, and
run it in real time in Rust, with every step checked against an independent reference.

**Status:** M1 (sweep measurement) done. See the milestones below and
[docs/measurement.md](docs/measurement.md).

```bash
uv run sweepeq simulate                      # measure a simulated device, no hardware
uv sync --extra audio
uv run sweepeq measure --device 2 --out m.npz  # through a sound card
```

| Part | What it does |
|---|---|
| `src/sweepeq/` | Python reference: sweep measurement, analysis, inverse-filter design, verification |
| `engine/` | Rust real-time engine: SIMD FIR and partitioned FFT convolution |
| `testdata/golden/` | Golden vectors that keep the Python and Rust implementations in agreement |
| `bench/` | Benchmarks and per-machine baselines |

## Milestones

1. ✅ Sweep measurement (synchronized exponential sweep, impulse response, harmonic distortion)
2. Real measurements and analysis, validated against REW
3. Correction filter design (regularised inversion, target curves, boost limits)
4. Rust real-time engine
5. Benchmarks and latency
6. Release

Every algorithm is implemented from the published sources listed in
[docs/references.md](docs/references.md).

## Development

```bash
uv sync
uv run pytest
cargo test --manifest-path engine/Cargo.toml
pre-commit install
```

## License

MIT
