# sweepeq

Measure a loudspeaker's or headphone's frequency response, design a correction filter, and
run it in real time in Rust, with every step checked against an independent reference.

**Status:** early development (setup). See the milestones below.

| Part | What it does |
|---|---|
| `src/sweepeq/` | Python reference: sweep measurement, analysis, inverse-filter design, verification |
| `engine/` | Rust real-time engine: SIMD FIR and partitioned FFT convolution |
| `testdata/golden/` | Golden vectors that keep the Python and Rust implementations in agreement |
| `bench/` | Benchmarks and per-machine baselines |

## Milestones

1. Sweep measurement (exponential sine sweep, impulse response, harmonic distortion)
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
