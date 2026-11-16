# sweepeq

Measure a loudspeaker's or headphone's frequency response, design a correction filter, and
run it in real time in Rust, with every step checked against an independent reference.

**Status:** M1 (measurement), M3 (filter design), M4 (real-time engine) and M5 (benchmarks) done; M2 waits on a measurement microphone. See the milestones below and
[docs/measurement.md](docs/measurement.md).

```bash
uv run sweepeq simulate                      # measure a simulated device, no hardware
uv sync --extra audio
uv run sweepeq measure --device pipewire --out m.npz  # through a sound card
uv run sweepeq design m.npz --fir eq.wav --peq 8 --apo eq.txt  # a correction filter
cargo build --release --manifest-path engine/Cargo.toml
engine/target/release/sweepeq-engine process --filter eq.wav --input in.wav --output out.wav
```

| Part | What it does |
|---|---|
| `src/sweepeq/` | Python reference: sweep measurement, analysis, inverse-filter design, verification |
| `engine/` | Rust real-time engine: SIMD FIR, partitioned FFT convolution, lock-free filter swap, CLI, Python bindings |
| `testdata/golden/` | Golden vectors that keep the Python and Rust implementations in agreement |
| `bench/` | Benchmarks and per-machine baselines |

## Milestones

1. ✅ Sweep measurement (synchronized exponential sweep, impulse response, harmonic distortion)
2. Real measurements and analysis, validated against REW (software done; [procedure](docs/validation.md))
3. ✅ Correction filter design (regularised inversion, boost limit, FIR and parametric EQ; [docs](docs/design.md))
4. ✅ Rust real-time engine (SIMD FIR, partitioned FFT convolution, lock-free filter swap; [docs](docs/engine.md))
5. ✅ Benchmarks and latency (crossover at 128–256 taps; a 4096-tap correction under 1% of a core; [docs](docs/benchmarks.md))
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
