# Real-time engine (M4)

The Rust workspace in [`engine/`](../engine) applies the correction filters that
`sweepeq design` writes, to files or live audio.

| Crate | What it is |
|---|---|
| `sweepeq-core` | The convolvers and the filter swap; no audio I/O |
| `sweepeq-cli` | `sweepeq-engine`: process WAV files, or live audio through cpal |
| `sweepeq-py` | Python bindings (PyO3), installed as the `sweepeq-engine` package |

```bash
cargo build --release --manifest-path engine/Cargo.toml
engine/target/release/sweepeq-engine info    --filter eq.wav
engine/target/release/sweepeq-engine process --filter eq.wav --input in.wav --output out.wav
engine/target/release/sweepeq-engine live    --filter eq.wav --watch     # edit eq.wav to swap
uv pip install ./engine/sweepeq-py          # then: import sweepeq_engine
```

## Convolvers

- **`DirectFir`:** one dot product per output sample, no latency. The taps are stored
  reversed and the input history and new samples share one linear buffer, so every window
  is contiguous. The dot product uses AVX2 with FMA on x86-64 or NEON on AArch64 (8 or 4
  lanes, four independent accumulators), chosen once at construction and called through a
  function pointer; a portable kernel with the same summation structure is the fallback.
- **`Partitioned`:** uniformly partitioned overlap-save convolution (Wefers 2015). The
  filter is cut into partitions of `B` taps, each transformed once to a `2B`-point
  spectrum; each block of input is transformed, stored in a frequency-domain delay line,
  multiplied and accumulated with the partitions, and transformed back. Spectra are kept
  as separate real and imaginary arrays so the multiply-accumulate vectorises. Latency:
  exactly `B` samples, because input is collected into blocks; any number of samples can
  be passed per call.

Both implement `Convolver` (`process`, `latency`, `reset`, `taps`), and `build` picks one
from a `Method`.

## Real-time rules

Once a convolver is built, `process` never allocates, frees, locks or blocks. Building one
(planning FFTs, transforming the filter) allocates, and belongs on another thread.
`tests/no_alloc.rs` enforces this with a global allocator that counts allocations made
while `process` runs, for both convolvers and during filter swaps; a guard test checks the
counter does count.

### Changing the filter while running

`engine(first, fade_len, max_block)` returns an `Engine`, owned by the audio thread, and a
`Controller`, owned by any other thread:

1. The controller builds the new convolver (allocating there) and sends it through a
   lock-free single-producer single-consumer ring (`rtrb`).
2. At its next block, the engine takes it and crossfades from the old convolver to the new
   one over `fade_len` samples, running both meanwhile.
3. The old convolver goes back through a second ring, and the controller drops it on its
   own thread: the audio thread never frees memory.

The engine only starts a swap when the return ring has room, so returning the old
convolver cannot fail; if the controller never collects, swaps wait. Swaps between
convolvers of different latency are refused, since the crossfade would mix misaligned
signals. The new convolver starts with empty history; the crossfade fades it in from
silence, which hides that as long as the fade is longer than the filter's significant
length.

`sweepeq-engine live --watch` uses this: editing the filter file swaps it in, crossfaded.

## Verification

| Check | What it covers | Result |
|---|---|---|
| Unit and integration tests | every kernel, block sizes 2–1024, filter lengths around the partition boundaries, input in chunks of 1 to 4096 samples, swaps, reset | error ≤ 1.3e-7 of the worst-case output `Σ|h|·max|x|` (tolerance 5e-7) |
| Golden vectors (`tools/make_golden.py`) | real correction filters designed by the Python reference (minimum and linear phase, 37–4096 taps, the HD 600 correction), expected outputs in float64 | table below |
| Python bindings vs SciPy (`tests/test_engine.py`) | the engine through PyO3, in uneven chunks | ≤ 5e-7 |
| CLI end to end | stereo WAV in and out, latency removed, wrong sample rate refused | exact to 1e-6 |
| No allocation | `process` and swaps | 0 allocations |
| Miri | the safe code and `rtrb`'s and `realfft`'s unsafe code, on small cases | no undefined behaviour |
| AddressSanitizer | the whole core test suite, including the AVX2 kernel's unaligned loads | clean |
| Fuzzing (libFuzzer, 120 s in CI) | random filters, block sizes, chunk patterns and signals against a naive f64 convolution, and a swap | 2.3 million runs clean locally; a planted off-by-one in the delay line was caught at once |

Golden vectors, largest error relative to `Σ|h|·max|x|`:

| Case | Taps | Direct | Partitioned, B = 64 | Partitioned, B = 1024 |
|---|---|---|---|---|
| device_lin1023_sweep | 1023 | 1.2e-7 | 1.5e-7 | 2.8e-7 |
| device_min4096_noise | 4096 | 4.9e-8 | 1.8e-7 | 1.3e-7 |
| hd600_min2048_impulses | 2048 | 2.3e-9 | 9.2e-9 | 1.4e-8 |
| short_37_noise | 37 | 6.7e-8 | 1.6e-7 | 1.8e-7 |

That is about the rounding of single precision (6e-8): the engine computes in f32 and the
reference in f64.

## Speed (first look)

On a laptop (one core), filtering 5 s of stereo audio with the 4096-tap correction designed
from a real headphone measurement: direct FIR (AVX2+FMA) 82× real time, partitioned
(B = 256) 485× real time. M5 measures this properly: the crossover between the methods
over filter length and block size, latency, CPU load per channel, and x86 against ARM.

## Review fixes

A review after M4 (and its first CI run) found three problems, each now covered by tests:

- **The allocation test could fail spuriously.** Its counter was a global shared by all
  test threads, so on 2-core CI runners another test's deliberate allocations (the
  counter's own guard test) landed in its window: "allocated 2 times". Locally, with 20
  cores, the timing never overlapped. The count is now thread-local, with a test that
  allocations on another thread are not counted. The engine itself never allocated.
- **Live mode could swap the stereo channels for good.** Samples crossed from the input to
  the output callback one at a time, so a full ring or an underrun in the middle of a
  frame moved every later sample by one position. Whole frames now move or none do
  (`sweepeq-cli/src/frames.rs`, unit-tested without audio devices).
- **The Python bindings rejected NumPy's default float64 arrays** (what `sweepeq.design`
  returns), lists and strided views, with an unhelpful TypeError. Any real 1-D array-like
  is now converted to float32.

## Not yet verified

- **`live`** compiles and shares all its processing with the tested code, but has not been
  run against real audio devices.
- The Miri job skips the SIMD kernels (Miri does not execute those intrinsics); the
  sanitizer and the tests cover them.
