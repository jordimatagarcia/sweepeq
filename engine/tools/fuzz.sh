#!/usr/bin/env bash
# Coverage-guided fuzzing of the convolvers with libFuzzer on the stable toolchain.
#
#   engine/tools/fuzz.sh [seconds]      (default 120)
#
# The target is fuzz/fuzz_targets/convolve.rs. Crashes are written to the working directory
# as crash-*; the corpus is kept in ./corpus.
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
seconds="${1:-120}"
host="$(rustc -vV | sed -n 's/^host: //p')"
export RUSTFLAGS="-C passes=sancov-module \
    -C llvm-args=-sanitizer-coverage-level=4 \
    -C llvm-args=-sanitizer-coverage-inline-8bit-counters \
    -C llvm-args=-sanitizer-coverage-pc-table \
    -C llvm-args=-sanitizer-coverage-trace-compares \
    -C debug-assertions -C overflow-checks"
# An explicit --target keeps the instrumentation out of build scripts.
cargo build --quiet --release --manifest-path "$here/fuzz/Cargo.toml" --bin convolve --target "$host"
mkdir -p corpus
"$here/fuzz/target/$host/release/convolve" -max_total_time="$seconds" -max_len=1200 corpus
