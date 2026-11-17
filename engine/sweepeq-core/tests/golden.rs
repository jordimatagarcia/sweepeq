//! The engine against golden vectors from the Python reference (tools/make_golden.py):
//! real correction filters designed by `sweepeq.design`, expected outputs in f64.

mod common;

use std::fs;
use std::path::{Path, PathBuf};

use common::run;
use sweepeq_core::{Convolver, DirectFir, Kernel, Partitioned};

/// Largest error relative to the worst-case output magnitude `Σ|h| · max|x|`.
const TOL: f64 = 5e-7;

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/golden")
}

fn read_f32(p: &Path) -> Vec<f32> {
    let bytes = fs::read(p).unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn read_f64(p: &Path) -> Vec<f64> {
    let bytes = fs::read(p).unwrap();
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| f64::from_le_bytes(*b))
        .collect()
}

struct Case {
    name: String,
    taps: Vec<f32>,
    input: Vec<f32>,
    expected: Vec<f64>,
}

fn cases() -> Vec<Case> {
    let mut out: Vec<Case> = fs::read_dir(golden_dir())
        .expect("testdata/golden is missing: run tools/make_golden.py")
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("meta.json").exists())
        .map(|d| Case {
            name: d.file_name().unwrap().to_string_lossy().into_owned(),
            taps: read_f32(&d.join("taps.f32")),
            input: read_f32(&d.join("input.f32")),
            expected: read_f64(&d.join("expected.f64")),
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    assert!(
        out.len() >= 4,
        "expected at least 4 golden cases, found {}",
        out.len()
    );
    out
}

fn rel_error(got: &[f32], want: &[f64], c: &Case) -> f64 {
    let scale = c.taps.iter().map(|&v| f64::from(v.abs())).sum::<f64>()
        * c.input
            .iter()
            .map(|&v| f64::from(v.abs()))
            .fold(0.0, f64::max);
    got.iter()
        .zip(want)
        .map(|(&g, &w)| (f64::from(g) - w).abs())
        .fold(0.0, f64::max)
        / scale
}

#[test]
fn direct_fir_reproduces_the_golden_vectors() {
    for c in cases() {
        for kernel in [Kernel::Scalar, Kernel::Avx2Fma, Kernel::Neon] {
            if !kernel.is_available() {
                continue;
            }
            let mut f = DirectFir::using_kernel(&c.taps, 512, kernel);
            let got = run(&mut f, &c.input, &[480, 17, 1024]);
            let e = rel_error(&got, &c.expected, &c);
            assert!(e < TOL, "{} {kernel:?}: {e:e}", c.name);
        }
    }
}

#[test]
fn partitioned_reproduces_the_golden_vectors() {
    for c in cases() {
        for block in [64usize, 256, 1024] {
            let mut p = Partitioned::new(&c.taps, block);
            let got = run(&mut p, &c.input, &[480, 17, 1024]);
            let n = c.input.len() - block;
            let e = rel_error(&got[block..], &c.expected[..n], &c);
            assert!(e < TOL, "{} block {block}: {e:e}", c.name);
            assert_eq!(p.latency(), block);
        }
    }
}
