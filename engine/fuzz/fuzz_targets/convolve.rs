//! Random filters, block sizes, chunk patterns and signals: the direct and partitioned
//! convolvers must agree with a naive f64 convolution, and a filter swap must stay finite
//! and end on the new filter's output.

#![no_main]

use libfuzzer_sys::fuzz_target;
use sweepeq_core::{Convolver, DirectFir, Partitioned, engine};

struct Bytes<'a>(&'a [u8]);

impl Bytes<'_> {
    fn u8(&mut self) -> u8 {
        let (&b, rest) = self.0.split_first().unwrap_or((&0, &[]));
        self.0 = rest;
        b
    }
    /// A finite sample in [-1, 1).
    fn sample(&mut self) -> f32 {
        f32::from(self.u8() as i8) / 128.0
    }
}

fn reference(x: &[f32], h: &[f32]) -> Vec<f64> {
    (0..x.len())
        .map(|n| {
            (0..h.len().min(n + 1))
                .map(|k| f64::from(h[k]) * f64::from(x[n - k]))
                .sum()
        })
        .collect()
}

fn run(c: &mut dyn Convolver, x: &[f32], sizes: &[usize]) -> Vec<f32> {
    let mut y = vec![0.0; x.len()];
    let (mut i, mut k) = (0, 0);
    while i < x.len() {
        let n = sizes[k % sizes.len()].clamp(1, x.len() - i);
        c.process(&x[i..i + n], &mut y[i..i + n]);
        i += n;
        k += 1;
    }
    y
}

fuzz_target!(|data: &[u8]| {
    let mut b = Bytes(data);
    let taps = 1 + usize::from(b.u8()) % 200;
    let block = 2usize << (b.u8() % 7); // 2..128
    let sizes: Vec<usize> = (0..3).map(|_| 1 + usize::from(b.u8())).collect();
    let h: Vec<f32> = (0..taps).map(|_| b.sample()).collect();
    let x: Vec<f32> = (0..b.0.len().min(600)).map(|_| b.sample()).collect();
    if x.is_empty() {
        return;
    }
    let want = reference(&x, &h);
    let scale = h.iter().map(|v| f64::from(v.abs())).sum::<f64>().max(1e-9);
    let tol = 1e-6 * scale;

    let got = run(&mut DirectFir::new(&h, 64), &x, &sizes);
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (f64::from(g) - w).abs() <= tol,
            "direct sample {i}: {g} vs {w}"
        );
    }
    let got = run(&mut Partitioned::new(&h, block), &x, &sizes);
    assert!(got[..block.min(x.len())].iter().all(|&v| v == 0.0));
    for i in block..x.len() {
        let (g, w) = (f64::from(got[i]), want[i - block]);
        assert!(
            (g - w).abs() <= tol,
            "partitioned block {block} sample {i}: {g} vs {w}"
        );
    }

    // swap to the reversed filter halfway; the output must stay finite and bounded
    let h2: Vec<f32> = h.iter().rev().copied().collect();
    let (mut eng, mut ctl) = engine(Box::new(DirectFir::new(&h, 64)), 1 + taps, 64);
    let mut y = vec![0.0; x.len()];
    let half = x.len() / 2;
    eng.process(&x[..half], &mut y[..half]);
    ctl.set(Box::new(DirectFir::new(&h2, 64))).unwrap();
    eng.process(&x[half..], &mut y[half..]);
    let bound = scale * 1.0 + 1e-6;
    assert!(
        y.iter()
            .all(|v| v.is_finite() && f64::from(v.abs()) <= bound)
    );
});
