#![allow(dead_code)]

/// Deterministic uniform noise in [-0.5, 0.5).
pub fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) as f32) / ((1u64 << 24) as f32) - 0.5
        })
        .collect()
}

/// A decaying-noise filter, like a real correction filter's impulse response.
pub fn filter(n: usize, seed: u64) -> Vec<f32> {
    noise(n, seed)
        .iter()
        .enumerate()
        .map(|(i, &v)| v * (-(i as f32) / (n as f32 / 4.0 + 1.0)).exp())
        .collect()
}

/// Causal convolution in f64: the first `x.len()` samples of `x * h`.
pub fn reference(x: &[f32], h: &[f32]) -> Vec<f64> {
    (0..x.len())
        .map(|n| {
            (0..h.len().min(n + 1))
                .map(|k| f64::from(h[k]) * f64::from(x[n - k]))
                .sum()
        })
        .collect()
}

/// Largest error relative to the worst case output magnitude `Σ|h| · max|x|`.
pub fn rel_error(got: &[f32], want: &[f64], x: &[f32], h: &[f32]) -> f64 {
    let scale = h.iter().map(|&v| f64::from(v.abs())).sum::<f64>()
        * x.iter().map(|&v| f64::from(v.abs())).fold(0.0, f64::max);
    got.iter()
        .zip(want)
        .map(|(&g, &w)| (f64::from(g) - w).abs())
        .fold(0.0, f64::max)
        / scale.max(1e-30)
}

/// Run a convolver over `x` in chunks whose sizes cycle through `sizes`.
pub fn run(c: &mut dyn sweepeq_core::Convolver, x: &[f32], sizes: &[usize]) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len()];
    let mut i = 0;
    let mut k = 0;
    while i < x.len() {
        let n = sizes[k % sizes.len()].min(x.len() - i);
        c.process(&x[i..i + n], &mut y[i..i + n]);
        i += n;
        k += 1;
    }
    y
}
