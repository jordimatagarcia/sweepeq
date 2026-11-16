//! Dot-product kernels: the inner loop of the direct FIR filter.
//!
//! A kernel is chosen once, when a filter is built ([`Kernel::detect`]), and called through
//! a plain function pointer, so the audio path never repeats feature detection. The SIMD
//! kernels keep several independent accumulators so consecutive fused multiply-adds don't
//! wait on each other.

/// `fn(a, b) -> Σ a[i]·b[i]` over two slices of equal length.
pub type DotFn = fn(&[f32], &[f32]) -> f32;

/// The dot-product implementations available on this machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    Scalar,
    /// x86-64 AVX2 with FMA: 8 lanes, 4 accumulators.
    Avx2Fma,
    /// AArch64 NEON: 4 lanes, 4 accumulators.
    Neon,
}

impl Kernel {
    /// The fastest kernel the CPU supports (scalar under Miri, which does not model SIMD
    /// intrinsics fully).
    pub fn detect() -> Self {
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                return Kernel::Avx2Fma;
            }
        }
        #[cfg(all(target_arch = "aarch64", not(miri)))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return Kernel::Neon;
            }
        }
        Kernel::Scalar
    }

    /// Whether this machine can run the kernel.
    pub fn is_available(self) -> bool {
        match self {
            Kernel::Scalar => true,
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            Kernel::Avx2Fma => {
                std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
            }
            #[cfg(all(target_arch = "aarch64", not(miri)))]
            Kernel::Neon => std::arch::is_aarch64_feature_detected!("neon"),
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    /// The kernel's function. Panics if the CPU can't run it (check [`Kernel::is_available`]).
    pub fn function(self) -> DotFn {
        assert!(self.is_available(), "{self:?} is not supported on this CPU");
        match self {
            Kernel::Scalar => dot_scalar,
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            Kernel::Avx2Fma => dot_avx2_fma_checked,
            #[cfg(all(target_arch = "aarch64", not(miri)))]
            Kernel::Neon => dot_neon_checked,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

/// Portable dot product with four accumulators (the same summation structure as the SIMD
/// kernels, so the compiler can vectorise it too).
pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; 4];
    let chunks = a.len() / 4;
    for i in 0..chunks {
        for (lane, sum) in acc.iter_mut().enumerate() {
            *sum += a[4 * i + lane] * b[4 * i + lane];
        }
    }
    let mut tail = 0.0f32;
    for i in 4 * chunks..a.len() {
        tail += a[i] * b[i];
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
fn dot_avx2_fma_checked(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    // SAFETY: this function is only handed out by `Kernel::function` after
    // `Kernel::is_available` confirmed AVX2 and FMA, and the lengths are equal.
    unsafe { dot_avx2_fma(a, b) }
}

/// # Safety
/// The CPU must support AVX2 and FMA, and `a.len() == b.len()`.
#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc = [_mm256_setzero_ps(); 4];
    let mut i = 0;
    // SAFETY (all loads): `i + 32 <= n` (or `i + 8 <= n` below) keeps every unaligned
    // 8-lane load inside both slices, which have length `n`.
    while i + 32 <= n {
        for (k, sum) in acc.iter_mut().enumerate() {
            let x = unsafe { _mm256_loadu_ps(pa.add(i + 8 * k)) };
            let y = unsafe { _mm256_loadu_ps(pb.add(i + 8 * k)) };
            *sum = _mm256_fmadd_ps(x, y, *sum);
        }
        i += 32;
    }
    while i + 8 <= n {
        let x = unsafe { _mm256_loadu_ps(pa.add(i)) };
        let y = unsafe { _mm256_loadu_ps(pb.add(i)) };
        acc[0] = _mm256_fmadd_ps(x, y, acc[0]);
        i += 8;
    }
    let s = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
    // horizontal sum of 8 lanes
    let hi = _mm256_extractf128_ps(s, 1);
    let lo = _mm256_castps256_ps128(s);
    let q = _mm_add_ps(lo, hi);
    let d = _mm_add_ps(q, _mm_movehl_ps(q, q));
    let r = _mm_add_ss(d, _mm_shuffle_ps(d, d, 0x55));
    let mut total = _mm_cvtss_f32(r);
    while i < n {
        total += a[i] * b[i];
        i += 1;
    }
    total
}

#[cfg(all(target_arch = "aarch64", not(miri)))]
fn dot_neon_checked(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    // SAFETY: only handed out after `is_available` confirmed NEON; lengths are equal.
    unsafe { dot_neon(a, b) }
}

/// # Safety
/// The CPU must support NEON, and `a.len() == b.len()`.
#[cfg(all(target_arch = "aarch64", not(miri)))]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc = [vdupq_n_f32(0.0); 4];
    let mut i = 0;
    // SAFETY (all loads): `i + 16 <= n` (or `i + 4 <= n`) keeps every 4-lane load in bounds.
    while i + 16 <= n {
        for (k, sum) in acc.iter_mut().enumerate() {
            let x = unsafe { vld1q_f32(pa.add(i + 4 * k)) };
            let y = unsafe { vld1q_f32(pb.add(i + 4 * k)) };
            *sum = vfmaq_f32(*sum, x, y);
        }
        i += 16;
    }
    while i + 4 <= n {
        let x = unsafe { vld1q_f32(pa.add(i)) };
        let y = unsafe { vld1q_f32(pb.add(i)) };
        acc[0] = vfmaq_f32(acc[0], x, y);
        i += 4;
    }
    let s = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
    let mut total = vaddvq_f32(s);
    while i < n {
        total += a[i] * b[i];
        i += 1;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum()
    }

    fn data(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    #[test]
    fn all_available_kernels_agree_with_f64_for_every_length() {
        for kernel in [Kernel::Scalar, Kernel::Avx2Fma, Kernel::Neon] {
            if !kernel.is_available() {
                continue;
            }
            let dot = kernel.function();
            // every tail length around the 8-, 16- and 32-lane block boundaries (fewer under
            // Miri, which interprets every instruction)
            let lengths: Vec<usize> = if cfg!(miri) {
                (0..40).collect()
            } else {
                (0..80).chain([255, 256, 257, 1023, 4096]).collect()
            };
            for n in lengths {
                let (a, b) = (data(n, 1), data(n, 2));
                let exact = reference(&a, &b);
                let bound = 1e-6 * (n as f64).max(1.0) * 0.25;
                let got = f64::from(dot(&a, &b));
                assert!(
                    (got - exact).abs() <= bound,
                    "{kernel:?} n={n}: {got} vs {exact}"
                );
            }
        }
    }

    #[test]
    fn detect_returns_an_available_kernel() {
        assert!(Kernel::detect().is_available());
    }
}
