//! Dot-product kernels: the inner loop of the direct FIR filter.
//!
//! A kernel is chosen once, when a filter is built ([`Kernel::detect`]), and called through
//! a plain function pointer, so the audio path never repeats feature detection. The SIMD
//! kernels keep several independent accumulators so consecutive fused multiply-adds don't
//! wait on each other.

/// `fn(a, b) -> Σ a[i]·b[i]` over two slices of equal length.
pub type InnerProductFn = fn(&[f32], &[f32]) -> f32;

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
    pub fn function(self) -> InnerProductFn {
        assert!(self.is_available(), "{self:?} is not supported on this CPU");
        match self {
            Kernel::Scalar => inner_product_scalar,
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            Kernel::Avx2Fma => inner_product_avx2_safe,
            #[cfg(all(target_arch = "aarch64", not(miri)))]
            Kernel::Neon => inner_product_neon_safe,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

/// Portable dot product with four accumulators (the same summation structure as the SIMD
/// kernels, so the compiler can vectorise it too).
pub fn inner_product_scalar(a: &[f32], b: &[f32]) -> f32 {
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
fn inner_product_avx2_safe(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    // SAFETY: `Kernel::function` only returns this after `Kernel::is_available` has found
    // AVX2 and FMA on this CPU.
    unsafe { inner_product_avx2(a, b) }
}

/// Inner product in 8-lane AVX2 registers, four independent sums in flight.
///
/// # Safety
/// Requires AVX2 and FMA at run time.
#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2,fma")]
unsafe fn inner_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    // Loads take `&[f32; 8]`: the array type guarantees each unaligned load is in bounds.
    // SAFETY: `x` is an array of exactly 8 floats.
    let load = |x: &[f32; 8]| unsafe { _mm256_loadu_ps(x.as_ptr()) };
    let (wide_a, rest_a) = a.as_chunks::<32>();
    let (wide_b, rest_b) = b.as_chunks::<32>();
    let mut sums = [_mm256_setzero_ps(); 4];
    for (xa, xb) in wide_a.iter().zip(wide_b) {
        let (xs, ys) = (xa.as_chunks::<8>().0, xb.as_chunks::<8>().0);
        for (lane, (x, y)) in xs.iter().zip(ys).enumerate() {
            sums[lane] = _mm256_fmadd_ps(load(x), load(y), sums[lane]);
        }
    }
    let (narrow_a, tail_a) = rest_a.as_chunks::<8>();
    let (narrow_b, tail_b) = rest_b.as_chunks::<8>();
    for (x, y) in narrow_a.iter().zip(narrow_b) {
        sums[0] = _mm256_fmadd_ps(load(x), load(y), sums[0]);
    }
    let total = _mm256_add_ps(
        _mm256_add_ps(sums[0], sums[1]),
        _mm256_add_ps(sums[2], sums[3]),
    );
    let mut lanes = [0.0f32; 8];
    // SAFETY: `lanes` holds exactly 8 floats.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), total) };
    let tail: f32 = tail_a.iter().zip(tail_b).map(|(x, y)| x * y).sum();
    (lanes[0] + lanes[4])
        + (lanes[1] + lanes[5])
        + (lanes[2] + lanes[6])
        + (lanes[3] + lanes[7])
        + tail
}

#[cfg(all(target_arch = "aarch64", not(miri)))]
fn inner_product_neon_safe(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    // SAFETY: only returned by `Kernel::function` after NEON was found on this CPU.
    unsafe { inner_product_neon(a, b) }
}

/// Inner product in 4-lane NEON registers, four independent sums in flight.
///
/// # Safety
/// Requires NEON at run time.
#[cfg(all(target_arch = "aarch64", not(miri)))]
#[target_feature(enable = "neon")]
unsafe fn inner_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    // SAFETY: `x` is an array of exactly 4 floats.
    let load = |x: &[f32; 4]| unsafe { vld1q_f32(x.as_ptr()) };
    let (wide_a, rest_a) = a.as_chunks::<16>();
    let (wide_b, rest_b) = b.as_chunks::<16>();
    let mut sums = [vdupq_n_f32(0.0); 4];
    for (xa, xb) in wide_a.iter().zip(wide_b) {
        let (xs, ys) = (xa.as_chunks::<4>().0, xb.as_chunks::<4>().0);
        for (lane, (x, y)) in xs.iter().zip(ys).enumerate() {
            sums[lane] = vfmaq_f32(sums[lane], load(x), load(y));
        }
    }
    let (narrow_a, tail_a) = rest_a.as_chunks::<4>();
    let (narrow_b, tail_b) = rest_b.as_chunks::<4>();
    for (x, y) in narrow_a.iter().zip(narrow_b) {
        sums[0] = vfmaq_f32(sums[0], load(x), load(y));
    }
    let total = vaddq_f32(vaddq_f32(sums[0], sums[1]), vaddq_f32(sums[2], sums[3]));
    let tail: f32 = tail_a.iter().zip(tail_b).map(|(x, y)| x * y).sum();
    vaddvq_f32(total) + tail
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
