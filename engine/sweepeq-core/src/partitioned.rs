//! Uniformly partitioned overlap-save convolution (UPOLS; Wefers 2015, ch. 5).
//!
//! The filter is cut into `P = ⌈N / B⌉` partitions of `B` taps, each transformed once to a
//! `2B`-point spectrum. Every block of `B` input samples:
//!
//! 1. the last `2B` input samples (previous block, current block) are transformed;
//! 2. that spectrum enters a frequency-domain delay line (FDL) of the last `P` spectra;
//! 3. the output spectrum is `Σ_p X[t - p] · H[p]`, a complex multiply-accumulate over the
//!    FDL and the filter partitions;
//! 4. its inverse transform's second half is the output block (overlap-save).
//!
//! Cost per sample: two FFTs of size `2B` per `B` samples plus `P` complex MACs per bin,
//! instead of `N` multiply-adds for the direct form. Spectra are stored as separate real
//! and imaginary arrays so the MAC loop vectorises.
//!
//! The algorithm works on whole blocks of `B` samples; [`Partitioned`] accepts any number of
//! samples per call by buffering, which delays the output by exactly `B` samples.

use std::sync::Arc;

use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use crate::Convolver;

pub struct Partitioned {
    b: usize,
    parts: usize,
    taps: usize,
    fwd: Arc<dyn RealToComplex<f32>>,
    inv: Arc<dyn ComplexToReal<f32>>,
    /// Filter spectra, partition-major: `h_re[p * bins + k]`, scaled by `1/(2B)`.
    h_re: Box<[f32]>,
    h_im: Box<[f32]>,
    /// Frequency-domain delay line of input spectra, same layout; `head` is the newest.
    fdl_re: Box<[f32]>,
    fdl_im: Box<[f32]>,
    head: usize,
    /// Time-domain window of the last 2B input samples.
    window: Box<[f32]>,
    /// FFT work buffers.
    time: Box<[f32]>,
    spec: Box<[Complex32]>,
    scratch_fwd: Box<[Complex32]>,
    scratch_inv: Box<[Complex32]>,
    acc_re: Box<[f32]>,
    acc_im: Box<[f32]>,
    /// Streaming: input samples collected for the current block, output of the previous.
    in_block: Box<[f32]>,
    out_block: Box<[f32]>,
    pos: usize,
}

impl Partitioned {
    /// A convolver for `taps` with block (partition) size `block`, a power of two ≥ 2.
    pub fn new(taps: &[f32], block: usize) -> Self {
        assert!(!taps.is_empty(), "a filter needs at least one tap");
        assert!(
            block >= 2 && block.is_power_of_two(),
            "block must be a power of two >= 2"
        );
        let b = block;
        let n_fft = 2 * b;
        let bins = b + 1;
        let parts = taps.len().div_ceil(b);
        let mut planner = RealFftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(n_fft);
        let inv = planner.plan_fft_inverse(n_fft);

        let mut h_re = vec![0.0; parts * bins].into_boxed_slice();
        let mut h_im = vec![0.0; parts * bins].into_boxed_slice();
        let mut time = vec![0.0; n_fft].into_boxed_slice();
        let mut spec = fwd.make_output_vec().into_boxed_slice();
        let mut scratch_fwd = fwd.make_scratch_vec().into_boxed_slice();
        let scale = 1.0 / n_fft as f32; // realfft does not normalise the inverse
        for (p, chunk) in taps.chunks(b).enumerate() {
            time.fill(0.0);
            time[..chunk.len()].copy_from_slice(chunk);
            fwd.process_with_scratch(&mut time, &mut spec, &mut scratch_fwd)
                .expect("FFT buffer sizes match the plan");
            for (k, c) in spec.iter().enumerate() {
                h_re[p * bins + k] = c.re * scale;
                h_im[p * bins + k] = c.im * scale;
            }
        }
        let scratch_inv = inv.make_scratch_vec().into_boxed_slice();
        Self {
            b,
            parts,
            taps: taps.len(),
            fwd,
            inv,
            h_re,
            h_im,
            fdl_re: vec![0.0; parts * bins].into_boxed_slice(),
            fdl_im: vec![0.0; parts * bins].into_boxed_slice(),
            head: 0,
            window: vec![0.0; n_fft].into_boxed_slice(),
            time,
            spec,
            scratch_fwd,
            scratch_inv,
            acc_re: vec![0.0; bins].into_boxed_slice(),
            acc_im: vec![0.0; bins].into_boxed_slice(),
            in_block: vec![0.0; b].into_boxed_slice(),
            out_block: vec![0.0; b].into_boxed_slice(),
            pos: 0,
        }
    }

    pub fn block(&self) -> usize {
        self.b
    }

    pub fn partitions(&self) -> usize {
        self.parts
    }

    /// Process exactly one block of `B` samples from `in_block` into `out_block`.
    fn process_block(&mut self) {
        let b = self.b;
        let bins = b + 1;
        // 1. slide the window: [previous block | current block]
        self.window.copy_within(b.., 0);
        self.window[b..].copy_from_slice(&self.in_block);
        self.time.copy_from_slice(&self.window);
        self.fwd
            .process_with_scratch(&mut self.time, &mut self.spec, &mut self.scratch_fwd)
            .expect("FFT buffer sizes match the plan");
        // 2. newest spectrum into the delay line
        self.head = if self.head == 0 {
            self.parts - 1
        } else {
            self.head - 1
        };
        let base = self.head * bins;
        for (k, c) in self.spec.iter().enumerate() {
            self.fdl_re[base + k] = c.re;
            self.fdl_im[base + k] = c.im;
        }
        // 3. complex multiply-accumulate over partitions: X[t - p] lives at slot head + p
        self.acc_re.fill(0.0);
        self.acc_im.fill(0.0);
        for p in 0..self.parts {
            let slot = (self.head + p) % self.parts;
            let (xr, xi) = (
                &self.fdl_re[slot * bins..(slot + 1) * bins],
                &self.fdl_im[slot * bins..(slot + 1) * bins],
            );
            let (hr, hi) = (
                &self.h_re[p * bins..(p + 1) * bins],
                &self.h_im[p * bins..(p + 1) * bins],
            );
            for k in 0..bins {
                self.acc_re[k] += xr[k] * hr[k] - xi[k] * hi[k];
                self.acc_im[k] += xr[k] * hi[k] + xi[k] * hr[k];
            }
        }
        // 4. back to time; the second half is the valid (overlap-save) output
        for (k, c) in self.spec.iter_mut().enumerate() {
            *c = Complex32::new(self.acc_re[k], self.acc_im[k]);
        }
        // a real signal's DC and Nyquist bins are real; drop rounding residue there
        self.spec[0].im = 0.0;
        self.spec[b].im = 0.0;
        self.inv
            .process_with_scratch(&mut self.spec, &mut self.time, &mut self.scratch_inv)
            .expect("FFT buffer sizes match the plan");
        self.out_block.copy_from_slice(&self.time[b..]);
    }
}

impl Convolver for Partitioned {
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(
            input.len(),
            output.len(),
            "input and output must have the same length"
        );
        let mut done = 0;
        while done < input.len() {
            let n = (self.b - self.pos).min(input.len() - done);
            let (from, to) = (self.pos, self.pos + n);
            // the previous block's output goes out while this block's input comes in
            output[done..done + n].copy_from_slice(&self.out_block[from..to]);
            self.in_block[from..to].copy_from_slice(&input[done..done + n]);
            self.pos = to;
            done += n;
            if self.pos == self.b {
                self.process_block();
                self.pos = 0;
            }
        }
    }

    fn latency(&self) -> usize {
        self.b
    }

    fn reset(&mut self) {
        self.fdl_re.fill(0.0);
        self.fdl_im.fill(0.0);
        self.window.fill(0.0);
        self.in_block.fill(0.0);
        self.out_block.fill(0.0);
        self.pos = 0;
        self.head = 0;
    }

    fn taps(&self) -> usize {
        self.taps
    }
}
