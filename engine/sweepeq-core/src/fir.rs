//! Direct-form FIR filter: one dot product per output sample.
//!
//! The taps are stored reversed, so each output is a dot product of a contiguous window of
//! the input history with them. Input history and each new block live in one linear buffer
//! (``taps - 1`` samples of history followed by up to ``max_block`` new samples), so every
//! window is contiguous and SIMD loads need no wrap-around. Larger blocks are processed in
//! ``max_block`` pieces. Latency: none.

use crate::Convolver;
use crate::kernels::{InnerProductFn, Kernel};

pub struct DirectFir {
    reversed_taps: Box<[f32]>,
    buf: Box<[f32]>,
    max_block: usize,
    inner_product: InnerProductFn,
    kernel: Kernel,
}

impl DirectFir {
    /// A filter with the given taps, processing blocks of up to `max_block` samples per
    /// internal step, with the fastest available kernel.
    pub fn new(taps: &[f32], max_block: usize) -> Self {
        Self::using_kernel(taps, max_block, Kernel::detect())
    }

    /// As [`DirectFir::new`] with a specific kernel. Panics if the CPU can't run it.
    pub fn using_kernel(taps: &[f32], max_block: usize, kernel: Kernel) -> Self {
        assert!(!taps.is_empty(), "a filter needs at least one tap");
        assert!(max_block > 0, "max_block must be positive");
        let reversed_taps: Box<[f32]> = taps.iter().rev().copied().collect();
        let buf = vec![0.0; taps.len() - 1 + max_block].into_boxed_slice();
        Self {
            reversed_taps,
            buf,
            max_block,
            inner_product: kernel.function(),
            kernel,
        }
    }

    pub fn kernel(&self) -> Kernel {
        self.kernel
    }

    fn process_piece(&mut self, input: &[f32], output: &mut [f32]) {
        let n = self.reversed_taps.len();
        let hist = n - 1;
        let m = input.len();
        self.buf[hist..hist + m].copy_from_slice(input);
        for (i, y) in output.iter_mut().enumerate() {
            *y = (self.inner_product)(&self.buf[i..i + n], &self.reversed_taps);
        }
        // keep the last n - 1 samples as the next piece's history
        self.buf.copy_within(m..m + hist, 0);
    }
}

impl Convolver for DirectFir {
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(
            input.len(),
            output.len(),
            "input and output must have the same length"
        );
        for (x, y) in input
            .chunks(self.max_block)
            .zip(output.chunks_mut(self.max_block))
        {
            self.process_piece(x, y);
        }
    }

    fn latency(&self) -> usize {
        0
    }

    fn reset(&mut self) {
        self.buf.fill(0.0);
    }

    fn taps(&self) -> usize {
        self.reversed_taps.len()
    }
}
