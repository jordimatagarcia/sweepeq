//! Real-time convolution for loudspeaker and headphone correction filters.
//!
//! - [`DirectFir`]: direct-form FIR with SIMD dot products (AVX2+FMA, NEON), no latency.
//! - [`Partitioned`]: uniformly partitioned overlap-save FFT convolution, latency of one
//!   block, far cheaper for long filters.
//! - [`engine`]: swapping filters while audio runs, with a crossfade, lock-free.
//!
//! Real-time rule: once built, nothing in `process` allocates, frees, locks or blocks.
//! Construction (planning FFTs, transforming the filter) allocates and belongs on a
//! non-audio thread. `tests/no_alloc.rs` enforces the rule with a counting allocator.

#![deny(unsafe_op_in_unsafe_fn)]

pub mod engine;
pub mod fir;
pub mod kernels;
pub mod partitioned;

pub use engine::{Controller, Engine, SwapError, engine};
pub use fir::DirectFir;
pub use kernels::Kernel;
pub use partitioned::Partitioned;

/// A streaming convolver: feed it input in chunks of any size, get output of the same size.
pub trait Convolver: Send {
    /// Filter `input` into `output` (same length), continuing from the previous call.
    fn process(&mut self, input: &[f32], output: &mut [f32]);
    /// Output delay in samples, on top of the filter's own response.
    fn latency(&self) -> usize;
    /// Forget all history (as if just built).
    fn reset(&mut self);
    /// Filter length.
    fn taps(&self) -> usize;
}

/// How to convolve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// [`DirectFir`].
    Direct,
    /// [`Partitioned`] with the given block size.
    Partitioned { block: usize },
}

/// Build a convolver for `taps`. `max_block` bounds the direct filter's internal step.
pub fn build(taps: &[f32], method: Method, max_block: usize) -> Box<dyn Convolver> {
    match method {
        Method::Direct => Box::new(DirectFir::new(taps, max_block)),
        Method::Partitioned { block } => Box::new(Partitioned::new(taps, block)),
    }
}
