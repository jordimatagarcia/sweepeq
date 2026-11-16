//! Changing the filter while audio runs, without locks, allocation or clicks.
//!
//! [`engine`] splits into an [`Engine`], owned by the audio thread, and a [`Controller`],
//! owned by any other thread. The controller builds a new convolver (all allocation happens
//! there) and sends it through a lock-free single-producer single-consumer ring (`rtrb`).
//! At the start of its next block, the engine picks it up and crossfades from the old
//! convolver to the new one over `fade_len` samples, running both meanwhile. When the fade
//! is over, the old convolver is sent back through a second ring, and the controller drops
//! it on its own thread, so the audio thread never frees memory either.
//!
//! The engine only starts a swap when the return ring has room, so sending the old convolver
//! back can never fail. If the controller never calls [`Controller::collect`], swaps simply
//! wait.
//!
//! The new convolver starts with empty history, so its first output lacks the tail of
//! earlier input. The crossfade fades it in from silence, which hides this as long as the
//! fade is longer than the filter's significant length (minimum-phase filters concentrate
//! their energy at the start).

use rtrb::{Consumer, Producer, RingBuffer};

use crate::Convolver;

/// Convolvers in flight from the controller to the engine.
const QUEUE: usize = 2;

pub struct Engine {
    current: Box<dyn Convolver>,
    fading: Option<Fade>,
    incoming: Consumer<Box<dyn Convolver>>,
    retired: Producer<Box<dyn Convolver>>,
    old_out: Box<[f32]>,
    fade_len: usize,
    max_block: usize,
}

struct Fade {
    old: Box<dyn Convolver>,
    done: usize,
}

pub struct Controller {
    outgoing: Producer<Box<dyn Convolver>>,
    retired: Consumer<Box<dyn Convolver>>,
    latency: usize,
}

#[derive(Debug)]
pub enum SwapError {
    /// The new convolver's latency differs from the running one's: a crossfade between
    /// them would mix misaligned signals.
    LatencyMismatch { running: usize, new: usize },
    /// Swaps are already queued; try again after the engine has taken them.
    Full(Box<dyn Convolver>),
}

impl std::fmt::Debug for dyn Convolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Convolver({} taps, latency {})",
            self.taps(),
            self.latency()
        )
    }
}

/// An engine running `first`, crossfading over `fade_len` samples on every swap, and
/// processing at most `max_block` samples per internal step (larger calls are split).
pub fn engine(
    first: Box<dyn Convolver>,
    fade_len: usize,
    max_block: usize,
) -> (Engine, Controller) {
    assert!(fade_len > 0 && max_block > 0);
    let (outgoing, incoming) = RingBuffer::new(QUEUE);
    let (retired_tx, retired_rx) = RingBuffer::new(QUEUE + 1);
    let latency = first.latency();
    let engine = Engine {
        current: first,
        fading: None,
        incoming,
        retired: retired_tx,
        old_out: vec![0.0; max_block].into_boxed_slice(),
        fade_len,
        max_block,
    };
    (
        engine,
        Controller {
            outgoing,
            retired: retired_rx,
            latency,
        },
    )
}

impl Controller {
    /// Queue `next` to replace the running convolver.
    pub fn set(&mut self, next: Box<dyn Convolver>) -> Result<(), SwapError> {
        if next.latency() != self.latency {
            return Err(SwapError::LatencyMismatch {
                running: self.latency,
                new: next.latency(),
            });
        }
        self.outgoing
            .push(next)
            .map_err(|rtrb::PushError::Full(c)| SwapError::Full(c))
    }

    /// Drop the convolvers the engine has finished with. Returns how many.
    pub fn collect(&mut self) -> usize {
        let mut n = 0;
        while let Ok(old) = self.retired.pop() {
            drop(old);
            n += 1;
        }
        n
    }

    /// Whether every queued swap has been taken by the engine.
    pub fn idle(&self) -> bool {
        self.outgoing.slots() == QUEUE
    }
}

impl Engine {
    pub fn is_fading(&self) -> bool {
        self.fading.is_some()
    }

    fn process_piece(&mut self, input: &[f32], output: &mut [f32]) {
        if self.fading.is_none()
            && self.retired.slots() > 0
            && let Ok(next) = self.incoming.pop()
        {
            let old = std::mem::replace(&mut self.current, next);
            self.fading = Some(Fade { old, done: 0 });
        }
        self.current.process(input, output);
        let Some(fade) = self.fading.as_mut() else {
            return;
        };
        let old_out = &mut self.old_out[..input.len()];
        fade.old.process(input, old_out);
        let len = self.fade_len as f32;
        for (i, (y, &o)) in output.iter_mut().zip(old_out.iter()).enumerate() {
            let g = (((fade.done + i + 1) as f32) / len).min(1.0);
            *y = g * *y + (1.0 - g) * o;
        }
        fade.done += input.len();
        if fade.done >= self.fade_len {
            let old = self.fading.take().map(|f| f.old);
            if let Some(old) = old {
                // Room was checked before the swap started, and only one swap runs at a
                // time, so this cannot fail.
                let pushed = self.retired.push(old);
                debug_assert!(pushed.is_ok());
            }
        }
    }
}

impl Convolver for Engine {
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(
            input.len(),
            output.len(),
            "input and output must have the same length"
        );
        let m = self.max_block;
        for (x, y) in input.chunks(m).zip(output.chunks_mut(m)) {
            self.process_piece(x, y);
        }
    }

    fn latency(&self) -> usize {
        self.current.latency()
    }

    fn reset(&mut self) {
        self.current.reset();
        if let Some(f) = self.fading.as_mut() {
            f.old.reset();
        }
    }

    fn taps(&self) -> usize {
        self.current.taps()
    }
}
