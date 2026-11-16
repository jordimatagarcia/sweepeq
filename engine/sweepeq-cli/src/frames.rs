//! Moving interleaved audio between the input and output callbacks through a lock-free
//! single-producer single-consumer ring, whole frames at a time.
//!
//! Moving samples one by one let a frame be split: a full ring mid-frame dropped the rest of
//! that frame, and an underrun mid-frame took only part of one. Every later sample was then
//! off by one position, which swaps left and right for good. With one producer and one
//! consumer, a check that a whole frame fits (or is available) stays true until it is used:
//! only the other side can change the count, and only in the safe direction.

use rtrb::{Consumer, Producer};

/// Push the first `channels` samples of each `in_ch`-sample frame in `data`. Frames that
/// don't fit are dropped whole. Returns how many were dropped.
pub fn push_frames(tx: &mut Producer<f32>, data: &[f32], in_ch: usize, channels: usize) -> usize {
    let mut dropped = 0;
    for frame in data.chunks_exact(in_ch) {
        if tx.slots() < channels {
            dropped += 1;
            continue;
        }
        for &v in &frame[..channels] {
            // cannot fail: the room was checked above and only we push
            let _ = tx.push(v);
        }
    }
    dropped
}

/// Pop one frame of `out.len()` samples, or leave `out` as silence and return false if a
/// whole frame isn't available (an underrun).
pub fn pop_frame(rx: &mut Consumer<f32>, out: &mut [f32]) -> bool {
    if rx.slots() < out.len() {
        out.fill(0.0);
        return false;
    }
    for v in out.iter_mut() {
        *v = rx.pop().unwrap_or(0.0); // cannot fail: availability was checked above
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use rtrb::RingBuffer;

    /// Interleaved stereo where left is +n and right is -n for frame n.
    fn stereo(frames: usize, start: usize) -> Vec<f32> {
        (start..start + frames)
            .flat_map(|n| [n as f32 + 1.0, -(n as f32) - 1.0])
            .collect()
    }

    fn assert_aligned(frame: &[f32]) {
        assert_eq!(frame[0], -frame[1], "channels misaligned: {frame:?}");
    }

    #[test]
    fn a_full_ring_drops_whole_frames() {
        // capacity 5 holds two stereo frames and one spare sample
        let (mut tx, mut rx) = RingBuffer::<f32>::new(5);
        let dropped = push_frames(&mut tx, &stereo(4, 0), 2, 2);
        assert_eq!(dropped, 2);
        let mut f = [0.0; 2];
        while pop_frame(&mut rx, &mut f) {
            assert_aligned(&f);
        }
        // after draining, new frames stay aligned too
        push_frames(&mut tx, &stereo(2, 10), 2, 2);
        while pop_frame(&mut rx, &mut f) {
            assert_aligned(&f);
        }
    }

    #[test]
    fn an_underrun_takes_no_partial_frame() {
        let (mut tx, mut rx) = RingBuffer::<f32>::new(16);
        // a lone sample: half a frame
        tx.push(1.0).unwrap();
        let mut f = [9.0; 2];
        assert!(!pop_frame(&mut rx, &mut f));
        assert_eq!(f, [0.0, 0.0]);
        // its other half arrives: the frame comes out whole and in order
        tx.push(-1.0).unwrap();
        assert!(pop_frame(&mut rx, &mut f));
        assert_eq!(f, [1.0, -1.0]);
    }

    #[test]
    fn extra_input_channels_are_skipped() {
        let (mut tx, mut rx) = RingBuffer::<f32>::new(16);
        // four input channels, keep the first two
        push_frames(&mut tx, &[1.0, -1.0, 7.0, 7.0, 2.0, -2.0, 7.0, 7.0], 4, 2);
        let mut f = [0.0; 2];
        assert!(pop_frame(&mut rx, &mut f) && f == [1.0, -1.0]);
        assert!(pop_frame(&mut rx, &mut f) && f == [2.0, -2.0]);
        assert!(!pop_frame(&mut rx, &mut f));
    }
}
