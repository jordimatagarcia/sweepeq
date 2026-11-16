//! The real-time rule: once built, `process` never allocates or frees memory, including
//! while swapping filters. Checked with a global allocator that counts calls made while a
//! thread-local flag is set.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::{filter, noise};
use sweepeq_core::{Convolver, DirectFir, Partitioned, engine};

struct Counting;

static CALLS: AtomicUsize = AtomicUsize::new(0);
thread_local! { static WATCH: Cell<bool> = const { Cell::new(false) }; }

fn note() {
    if WATCH.with(Cell::get) {
        CALLS.fetch_add(1, Ordering::SeqCst);
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note();
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        note();
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note();
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn watched<F: FnOnce()>(f: F) -> usize {
    CALLS.store(0, Ordering::SeqCst);
    WATCH.with(|w| w.set(true));
    f();
    WATCH.with(|w| w.set(false));
    CALLS.load(Ordering::SeqCst)
}

#[test]
fn process_does_not_allocate() {
    let h = filter(4096, 1);
    let x = noise(48_000, 2);
    let mut y = vec![0.0f32; x.len()];
    let mut convolvers: Vec<Box<dyn Convolver>> = vec![
        Box::new(DirectFir::new(&h, 256)),
        Box::new(Partitioned::new(&h, 256)),
    ];
    for c in &mut convolvers {
        let n = watched(|| {
            for (xi, yi) in x.chunks(333).zip(y.chunks_mut(333)) {
                c.process(xi, yi);
            }
            c.reset();
        });
        assert_eq!(n, 0, "{c:?} allocated {n} times");
    }
}

#[test]
fn swapping_does_not_allocate_on_the_audio_thread() {
    let x = noise(48_000, 3);
    let mut y = vec![0.0f32; x.len()];
    let (mut eng, mut ctl) = engine(Box::new(Partitioned::new(&filter(2048, 1), 128)), 4096, 512);
    for seed in 0..3 {
        ctl.set(Box::new(Partitioned::new(&filter(2048, seed + 2), 128)))
            .unwrap();
        let n = watched(|| {
            for (xi, yi) in x.chunks(256).zip(y.chunks_mut(256)) {
                eng.process(xi, yi);
            }
        });
        assert_eq!(
            n, 0,
            "swap {seed}: the audio thread allocated or freed {n} times"
        );
        assert_eq!(ctl.collect(), 1); // the old filter is freed here, off the audio thread
    }
}

#[test]
fn the_counter_sees_allocations() {
    // guards the tests above against a counter that never counts
    let n = watched(|| drop(std::hint::black_box(vec![1u8; 16])));
    assert_eq!(n, 2);
}
