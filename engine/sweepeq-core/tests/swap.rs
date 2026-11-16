//! Changing the filter while running: crossfade, latency checks, returning old filters.

mod common;

use common::{filter, noise, reference, run};
use sweepeq_core::{Convolver, DirectFir, Partitioned, SwapError, engine};

#[test]
fn swap_crossfades_then_runs_the_new_filter() {
    let (h1, h2) = (filter(200, 1), filter(200, 2));
    let x = noise(20_000, 9);
    let fade = 1000;
    let (mut eng, mut ctl) = engine(Box::new(DirectFir::new(&h1, 256)), fade, 256);
    let before = run(&mut eng, &x[..5000], &[256]);
    let want1 = reference(&x[..5000], &h1);
    assert!(
        before
            .iter()
            .zip(&want1)
            .all(|(&g, &w)| (f64::from(g) - w).abs() < 1e-5)
    );

    ctl.set(Box::new(DirectFir::new(&h2, 256))).unwrap();
    let mut after = vec![0.0f32; 15_000];
    eng.process(&x[5000..], &mut after);
    assert!(!eng.is_fading());
    // once the fade and the new filter's history have filled, the output is h2's
    let want2 = reference(&x, &h2);
    for i in (fade + 200)..15_000 {
        assert!(
            (f64::from(after[i]) - want2[5000 + i]).abs() < 1e-5,
            "sample {i}"
        );
    }
    assert_eq!(ctl.collect(), 1);
}

#[test]
fn swapping_to_the_same_filter_is_seamless() {
    let h = filter(64, 4);
    let x = noise(8000, 5);
    let (mut eng, mut ctl) = engine(Box::new(DirectFir::new(&h, 128)), 2048, 128);
    let mut y = run(&mut eng, &x[..1000], &[128]);
    ctl.set(Box::new(DirectFir::new(&h, 128))).unwrap();
    y.extend(run(&mut eng, &x[1000..], &[100]));
    let want = reference(&x, &h);
    // during the fade the new copy lacks history for its first 63 samples, so the output
    // dips there; after that, both copies agree and the crossfade is exact
    for i in 1000 + 64..x.len() {
        assert!((f64::from(y[i]) - want[i]).abs() < 1e-5, "sample {i}");
    }
}

#[test]
fn latency_must_match() {
    let h = filter(100, 1);
    let (_eng, mut ctl) = engine(Box::new(Partitioned::new(&h, 64)), 512, 256);
    match ctl.set(Box::new(DirectFir::new(&h, 64))) {
        Err(SwapError::LatencyMismatch {
            running: 64,
            new: 0,
        }) => {}
        other => panic!("expected a latency mismatch, got {other:?}"),
    }
    assert!(ctl.set(Box::new(Partitioned::new(&h, 64))).is_ok());
}

#[test]
fn queue_full_returns_the_convolver() {
    let h = filter(10, 1);
    let (_eng, mut ctl) = engine(Box::new(DirectFir::new(&h, 64)), 64, 64);
    ctl.set(Box::new(DirectFir::new(&h, 64))).unwrap();
    ctl.set(Box::new(DirectFir::new(&h, 64))).unwrap();
    match ctl.set(Box::new(DirectFir::new(&h, 64))) {
        Err(SwapError::Full(c)) => assert_eq!(c.taps(), 10),
        other => panic!("expected Full, got {other:?}"),
    }
}

#[test]
fn swaps_wait_while_old_filters_are_not_collected() {
    let h = filter(10, 1);
    let x = noise(64, 1);
    let mut y = vec![0.0f32; 64];
    let (mut eng, mut ctl) = engine(Box::new(DirectFir::new(&h, 64)), 16, 64);
    // three swaps fill the return ring (capacity 3); the fourth must wait
    for _ in 0..3 {
        ctl.set(Box::new(DirectFir::new(&h, 64))).unwrap();
        eng.process(&x, &mut y);
        eng.process(&x, &mut y);
    }
    ctl.set(Box::new(DirectFir::new(&h, 64))).unwrap();
    eng.process(&x, &mut y);
    assert!(
        !eng.is_fading() && !ctl.idle(),
        "the engine must not start a swap it can't finish"
    );
    assert_eq!(ctl.collect(), 3);
    eng.process(&x, &mut y);
    assert!(ctl.idle());
}
