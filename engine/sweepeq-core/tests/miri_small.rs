//! Small cases of every convolver and the swap, sized to run under Miri
//! (`cargo +nightly miri test -p sweepeq-core --lib --test miri_small`), which checks the
//! unsafe code paths (and rtrb's, realfft's) for undefined behaviour.

mod common;

use common::{filter, noise, reference, rel_error, run};
use sweepeq_core::{Convolver, DirectFir, Kernel, Partitioned, engine};

#[test]
fn small_direct_and_partitioned() {
    let h = filter(37, 1);
    let x = noise(300, 2);
    let want = reference(&x, &h);
    let mut d = DirectFir::with_kernel(&h, 16, Kernel::detect());
    assert!(rel_error(&run(&mut d, &x, &[1, 5, 16, 33]), &want, &x, &h) < 5e-7);
    let mut p = Partitioned::new(&h, 8);
    let got = run(&mut p, &x, &[3, 8, 20]);
    assert!(rel_error(&got[8..], &want[..x.len() - 8], &x, &h) < 5e-7);
}

#[test]
fn small_swap() {
    let x = noise(400, 3);
    let mut y = vec![0.0f32; 400];
    let (mut eng, mut ctl) = engine(Box::new(Partitioned::new(&filter(20, 1), 8)), 32, 64);
    ctl.set(Box::new(Partitioned::new(&filter(20, 2), 8)))
        .unwrap();
    eng.process(&x, &mut y);
    assert_eq!(ctl.collect(), 1);
    assert!(y.iter().all(|v| v.is_finite()));
}
