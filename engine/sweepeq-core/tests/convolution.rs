//! Every convolver against a naive f64 convolution.

mod common;

use common::{filter, noise, reference, rel_error, run};
use sweepeq_core::{Convolver, DirectFir, Kernel, Method, Partitioned, build};

const TOL: f64 = 5e-7;
const CHUNKS: [&[usize]; 4] = [&[1], &[64], &[7, 300, 1, 128, 513], &[4096]];

#[test]
fn direct_fir_matches_reference_for_every_kernel() {
    for kernel in [Kernel::Scalar, Kernel::Avx2Fma, Kernel::Neon] {
        if !kernel.is_available() {
            continue;
        }
        for taps in [1usize, 2, 7, 8, 31, 32, 33, 100, 513] {
            let h = filter(taps, taps as u64);
            let x = noise(3000, 7);
            let want = reference(&x, &h);
            for sizes in CHUNKS {
                let mut c = DirectFir::with_kernel(&h, 256, kernel);
                let got = run(&mut c, &x, sizes);
                let e = rel_error(&got, &want, &x, &h);
                assert!(e < TOL, "{kernel:?} taps={taps} chunks={sizes:?}: {e:e}");
            }
        }
    }
}

#[test]
fn partitioned_matches_reference_delayed_by_one_block() {
    for block in [2usize, 16, 64, 256] {
        for taps in [1, block - 1, block, block + 1, 3 * block, 1000] {
            let h = filter(taps, 11 + taps as u64);
            let x = noise(4000, 3);
            let want = reference(&x, &h);
            for sizes in CHUNKS {
                let mut c = Partitioned::new(&h, block);
                assert_eq!(c.latency(), block);
                let got = run(&mut c, &x, sizes);
                // output is delayed by exactly one block, which starts as silence
                assert!(got[..block].iter().all(|&v| v == 0.0));
                let e = rel_error(&got[block..], &want[..x.len() - block], &x, &h);
                assert!(e < TOL, "block={block} taps={taps} chunks={sizes:?}: {e:e}");
            }
        }
    }
}

#[test]
fn reset_forgets_history() {
    let h = filter(300, 1);
    let x = noise(2000, 2);
    for method in [Method::Direct, Method::Partitioned { block: 64 }] {
        let mut c = build(&h, method, 128);
        let first = run(c.as_mut(), &x, &[100]);
        c.reset();
        let again = run(c.as_mut(), &x, &[37]);
        let e = first
            .iter()
            .zip(&again)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(e < 1e-6, "{method:?}: {e}");
    }
}

#[test]
fn impulse_response_is_the_filter() {
    let h = filter(777, 5);
    let mut x = vec![0.0f32; 2048];
    x[0] = 1.0;
    let mut d = DirectFir::new(&h, 512);
    let got = run(&mut d, &x, &[512]);
    for (g, w) in got.iter().zip(&h) {
        assert!((g - w).abs() < 1e-7);
    }
    let mut p = Partitioned::new(&h, 128);
    let got = run(&mut p, &x, &[512]);
    for (g, w) in got[128..].iter().zip(&h) {
        assert!((g - w).abs() < 1e-6);
    }
}

#[test]
#[should_panic(expected = "power of two")]
fn partitioned_rejects_bad_block() {
    let _ = Partitioned::new(&[1.0], 48);
}
