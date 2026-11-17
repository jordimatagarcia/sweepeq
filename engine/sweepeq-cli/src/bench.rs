//! `sweepeq-engine bench`: time every convolution method over filter lengths and block
//! sizes, as an audio callback would drive it, and write the results as JSON.
//!
//! Each case processes one second of audio at 48 kHz in chunks of `chunk` samples (an
//! audio callback's size), after a warm-up pass. It repeats until it has run for at least
//! `min_time` and at least three times, and reports the median, so a single slow run
//! (another process, a frequency change) does not decide the result.
//!
//! Averages are not what a real-time thread cares about: every callback has a deadline of
//! `chunk / fs`. A partitioned convolver with a block larger than the callback does all its
//! work in one callback out of `block / chunk`, so it also records the slowest single call
//! (median over the runs of each run's slowest), as a fraction of that deadline.

use std::time::{Duration, Instant};

use anyhow::Result;
use sweepeq_core::{Convolver, DirectFir, Kernel, Partitioned};

pub const FS: f64 = 48_000.0;

pub struct Options {
    pub quick: bool,
    pub chunk: usize,
    pub min_time: Duration,
}

struct Case {
    method: String,
    block: usize,
    taps: usize,
    ns_per_sample: f64,
    worst_call_ns: f64,
}

struct Timing {
    ns_per_sample: f64,
    worst_call_ns: f64,
}

fn signal(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) as f32) / ((1u64 << 24) as f32) - 0.5
        })
        .collect()
}

fn time(c: &mut dyn Convolver, x: &[f32], chunk: usize, min_time: Duration) -> Timing {
    let mut y = vec![0.0f32; x.len()];
    // one pass over the signal; returns the slowest single call (ns)
    let pass = |c: &mut dyn Convolver, y: &mut [f32]| {
        let mut worst = 0.0f64;
        for (xi, yi) in x.chunks(chunk).zip(y.chunks_mut(chunk)) {
            let t = Instant::now();
            c.process(xi, yi);
            worst = worst.max(t.elapsed().as_nanos() as f64);
        }
        worst
    };
    pass(c, &mut y); // warm-up: caches, branch predictors, page faults
    let mut runs = Vec::new();
    let mut worsts = Vec::new();
    let started = Instant::now();
    while runs.len() < 3 || started.elapsed() < min_time {
        let t = Instant::now();
        worsts.push(pass(c, &mut y));
        runs.push(t.elapsed().as_secs_f64());
        std::hint::black_box(&y);
    }
    runs.sort_by(f64::total_cmp);
    worsts.sort_by(f64::total_cmp);
    Timing {
        ns_per_sample: runs[runs.len() / 2] * 1e9 / x.len() as f64,
        worst_call_ns: worsts[worsts.len() / 2],
    }
}

pub fn run(opts: &Options) -> Result<String> {
    let taps_list: &[usize] = if opts.quick {
        &[64, 1024, 4096]
    } else {
        &[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    };
    let blocks: &[usize] = if opts.quick {
        &[64, 256]
    } else {
        &[32, 64, 128, 256, 512, 1024, 2048]
    };
    let x = signal(FS as usize, 1);
    let simd = Kernel::detect();
    let mut cases = Vec::new();
    for &taps in taps_list {
        let h = signal(taps, taps as u64);
        let mut push = |method: String, block: usize, c: &mut dyn Convolver| {
            let t = time(c, &x, opts.chunk, opts.min_time);
            eprintln!(
                "{taps:>6} taps  {method:<16} B={block:<5} {:>9.2} ns/sample, worst call {:>8.1} us",
                t.ns_per_sample,
                t.worst_call_ns / 1e3
            );
            cases.push(Case {
                method,
                block,
                taps,
                ns_per_sample: t.ns_per_sample,
                worst_call_ns: t.worst_call_ns,
            });
        };
        push(
            format!("direct-{simd:?}"),
            0,
            &mut DirectFir::using_kernel(&h, 1024, simd),
        );
        if simd != Kernel::Scalar {
            push(
                "direct-Scalar".into(),
                0,
                &mut DirectFir::using_kernel(&h, 1024, Kernel::Scalar),
            );
        }
        for &b in blocks {
            push("partitioned".into(), b, &mut Partitioned::new(&h, b));
        }
    }
    Ok(to_json(&cases, opts, simd))
}

fn processor_name() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name") || l.starts_with("Model"))
                .and_then(|l| l.split(':').nth(1))
                .map(|m| m.trim().to_owned())
        })
        .unwrap_or_else(|| std::env::consts::ARCH.to_owned())
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn to_json(cases: &[Case], opts: &Options, simd: Kernel) -> String {
    let rows: Vec<String> = cases
        .iter()
        .map(|c| {
            format!(
                "    {{\"method\": \"{}\", \"block\": {}, \"taps\": {}, \"ns_per_sample\": {:.4}, \
                 \"realtime_factor\": {:.1}, \"cpu_percent_per_channel\": {:.4}, \
                 \"worst_call_us\": {:.3}, \"worst_call_deadline_fraction\": {:.5}}}",
                escape(&c.method),
                c.block,
                c.taps,
                c.ns_per_sample,
                1e9 / (c.ns_per_sample * FS),
                c.ns_per_sample * FS / 1e7,
                c.worst_call_ns / 1e3,
                c.worst_call_ns / (opts.chunk as f64 / FS * 1e9)
            )
        })
        .collect();
    format!(
        "{{\n  \"cpu\": \"{}\",\n  \"arch\": \"{}\",\n  \"simd\": \"{simd:?}\",\n  \"sample_rate\": {FS},\n  \
         \"chunk\": {},\n  \"quick\": {},\n  \"cases\": [\n{}\n  ]\n}}\n",
        escape(&processor_name()),
        std::env::consts::ARCH,
        opts.chunk,
        opts.quick,
        rows.join(",\n")
    )
}
