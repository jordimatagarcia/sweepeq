//! `sweepeq-engine`: apply an FIR correction filter (a WAV file of taps, as written by
//! `sweepeq design --fir`) to a WAV file, or to live audio.

mod frames;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, Subcommand, ValueEnum};
use sweepeq_core::{Convolver, Method, build, engine};

#[derive(Parser)]
#[command(name = "sweepeq-engine", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, ValueEnum)]
enum MethodArg {
    Direct,
    Partitioned,
}

#[derive(Subcommand)]
enum Command {
    /// Filter a WAV file; the output is aligned with the input (latency removed).
    Process {
        #[arg(long)]
        filter: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, value_enum, default_value = "partitioned")]
        method: MethodArg,
        /// Partition size (partitioned method).
        #[arg(long, default_value_t = 256)]
        block: usize,
    },
    /// Show a filter's length, gain and peak.
    Info {
        #[arg(long)]
        filter: PathBuf,
    },
    /// Filter live audio from an input device to an output device. With --watch, edits to
    /// the filter file are picked up and crossfaded in while running.
    Live {
        #[arg(long)]
        filter: PathBuf,
        #[arg(long, value_enum, default_value = "partitioned")]
        method: MethodArg,
        #[arg(long, default_value_t = 256)]
        block: usize,
        /// Input device name (default: the system default).
        #[arg(long)]
        input_device: Option<String>,
        /// Output device name (default: the system default).
        #[arg(long)]
        output_device: Option<String>,
        /// Stop after this many seconds (default: run until interrupted).
        #[arg(long)]
        seconds: Option<f64>,
        /// Reload the filter when the file changes.
        #[arg(long)]
        watch: bool,
        /// Crossfade length when swapping filters (ms).
        #[arg(long, default_value_t = 100.0)]
        fade_ms: f64,
    },
}

fn method(m: MethodArg, block: usize) -> Method {
    match m {
        MethodArg::Direct => Method::Direct,
        MethodArg::Partitioned => Method::Partitioned { block },
    }
}

/// Filter taps per channel, and the sample rate, from a WAV file.
fn read_filter(path: &Path) -> Result<(Vec<Vec<f32>>, u32)> {
    let (data, spec) = read_wav(path)?;
    ensure!(
        !data.is_empty() && !data[0].is_empty(),
        "{} has no samples",
        path.display()
    );
    Ok((data, spec.sample_rate))
}

/// Deinterleaved samples as f32 and the file's spec.
fn read_wav(path: &Path) -> Result<(Vec<Vec<f32>>, hound::WavSpec)> {
    let mut r =
        hound::WavReader::open(path).with_context(|| format!("reading {}", path.display()))?;
    let spec = r.spec();
    let ch = usize::from(spec.channels);
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    };
    let mut out = vec![Vec::with_capacity(samples.len() / ch); ch];
    for frame in samples.chunks_exact(ch) {
        for (c, &v) in frame.iter().enumerate() {
            out[c].push(v);
        }
    }
    Ok((out, spec))
}

fn filter_for_channel(filters: &[Vec<f32>], c: usize) -> &[f32] {
    &filters[if filters.len() == 1 { 0 } else { c }]
}

fn process(filter: &Path, input: &Path, output: &Path, m: Method) -> Result<()> {
    let (filters, filter_fs) = read_filter(filter)?;
    let (channels, spec) = read_wav(input)?;
    ensure!(
        filters.len() == 1 || filters.len() == channels.len(),
        "the filter has {} channels and the input {}: use a mono filter or one per channel",
        filters.len(),
        channels.len()
    );
    if filter_fs != spec.sample_rate {
        bail!(
            "the filter is for {filter_fs} Hz but the input is {} Hz",
            spec.sample_rate
        );
    }
    let started = Instant::now();
    let mut out = Vec::with_capacity(channels.len());
    for (c, x) in channels.iter().enumerate() {
        let mut conv = build(filter_for_channel(&filters, c), m, 4096);
        let lat = conv.latency();
        // feed `latency` extra zeros so the aligned output covers the whole input
        let mut padded = x.clone();
        padded.resize(x.len() + lat, 0.0);
        let mut y = vec![0.0f32; padded.len()];
        conv.process(&padded, &mut y);
        out.push(y[lat..].to_vec());
    }
    let elapsed = started.elapsed().as_secs_f64();
    let wav_spec = hound::WavSpec {
        channels: spec.channels,
        sample_rate: spec.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(output, wav_spec)?;
    let mut peak = 0.0f32;
    for i in 0..out[0].len() {
        for ch in &out {
            peak = peak.max(ch[i].abs());
            w.write_sample(ch[i])?;
        }
    }
    w.finalize()?;
    let seconds = out[0].len() as f64 / f64::from(spec.sample_rate);
    println!(
        "wrote {} ({:.1} s, {} ch), {:.0}x real time, peak {:.2} dBFS",
        output.display(),
        seconds,
        out.len(),
        seconds / elapsed.max(1e-9),
        20.0 * f64::from(peak.max(1e-12)).log10()
    );
    if peak > 1.0 {
        println!("warning: the output exceeds full scale; lower the input or add a preamp");
    }
    Ok(())
}

fn info(filter: &Path) -> Result<()> {
    let (filters, fs) = read_filter(filter)?;
    for (c, h) in filters.iter().enumerate() {
        let dc: f64 = h.iter().map(|&v| f64::from(v)).sum();
        let l1: f64 = h.iter().map(|&v| f64::from(v.abs())).sum();
        let peak =
            h.iter().enumerate().fold(
                (0, 0.0f32),
                |a, (i, &v)| if v.abs() > a.1 { (i, v.abs()) } else { a },
            );
        println!(
            "channel {c}: {} taps ({:.1} ms at {fs} Hz), DC gain {:+.2} dB, worst-case gain {:+.2} dB, peak at tap {}",
            h.len(),
            h.len() as f64 / f64::from(fs) * 1e3,
            20.0 * dc.abs().max(1e-12).log10(),
            20.0 * l1.max(1e-12).log10(),
            peak.0
        );
    }
    Ok(())
}

fn live(
    filter: &Path,
    m: Method,
    input_device: Option<&str>,
    output_device: Option<&str>,
    seconds: Option<f64>,
    watch: bool,
    fade_ms: f64,
) -> Result<()> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    let host = cpal::default_host();
    let find = |name: Option<&str>, input: bool| -> Result<cpal::Device> {
        match name {
            None => if input {
                host.default_input_device()
            } else {
                host.default_output_device()
            }
            .context("no default audio device"),
            Some(n) => {
                let mut devices = if input {
                    host.input_devices()?
                } else {
                    host.output_devices()?
                };
                devices
                    .find(|d| d.name().map(|x| x == n).unwrap_or(false))
                    .with_context(|| format!("no audio device named {n:?}"))
            }
        }
    };
    let in_dev = find(input_device, true)?;
    let out_dev = find(output_device, false)?;
    let in_cfg: cpal::StreamConfig = in_dev.default_input_config()?.into();
    let out_cfg: cpal::StreamConfig = out_dev.default_output_config()?.into();
    ensure!(
        in_cfg.sample_rate == out_cfg.sample_rate,
        "input runs at {} Hz and output at {} Hz; set both devices to the same rate",
        in_cfg.sample_rate.0,
        out_cfg.sample_rate.0
    );
    let fs = out_cfg.sample_rate.0;
    let (filters, filter_fs) = read_filter(filter)?;
    ensure!(
        filter_fs == fs,
        "the filter is for {filter_fs} Hz but the devices run at {fs} Hz"
    );
    let in_ch = usize::from(in_cfg.channels);
    let out_ch = usize::from(out_cfg.channels);
    let channels = in_ch.min(out_ch).min(2);
    ensure!(
        filters.len() == 1 || filters.len() >= channels,
        "not enough filter channels"
    );

    // one engine per channel; controllers stay on this thread for swaps
    const MAX_BLOCK: usize = 8192;
    let fade = ((fade_ms / 1e3) * f64::from(fs)).max(1.0) as usize;
    let mut engines = Vec::new();
    let mut controllers = Vec::new();
    for c in 0..channels {
        let (e, ctl) = engine(
            build(filter_for_channel(&filters, c), m, MAX_BLOCK),
            fade,
            MAX_BLOCK,
        );
        engines.push(e);
        controllers.push(ctl);
    }

    // input -> output through a lock-free ring of interleaved frames (~0.5 s)
    let (mut tx, mut rx) = rtrb::RingBuffer::<f32>::new(fs as usize * channels / 2);
    let in_stream = in_dev.build_input_stream(
        &in_cfg,
        move |data: &[f32], _| {
            // a full ring drops whole frames rather than blocking
            frames::push_frames(&mut tx, data, in_ch, channels);
        },
        |e| eprintln!("input error: {e}"),
        None,
    )?;
    let mut x = vec![vec![0.0f32; MAX_BLOCK]; channels];
    let mut frame = vec![0.0f32; channels];
    let mut y = vec![vec![0.0f32; MAX_BLOCK]; channels];
    let out_stream = out_dev.build_output_stream(
        &out_cfg,
        move |data: &mut [f32], _| {
            for chunk in data.chunks_mut(MAX_BLOCK * out_ch) {
                let n_frames = chunk.len() / out_ch;
                for i in 0..n_frames {
                    // underrun: a frame of silence, never part of one
                    frames::pop_frame(&mut rx, &mut frame);
                    for (xc, &v) in x.iter_mut().zip(&frame) {
                        xc[i] = v;
                    }
                }
                for ((e, xc), yc) in engines.iter_mut().zip(&x).zip(y.iter_mut()) {
                    e.process(&xc[..n_frames], &mut yc[..n_frames]);
                }
                for (i, frame) in chunk.chunks_mut(out_ch).enumerate() {
                    for (k, s) in frame.iter_mut().enumerate() {
                        *s = y[k.min(channels - 1)][i];
                    }
                }
            }
        },
        |e| eprintln!("output error: {e}"),
        None,
    )?;
    in_stream.play()?;
    out_stream.play()?;
    println!(
        "running: {} -> {}, {fs} Hz, {channels} ch, {} taps, added latency {} samples{}",
        in_dev.name().unwrap_or_default(),
        out_dev.name().unwrap_or_default(),
        filters[0].len(),
        build(&filters[0], m, MAX_BLOCK).latency(),
        if watch {
            ", watching the filter file"
        } else {
            ""
        }
    );

    let start = Instant::now();
    let mut stamp = modified(filter);
    loop {
        std::thread::sleep(Duration::from_millis(200));
        for ctl in &mut controllers {
            ctl.collect();
        }
        if watch && modified(filter) != stamp {
            stamp = modified(filter);
            match read_filter(filter) {
                Ok((new, nfs)) if nfs == fs => {
                    for (c, ctl) in controllers.iter_mut().enumerate() {
                        let conv = build(filter_for_channel(&new, c), m, MAX_BLOCK);
                        if let Err(e) = ctl.set(conv) {
                            eprintln!("swap failed on channel {c}: {e:?}");
                        }
                    }
                    println!("filter reloaded ({} taps), crossfading", new[0].len());
                }
                Ok((_, nfs)) => eprintln!("ignored: the new filter is for {nfs} Hz"),
                Err(e) => eprintln!("ignored: {e:#}"),
            }
        }
        if seconds.is_some_and(|s| start.elapsed().as_secs_f64() >= s) {
            break;
        }
    }
    Ok(())
}

fn modified(p: &Path) -> Option<SystemTime> {
    std::fs::metadata(p).and_then(|m| m.modified()).ok()
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Process {
            filter,
            input,
            output,
            method: m,
            block,
        } => process(&filter, &input, &output, method(m, block)),
        Command::Info { filter } => info(&filter),
        Command::Live {
            filter,
            method: m,
            block,
            input_device,
            output_device,
            seconds,
            watch,
            fade_ms,
        } => live(
            &filter,
            method(m, block),
            input_device.as_deref(),
            output_device.as_deref(),
            seconds,
            watch,
            fade_ms,
        ),
    }
}
