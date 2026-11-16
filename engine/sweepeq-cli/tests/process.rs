//! `sweepeq-engine process` end to end: WAV in, WAV out, latency removed.

use std::path::Path;
use std::process::Command;

fn write_wav(path: &Path, channels: &[Vec<f32>], fs: u32) {
    let spec = hound::WavSpec {
        channels: channels.len() as u16,
        sample_rate: fs,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    for i in 0..channels[0].len() {
        for c in channels {
            w.write_sample(c[i]).unwrap();
        }
    }
    w.finalize().unwrap();
}

fn read_wav(path: &Path) -> (Vec<Vec<f32>>, u32) {
    let mut r = hound::WavReader::open(path).unwrap();
    let spec = r.spec();
    let ch = usize::from(spec.channels);
    let s: Vec<f32> = r.samples::<f32>().map(Result::unwrap).collect();
    let out = (0..ch)
        .map(|c| s.iter().skip(c).step_by(ch).copied().collect())
        .collect();
    (out, spec.sample_rate)
}

#[test]
fn process_aligns_output_with_input_for_both_methods() {
    let dir = std::env::temp_dir().join(format!("sweepeq-cli-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // a filter that delays by 3 samples and halves: y[n] = 0.5 x[n - 3]
    let mut h = vec![0.0f32; 300];
    h[3] = 0.5;
    write_wav(&dir.join("f.wav"), &[h], 48_000);
    let left: Vec<f32> = (0..5000)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 100.0)
        .collect();
    let right: Vec<f32> = left.iter().map(|v| -v).collect();
    write_wav(&dir.join("in.wav"), &[left.clone(), right.clone()], 48_000);

    for method in ["direct", "partitioned"] {
        let out = dir.join(format!("out_{method}.wav"));
        let status = Command::new(env!("CARGO_BIN_EXE_sweepeq-engine"))
            .args(["process", "--method", method, "--block", "64", "--filter"])
            .arg(dir.join("f.wav"))
            .arg("--input")
            .arg(dir.join("in.wav"))
            .arg("--output")
            .arg(&out)
            .status()
            .unwrap();
        assert!(status.success());
        let (y, fs) = read_wav(&out);
        assert_eq!(fs, 48_000);
        assert_eq!(y.len(), 2);
        assert_eq!(y[0].len(), left.len(), "{method}: output length");
        for i in 3..left.len() {
            assert!((y[0][i] - 0.5 * left[i - 3]).abs() < 1e-6, "{method} L {i}");
            assert!(
                (y[1][i] - 0.5 * right[i - 3]).abs() < 1e-6,
                "{method} R {i}"
            );
        }
    }

    // a filter for another sample rate is refused
    write_wav(&dir.join("f44.wav"), &[vec![1.0]], 44_100);
    let status = Command::new(env!("CARGO_BIN_EXE_sweepeq-engine"))
        .args(["process", "--filter"])
        .arg(dir.join("f44.wav"))
        .arg("--input")
        .arg(dir.join("in.wav"))
        .arg("--output")
        .arg(dir.join("bad.wav"))
        .output()
        .unwrap();
    assert!(!status.status.success());
    assert!(String::from_utf8_lossy(&status.stderr).contains("44100 Hz"));
    std::fs::remove_dir_all(&dir).ok();
}
