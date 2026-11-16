"""Measurement sessions: play a sweep, record the response, compute impulse responses.

The audio I/O is behind the ``Recorder`` protocol, so the same code runs against a sound card
(``SoundDeviceRecorder``, needs the ``audio`` extra) or a simulated device (tests).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from sweepeq.deconv import HarmonicIr, deconvolve, separate
from sweepeq.simulated import SimulatedDevice
from sweepeq.sweep import SweepSpec, synchronized_sweep

FloatArray = NDArray[np.float64]

TAIL_S = 1.0
"""Silence recorded after the sweep, for the device's decay and latency."""


class Recorder(Protocol):
    def play_record(self, out: FloatArray, fs: int) -> FloatArray:
        """Play ``out`` and return the simultaneous recording (same length or longer)."""
        ...


@dataclass(frozen=True)
class SimulatedRecorder:
    device: SimulatedDevice
    seed: int = 0

    def play_record(self, out: FloatArray, fs: int) -> FloatArray:
        if fs != self.device.fs:
            raise ValueError("sample rate differs from the device's")
        return self.device.respond(out, np.random.default_rng(self.seed))


@dataclass(frozen=True)
class SoundDeviceRecorder:
    """A sound card through PortAudio. Input and output device may differ; ``channel`` is
    the input channel the microphone is on (0-based)."""

    device: int | str | None = None
    channel: int = 0

    def play_record(self, out: FloatArray, fs: int) -> FloatArray:
        import sounddevice as sd  # optional dependency

        rec = sd.playrec(
            out.astype(np.float32),
            samplerate=fs,
            channels=self.channel + 1,
            device=self.device,
            blocking=True,
        )
        return np.asarray(rec[:, self.channel], dtype=np.float64)


@dataclass(frozen=True)
class Measurement:
    spec: SweepSpec
    recorded: FloatArray
    harmonics: list[HarmonicIr]
    """Linear response first, then harmonics 2, 3, …"""
    peak_level: float
    """Recording peak (full scale = 1): close to 1 means it may have clipped."""
    latency_s: float
    """Arrival time of the linear impulse response's peak: the playback-to-recording
    latency plus the acoustic path (and a few samples of the measurement's own band limit)."""
    peak_to_noise_db: float
    """Linear impulse response peak relative to the noise floor of the deconvolution, taken
    where neither the linear nor the harmonic responses are. Below ``LOW_PNR_DB`` the
    measurement is mostly noise (e.g. the microphone isn't hearing the device)."""

    @property
    def linear(self) -> HarmonicIr:
        return self.harmonics[0]

    def save(self, path: Path) -> None:
        arrays: dict[str, FloatArray] = {
            "fs": np.array(self.spec.fs),
            "f1": np.array(self.spec.f1),
            "f2": np.array(self.spec.f2),
            "rate_s": np.array(self.spec.rate_s),
            "recorded": self.recorded,
        }
        for h in self.harmonics:
            arrays[f"ir_{h.order}"] = h.ir
            arrays[f"offset_{h.order}"] = np.array(h.offset_s)
        np.savez_compressed(path, allow_pickle=False, **arrays)


def load_linear(path: Path) -> tuple[HarmonicIr, int, float, float]:
    """Linear impulse response, sample rate and sweep band of a saved measurement."""
    with np.load(path, allow_pickle=False) as d:
        hir = HarmonicIr(1, np.asarray(d["ir_1"], dtype=np.float64), float(d["offset_1"]))
        return hir, int(d["fs"]), float(d["f1"]), float(d["f2"])


CLIP_LEVEL = 0.99
LOW_PNR_DB = 40.0


def measure(spec: SweepSpec, recorder: Recorder, n_harmonics: int = 5) -> Measurement:
    """Play the sweep (followed by ``TAIL_S`` of silence) and separate the responses."""
    sweep = synchronized_sweep(spec)
    out = np.concatenate([sweep, np.zeros(round(TAIL_S * spec.fs))])
    recorded = recorder.play_record(out, spec.fs)
    peak = float(np.max(np.abs(recorded)))
    h = deconvolve(recorded, sweep, spec)
    harmonics = separate(h, spec, n_max=n_harmonics)
    linear = harmonics[0]
    k = int(np.argmax(np.abs(linear.ir)))
    latency = k / spec.fs + linear.offset_s
    return Measurement(
        spec, recorded, harmonics, peak, latency, _peak_to_noise(h, linear, spec, n_harmonics)
    )


def _peak_to_noise(h: FloatArray, linear: HarmonicIr, spec: SweepSpec, n_max: int) -> float:
    """Peak of the linear response over the RMS of the deconvolution result between the
    end of the linear window and the start of the highest harmonic's (nan if that gap is
    too short to estimate from)."""
    start = len(linear.ir) + round(linear.offset_s * spec.fs) + round(0.01 * spec.fs)
    stop = len(h) - round(spec.harmonic_delay_s(n_max) * spec.fs) - round(0.01 * spec.fs)
    if stop - start < round(0.05 * spec.fs):
        return float("nan")
    noise = float(np.sqrt(np.mean(h[start:stop] ** 2)))
    peak = float(np.max(np.abs(linear.ir)))
    return float("inf") if noise == 0.0 else 20.0 * float(np.log10(peak / noise))
