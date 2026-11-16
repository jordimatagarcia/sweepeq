"""Measurement sessions: play a sweep, record the response, compute impulse responses.

The audio I/O is behind the ``Recorder`` protocol, so the same code runs against a sound card
(``SoundDeviceRecorder``, needs the ``audio`` extra) or the simulated plant (tests).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from sweepeq.deconv import HarmonicIr, deconvolve, separate
from sweepeq.plant import Plant
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
    plant: Plant
    seed: int = 0

    def play_record(self, out: FloatArray, fs: int) -> FloatArray:
        if fs != self.plant.fs:
            raise ValueError("sample rate differs from the plant's")
        return self.plant.respond(out, np.random.default_rng(self.seed))


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


CLIP_LEVEL = 0.99


def measure(spec: SweepSpec, recorder: Recorder, n_harmonics: int = 5) -> Measurement:
    """Play the sweep (followed by ``TAIL_S`` of silence) and separate the responses."""
    sweep = synchronized_sweep(spec)
    out = np.concatenate([sweep, np.zeros(round(TAIL_S * spec.fs))])
    recorded = recorder.play_record(out, spec.fs)
    peak = float(np.max(np.abs(recorded)))
    h = deconvolve(recorded, sweep, spec)
    return Measurement(spec, recorded, separate(h, spec, n_max=n_harmonics), peak)
