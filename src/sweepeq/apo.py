"""Parametric EQ presets in Equalizer APO's text format.

This is the format AutoEQ publishes and EasyEffects imports and exports, e.g.::

    Preamp: -4.2 dB
    Filter 1: ON LSC Fc 105 Hz Gain -1.7 dB Q 0.70
    Filter 2: ON PK Fc 217.4 Hz Gain 4.4 dB Q 2.66
    Filter 3: ON HSC Fc 10000 Hz Gain -4.0 dB Q 0.70

Supported filter types: PK (peaking), LSC / LS (low shelf), HSC / HS (high shelf), all as
RBJ cookbook biquads with the given Q. Disabled (OFF) filters and lines that are not
filters are ignored; an enabled filter of any other type (high-pass, notch, ...) is an
error, so a preset is never silently read as a different curve.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from sweepeq.simulated import peaking_biquad, shelf_biquad

FloatArray = NDArray[np.float64]

_FILTER = re.compile(
    r"Filter\s*\d*\s*:\s*(?P<on>ON|OFF)\s+(?P<type>PK|LSC|LS|HSC|HS)\s+"
    r"Fc\s+(?P<fc>[-\d.eE+]+)\s*Hz\s+Gain\s+(?P<gain>[-\d.eE+]+)\s*dB"
    r"(?:\s+Q\s+(?P<q>[-\d.eE+]+))?",
    re.IGNORECASE,
)
_ANY_FILTER = re.compile(r"Filter\s*\d*\s*:\s*ON\b", re.IGNORECASE)
_PREAMP = re.compile(r"Preamp\s*:\s*(?P<db>[-\d.eE+]+)\s*dB", re.IGNORECASE)
DEFAULT_SHELF_Q = 0.7071


@dataclass(frozen=True)
class Band:
    kind: str
    """"PK", "LS" or "HS"."""
    fc: float
    gain_db: float
    q: float


@dataclass(frozen=True)
class Preset:
    preamp_db: float
    bands: tuple[Band, ...]

    def sos(self, fs: float) -> FloatArray:
        rows = []
        for b in self.bands:
            if b.kind == "PK":
                rows.append(peaking_biquad(b.fc, b.gain_db, b.q, fs))
            else:
                rows.append(
                    shelf_biquad("low" if b.kind == "LS" else "high", b.fc, b.gain_db, b.q, fs)
                )
        return np.vstack(rows) if rows else np.empty((0, 6))

    def response_db(self, freqs: FloatArray, fs: float) -> FloatArray:
        """Magnitude (dB) of preamp plus bands at ``freqs``, at sample rate ``fs``."""
        db = np.full(len(freqs), self.preamp_db)
        sos = self.sos(fs)
        if len(sos):
            _, h = signal.sosfreqz(sos, worN=freqs, fs=fs)
            db = db + 20.0 * np.log10(np.abs(h))
        return np.asarray(db, dtype=np.float64)


def parse_apo(text: str) -> Preset:
    preamp = 0.0
    bands: list[Band] = []
    for line in text.splitlines():
        if m := _PREAMP.search(line):
            preamp += float(m["db"])
        elif m := _FILTER.search(line):
            if m["on"].upper() == "OFF":
                continue
            kind = {"PK": "PK", "LSC": "LS", "LS": "LS", "HSC": "HS", "HS": "HS"}[m["type"].upper()]
            if m["q"] is None and kind == "PK":
                raise ValueError(f"peaking filter without Q: {line.strip()!r}")
            q = float(m["q"]) if m["q"] is not None else DEFAULT_SHELF_Q
            bands.append(Band(kind, float(m["fc"]), float(m["gain"]), q))
        elif _ANY_FILTER.search(line):
            raise ValueError(
                f"unsupported filter (only PK, LSC/LS and HSC/HS are): {line.strip()!r}"
            )
    return Preset(preamp, tuple(bands))


def read_apo(path: Path) -> Preset:
    return parse_apo(path.read_text(encoding="utf-8", errors="replace"))


def format_apo(preset: Preset) -> str:
    """The preset as Equalizer APO text (what ``parse_apo`` reads back)."""
    names = {"PK": "PK", "LS": "LSC", "HS": "HSC"}
    lines = [f"Preamp: {preset.preamp_db:.1f} dB"]
    for i, b in enumerate(preset.bands, start=1):
        lines.append(
            f"Filter {i}: ON {names[b.kind]} Fc {b.fc:.1f} Hz Gain {b.gain_db:.1f} dB Q {b.q:.2f}"
        )
    return "\n".join(lines) + "\n"
