# Validation against REW (M2)

Target: sweepeq's measurement of a real loudspeaker agrees with Room EQ Wizard (REW) on the
same setup to within ±1 dB from 50 Hz to 10 kHz, both with 1/6-octave smoothing and
normalised at 1 kHz.

## Setup

- Loudspeaker, a USB measurement microphone with its calibration file, and a sound card
  (the USB microphone's own interface is fine for the input).
- Microphone on a stand at about 0.5 m from the speaker, on its axis, away from walls.
- Do not move anything between the sweepeq and REW measurements.
- Set the level once so the sweep is clearly audible but nowhere near clipping (sweepeq
  warns if the recording peaks above −0.1 dBFS).

## Procedure

```bash
uv sync --all-extras
uv run sweepeq devices                       # find the device index
uv run sweepeq measure --device N --out recordings/spk_pos1.npz
# optional: two more microphone positions, a few cm apart
uv run sweepeq measure --device N --out recordings/spk_pos2.npz
uv run sweepeq measure --device N --out recordings/spk_pos3.npz
```

In REW: measure the same speaker with the same microphone, calibration file and positions
(a 20 Hz–20 kHz sweep), average the positions if more than one, then *File → Export →
Export measurement as text*, saved as `recordings/rew.txt`.

```bash
uv run sweepeq analyse recordings/spk_pos*.npz --cal <mic calibration file> \
    --compare recordings/rew.txt --plot docs/images/rew_comparison.png
```

The command prints the largest difference over 50 Hz–10 kHz. `recordings/` is ignored by
git; only the plot and the numbers below are published.

## Results

To be filled in after the measurement:

| Device | Positions | Max difference, 50 Hz–10 kHz |
|---|---|---|
| | | |

![sweepeq vs REW](images/rew_comparison.png)

## Known differences to expect

- **Windowing.** Neither tool gates out room reflections by default, so both measure the
  speaker in the room. Small differences at low frequencies come from REW's and sweepeq's
  different impulse-response window lengths.
- **Smoothing.** Both smooth in power over 1/6 octave, but window shapes may differ
  slightly; differences show at sharp peaks and notches.
- **Absolute level.** Not compared: REW calibrates to SPL, sweepeq reports relative level.
