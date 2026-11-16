# Measurement (M1)

How `sweepeq` turns a sweep recording into a linear impulse response and harmonic distortion
levels, the design decisions behind it, and how it is validated.

## Method

1. **Synchronized exponential sweep** (Farina 2000; Novak et al. 2015).
   `x(t) = a·sin(2π·f1·L·exp(t/L))`, with the rate `L` rounded so that `f1·L` is an integer.
   An exponential sweep spends equal time per octave, so after deconvolution the n-th
   harmonic's impulse response appears `L·ln(n)` seconds *before* the linear one. The
   rounding makes the harmonics start in phase, so their responses are exact, not just their
   magnitudes.
2. **Deconvolution by regularised spectral division:** `H = Y·X* / (|X|² + ε)`, with ε
   negligible in the band and large outside it (Müller and Massarani 2001). The FFT is long
   enough for linear, not circular, deconvolution; the harmonics wrap to the end.
3. **Separation:** each response is cut out with a window from 2 ms before its start to
   2 ms before the next one's, with short half-Hann tapers.
4. **Analysis:** the frequency response by a direct DFT of the windowed response, delay from
   the interpolated peak, and the n-th harmonic level as `|H_n(n·f)| / |H_1(f)|`.

## Design decisions found by testing

Each of these was found by a failing acceptance test against the simulated device.

- **Smooth regularisation edges.** A step in ε at the band edges made the impulse response
  ring at 20 Hz and 20 kHz. ε now changes over a raised-cosine taper in log frequency
  (half an octave above f1, a sixth of an octave below f2).
- **A minimum-phase band limit.** The regularised division is band-limited by the real,
  zero-phase factor `|X|²/(|X|² + ε)`. Its impulse response rings *before* time zero as
  much as after; the window cut that pre-ringing off, which cost 0.3 dB at 40–300 Hz, and
  the pre-ringing leaked into the harmonic windows (2nd harmonic off by up to 3 dB). The
  band limit is now given its minimum phase (same magnitude, causal), computed from the real
  cepstrum (Oppenheim and Schafer). The measurement then behaves like the device followed
  by a causal band-pass, as a real measurement chain does. Error after the change:
  0.003 dB.
- **Delay against a loopback reference.** The causal band-pass delays the response peak by
  a few samples, so the delay is measured relative to a loopback measurement through the
  same chain, as with a hardware loopback.
- **Fade-in in octaves.** A device with even-order distortion has a DC offset that switches
  on with the sweep. With a 20 ms fade-in, that switch-on transient has content up to about
  50 Hz, which deconvolution maps to around −0.4 s, between the 2nd and 3rd harmonic
  responses (3rd harmonic off by 2.4 dB at low frequencies). The fade-in now lasts a
  quarter octave of the sweep, so the transient is slower and only touches frequencies below
  the passband.
- **Where harmonic levels are valid.** The n-th harmonic sweep starts at `n·f1`, so its
  response has its own band edge there. Harmonic levels are reported for fundamentals from
  `4·f1` (two octaves above the sweep start) to `f2/(2n)`; for lower frequencies, start the
  sweep lower.

## Why the harmonic level needs no correction factor

An exponential sweep spends time `L/f` per hertz at frequency `f`, so by the
stationary-phase approximation its spectrum magnitude is proportional to `√(L/f)`. The n-th
harmonic of a tone sweeping through `f` is a sweep through `n·f` that moves n times faster,
so its spectrum at `n·f` is proportional to `√(L/(n·f))`: the same as the input sweep's at
`n·f`. Dividing by `X(n·f)` therefore leaves exactly the harmonic's amplitude relative to the
input, and `|H_n(n·f)| / |H_1(f)|` is the harmonic level relative to the fundamental. The
tests check this against the closed form for a polynomial device.

## Validation (tests/test_measurement.py)

The simulated device is a polynomial nonlinearity (`u = x + a2·x² + a3·x³`), followed by
peaking EQ biquads (RBJ cookbook), a delay and white noise. Its linear response and, for a
flat filter, its harmonic levels are known exactly. A 3 s sweep at 48 kHz is used.

| Target | Result |
|---|---|
| Magnitude within 0.1 dB in the passband [2·f1, f2/2] (three EQ bands, gain 0.7, noise −80 dBFS, delays up to 0.1 s) | max error 0.003 dB |
| Flat system | within 0.01 dB |
| Delay to the sample (pure delay, and with EQ relative to the same EQ) | within 0.01 samples |
| 2nd and 3rd harmonic levels within 1 dB (four mixes of a2 and a3, down to −64 dB) | within 1 dB over their valid range |
| Linear device shows no harmonics | below −80 dB |

M2 validates the same pipeline on real hardware against REW.
