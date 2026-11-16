# References

Every algorithm in this project is implemented from the published sources below. Each module
cites the source it follows.

## Measurement
- A. Farina, "Simultaneous measurement of impulse response and distortion with a swept-sine
  technique", 108th AES Convention, 2000.
- A. Farina, "Advancements in impulse response measurements by sine sweeps", 122nd AES
  Convention, 2007.
- S. Müller and P. Massarani, "Transfer-function measurement with sweeps", J. Audio Eng. Soc.
  49(6), 2001.

## Inverse filter design
- O. Kirkeby, P. A. Nelson, H. Hamada and F. Orduna-Bustamante, "Fast deconvolution of
  multichannel systems using regularization", IEEE Trans. Speech Audio Process. 6(2), 1998.
- S. G. Norcross, G. A. Soulodre and M. C. Lavoie, "Subjective investigations of inverse
  filtering", J. Audio Eng. Soc. 52(10), 2004.
- A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed., 2010
  (minimum phase via the real cepstrum).

## Target curves and reference data
- S. E. Olive et al., "A statistical model that predicts listeners' preference ratings of
  around-ear and on-ear headphones", 144th AES Convention, 2018 (Harman target).
- J. Pasanen, AutoEQ: https://github.com/jaakkopasanen/AutoEq (public measurements and
  corrections used for validation).
- Room EQ Wizard (REW): https://www.roomeqwizard.com (reference measurements).

## Real-time convolution
- F. Wefers, *Partitioned convolution algorithms for real-time auralization*, Logos Verlag,
  2015.
- W. G. Gardner, "Efficient convolution without input-output delay", J. Audio Eng. Soc.
  43(3), 1995.
