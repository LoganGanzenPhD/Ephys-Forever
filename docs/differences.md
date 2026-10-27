# Differences from the commercial reference tool

Where Ephys-Forever's results differ from exports of the commercial tool in
`tests/reference/clampfit/`, the difference and its cause are recorded here.

## 8-pole Bessel low-pass filter

Cases: `vclamp_iv_steps_001_bessel` and `vclamp_iv_steps_001_bessel_1000hz` (8-pole
Bessel at 1 kHz; the same export, made twice), and `vclamp_iv_steps_001_bessel_2000hz`
(2 kHz). All at 20 kHz sampling, on IN 0C.

- **Design.** The reference output is reproduced to its 6 significant digits by an
  8-pole Bessel prototype normalised to -3 dB, mapped to digital form with the matched
  z-transform (each analog pole `s` becomes `exp(s / fs)`, no zeros, unit gain at 0 Hz).
  This is `bessel_lowpass(..., design="matched_z")`. The default,
  `design="bilinear"`, is a different digital filter: it puts -3 dB exactly at the
  stated cutoff, while the matched-z filter is -3 dB near 1045 Hz for a 1 kHz setting
  and has a shorter delay (about 0.32 ms against 0.5 ms). On this recording the two
  designs differ by up to 14 pA at the capacitive transients (rms about 1 pA).
- **Cutoff scale.** The match is exact only with the cutoff multiplied by 1.0016677
  (1001.67 Hz for 1000 Hz). At the nominal cutoff the largest difference is 0.45 % of
  the sweep's largest value (0.07 pA). The 2 kHz export needs the same factor: fitted
  on its own it gives 1.0016681, against 1.0016675 at 1 kHz, a spread within the
  exports' 6-digit rounding. The factor is therefore constant across cutoffs at this
  sampling rate, and its cause is still unknown.
- **Start of the trace.** The reference's first two samples equal the unfiltered input
  and the next ~20 samples carry a start-up transient. Ephys-Forever starts the filter
  in its steady state for the first sample, so a trace that begins at a constant level
  shows no transient. The first 50 samples are left out of the comparison.

## Exponential fits

Case: `vclamp_iv_steps_001_exp_fits` (1- and 2-term exponentials, 984 points per
sweep, on the filtered recording above).

- **Parameters.** For most sweeps the reference parameters are not the least-squares
  optimum: Ephys-Forever's fits reach a lower sum of squared errors (SSE) on every
  sweep they can fit, by up to 72 % (single exponential, sweep 7: SSE 18714 against
  67847 pA²). The cause is the fit method: the tool chose its Chebyshev search
  automatically, which is not a least-squares minimisation. These parameters are
  therefore compared through SSE only.
- **Least-squares reference.** Case `vclamp_iv_steps_001_exp_fits_simplex` repeats the
  fits with the tool's Simplex search (sum of squared errors, no weighting). Where the
  fit is well determined (single exponential: 8 of 9 sweeps; double: 4 of 9) the
  parameters agree to about 1e-5. On the rest the tool's Simplex stops early (single
  sweep 8), or the second term is not determined by the data and the two programs
  return different, equally meaningless pairs of terms (double sweeps 1, 2, 3, 5, 6).
- **Negative time constants.** Single-exponential sweep 2 is a rising exponential:
  tau = -730.6 ms (Chebyshev) and -9.48 ms (Simplex) in the references. Like the
  reference tool, Ephys-Forever allows negative tau, and matches the Simplex fit.
- **Term order.** The reference does not order 2-term results; Ephys-Forever sorts
  terms by increasing tau.
- **Standard deviation.** The reference's "Standard Deviation" is `sqrt(SSE / (n - 1))`.
  Ephys-Forever's `residual_sd` is `sqrt(SSE / (n - p))`, with `p` fitted parameters.
- **Not compared.** The reference's "Correlation Coefficient" and "Fitted Area" do not
  match the usual definitions (Pearson r of data and fit; area under the fitted
  curve), and are not compared.
