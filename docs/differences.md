# Differences from the commercial reference tool

Where Ephys-Forever's results differ from exports of the commercial tool in
`tests/reference/clampfit/`, the difference and its cause are recorded here.

## 8-pole Bessel low-pass filter

Case: `vclamp_iv_steps_001_bessel` (8-pole Bessel at 1 kHz, 20 kHz sampling, IN 0C).

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
  the sweep's largest value (0.07 pA). The cause of the factor is unknown; a second
  export at another cutoff would show whether it is constant.
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
  67847 pA²). Parameters are therefore compared through SSE, not value by value, until
  the reference tool's fit method (search method, minimisation and weighting) is known.
- **Negative time constants.** Single-exponential sweep 2 in the reference has
  tau = -730.6 ms. Ephys-Forever keeps tau > 0, so that sweep is not compared.
- **Term order.** The reference does not order 2-term results; Ephys-Forever sorts
  terms by increasing tau.
- **Standard deviation.** The reference's "Standard Deviation" is `sqrt(SSE / (n - 1))`.
  Ephys-Forever's `residual_sd` is `sqrt(SSE / (n - p))`, with `p` fitted parameters.
- **Not compared.** The reference's "Correlation Coefficient" and "Fitted Area" do not
  match the usual definitions (Pearson r of data and fit; area under the fitted
  curve), and are not compared.
