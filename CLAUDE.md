# CLAUDE.md

Guidance for Claude Code when working in this repository. Read this fully at the start of every session.

> Placeholder names: `<pkg>` is the Python package name and `<App>` is the product name. Replace both once chosen. Do not use any name that resembles "Clampfit", "pCLAMP", "Clampex", or "Axon".

## 1. What this project is

A standalone, open-source, cross-platform application for analyzing patch-clamp and other electrophysiology recordings. The goal is to match and then exceed the analysis capabilities of commercial tools such as Clampfit, with a modern, fast, reproducible workflow.

Primary users: electrophysiologists who need to open recordings, inspect sweeps, measure, fit, detect events, and batch-process experiments without a Windows-only commercial license.

## 2. Non-negotiable rules

1. **Clean-room development.** Never use, request, read, or reproduce code, decompiled output, binaries, or license-restricted internals of any commercial software. Implement only from public file-format documentation, published scientific methods, and the behavior described in tests. If a task seems to require proprietary internals, stop and ask.
2. **Core is GUI-free.** Nothing in `<pkg>/core/` may import Qt, pyqtgraph, or any GUI module. Every analysis must be callable from a script or notebook.
3. **Tests define correctness.** A feature is done only when its tests pass, including comparison against reference outputs in `tests/reference/`. Never edit a reference file or loosen a tolerance to make a test pass. If a reference looks wrong, stop and report it.
4. **No silent unit conversions.** All internal data uses SI units (seconds, amperes, volts, siemens, ohms, farads) as float64. Display units (pA, mV, ms, MΩ, pF) are a presentation concern handled only in the GUI and export layers.
5. **Small, reviewable changes.** One feature or fix per change. Do not refactor unrelated code in the same change.
6. **Ask before adding dependencies.** Prefer the approved stack below. Any new runtime dependency needs explicit approval.

## 3. Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.12+ |
| Numerics | NumPy, SciPy |
| Fitting | SciPy `least_squares` / `curve_fit`; lmfit if approved |
| Tables | pandas (results), pyarrow (Parquet export) |
| Project files | HDF5 via h5py |
| GUI | PySide6 + pyqtgraph |
| Tests | pytest, pytest-benchmark, hypothesis |
| Lint / format | ruff (lint + format), mypy (strict on `core/`) |
| Packaging | pyproject.toml (hatchling); PyInstaller for desktop builds |

File readers (ABF, ATF, and later formats) are written in this project. Do not add third-party ephys I/O libraries as dependencies.

## 4. Repository layout

```
<pkg>/
  core/
    model.py          # Recording, Sweep, Channel, Epoch dataclasses
    units.py          # SI handling and display conversion helpers
    io/
      abf1.py         # ABF 1.x reader
      abf2.py         # ABF 2.x reader
      atf.py          # ATF text reader/writer
      registry.py     # format detection and dispatch
    analysis/
      baseline.py
      measure.py      # cursor-region statistics (peak, mean, area, rise time, ...)
      filters.py      # Bessel, Gaussian, notch, boxcar
      fit/            # exponentials, Boltzmann, Hill, custom expressions
      events.py       # threshold and template-matching detection
      iv.py
      spikes.py
      single_channel/ # idealization, dwell times (later phase)
    project.py        # HDF5 project save/load
    log.py            # analysis operation log for replay
  batch/
    cli.py            # headless batch entry point
  gui/
    app.py
    views/
    widgets/
tests/
  (see tests/README.md)
docs/
scripts/
  make_synthetic.py   # deterministic synthetic test data generator
```

## 5. Data model conventions

- `Recording` holds metadata, a list of `Sweep`s, and channel definitions.
- `Sweep.data` is a 2D float64 array shaped `(n_channels, n_samples)` in SI units.
- Time is derived from `sample_rate` and `t0`, not stored per sample.
- Every channel carries `name`, `unit` (SI symbol), and `kind` (`"current"`, `"voltage"`, `"other"`).
- Raw file values and scaling factors are kept in metadata so readers can be audited.
- Analysis functions are pure: they take arrays/model objects plus explicit parameters and return results. No global state. No hidden defaults that are not documented in the docstring.

## 6. Reproducibility

Every analysis run from the GUI must be recorded as an entry in the operation log (`core/log.py`) containing the function name, all parameters, the input file hash, and the package version. A log must replay headlessly through `batch/cli.py` and produce identical results.

## 7. Coding standards

- Type hints everywhere. `mypy --strict` must pass for `core/`.
- NumPy-style docstrings. Every analysis function documents its algorithm, units, edge cases, and a citation when the method comes from the literature.
- Vectorize with NumPy. Avoid Python loops over samples.
- Raise specific exceptions with actionable messages (for example, "cursor 2 precedes cursor 1").
- No print statements in `core/`; use `logging`.

## 8. Workflow for each task

1. Restate the task and list the files you expect to touch.
2. Find or write the tests first. If reference data is missing for this feature, stop and ask for it rather than inventing expected values.
3. Implement the minimum code to pass.
4. Run: `ruff check . && ruff format --check . && mypy <pkg>/core && pytest`.
5. Summarize what changed, any assumptions made, and anything that needs the maintainer's scientific review.

## 9. Commands

```
pip install -e ".[dev]"          # install with dev dependencies
pytest                            # full test suite
pytest tests/unit -q              # fast unit tests only
pytest -m reference               # comparisons against reference outputs
pytest --benchmark-only           # performance tests
python scripts/make_synthetic.py  # regenerate synthetic data (deterministic)
python -m <pkg>.gui.app           # launch GUI
python -m <pkg>.batch.cli --help  # headless batch
```

## 10. Roadmap (build in this order)

1. Data model, ABF2 reader, ABF1 reader, ATF reader. Validated against real files.
2. Minimal viewer: open file, browse sweeps and channels, zoom, pan, overlay. Must open and scroll faster than the commercial tool on the benchmark files.
3. Cursors, baseline subtraction, region measurements, results table, CSV/Excel/Parquet export.
4. Filters and curve fitting (exponentials 1 to 3 terms, Boltzmann, Hill, custom expressions).
5. I-V analysis and spike/action potential analysis.
6. Event detection (threshold and template matching) with review and accept/reject UI.
7. Batch mode and operation-log replay.
8. Project files (HDF5) and NWB-compatible export.
9. Single-channel idealization and dwell-time analysis.
10. Additional readers (HEKA, AxoGraph, WinWCP) and installers for Windows, macOS, and Linux.

Do not start a phase until the previous phase's tests pass.

## 11. Things that need human scientific review

Flag these explicitly in your summary instead of deciding silently:

- Sign conventions for currents and holding potentials
- Baseline definitions and where the baseline window sits
- Rise/decay time definitions (10-90%, 20-80%, time constants)
- Any filter whose cutoff definition could differ between conventions (for example, -3 dB vs. Gaussian sigma)
- Any result that differs from a reference output, even within tolerance, in a systematic direction
