"""Deterministic synthetic test-data generator.

Writes one ATF file per case to ``tests/data/synthetic/`` and the analytic ground truth
for that case to ``tests/reference/synthetic/<case>.yaml``, plus
``tests/data/synthetic/MANIFEST.yaml``. Every case draws its noise from its own fixed
seed, so the output is byte-identical across runs and adding a case never changes the
others.

All values in the files and in the ground truth are SI (s, A, V, S). The ATF files are
written with ``%.9g`` sample values, so ``realized`` statistics in the ground truth are
computed from the values as written, not from the float64 arrays before formatting.

Usage::

    python scripts/make_synthetic.py            # regenerate everything
    python scripts/make_synthetic.py --check    # verify the committed files are current
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import signal

FloatArray = NDArray[np.float64]
YamlValue = None | bool | int | float | str | list["YamlValue"] | dict[str, "YamlValue"]

SEED = 20261024
GENERATOR_VERSION = 1
SAMPLE_FORMAT = "%.9g"
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path("tests/data/synthetic")
TRUTH_DIR = Path("tests/reference/synthetic")


@dataclass(frozen=True)
class Column:
    """One ATF data column."""

    title: str
    unit: str
    data: FloatArray


@dataclass(frozen=True)
class Channel:
    """Channel description for the manifest and ground truth."""

    name: str
    unit: str
    kind: str


@dataclass(frozen=True)
class Case:
    """A generated case: file contents plus analytic truth."""

    name: str
    mode: str
    description: str
    channel: Channel
    sample_rate_hz: float | None
    columns: list[Column]
    truth: dict[str, YamlValue]


def case_rng(case_id: int) -> np.random.Generator:
    """Return the generator for one case, independent of every other case."""
    return np.random.default_rng([SEED, case_id])


def time_axis(sample_rate_hz: float, n_samples: int) -> FloatArray:
    """Sample times ``i / fs`` starting at 0 s."""
    return np.arange(n_samples, dtype=np.float64) / sample_rate_hz


def as_written(data: FloatArray) -> FloatArray:
    """Round-trip values through the ATF number format."""
    text = "\n".join(SAMPLE_FORMAT % v for v in data)
    return np.loadtxt(io.StringIO(text), dtype=np.float64, ndmin=1)


def sweep_columns(t: FloatArray, sweeps: list[FloatArray], unit: str) -> list[Column]:
    """Time column followed by one column per sweep."""
    cols = [Column("Time", "s", t)]
    cols += [Column(f"Trace #{i + 1}", unit, s) for i, s in enumerate(sweeps)]
    return cols


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def step_known_amplitude() -> Case:
    """Square current step of known amplitude on a baseline, Gaussian noise per sweep."""
    rng = case_rng(1)
    fs, n, n_sweeps = 20_000.0, 10_000, 5
    baseline, amplitude, sigma = -20e-12, -100e-12, 2e-12
    i_on, i_off = 2_000, 6_000  # step covers samples [i_on, i_off)
    t = time_axis(fs, n)
    clean = np.full(n, baseline)
    clean[i_on:i_off] += amplitude
    sweeps = [clean + rng.normal(0.0, sigma, n) for _ in range(n_sweeps)]

    realized: list[YamlValue] = []
    for s in sweeps:
        w = as_written(s)
        base_mean = float(np.mean(w[:i_on]))
        realized.append(
            {
                "baseline_mean": base_mean,
                "step_mean": float(np.mean(w[i_on:i_off])),
                "step_min": float(np.min(w[i_on:i_off])),
                "step_area_vs_baseline_mean": float(np.sum(w[i_on:i_off] - base_mean) / fs),
            }
        )
    truth: dict[str, YamlValue] = {
        "signal": {
            "baseline_level": baseline,
            "step_amplitude": amplitude,
            "step_level": baseline + amplitude,
            "step_start_s": i_on / fs,
            "step_end_s": i_off / fs,
            "step_samples": [i_on, i_off],
            "step_area_noise_free": amplitude * (i_off - i_on) / fs,
        },
        "noise": {"kind": "gaussian", "sigma": sigma, "independent_per_sweep": True},
        "realized_definitions": {
            "baseline_mean": "mean of samples [0, step_start)",
            "step_mean": "mean of samples [step_start, step_end)",
            "step_min": "most negative raw sample in [step_start, step_end)",
            "step_area_vs_baseline_mean": (
                "sum over [step_start, step_end) of (x - baseline_mean) / fs, A*s"
            ),
        },
        "realized_per_sweep": realized,
    }
    return Case(
        "step_known_amplitude",
        "voltage_clamp",
        "Square step with known amplitude plus Gaussian noise",
        Channel("IN0", "A", "current"),
        fs,
        sweep_columns(t, sweeps, "A"),
        truth,
    )


def exp_decay_single() -> Case:
    """``A*exp(-t/tau) + C``; sweep 1 noise-free, sweep 2 with Gaussian noise."""
    rng = case_rng(2)
    fs, n = 20_000.0, 5_000
    a, tau, c, sigma = 100e-12, 20e-3, -10e-12, 0.5e-12
    t = time_axis(fs, n)
    clean = a * np.exp(-t / tau) + c
    sweeps = [clean, clean + rng.normal(0.0, sigma, n)]
    truth: dict[str, YamlValue] = {
        "model": "A * exp(-t / tau) + C, t from the first sample",
        "params": {"A": a, "tau": tau, "C": c},
        "sweeps": [{"sweep": 1, "noise_sigma": 0.0}, {"sweep": 2, "noise_sigma": sigma}],
    }
    return Case(
        "exp_decay_single",
        "voltage_clamp",
        "Single exponential decay with offset",
        Channel("IN0", "A", "current"),
        fs,
        sweep_columns(t, sweeps, "A"),
        truth,
    )


def exp_decay_double() -> Case:
    """Two-exponential decays with tau ratios 5 and 20, each noise-free and noisy."""
    rng = case_rng(3)
    fs, n = 20_000.0, 6_000
    a1, a2, c, sigma = 80e-12, 40e-12, -5e-12, 0.5e-12
    tau1 = 2e-3
    t = time_axis(fs, n)
    sweeps: list[FloatArray] = []
    meta: list[YamlValue] = []
    for ratio in (5.0, 20.0):
        tau2 = tau1 * ratio
        clean = a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + c
        for noise in (0.0, sigma):
            y = clean + rng.normal(0.0, sigma, n) if noise > 0 else clean
            sweeps.append(y)
            meta.append(
                {
                    "sweep": len(sweeps),
                    "params": {"A1": a1, "tau1": tau1, "A2": a2, "tau2": tau2, "C": c},
                    "tau_ratio": ratio,
                    "noise_sigma": noise,
                }
            )
    truth: dict[str, YamlValue] = {
        "model": "A1 * exp(-t / tau1) + A2 * exp(-t / tau2) + C, t from the first sample",
        "sweeps": meta,
    }
    return Case(
        "exp_decay_double",
        "voltage_clamp",
        "Two exponentials, tau ratio 5 and 20",
        Channel("IN0", "A", "current"),
        fs,
        sweep_columns(t, sweeps, "A"),
        truth,
    )


def boltzmann_activation() -> Case:
    """G-V table: ``Gmax / (1 + exp((V_half - V) / k))``, noise-free and noisy columns."""
    rng = case_rng(4)
    g_max, v_half, k = 5e-9, -25e-3, 6e-3
    sigma = 0.02 * g_max
    v = np.round(np.arange(-100e-3, 40e-3 + 1e-9, 5e-3), 12)
    clean = g_max / (1.0 + np.exp((v_half - v) / k))
    noisy = clean + rng.normal(0.0, sigma, v.size)
    truth: dict[str, YamlValue] = {
        "model": "Gmax / (1 + exp((V_half - V) / k)), k > 0 for activation",
        "params": {"Gmax": g_max, "V_half": v_half, "k": k},
        "columns": {
            "Voltage": "test potential, V",
            "Conductance": "noise-free",
            "Conductance noisy": f"Gaussian noise, sigma = {sigma!r} S",
        },
        "noise": {"kind": "gaussian", "sigma": sigma},
    }
    cols = [
        Column("Voltage", "V", v),
        Column("Conductance", "S", clean),
        Column("Conductance noisy", "S", noisy),
    ]
    return Case(
        "boltzmann_activation",
        "table",
        "G-V curve with known V_half and slope",
        Channel("G", "S", "other"),
        None,
        cols,
        truth,
    )


def epsc_train_known_times() -> Case:
    """Biexponential inward events at known onsets and amplitudes in Gaussian noise."""
    rng = case_rng(5)
    fs, n = 20_000.0, 60_000
    tau_r, tau_d, sigma = 0.5e-3, 5e-3, 2e-12
    t = time_axis(fs, n)
    t_peak = tau_r * tau_d / (tau_d - tau_r) * math.log(tau_d / tau_r)
    norm = 1.0 / (math.exp(-t_peak / tau_d) - math.exp(-t_peak / tau_r))

    onsets: list[float] = []
    t_next = 0.05
    while t_next < t[-1] - 0.05:
        onsets.append(t_next)
        t_next += 0.025 + float(rng.exponential(0.03))
    amps = rng.uniform(-80e-12, -15e-12, len(onsets))

    clean = np.zeros(n)
    span = int(10 * tau_d * fs)
    for t0, amp in zip(onsets, amps, strict=True):
        i0 = math.ceil(t0 * fs)
        s = t[i0 : i0 + span] - t0
        clean[i0 : i0 + span] += amp * norm * (np.exp(-s / tau_d) - np.exp(-s / tau_r))
    sweep = clean + rng.normal(0.0, sigma, n)

    truth: dict[str, YamlValue] = {
        "kernel": {
            "model": "amp * norm * (exp(-s / tau_decay) - exp(-s / tau_rise)), s = t - onset >= 0",
            "tau_rise": tau_r,
            "tau_decay": tau_d,
            "norm": norm,
            "time_to_peak": t_peak,
            "truncated_after_s": span / fs,
        },
        "amplitude_definition": "peak of the isolated noise-free event; overlapping tails add",
        "noise": {"kind": "gaussian", "sigma": sigma},
        "n_events": len(onsets),
        "onset_s": [float(x) for x in onsets],
        "peak_s": [float(x) + t_peak for x in onsets],
        "amplitude": [float(x) for x in amps],
    }
    return Case(
        "epsc_train_known_times",
        "gap_free",
        "Biexponential events at known times and amplitudes in noise",
        Channel("IN0", "A", "current"),
        fs,
        sweep_columns(t, [sweep], "A"),
        truth,
    )


def ap_train() -> Case:
    """Spikes built from raised-cosine rise and fall with known threshold, peak, width.

    Between spikes the membrane ramps linearly from the AHP level to threshold. Each spike
    rises from threshold to its peak over ``t_rise`` and falls from the peak to the AHP
    level over ``t_fall``, both as half-cosines, so the half-amplitude crossings have
    closed forms. Sweep 1 is noise-free, sweep 2 adds Gaussian noise.
    """
    rng = case_rng(6)
    fs, n = 50_000.0, 50_000
    v_thr, v_ahp, sigma = -50e-3, -65e-3, 0.2e-3
    t_rise, t_fall = 0.4e-3, 0.8e-3
    onsets = [0.1, 0.18, 0.27, 0.35, 0.45, 0.53, 0.62, 0.71, 0.82, 0.9]
    peaks = rng.uniform(20e-3, 35e-3, len(onsets))
    t = time_axis(fs, n)

    v = np.full(n, v_ahp)
    seg_start = 0.0
    for t0, v_pk in zip(onsets, peaks, strict=True):
        ramp = (t >= seg_start) & (t < t0)
        v[ramp] = v_ahp + (v_thr - v_ahp) * (t[ramp] - seg_start) / (t0 - seg_start)
        rise = (t >= t0) & (t < t0 + t_rise)
        v[rise] = v_thr + (v_pk - v_thr) * (1.0 - np.cos(np.pi * (t[rise] - t0) / t_rise)) / 2
        t1 = t0 + t_rise
        fall = (t >= t1) & (t < t1 + t_fall)
        v[fall] = v_ahp + (v_pk - v_ahp) * (1.0 + np.cos(np.pi * (t[fall] - t1) / t_fall)) / 2
        seg_start = t1 + t_fall
    sweeps = [v, v + rng.normal(0.0, sigma, n)]

    spikes: list[YamlValue] = []
    for t0, v_pk in zip(onsets, peaks, strict=True):
        v_half = (v_thr + float(v_pk)) / 2
        fall_cross = t_fall / math.pi * math.acos(2 * (v_half - v_ahp) / (v_pk - v_ahp) - 1)
        spikes.append(
            {
                "onset_s": t0,
                "threshold": v_thr,
                "peak": float(v_pk),
                "peak_time_s": t0 + t_rise,
                "amplitude_from_threshold": float(v_pk) - v_thr,
                "half_width_s": t_rise / 2 + fall_cross,
                "max_rise_slope": (float(v_pk) - v_thr) * math.pi / (2 * t_rise),
                "max_fall_slope": -(float(v_pk) - v_ahp) * math.pi / (2 * t_fall),
                "ahp_level": v_ahp,
            }
        )
    truth: dict[str, YamlValue] = {
        "construction": {
            "interspike": "linear ramp from ahp_level to threshold",
            "rise": "threshold + (peak - threshold) * (1 - cos(pi * s / t_rise)) / 2",
            "fall": "ahp + (peak - ahp) * (1 + cos(pi * s / t_fall)) / 2",
            "t_rise": t_rise,
            "t_fall": t_fall,
        },
        "definitions": {
            "threshold": "voltage where the ramp hands over to the spike (onset_s)",
            "half_width_s": "width at (threshold + peak) / 2",
        },
        "sweeps": [{"sweep": 1, "noise_sigma": 0.0}, {"sweep": 2, "noise_sigma": sigma}],
        "n_spikes": len(onsets),
        "spikes": spikes,
    }
    return Case(
        "ap_train",
        "current_clamp",
        "Spike waveforms with known threshold, peak, width",
        Channel("IN0", "V", "voltage"),
        fs,
        sweep_columns(t, sweeps, "V"),
        truth,
    )


def line_noise_60hz() -> Case:
    """3 Hz sinusoid plus 60, 120 and 180 Hz line components; whole cycles of each."""
    rng = case_rng(7)
    fs, n = 10_000.0, 20_000
    base_amp, base_freq, sigma = 20e-12, 3.0, 0.5e-12
    harmonics = [(60.0, 10e-12, 0.3), (120.0, 4e-12, 1.1), (180.0, 2e-12, 2.0)]
    t = time_axis(fs, n)
    base = base_amp * np.sin(2 * np.pi * base_freq * t)
    line = sum(
        (a * np.sin(2 * np.pi * f * t + p) for f, a, p in harmonics),
        start=np.zeros(n),
    )
    clean = base + line
    sweeps = [clean, clean + rng.normal(0.0, sigma, n)]
    truth: dict[str, YamlValue] = {
        "base_signal": {
            "model": "amplitude * sin(2 * pi * frequency_hz * t)",
            "amplitude": base_amp,
            "frequency_hz": base_freq,
        },
        "line_components": [
            {"frequency_hz": f, "amplitude": a, "phase_rad": p} for f, a, p in harmonics
        ],
        "line_model": "sum of amplitude * sin(2 * pi * frequency_hz * t + phase_rad)",
        "duration_note": "2 s holds whole cycles of every component, so the DFT has no leakage",
        "sweeps": [{"sweep": 1, "noise_sigma": 0.0}, {"sweep": 2, "noise_sigma": sigma}],
    }
    return Case(
        "line_noise_60hz",
        "gap_free",
        "Signal plus 60 Hz and harmonics",
        Channel("IN0", "A", "current"),
        fs,
        sweep_columns(t, sweeps, "A"),
        truth,
    )


def bessel_step_response() -> Case:
    """Ideal current step through analog 4-pole Bessel low-pass filters, no noise.

    The response is the exact continuous-time step response of the analog filter,
    sampled at the sample times, for cutoffs of 500, 1000 and 2000 Hz (one per sweep).
    The cutoff is the -3 dB frequency (``scipy.signal.bessel(..., norm="mag")``).
    """
    fs, n = 50_000.0, 1_000
    amplitude, i_on, order = 100e-12, 250, 4
    cutoffs = [500.0, 1000.0, 2000.0]
    t = time_axis(fs, n)
    t_on = i_on / fs
    sweeps: list[FloatArray] = []
    meta: list[YamlValue] = []
    for fc in cutoffs:
        b, a = signal.bessel(order, 2 * np.pi * fc, btype="low", analog=True, norm="mag")
        system = signal.lti(b, a)
        y = np.zeros(n)
        _, resp = signal.step(system, T=t[i_on:] - t_on)
        y[i_on:] = amplitude * resp
        sweeps.append(y)

        dense = np.linspace(0.0, 10.0 / fc, 200_001)
        _, unit = signal.step(system, T=dense)
        rising = slice(0, int(np.argmax(unit)) + 1)  # monotonic up to the overshoot peak
        t10 = float(np.interp(0.1, unit[rising], dense[rising]))
        t50 = float(np.interp(0.5, unit[rising], dense[rising]))
        t90 = float(np.interp(0.9, unit[rising], dense[rising]))
        meta.append(
            {
                "sweep": len(sweeps),
                "cutoff_hz": fc,
                "rise_time_10_90_s": t90 - t10,
                "delay_to_50_percent_s": t50,
                "overshoot_fraction": float(np.max(unit) - 1.0),
            }
        )
    truth: dict[str, YamlValue] = {
        "input": {"amplitude": amplitude, "step_time_s": t_on, "step_sample": i_on},
        "filter": {
            "family": "bessel",
            "order": order,
            "type": "analog low-pass",
            "cutoff_definition": "-3 dB magnitude (scipy.signal.bessel norm='mag')",
            "response": "exact analog step response sampled at sample times",
        },
        "derived_values_note": (
            "rise time, delay and overshoot are computed numerically from the analog "
            "step response on a 200001-point grid over 10 / cutoff_hz"
        ),
        "sweeps": meta,
    }
    return Case(
        "bessel_step_response",
        "voltage_clamp",
        "Step through a known filter",
        Channel("IN0", "A", "current"),
        fs,
        sweep_columns(t, sweeps, "A"),
        truth,
    )


CASES: list[Callable[[], Case]] = [
    step_known_amplitude,
    exp_decay_single,
    exp_decay_double,
    boltzmann_activation,
    epsc_train_known_times,
    ap_train,
    line_noise_60hz,
    bessel_step_response,
]


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

ATF_MODES = {
    "voltage_clamp": "Episodic Stimulation",
    "current_clamp": "Episodic Stimulation",
    "gap_free": "Gap Free",
    "table": "Table",
}


def atf_text(case: Case) -> str:
    """Render a case as ATF 1.0 text: header records, column titles, tab-separated data."""
    n_data = len(case.columns) - 1
    signals = "\t".join(f'"{case.channel.name}"' for _ in range(n_data))
    records = [
        f'"AcquisitionMode={ATF_MODES[case.mode]}"',
        f'"Comment=Ephys-Forever synthetic case {case.name}, generator v{GENERATOR_VERSION}"',
        f'"SignalsExported={case.channel.name}"',
        f'"Signals="\t{signals}',
    ]
    titles = "\t".join(f'"{c.title} ({c.unit})"' for c in case.columns)
    table = np.column_stack([c.data for c in case.columns])
    buf = io.StringIO()
    np.savetxt(buf, table, fmt=SAMPLE_FORMAT, delimiter="\t", newline="\n")
    header = ["ATF\t1.0", f"{len(records)}\t{len(case.columns)}", *records, titles]
    return "\n".join(header) + "\n" + buf.getvalue()


def yaml_scalar(value: YamlValue) -> str:
    """Format a scalar so YAML 1.1 and 1.2 loaders read back the same value and type."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return ".nan"
        if math.isinf(value):
            return ".inf" if value > 0 else "-.inf"
        text = repr(value)
        mantissa, _, exponent = text.partition("e")
        if "." not in mantissa:
            mantissa += ".0"
        return f"{mantissa}e{exponent}" if exponent else mantissa
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    raise TypeError(f"not a YAML scalar: {value!r}")


def yaml_lines(value: YamlValue, indent: int) -> list[str]:
    """Block-style YAML for mappings and lists of mappings, flow style for scalar lists."""
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, dict | list) and item and not is_scalar_list(item):
                lines.append(f"{pad}{key}:")
                lines += yaml_lines(item, indent + 1)
            else:
                lines.append(f"{pad}{key}: {yaml_inline(item)}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and item:
                sub = yaml_lines(item, indent + 1)
                lines.append(f"{pad}- {sub[0].lstrip()}")
                lines += sub[1:]
            else:
                lines.append(f"{pad}- {yaml_inline(item)}")
    else:
        lines.append(pad + yaml_scalar(value))
    return lines


def is_scalar_list(value: YamlValue) -> bool:
    return isinstance(value, list) and all(not isinstance(v, dict | list) for v in value)


def yaml_inline(value: YamlValue) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(yaml_inline(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {yaml_inline(v)}" for k, v in value.items()) + "}"
    return yaml_scalar(value)


def yaml_text(doc: YamlValue, comment: str) -> str:
    return f"# {comment}\n" + "\n".join(yaml_lines(doc, 0)) + "\n"


def channel_doc(ch: Channel) -> dict[str, YamlValue]:
    return {"name": ch.name, "unit": ch.unit, "kind": ch.kind}


def truth_doc(case: Case) -> dict[str, YamlValue]:
    n_sweeps = len(case.columns) - 1 if case.mode != "table" else 0
    doc: dict[str, YamlValue] = {
        "case": case.name,
        "description": case.description,
        "generator_version": GENERATOR_VERSION,
        "seed": SEED,
        "file": f"{case.name}.atf",
        "units": "SI",
        "mode": case.mode,
        "sample_rate_hz": case.sample_rate_hz,
        "n_samples": int(case.columns[0].data.size),
        "n_sweeps": n_sweeps,
        "channel": channel_doc(case.channel),
    }
    doc.update(case.truth)
    return doc


def manifest_doc(cases: list[Case]) -> list[YamlValue]:
    entries: list[YamlValue] = []
    for case in cases:
        entries.append(
            {
                "file": f"{case.name}.atf",
                "format": "atf",
                "mode": case.mode,
                "n_sweeps": len(case.columns) - 1 if case.mode != "table" else 0,
                "n_channels": 1,
                "channels": [channel_doc(case.channel)],
                "sample_rate_hz": case.sample_rate_hz,
                "acquired_with": f"scripts/make_synthetic.py v{GENERATOR_VERSION}",
                "notes": case.description,
                "shareable": True,
            }
        )
    return entries


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as fh:
        fh.write(text)


def generate(root: Path) -> list[Path]:
    """Write every case under ``root``; return the written paths."""
    header = "Generated by scripts/make_synthetic.py. Do not hand-edit."
    cases = [make() for make in CASES]
    written: list[Path] = []
    for case in cases:
        atf = root / DATA_DIR / f"{case.name}.atf"
        truth = root / TRUTH_DIR / f"{case.name}.yaml"
        write_text(atf, atf_text(case))
        write_text(truth, yaml_text(truth_doc(case), header))
        written += [atf, truth]
    manifest = root / DATA_DIR / "MANIFEST.yaml"
    write_text(manifest, yaml_text(manifest_doc(cases), header))
    written.append(manifest)
    return written


def check(root: Path) -> list[str]:
    """Regenerate into a temporary folder and list files that differ from ``root``."""
    with tempfile.TemporaryDirectory() as tmp:
        fresh = generate(Path(tmp))
        stale = []
        for path in fresh:
            rel = path.relative_to(tmp)
            current = root / rel
            if not current.exists() or current.read_bytes() != path.read_bytes():
                stale.append(rel.as_posix())
        return stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    parser.add_argument(
        "--check", action="store_true", help="fail if committed files are out of date"
    )
    args = parser.parse_args(argv)
    if args.check:
        stale = check(args.root)
        for rel in stale:
            print(f"out of date: {rel}", file=sys.stderr)
        return 1 if stale else 0
    for path in generate(args.root):
        print(path.relative_to(args.root).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
