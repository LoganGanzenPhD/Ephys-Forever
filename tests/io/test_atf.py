"""ATF reader against the ATF export of the IV steps recording and the synthetic files."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from ephys_forever.core.io.abf2 import read_abf2
from ephys_forever.core.io.atf import read_atf
from ephys_forever.core.io.errors import ReaderError, UnsupportedFeatureError
from ephys_forever.core.model import Recording
from ephys_forever.core.units import parse_unit

TESTS = Path(__file__).resolve().parents[1]
REFERENCE = TESTS / "reference" / "clampfit" / "vclamp_iv_steps_001_atf"
SYNTHETIC = TESTS / "data" / "synthetic"
SYNTHETIC_TRUTH = TESTS / "reference" / "synthetic"


def text_table(path: Path) -> tuple[list[str], np.ndarray]:
    """Independent parse: header lines and the data block as floats, row by row."""
    lines = path.read_text(encoding="latin-1").splitlines()
    n_records = int(lines[1].split("\t")[0])
    header = lines[: 3 + n_records]
    rows = [[float(v) for v in line.split("\t")] for line in lines[3 + n_records :]]
    return header, np.array(rows)


@pytest.fixture(scope="module")
def expected() -> dict[str, Any]:
    with (REFERENCE / "expected.json").open(encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


@pytest.fixture(scope="module")
def path(real_data_file: Callable[[str], Path]) -> Path:
    return real_data_file("vclamp_iv_steps_001.atf")


@pytest.fixture(scope="module")
def rec(path: Path) -> Recording:
    return read_atf(path)


@pytest.mark.realdata
@pytest.mark.reference
class TestIVStepsExport:
    def test_identity(self, rec: Recording, expected: dict[str, Any]) -> None:
        md = rec.metadata
        assert md["sha256"] == expected["sha256"]
        assert (md["format"], md["format_version"]) == ("atf", expected["atf_version"])
        assert md["acquisition_mode"] == expected["acquisition_mode"]
        assert md["comment"] == expected["comment"]

    def test_shape_and_timing(self, rec: Recording, expected: dict[str, Any]) -> None:
        assert rec.metadata["total_samples"] == expected["total_samples"]
        assert rec.n_sweeps == expected["sweep_count"]
        assert rec.n_channels == expected["channel_count"]
        assert rec.sample_rate == expected["sampling_rate_hz"]
        for sweep in rec.sweeps:
            assert sweep.n_samples == expected["samples_per_sweep_per_channel"]
            assert sweep.duration * 1e3 == pytest.approx(expected["sweep_duration_ms"])
        assert rec.metadata["sweep_start_source"] == "SweepStartTimesMS"
        np.testing.assert_allclose(
            np.diff([s.t0 for s in rec.sweeps]), expected["sweep_start_interval_s"], rtol=1e-12
        )

    def test_channels(self, rec: Recording, expected: dict[str, Any]) -> None:
        for channel, exp in zip(rec.channels, expected["channels"], strict=True):
            assert channel.name == exp["name"]
            assert channel.metadata["file_unit"] == exp["units"]
            assert channel.unit == parse_unit(exp["units"]).si_unit

    def test_values_equal_file_text_exactly(self, rec: Recording, path: Path) -> None:
        _, table = text_table(path)
        for s, sweep in enumerate(rec.sweeps):
            for c, channel in enumerate(rec.channels):
                column = table[:, 1 + s * rec.n_channels + c]
                factor = channel.metadata["scaling"]["si_factor"]
                np.testing.assert_array_equal(sweep.data[c], column * factor)

    def test_matches_source_abf_to_export_precision(
        self, rec: Recording, expected: dict[str, Any], real_data_file: Callable[[str], Path]
    ) -> None:
        """Each ATF value is the ABF value rounded to the export's significant digits.

        The exporter computes values in single precision, so its pre-rounding value can
        differ from this reader's float64 value by about one float32 ulp. For example raw
        96 LSB on IN 0 is 58.5937472 here and was written as 58.5938, the tie-rounding of
        58.59375. The bound is half a unit in the last written digit plus one float32 ulp.
        """
        abf = read_abf2(real_data_file("vclamp_iv_steps_001.abf"))
        digits = expected["value_precision_significant_digits"]
        assert [c.name for c in abf.channels] == [c.name for c in rec.channels]
        for a_sweep, t_sweep in zip(abf.sweeps, rec.sweeps, strict=True):
            for c, channel in enumerate(rec.channels):
                factor = channel.metadata["scaling"]["si_factor"]
                ref = a_sweep.data[c] / factor
                got = t_sweep.data[c] / factor
                magnitude = np.maximum(np.abs(ref), np.abs(got))
                with np.errstate(divide="ignore"):
                    exponent = np.floor(np.log10(np.where(magnitude > 0, magnitude, 1.0)))
                float32_ulp = np.abs(ref) * float(np.finfo(np.float32).eps)
                bound = 0.5 * 10.0 ** (exponent - (digits - 1)) * (1 + 1e-9) + float32_ulp
                assert np.all(np.abs(got - ref) <= bound), channel.name
        assert [s.t0 for s in rec.sweeps] == pytest.approx([s.t0 for s in abf.sweeps])


class TestSyntheticFiles:
    @pytest.mark.parametrize(
        "case", ["step_known_amplitude", "exp_decay_double", "bessel_step_response"]
    )
    def test_reads_generator_output(self, case: str) -> None:
        path = SYNTHETIC / f"{case}.atf"
        with (SYNTHETIC_TRUTH / f"{case}.yaml").open(encoding="utf-8") as fh:
            truth = yaml.safe_load(fh)
        rec = read_atf(path)
        assert rec.sample_rate == truth["sample_rate_hz"]
        assert rec.n_sweeps == truth["n_sweeps"]
        assert [c.name for c in rec.channels] == [truth["channel"]["name"]]
        assert rec.channels[0].unit == truth["channel"]["unit"]
        _, table = text_table(path)
        for s, sweep in enumerate(rec.sweeps):
            np.testing.assert_array_equal(sweep.data[0], table[:, 1 + s])
            assert sweep.n_samples == truth["n_samples"]

    def test_table_files_are_rejected(self) -> None:
        with pytest.raises(UnsupportedFeatureError, match="not time"):
            read_atf(SYNTHETIC / "boltzmann_activation.atf")


HEADER = 'ATF\t1.0\n1\t3\n"AcquisitionMode=Gap Free"\n"Time (ms)"\t"IN 0 (pA)"\t"IN 1 (mV)"\n'


def test_gap_free_columns_and_ms_time(tmp_path: Path) -> None:
    path = tmp_path / "gf.atf"
    path.write_text(HEADER + "0\t1\t-60\n0.1\t2\t-61\n0.2\t3\t-62\n")
    rec = read_atf(path)
    assert rec.n_sweeps == 1
    assert rec.sample_rate == 10_000.0
    assert [(c.name, c.unit, c.kind) for c in rec.channels] == [
        ("IN 0", "A", "current"),
        ("IN 1", "V", "voltage"),
    ]
    np.testing.assert_array_equal(rec.sweeps[0].data[1], np.array([-60.0, -61.0, -62.0]) * 1e-3)


@pytest.mark.parametrize(
    ("content", "error", "match"),
    [
        ("ABF2\n", ReaderError, "too short"),
        ('XYZ\t1.0\n0\t2\n"Time (s)"\t"A (pA)"\n0\t1\n', ReaderError, "does not start"),
        (HEADER + "0\t1\n0.1\t2\n", ReaderError, "columns"),
        (HEADER + "0\t1\t2\n0.1\t1\t2\n0.5\t1\t2\n", ReaderError, "not uniformly"),
        (HEADER + "0\tx\t2\n0.1\t1\t2\n", ReaderError, "numeric"),
    ],
)
def test_malformed_files(tmp_path: Path, content: str, error: type[Exception], match: str) -> None:
    path = tmp_path / "bad.atf"
    path.write_text(content)
    with pytest.raises(error, match=match):
        read_atf(path)
