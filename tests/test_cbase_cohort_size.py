"""Tests for explicit complete-cohort size propagation through CBaSE."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from dialect import api
from dialect.bmr import _cbase_run
from dialect.bmr.cbase import CBaSEProvider
from dialect.cli.app import app

if TYPE_CHECKING:
    from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[1]
_COHORT_SIZE_MODULE = (
    _REPO_ROOT / "external" / "CBaSE" / "cbase_cohort_size.py"
)
_PARAMS_SCRIPT = _REPO_ROOT / "external" / "CBaSE" / "CBaSE_params_v1.2.py"
_QVALS_SCRIPT = _REPO_ROOT / "external" / "CBaSE" / "CBaSE_qvals_v1.2.py"


def _load_cohort_size_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_cbase_cohort_size_helper",
        _COHORT_SIZE_MODULE,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vendored_parser_preserves_legacy_inference_and_accepts_named_size() -> None:
    helper = _load_cohort_size_module()

    assert helper.parse_explicit_n_samples([]) is None
    assert helper.parse_explicit_n_samples(["--n-samples", "137"]) == 137


@pytest.mark.parametrize(
    "arguments",
    [
        ["137"],
        ["--unknown", "137"],
        ["--n-samples"],
        ["--n-samples", "0"],
        ["--n-samples", "-1"],
        ["--n-samples", "1.5"],
    ],
)
def test_vendored_parser_rejects_ambiguous_or_nonpositive_sizes(arguments) -> None:
    helper = _load_cohort_size_module()

    with pytest.raises(ValueError, match=r"positive integer|must be exactly"):
        helper.parse_explicit_n_samples(arguments)


def test_vendored_resolver_rejects_size_below_retained_membership() -> None:
    helper = _load_cohort_size_module()

    assert helper.resolve_n_samples(9, None) == 9
    assert helper.resolve_n_samples(9, 9) == 9
    assert helper.resolve_n_samples(9, 12) == 12
    with pytest.raises(ValueError, match="zero mutation-bearing"):
        helper.resolve_n_samples(0, None)
    with pytest.raises(ValueError, match=r"smaller than.*mutation-bearing"):
        helper.resolve_n_samples(9, 8)


def test_vendored_header_persists_size_in_qvals_compatible_field() -> None:
    helper = _load_cohort_size_module()

    header = helper.output_data_preparation_header(137)

    assert header.endswith("\tN_samples=137\n")
    assert helper.parse_output_data_preparation_n_samples(header) == 137
    with pytest.raises(ValueError, match="must be positive"):
        helper.output_data_preparation_header(0)


def test_vendored_denominator_seam_changes_only_the_per_sample_rate() -> None:
    helper = _load_cohort_size_module()

    explicit = helper.parse_explicit_n_samples(["--n-samples", "137"])
    resolved = helper.resolve_n_samples(9, explicit)
    header = helper.output_data_preparation_header(resolved)
    persisted = helper.parse_output_data_preparation_n_samples(header)

    assert persisted == 137
    assert helper.per_sample_rate(274.0, persisted) == 2.0
    assert helper.per_sample_rate(274.0, 274) == 1.0


@pytest.mark.parametrize(
    "header",
    [
        "gene\tmissing=4\n",
        "gene\tN_samples=0\n",
        "gene\tN_samples=not-an-integer\n",
    ],
)
def test_vendored_qvals_header_parser_fails_closed(header) -> None:
    helper = _load_cohort_size_module()

    with pytest.raises(ValueError, match="N_samples"):
        helper.parse_output_data_preparation_n_samples(header)


def test_vendored_params_entrypoint_parses_named_size_before_any_data_io(
    tmp_path,
) -> None:
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(_PARAMS_SCRIPT),
            "unused-input",
            "1",
            "hg19",
            "3",
            "0",
            "unused-output",
            "unused-reference",
            "unused-temp",
            "--n-samples",
            "0",
        ],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )

    assert result.returncode == 2
    assert "--n-samples must be a positive integer" in result.stderr
    assert "FileNotFoundError" not in result.stderr


def test_vendored_entrypoints_bind_cli_size_through_header_to_qvals() -> None:
    params_source = _PARAMS_SCRIPT.read_text(encoding="utf-8")
    qvals_source = _QVALS_SCRIPT.read_text(encoding="utf-8")

    parse_position = params_source.index(
        "EXPLICIT_N_SAMPLES = parse_explicit_n_samples(sys.argv[9:])",
    )
    resolve_position = params_source.index("N_samples = resolve_n_samples(")
    persist_position = params_source.index(
        "fout.write(output_data_preparation_header(N_samples))",
    )
    assert parse_position < resolve_position < persist_position
    assert (
        "N_samples = parse_output_data_preparation_n_samples(lines[0])"
        in qvals_source
    )
    assert "ratm_per_sample = per_sample_rate(ratm, N_samples)" in qvals_source
    assert "ratk_per_sample = per_sample_rate(ratk, N_samples)" in qvals_source


def test_sample_axis_file_preserves_exact_order_and_derives_denominator(
    tmp_path,
) -> None:
    sample_axis = tmp_path / "sample_axis.txt"
    sample_axis.write_text("s3\ns1\ns2\n", encoding="utf-8")

    resolved_axis, n_samples = _cbase_run._resolve_sample_axis(  # noqa: SLF001
        n_samples=None,
        sample_ids=sample_axis,
    )

    assert resolved_axis == ("s3", "s1", "s2")
    assert n_samples == 3


@pytest.mark.parametrize(
    ("sample_ids", "error_type", "match"),
    [
        ([], api.SampleAxisError, "at least one"),
        (["s1", ""], api.SampleAxisError, "nonempty"),
        (["s1", " s2"], api.SampleAxisError, "surrounding whitespace"),
        (["s1", "s1"], api.SampleAxisError, "duplicate"),
        (["s1", 2], api.SampleAxisError, "must be a string"),
        ({"s1", "s2"}, api.SampleAxisError, "ordered sequence"),
    ],
)
def test_sample_axis_rejects_invalid_membership(sample_ids, error_type, match) -> None:
    with pytest.raises(error_type, match=match):
        _cbase_run._resolve_sample_axis(  # noqa: SLF001
            n_samples=None,
            sample_ids=sample_ids,
        )


def test_explicit_denominator_requires_matching_exact_axis() -> None:
    with pytest.raises(ValueError, match="sample_ids is required"):
        _cbase_run._resolve_sample_axis(  # noqa: SLF001
            n_samples=3,
            sample_ids=None,
        )
    with pytest.raises(ValueError, match=r"n_samples \(2\).*axis length \(3\)"):
        _cbase_run._resolve_sample_axis(  # noqa: SLF001
            n_samples=2,
            sample_ids=["s1", "s2", "s3"],
        )


def _write_kept_mutations(tmp_path, rows) -> None:
    cbase_output = tmp_path / "CBaSE_output"
    cbase_output.mkdir()
    pd.DataFrame(rows).to_csv(
        cbase_output / "kept_mutations.csv",
        sep="\t",
        index=False,
    )


def test_count_generation_zero_completes_both_matrices_on_exact_axis(
    tmp_path,
) -> None:
    _write_kept_mutations(
        tmp_path,
        [
            {"sample": "s2", "gene": "G", "effect": "missense"},
            {"sample": "s1", "gene": "G", "effect": "nonsense"},
            {"sample": "s1", "gene": "H", "effect": "coding-synon"},
        ],
    )

    _cbase_run.generate_counts_from_cbase_output(
        str(tmp_path),
        sample_ids=["s3", "s1", "s2"],
    )

    effect_counts = pd.read_csv(tmp_path / "count_matrix.csv", index_col=0)
    gene_counts = pd.read_csv(
        tmp_path / "gene_level_count_matrix.csv",
        index_col=0,
    )
    assert list(effect_counts.index) == ["s3", "s1", "s2"]
    assert list(gene_counts.index) == ["s3", "s1", "s2"]
    assert (effect_counts.loc["s3"] == 0).all()
    assert (gene_counts.loc["s3"] == 0).all()
    assert effect_counts.loc["s1", "G_N"] == 1
    assert effect_counts.loc["s2", "G_M"] == 1
    assert gene_counts.loc["s1", "G"] == 1
    assert gene_counts.loc["s2", "G"] == 1


def test_count_generation_preserves_na_like_and_zero_padded_sample_ids(
    tmp_path,
) -> None:
    _write_kept_mutations(
        tmp_path,
        [
            {"sample": "NA", "gene": "G", "effect": "missense"},
            {"sample": "001", "gene": "H", "effect": "nonsense"},
        ],
    )

    _cbase_run.generate_counts_from_cbase_output(
        str(tmp_path),
        sample_ids=["NA", "001", "zero-event"],
    )

    effect_counts = pd.read_csv(
        tmp_path / "count_matrix.csv",
        index_col=0,
        keep_default_na=False,
        dtype={0: "string"},
    )
    assert list(effect_counts.index) == ["NA", "001", "zero-event"]
    assert effect_counts.loc["NA", "G_M"] == 1
    assert effect_counts.loc["001", "H_N"] == 1
    assert (effect_counts.loc["zero-event"] == 0).all()


def test_inferred_axis_keeps_synonymous_only_samples_as_zero_rows(tmp_path) -> None:
    _write_kept_mutations(
        tmp_path,
        [
            {"sample": "s2", "gene": "G", "effect": "missense"},
            {"sample": "s1", "gene": "H", "effect": "coding-synon"},
        ],
    )

    _cbase_run.generate_counts_from_cbase_output(str(tmp_path))

    effect_counts = pd.read_csv(tmp_path / "count_matrix.csv", index_col=0)
    assert list(effect_counts.index) == ["s1", "s2"]
    assert (effect_counts.loc["s1"] == 0).all()


def test_count_generation_rejects_any_retained_sample_outside_axis(
    tmp_path,
) -> None:
    _write_kept_mutations(
        tmp_path,
        [
            {"sample": "s1", "gene": "G", "effect": "missense"},
            {"sample": "outside", "gene": "H", "effect": "coding-synon"},
        ],
    )

    with pytest.raises(ValueError, match=r"outside the exact sample axis.*outside"):
        _cbase_run.generate_counts_from_cbase_output(
            str(tmp_path),
            sample_ids=["s1", "s2"],
        )
    assert not (tmp_path / "count_matrix.csv").exists()
    assert not (tmp_path / "gene_level_count_matrix.csv").exists()


def test_outside_axis_prevents_public_end_to_end_materialization(
    tmp_path,
    monkeypatch,
) -> None:
    maf = tmp_path / "cohort.maf"
    maf.write_text("fixture\n", encoding="utf-8")
    pmfs_published = False

    def fake_cbase_run(*_args, **_kwargs):
        _write_kept_mutations(
            tmp_path,
            [
                {"sample": "s1", "gene": "G", "effect": "missense"},
                {
                    "sample": "outside",
                    "gene": "H",
                    "effect": "coding-synon",
                },
            ],
        )

    def fake_bmr_materialization(*_args):
        nonlocal pmfs_published
        pmfs_published = True

    monkeypatch.setattr(_cbase_run, "generate_bmr_using_cbase", fake_cbase_run)
    monkeypatch.setattr(
        _cbase_run,
        "generate_bmr_files_from_cbase_output",
        fake_bmr_materialization,
    )

    with pytest.raises(api.SampleAxisError, match="outside the exact sample axis"):
        _cbase_run.generate_bmr_and_counts(
            str(maf),
            str(tmp_path),
            "hg19",
            "1e-100",
            n_samples=2,
            sample_ids=["s1", "s2"],
        )

    assert not pmfs_published
    assert not (tmp_path / "bmr_pmfs.csv").exists()
    assert not (tmp_path / "count_matrix.csv").exists()
    assert not (tmp_path / "gene_level_count_matrix.csv").exists()


@pytest.mark.parametrize("n_samples", [0, -1])
def test_python_runner_rejects_nonpositive_size_before_conversion(
    monkeypatch,
    n_samples,
) -> None:
    converted = False

    def fake_convert(*_args):
        nonlocal converted
        converted = True

    monkeypatch.setattr(_cbase_run, "convert_maf_to_cbase_input_file", fake_convert)

    with pytest.raises(ValueError, match="positive integer"):
        _cbase_run.generate_bmr_using_cbase(
            "cohort.maf",
            "out",
            "hg19",
            "1e-100",
            n_samples=n_samples,
        )
    assert not converted


@pytest.mark.parametrize("n_samples", [True, 4.2, "12"])
def test_python_runner_rejects_noninteger_size(tmp_path, n_samples) -> None:
    with pytest.raises(TypeError, match="positive integer"):
        _cbase_run.generate_bmr_using_cbase(
            "cohort.maf",
            str(tmp_path),
            "hg19",
            "1e-100",
            n_samples=n_samples,
        )


@pytest.mark.parametrize("n_samples", [None, 137, np.int64(137)])
def test_cbase_command_adds_named_size_only_when_explicit(
    tmp_path,
    monkeypatch,
    n_samples,
) -> None:
    cbase_input = tmp_path / "cbase_input.tsv"
    commands = []
    monkeypatch.setattr(
        _cbase_run,
        "convert_maf_to_cbase_input_file",
        lambda *_args: cbase_input,
    )
    monkeypatch.setattr(
        _cbase_run,
        "_run_cbase_step",
        lambda label, cmd: commands.append((label, cmd)),
    )

    _cbase_run.generate_bmr_using_cbase(
        "cohort.maf",
        str(tmp_path),
        "hg19",
        "1e-100",
        n_samples=n_samples,
    )

    params_command = commands[0][1]
    if n_samples is None:
        assert "--n-samples" not in params_command
    else:
        assert params_command[-2:] == ["--n-samples", "137"]
    assert "--n-samples" not in commands[1][1]


def test_cbase_pmf_only_limits_qvals_to_observed_genes(
    tmp_path,
    monkeypatch,
) -> None:
    cbase_input = tmp_path / "cbase_input.tsv"
    output = tmp_path / "CBaSE_output"
    commands = []
    monkeypatch.setattr(
        _cbase_run,
        "convert_maf_to_cbase_input_file",
        lambda *_args: cbase_input,
    )

    def fake_run(label, command):
        commands.append((label, command))
        if label == "CBaSE params":
            output.mkdir(exist_ok=True)
            pd.DataFrame({"gene": ["TP53", "KRAS", "TP53"]}).to_csv(
                output / "kept_mutations.csv",
                sep="\t",
                index=False,
            )
            (output / "q_values.txt").write_text("stale\n", encoding="utf-8")

    monkeypatch.setattr(_cbase_run, "_run_cbase_step", fake_run)

    _cbase_run.generate_bmr_using_cbase(
        "cohort.maf",
        str(tmp_path),
        "hg19",
        "1e-100",
        pmf_only=True,
    )

    assert not (output / "q_values.txt").exists()
    assert (output / "observed_genes.txt").read_text(encoding="utf-8") == (
        "KRAS\nTP53\n"
    )
    qvals_command = commands[1][1]
    assert qvals_command[-3:] == [
        "--pmf-only",
        "--genes-file",
        str(output / "observed_genes.txt"),
    ]


def test_end_to_end_runner_binds_axis_to_subprocess_and_count_generation(
    tmp_path,
    monkeypatch,
) -> None:
    maf = tmp_path / "cohort.maf"
    maf.write_text("fixture\n", encoding="utf-8")
    sample_axis = tmp_path / "sample_axis.txt"
    sample_axis.write_text("s3\ns1\ns2\n", encoding="utf-8")
    seen = {}

    def fake_run(*_args, n_samples):
        seen["n_samples"] = n_samples

    def fake_counts(*_args, sample_ids):
        seen["sample_ids"] = sample_ids

    monkeypatch.setattr(_cbase_run, "generate_bmr_using_cbase", fake_run)
    monkeypatch.setattr(
        _cbase_run,
        "generate_bmr_files_from_cbase_output",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        _cbase_run,
        "generate_counts_from_cbase_output",
        fake_counts,
    )

    _cbase_run.generate_bmr_and_counts(
        str(maf),
        str(tmp_path),
        "hg19",
        "1e-100",
        n_samples=3,
        sample_ids=sample_axis,
    )

    assert seen == {
        "n_samples": 3,
        "sample_ids": ("s3", "s1", "s2"),
    }


def test_axis_denominator_mismatch_fails_before_cbase_launch(monkeypatch) -> None:
    launched = False

    def fake_run(*_args, **_kwargs):
        nonlocal launched
        launched = True

    monkeypatch.setattr(_cbase_run, "generate_bmr_using_cbase", fake_run)

    with pytest.raises(ValueError, match=r"n_samples \(2\).*axis length \(3\)"):
        _cbase_run.generate_bmr_and_counts(
            "unused.maf",
            "unused-out",
            "hg19",
            "1e-100",
            n_samples=2,
            sample_ids=["s1", "s2", "s3"],
        )
    assert not launched


def test_cbase_provider_threads_explicit_size(monkeypatch) -> None:
    invocation = {}
    sentinel = object()

    def fake_generate(*args, n_samples, sample_ids):
        maf_path, out_dir, reference, threshold = args
        invocation.update(
            maf_path=maf_path,
            out_dir=out_dir,
            reference=reference,
            threshold=threshold,
            n_samples=n_samples,
            sample_ids=sample_ids,
        )

    monkeypatch.setattr("dialect.bmr.cbase.generate_bmr_and_counts", fake_generate)
    monkeypatch.setattr(CBaSEProvider, "load", lambda *_args: sentinel)

    sample_ids = ("s2", "s1")
    result = CBaSEProvider(
        threshold="1e-40",
        n_samples=2,
        sample_ids=sample_ids,
    ).estimate(
        "cohort.maf",
        "out",
        reference="hg38",
    )

    assert result is sentinel
    assert invocation == {
        "maf_path": "cohort.maf",
        "out_dir": "out",
        "reference": "hg38",
        "threshold": "1e-40",
        "n_samples": 2,
        "sample_ids": sample_ids,
    }


def test_cbase_provider_default_preserves_legacy_inference(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(
        "dialect.bmr.cbase.generate_bmr_and_counts",
        lambda *_args, **kwargs: seen.append(
            (kwargs["n_samples"], kwargs["sample_ids"]),
        ),
    )
    monkeypatch.setattr(CBaSEProvider, "load", lambda *_args: object())

    CBaSEProvider().estimate("cohort.maf", "out")

    assert seen == [(None, None)]


def test_cbase_provider_load_preserves_exact_string_sample_ids(tmp_path) -> None:
    pd.DataFrame(
        {"G_M": [1, 0]},
        index=pd.Index(["001", "NA"], name="sample"),
    ).to_csv(tmp_path / "count_matrix.csv")
    pd.DataFrame(
        [[1.0]],
        index=pd.Index(["G_M"], name="gene"),
    ).to_csv(tmp_path / "bmr_pmfs.csv")

    result = CBaSEProvider().load(str(tmp_path))

    assert list(result.counts.index) == ["001", "NA"]
    assert result.counts["G_M"].tolist() == [1, 0]


def test_public_api_can_construct_cbase_with_exact_axis(monkeypatch) -> None:
    sample_ids = ("s2", "s1")
    provider = CBaSEProvider(n_samples=2, sample_ids=sample_ids)
    sentinel = object()
    constructor_call = {}

    def fake_get_provider(name, **kwargs):
        constructor_call.update(name=name, kwargs=kwargs)
        return provider

    monkeypatch.setattr(api, "get_provider", fake_get_provider)
    monkeypatch.setattr(provider, "estimate", lambda *_args, **_kwargs: sentinel)

    result = api.estimate_bmr(
        "cohort.maf",
        "out",
        provider="cbase",
        n_samples=2,
        sample_ids=sample_ids,
    )

    assert result is sentinel
    assert constructor_call == {
        "name": "cbase",
        "kwargs": {"n_samples": 2, "sample_ids": sample_ids},
    }


def test_cbase_cli_forwards_exact_axis_and_denominator(tmp_path, monkeypatch) -> None:
    sample_axis = tmp_path / "sample_axis.txt"
    sample_axis.write_text("s2\ns1\n", encoding="utf-8")
    invocation = {}

    def fake_estimate_bmr(*args, **kwargs):
        invocation.update(args=args, kwargs=kwargs)

    monkeypatch.setattr("dialect.cli.app.api.estimate_bmr", fake_estimate_bmr)

    result = CliRunner().invoke(
        app,
        [
            "generate",
            "--maf",
            str(tmp_path / "cohort.maf"),
            "--out",
            str(tmp_path / "out"),
            "--bmr",
            "cbase",
            "--cbase-sample-axis",
            str(sample_axis),
            "--cbase-samples",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert invocation["kwargs"] == {
        "provider": "cbase",
        "reference": "hg19",
        "threshold": "1e-100",
        "n_samples": 2,
        "sample_ids": sample_axis,
    }


def test_cbase_cli_omission_preserves_inferred_axis(monkeypatch, tmp_path) -> None:
    invocation = {}
    monkeypatch.setattr(
        "dialect.cli.app.api.estimate_bmr",
        lambda *args, **kwargs: invocation.update(args=args, kwargs=kwargs),
    )

    result = CliRunner().invoke(
        app,
        [
            "generate",
            "--maf",
            str(tmp_path / "cohort.maf"),
            "--out",
            str(tmp_path / "out"),
            "--bmr",
            "cbase",
        ],
    )

    assert result.exit_code == 0, result.output
    assert invocation["kwargs"]["n_samples"] is None
    assert invocation["kwargs"]["sample_ids"] is None


@pytest.mark.parametrize("n_samples", ["0", "2"])
def test_cbase_cli_rejects_denominator_without_axis(
    n_samples,
    monkeypatch,
    tmp_path,
) -> None:
    called = False

    def fake_estimate_bmr(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("dialect.cli.app.api.estimate_bmr", fake_estimate_bmr)

    result = CliRunner().invoke(
        app,
        [
            "generate",
            "--maf",
            str(tmp_path / "cohort.maf"),
            "--out",
            str(tmp_path / "out"),
            "--bmr",
            "cbase",
            "--cbase-samples",
            n_samples,
        ],
    )

    assert result.exit_code == 2
    assert "--cbase-samples requires --cbase-sample-axis" in result.output
    assert not called


@pytest.mark.parametrize(
    ("axis_text", "n_samples", "message"),
    [
        ("s1\ns1\n", None, "duplicate identifier"),
        ("s1\ns2\ns3\n", "2", "does not match the exact sample axis length"),
    ],
)
def test_cbase_cli_fails_cleanly_on_invalid_axis_contract(
    axis_text,
    n_samples,
    message,
    tmp_path,
) -> None:
    sample_axis = tmp_path / "sample_axis.txt"
    sample_axis.write_text(axis_text, encoding="utf-8")
    arguments = [
        "generate",
        "--maf",
        str(tmp_path / "unused.maf"),
        "--out",
        str(tmp_path / "out"),
        "--bmr",
        "cbase",
        "--cbase-sample-axis",
        str(sample_axis),
    ]
    if n_samples is not None:
        arguments.extend(["--cbase-samples", n_samples])

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 2
    assert message in result.output
    assert not (tmp_path / "out" / "CBaSE_output").exists()


def test_cbase_cli_does_not_mask_unrelated_runtime_value_errors(
    monkeypatch,
    tmp_path,
) -> None:
    def fake_estimate_bmr(*_args, **_kwargs):
        msg = "invalid generated PMF"
        raise ValueError(msg)

    monkeypatch.setattr("dialect.cli.app.api.estimate_bmr", fake_estimate_bmr)

    result = CliRunner().invoke(
        app,
        [
            "generate",
            "--maf",
            str(tmp_path / "cohort.maf"),
            "--out",
            str(tmp_path / "out"),
            "--bmr",
            "cbase",
        ],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
    assert str(result.exception) == "invalid generated PMF"
