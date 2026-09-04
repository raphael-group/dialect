"""Tests for the focused immutable submission release."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import pytest

from analysis import build_tcga_revision_focused_release as release
from analysis import calibrate_tcga_revision_focused as calibration
from analysis import focused_revision_provenance as provenance
from analysis import postprocess_tcga_revision_focused as postprocess

if TYPE_CHECKING:
    from pathlib import Path


def _record(member: release.Member, path: str) -> dict[str, int | str]:
    return {"path": path, "bytes": member.size, "sha256": member.sha256}


def _closure_members(  # noqa: C901, PLR0912, PLR0915
    *,
    broken_postprocess_source: bool = False,
    broken_provider_artifact: bool = False,
    extra_payload: bool = False,
    sample_level_report: bool = False,
) -> list[release.Member]:
    members: dict[str, release.Member] = {}

    def add(name: str, content: bytes) -> release.Member:
        member = release._bytes_member(name, content)  # noqa: SLF001
        members[name] = member
        return member

    def add_json(name: str, value: object) -> release.Member:
        return add(name, release._canonical_json(value) + b"\n")  # noqa: SLF001

    analysis_config_name = "provenance/config/tcga_revision_config.json"
    analysis_config = add_json(analysis_config_name, {"analysis": "test"})
    calibration_config_name = "provenance/config/tcga_revision_calibration_config.json"
    calibration_config = add_json(
        calibration_config_name,
        {"calibration": "test"},
    )
    input_name = "provenance/input/input_manifest.json"
    input_member = add_json(
        input_name,
        {
            "contract": "input",
            "config": _record(
                analysis_config,
                "analysis/tcga_revision_config.json",
            ),
            "config_sha256": analysis_config.sha256,
            "cohorts": list(release.TCGA_COHORTS),
            "cohort_count": len(release.TCGA_COHORTS),
        },
    )
    provider_records = []
    for cohort in release.TCGA_COHORTS:
        files = {}
        for filename in release.RELEASED_PROVIDER_FILES:
            relative_path = f"cohorts/{cohort}/{filename}"
            provider_artifact = add(
                f"provenance/provider/{relative_path}",
                f"{cohort}\t{filename}\n".encode(),
            )
            files[filename] = _record(provider_artifact, relative_path)
            if (
                broken_provider_artifact
                and cohort == release.TCGA_COHORTS[0]
                and filename == release.RELEASED_PROVIDER_FILES[0]
            ):
                files[filename]["sha256"] = "0" * 64
        provider_records.append(
            {"cohort": cohort, "files": files, "mutsig_files": {}},
        )
    provider_name = "provenance/provider/provider_manifest.json"
    provider_member = add_json(
        provider_name,
        {
            "input_manifest": _record(input_member, "input_manifest.json"),
            "config_sha256": analysis_config.sha256,
            "cohorts": list(release.TCGA_COHORTS),
            "cohort_count": len(release.TCGA_COHORTS),
            "records": provider_records,
        },
    )
    run_name = "provenance/run/run_manifest.json"
    run_member = add_json(
        run_name,
        {
            "config": _record(
                analysis_config,
                "analysis/tcga_revision_config.json",
            ),
            "config_sha256": analysis_config.sha256,
            "provider_manifest": _record(
                provider_member,
                "provider_manifest.json",
            ),
            "cohorts": list(release.TCGA_COHORTS),
            "providers": ["cbase", "dig", "mutsig"],
            "top_k": 500,
        },
    )

    completion_tasks = []
    raw_tasks: dict[tuple[str, str], dict[str, Any]] = {}
    attested_tasks = []
    contract_evidence = []
    postprocess_manifest_records = []
    for cohort in release.TCGA_COHORTS:
        canonical_sha = hashlib.sha256(f"canonical-{cohort}".encode()).hexdigest()
        raw_sha = hashlib.sha256(f"raw-{cohort}".encode()).hexdigest()
        raw_bytes = len(cohort) + 100
        contract_evidence.append(
            {
                "path": f"contracts/{cohort}.json",
                "bytes": raw_bytes,
                "sha256": raw_sha,
                "canonical_sha256": canonical_sha,
            },
        )
        add_json(
            f"provenance/run/contracts/{cohort}.json",
            {
                "schema_version": provenance.SCHEMA_VERSION,
                "contract": provenance.PUBLIC_COHORT_CONTRACT,
                "cohort": cohort,
                "source_contract": {
                    "bytes": raw_bytes,
                    "sha256": raw_sha,
                    "canonical_sha256": canonical_sha,
                },
                "projection": {
                    "cohort": cohort,
                    "top_k": 500,
                    "focused_config_sha256": analysis_config.sha256,
                },
            },
        )
        sources = {}
        for provider in ("cbase", "dig", "mutsig"):
            pair_record = {
                "path": "pairwise_interaction_results.csv",
                "bytes": len(cohort) + len(provider) + 20,
                "sha256": hashlib.sha256(
                    f"pair-{cohort}-{provider}".encode(),
                ).hexdigest(),
            }
            single_record = {
                "path": "single_gene_results.csv",
                "bytes": len(cohort) + len(provider) + 10,
                "sha256": hashlib.sha256(
                    f"single-{cohort}-{provider}".encode(),
                ).hexdigest(),
            }
            task = {
                "cohort": cohort,
                "provider": provider,
                "top_k": 500,
                "config_sha256": analysis_config.sha256,
                "contract_sha256": canonical_sha,
                "outputs": {
                    "pairwise_interaction_results.csv": pair_record,
                    "single_gene_results.csv": single_record,
                },
            }
            task_name = f"provenance/run/tasks/{cohort}/{provider}/task_manifest.json"
            task_member = add_json(task_name, task)
            completion_tasks.append(
                {
                    "cohort": cohort,
                    "provider": provider,
                    "manifest": _record(
                        task_member,
                        f"tasks/{cohort}/{provider}/task_manifest.json",
                    ),
                },
            )
            attested_tasks.append(
                {
                    "cohort": cohort,
                    "provider": provider,
                    "manifest": _record(
                        task_member,
                        f"tasks/{cohort}/{provider}/task_manifest.json",
                    ),
                    "outputs": task["outputs"],
                },
            )
            raw_tasks[(cohort, provider)] = task
            sources[provider] = dict(pair_record)
        if broken_postprocess_source and cohort == "ACC":
            sources["mutsig"]["sha256"] = "0" * 64
        result_name = f"results/postprocess/{cohort}/{postprocess.RESULT_NAME}"
        result_member = add(result_name, b"gene_a,gene_b\n")
        cohort_manifest = add_json(
            f"results/postprocess/{cohort}/{postprocess.COHORT_MANIFEST_NAME}",
            {
                "cohort": cohort,
                "providers": ["cbase", "dig", "mutsig"],
                "sources": sources,
                "output": _record(result_member, f"{cohort}/{postprocess.RESULT_NAME}"),
            },
        )
        postprocess_manifest_records.append(
            _record(
                cohort_manifest,
                f"{cohort}/{postprocess.COHORT_MANIFEST_NAME}",
            ),
        )

    completion_name = "provenance/run/completion_manifest.json"
    completion_member = add_json(
        completion_name,
        {
            "config_sha256": analysis_config.sha256,
            "cohorts": list(release.TCGA_COHORTS),
            "run_manifest": _record(run_member, "run_manifest.json"),
            "tasks": completion_tasks,
            "task_count": len(completion_tasks),
        },
    )
    post_root_name = "results/postprocess/postprocess_manifest.json"
    post_root_member = add_json(
        post_root_name,
        {
            "run_completion": _record(
                completion_member,
                "completion_manifest.json",
            ),
            "cohorts": list(release.TCGA_COHORTS),
            "cohort_count": len(release.TCGA_COHORTS),
            "cohort_manifests": postprocess_manifest_records,
        },
    )

    calibration_run_name = "results/calibration/run_manifest.json"
    calibration_protocol = release._protocol_records()  # noqa: SLF001
    calibration_cells = tuple(
        (cohort, provider) for cohort, provider, _role in calibration_protocol
    )
    calibration_roles = {
        (cohort, provider): role for cohort, provider, role in calibration_protocol
    }
    calibration_run_member = add_json(
        calibration_run_name,
        {
            "config": _record(
                calibration_config,
                "analysis/tcga_revision_calibration_config.json",
            ),
            "run_completion": _record(completion_member, "completion_manifest.json"),
            "provider_manifest": _record(provider_member, "provider_manifest.json"),
            "cells": [
                {"cohort": cohort, "provider": provider, "role": role}
                for cohort, provider, role in calibration_protocol
            ],
        },
    )
    calibration_task_records = []
    for cohort, provider in calibration_cells:
        cell_root = f"results/calibration/tasks/{cohort}/{provider}"
        cell_data = add(
            f"{cell_root}/{calibration.TASK_DATA_NAME}",
            f"npz-{cohort}-{provider}".encode(),
        )
        cell_manifest_name = f"{cell_root}/{calibration.TASK_MANIFEST_NAME}"
        cell_manifest = add_json(
            cell_manifest_name,
            {
                "cohort": cohort,
                "provider": provider,
                "role": calibration_roles[(cohort, provider)],
                "config_sha256": calibration_config.sha256,
                "run_completion_sha256": completion_member.sha256,
                "output": _record(cell_data, calibration.TASK_DATA_NAME),
            },
        )
        calibration_task_records.append(
            _record(
                cell_manifest,
                f"tasks/{cohort}/{provider}/{calibration.TASK_MANIFEST_NAME}",
            ),
        )
    calibration_table_name = "results/calibration/calibration_cells.csv"
    calibration_table = add(calibration_table_name, b"cohort,provider\n")
    calibration_summary_name = "results/calibration/calibration_summary.json"
    calibration_summary = add_json(
        calibration_summary_name,
        {
            "config_sha256": calibration_config.sha256,
            "cell_count": len(calibration_cells),
            "run_manifest": _record(calibration_run_member, "run_manifest.json"),
            "task_manifests": calibration_task_records,
            "table": _record(calibration_table, calibration.SUMMARY_TABLE_NAME),
        },
    )

    rule_name = "results/reporting_rule.json"
    rule_member = add_json(
        rule_name,
        {
            "analysis_config_sha256": analysis_config.sha256,
            "calibration_config_sha256": calibration_config.sha256,
            "calibration_summary": _record(
                calibration_summary,
                calibration.SUMMARY_NAME,
            ),
            "postprocess_manifest": _record(
                post_root_member,
                postprocess.ROOT_MANIFEST_NAME,
            ),
            "primary_q_threshold": 0.01,
            "primary_adjustment": "BY",
        },
    )
    report_outputs = {}
    for name in release.REQUIRED_REPORT_OUTPUTS:
        if name == "figure6_burden_bins.csv":
            content = (
                b"cohort,provider,observed_log1p_bin_lower,observed_log1p_bin_upper,"
                b"expected_log1p_bin_lower,expected_log1p_bin_upper,tumor_count\n"
            )
        elif name == "table_s5.csv" and sample_level_report:
            content = b"cohort,cohort_row\n"
        elif name.endswith(".csv"):
            content = b"cohort\n"
        elif name.endswith(".pdf"):
            content = b"%PDF-test\n"
        else:
            content = b"report\n"
        report_output = add(f"results/report/{name}", content)
        report_outputs[name] = _record(report_output, name)
    report_manifest_member = add_json(
        "results/report/report_manifest.json",
        {
            "inputs": {
                "reporting_rule": _record(rule_member, "reporting_rule.json"),
            },
            "outputs": report_outputs,
        },
    )

    document_outputs = {}
    for name in release.REQUIRED_DOCUMENTS:
        document = add(f"documents/{name}", f"document {name}\n".encode())
        document_outputs[name] = _record(document, name)
    add_json(
        f"documents/{release.DOCUMENT_MANIFEST_NAME}",
        {
            "inputs": {
                "report_manifest": _record(
                    report_manifest_member,
                    "report_manifest.json",
                ),
            },
            "outputs": document_outputs,
        },
    )

    fit_commit = provenance.PRODUCTION_FIT_COMMIT
    release_commit = "b" * 40
    add_json(
        release.FIT_ATTESTATION_MEMBER,
        {
            "contract": provenance.FIT_ATTESTATION_CONTRACT,
            "source": {
                "fit_source_commit": fit_commit,
                "release_source_commit": release_commit,
                "fit_is_ancestor_of_release": True,
                "raw_fit_sources_unchanged_at_release": True,
            },
            "raw_chain": {
                "input_manifest": _record(input_member, "input_manifest.json"),
                "provider_manifest": _record(
                    provider_member,
                    "provider_manifest.json",
                ),
                "run_manifest": _record(run_member, "run_manifest.json"),
                "completion_manifest": _record(
                    completion_member,
                    "completion_manifest.json",
                ),
                "cohort_contracts": contract_evidence,
                "task_manifests": attested_tasks,
            },
            "privacy": {
                "raw_tumor_level_inputs_included": False,
                "sample_identifiers_included": False,
                "restricted_mutsig_source_included": False,
            },
        },
    )
    add_json(
        release.SOURCE_RECORD_NAME,
        {
            "contract": release.SOURCE_RECORD_CONTRACT,
            "fit_source_commit": fit_commit,
            "release_source_commit": release_commit,
            "fit_is_ancestor_of_release": True,
            "raw_tumor_level_inputs_included": False,
            "sample_identifiers_included": False,
            "restricted_mutsig_source_included": False,
        },
    )
    add(release.README_NAME, b"release\n")
    if extra_payload:
        add("provenance/provider/cohorts/ACC/unmanifested.csv", b"hidden\n")
    return list(members.values())


def _write_closure_archive(
    path: Path,
    *,
    broken_postprocess_source: bool = False,
    broken_provider_artifact: bool = False,
    extra_payload: bool = False,
    sample_level_report: bool = False,
) -> bytes:
    members = _closure_members(
        broken_postprocess_source=broken_postprocess_source,
        broken_provider_artifact=broken_provider_artifact,
        extra_payload=extra_payload,
        sample_level_report=sample_level_report,
    )
    manifest = release._manifest(  # noqa: SLF001
        members,
        fit_commit=provenance.PRODUCTION_FIT_COMMIT,
        release_commit="b" * 40,
    )
    release._write_archive(path, members, manifest)  # noqa: SLF001
    return manifest


def test_archive_is_deterministic_and_semantically_verified(tmp_path: Path) -> None:
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    manifest = _write_closure_archive(first)
    _write_closure_archive(second)

    assert first.read_bytes() == second.read_bytes()
    verified = release.verify_archive(first)
    assert verified["fit_source_commit"] == provenance.PRODUCTION_FIT_COMMIT
    assert verified["release_source_commit"] == "b" * 40

    receipt = {
        "schema_version": release.SCHEMA_VERSION,
        "contract": release.RECEIPT_CONTRACT,
        "archive": {
            "path": first.name,
            "bytes": first.stat().st_size,
            "sha256": release._sha256_path(first),  # noqa: SLF001
        },
        "release_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "fit_source_commit": provenance.PRODUCTION_FIT_COMMIT,
        "release_source_commit": "b" * 40,
        "member_count": len(_closure_members()),
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(release._canonical_json(receipt) + b"\n")  # noqa: SLF001
    assert release.verify_release(first, receipt_path)["release_source_commit"] == (
        "b" * 40
    )


def test_archive_verifier_rejects_semantically_broken_rehashed_archive(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "broken.tar.gz"
    _write_closure_archive(archive, broken_postprocess_source=True)

    with pytest.raises(ValueError, match="Postprocess source"):
        release.verify_archive(archive)


def test_archive_verifier_binds_released_pmfs_to_provider_manifest(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "broken-provider.tar.gz"
    _write_closure_archive(archive, broken_provider_artifact=True)

    with pytest.raises(ValueError, match="provider artifact"):
        release.verify_archive(archive)


def test_archive_verifier_rejects_unmanifested_extra_payload(tmp_path: Path) -> None:
    archive = tmp_path / "extra.tar.gz"
    _write_closure_archive(archive, extra_payload=True)

    with pytest.raises(ValueError, match="exact closed public inventory"):
        release.verify_archive(archive)


def test_archive_verifier_rejects_sample_level_csv_schema(tmp_path: Path) -> None:
    archive = tmp_path / "sample-level.tar.gz"
    _write_closure_archive(archive, sample_level_report=True)

    with pytest.raises(ValueError, match="sample-level row axis"):
        release.verify_archive(archive)


def test_document_plan_requires_exact_manifested_files(tmp_path: Path) -> None:
    for name in release.REQUIRED_DOCUMENTS:
        (tmp_path / name).write_text(f"final {name}\n", encoding="utf-8")
    outputs = {
        name: {
            "path": name,
            "bytes": (tmp_path / name).stat().st_size,
            "sha256": release._sha256_path(tmp_path / name),  # noqa: SLF001
        }
        for name in release.REQUIRED_DOCUMENTS
    }
    (tmp_path / release.DOCUMENT_MANIFEST_NAME).write_text(
        json.dumps({"outputs": outputs}),
        encoding="utf-8",
    )

    members = release._document_members(tmp_path)  # noqa: SLF001
    assert {member.name for member in members} == {
        *(f"documents/{name}" for name in release.REQUIRED_DOCUMENTS),
        f"documents/{release.DOCUMENT_MANIFEST_NAME}",
    }

    (tmp_path / "unexpected.txt").write_text("leak\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact required files"):
        release._document_members(tmp_path)  # noqa: SLF001


@pytest.mark.parametrize(
    "name",
    [
        "inputs/count_matrix.csv",
        "inputs/sample_axis.txt",
        "inputs/persample_genes.txt",
        "inputs/persample_lambda.f32",
        "inputs/persample_meta.txt",
        "inputs/persample_patients.txt",
        "inputs/persample_receipt.tsv",
        "inputs/CHOL.maf",
        "external/mutsig/source.m",
        "results/report/cohort_burden_source.csv",
        "results/report/figure6_burden_source.csv",
    ],
)
def test_release_plan_rejects_sample_level_and_restricted_members(name: str) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        release._bytes_member(name, b"sensitive\n")  # noqa: SLF001


def test_release_readme_uses_dynamic_rule_and_two_commits() -> None:
    content = release._readme(  # noqa: SLF001
        rule={
            "inference_status": "reportable",
            "primary_adjustment": "BY",
            "primary_q_threshold": 0.01,
        },
        fit_commit="a" * 40,
        release_commit="b" * 40,
    ).content

    assert content is not None
    text = content.decode("utf-8")
    assert "BY q <= 0.01" in text
    assert "Fit source commit" in text
    assert "Release source commit" in text
    assert "stochastic initialization" in text
    assert "exact cohort-level CBaSE and DIG PMFs" in text
    assert "hash-bound to the provider manifest" in text
    assert "fixed aggregate bins only" in text
    assert "honest post-run source" in text
    assert "not a task-level attestation" in text
    assert "q <= 0.10" not in text
