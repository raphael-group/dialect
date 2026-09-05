"""Tests for the focused immutable submission release."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import tarfile
from io import BytesIO
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw, ImageFont

from analysis import build_tcga_revision_focused_release as release
from analysis import calibrate_tcga_revision_focused as calibration
from analysis import focused_revision_provenance as provenance
from analysis import postprocess_tcga_revision_focused as postprocess

if TYPE_CHECKING:
    from pathlib import Path

_REAL_INFERENCE_VALIDATOR = release._validate_archived_inference_frame  # noqa: SLF001
_REAL_PDF_PRIVACY_SCANNER = release._scan_pdf_privacy  # noqa: SLF001
_REAL_RAW_TASK_VALIDATOR = release._validate_archived_raw_task  # noqa: SLF001


@pytest.fixture(autouse=True)
def _compact_archive_inference_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep receipt-closure fixtures small; pair-axis logic is tested directly."""

    def validate(frame: pd.DataFrame, *, cohort: str, features: object) -> None:
        del features
        if frame.empty:
            assert tuple(frame.columns) == postprocess.result_columns()
            return
        postprocess.validate_inference_frame(frame, cohort=cohort)

    monkeypatch.setattr(release, "_validate_archived_inference_frame", validate)

    def validate_raw_task(task, **kwargs) -> None:
        kwargs["pair_count"] = task["pairwise_rows"]
        _REAL_RAW_TASK_VALIDATOR(task, **kwargs)

    monkeypatch.setattr(release, "_validate_archived_raw_task", validate_raw_task)
    monkeypatch.setattr(
        release.reporting,
        "_plot_figure6",
        lambda *, output, **_kwargs: output.write_bytes(b"%PDF-test\n"),
    )
    monkeypatch.setattr(release, "_scan_pdf_privacy", lambda *_args, **_kwargs: None)


def _record(member: release.Member, path: str) -> dict[str, int | str]:
    return {"path": path, "bytes": member.size, "sha256": member.sha256}


def _submission_document_content(name: str, *, table_s5: bytes) -> bytes:
    if name in {"Fig1.tif", "Fig2.tif"}:
        return b"II*\x00\x08\x00\x00\x00"
    if name == "S1_Table.csv":
        return table_s5
    if name.endswith(".pdf"):
        return b"%PDF-test\n"
    return f"document {name}\n".encode()


def _closure_members(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    broken_postprocess_source: bool = False,
    broken_provider_artifact: bool = False,
    extra_payload: bool = False,
    sample_level_report: bool = False,
    calibration_gate: object = True,
    include_calibration_gate: bool = True,
    rule_gate: object = True,
    include_rule_gate: bool = True,
    tampered_table_s5: bool = False,
    tampered_table_tex: bool = False,
    tampered_figure: bool = False,
    tampered_fit_iterations: bool = False,
    tampered_fit_continuous: bool = False,
    tampered_document_s1: bool = False,
    invalid_portal_tiff: bool = False,
) -> list[release.Member]:
    members: dict[str, release.Member] = {}

    def add(name: str, content: bytes) -> release.Member:
        member = release._bytes_member(name, content)  # noqa: SLF001
        members[name] = member
        return member

    def add_json(name: str, value: object) -> release.Member:
        return add(name, release._canonical_json(value) + b"\n")  # noqa: SLF001

    def detached_record(path: str, token: str) -> dict[str, int | str]:
        return {
            "path": path,
            "bytes": len(token) + 1,
            "sha256": hashlib.sha256(token.encode()).hexdigest(),
        }

    sample_counts = dict.fromkeys(release.TCGA_COHORTS, 1)
    sample_counts[release.TCGA_COHORTS[0]] = (
        release.reporting.EXPECTED_TUMOR_COUNT - len(release.TCGA_COHORTS) + 1
    )

    analysis_config_name = "provenance/config/tcga_revision_config.json"
    analysis_config = add(
        analysis_config_name,
        release.preparation.CONFIG_PATH.read_bytes(),
    )
    calibration_config_name = "provenance/config/tcga_revision_calibration_config.json"
    calibration_config = add(
        calibration_config_name,
        calibration.CONFIG_PATH.read_bytes(),
    )
    input_name = "provenance/input/input_manifest.json"
    input_cohort_records = [
        {
            "canonical_maf": detached_record(
                f"mafs/{cohort}.maf",
                f"canonical-maf-{cohort}",
            ),
            "cohort": cohort,
            "duplicate_resolution_policy": release.preparation.asdict(
                release.preparation.variant_data.TCGA_DUPLICATE_RESOLUTION_POLICY,
            ),
            "raw_maf": {
                "bytes": len(cohort) + 1,
                "sha256": hashlib.sha256(
                    f"raw-maf-{cohort}".encode(),
                ).hexdigest(),
            },
            "row_accounting": {
                "raw_rows": 3,
                "selected_rows": 2,
                "canonical_rows": 1,
                "removed_duplicate_rows": 1,
                "multiallelic_groups_preserved": 0,
                "unresolved_semantic_conflicts": 0,
            },
            "sample_axis_sha256": hashlib.sha256(
                f"sample-axis-{cohort}".encode(),
            ).hexdigest(),
            "sample_count": sample_counts[cohort],
        }
        for cohort in release.TCGA_COHORTS
    ]
    input_member = add_json(
        input_name,
        {
            "schema_version": release.preparation.SCHEMA_VERSION,
            "contract": release.preparation.INPUT_CONTRACT,
            "config": _record(
                analysis_config,
                "analysis/tcga_revision_config.json",
            ),
            "config_sha256": analysis_config.sha256,
            "datahub_commit": release.preparation.TCGA_DATAHUB_COMMIT,
            "population_manifest": detached_record(
                "population-source/population_manifest.json",
                "population-manifest",
            ),
            "cohorts": list(release.TCGA_COHORTS),
            "cohort_count": len(release.TCGA_COHORTS),
            "participant_count": release.reporting.EXPECTED_TUMOR_COUNT,
            "cohort_records": input_cohort_records,
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
        files["count_matrix.csv"] = detached_record(
            f"cohorts/{cohort}/count_matrix.csv",
            f"count-matrix-{cohort}",
        )
        files["sample_axis.txt"] = detached_record(
            f"cohorts/{cohort}/sample_axis.txt",
            f"provider-axis-{cohort}",
        )
        mutsig_files = {
            name: detached_record(
                f"mutsig/{cohort}/{name}",
                f"mutsig-{cohort}-{name}",
            )
            for name in release.preparation.REQUIRED_MUTSIG_FILES
        }
        mutsig_files["persample_patients.txt"] = {
            **files["sample_axis.txt"],
            "path": f"mutsig/{cohort}/persample_patients.txt",
        }
        provider_records.append(
            {"cohort": cohort, "files": files, "mutsig_files": mutsig_files},
        )
    provider_name = "provenance/provider/provider_manifest.json"
    provider_member = add_json(
        provider_name,
        {
            "schema_version": release.preparation.SCHEMA_VERSION,
            "contract": release.preparation.PROVIDER_CONTRACT,
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
            "schema_version": release.provenance.runner.SCHEMA_VERSION,
            "contract": release.provenance.runner.RUN_CONTRACT,
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
            "resources": release.preparation._load_config()["execution"],  # noqa: SLF001
        },
    )

    completion_tasks = []
    raw_tasks: dict[tuple[str, str], dict[str, Any]] = {}
    attested_tasks = []
    contract_evidence = []
    postprocess_manifest_records = []
    features = [f"G{index:03d}_M" for index in range(500)]
    pair_policy = {
        "epsilon_pretest_filter": release.core.TESTED_FAMILY_NO_PRETEST_FILTER,
        "marginal_effect_pretest_filter": (
            release.core.TESTED_FAMILY_NO_PRETEST_FILTER
        ),
        "pair_construction": release.core.TESTED_FAMILY_PAIR_CONSTRUCTION,
        "same_base_missense_nonsense": (
            release.core.TESTED_FAMILY_SAME_BASE_POLICY
        ),
        **release.core._pair_contract(features),  # noqa: SLF001
    }
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
                    "features": features,
                    "pair_policy": pair_policy,
                    "samples": {"count": sample_counts[cohort]},
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
                "schema_version": release.provenance.runner.SCHEMA_VERSION,
                "contract": release.provenance.runner.TASK_CONTRACT,
                "cohort": cohort,
                "provider": provider,
                "top_k": 500,
                "config_sha256": analysis_config.sha256,
                "contract_sha256": canonical_sha,
                "single_gene_rows": 500,
                "pairwise_rows": 0,
                "resource_usage": {
                    "elapsed_seconds": 1.0,
                    "user_cpu_seconds": 0.0,
                    "system_cpu_seconds": 0.0,
                    "peak_rss": {
                        "bytes": 1,
                        "native_value": 1,
                        "native_unit": "bytes",
                        "platform": "darwin",
                        "source": (
                            "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss"
                        ),
                    },
                },
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
        result_member = add(
            result_name,
            (",".join(postprocess.result_columns()) + "\n").encode(),
        )
        diagnostics = {
            provider: {
                "full_affine_rank_count": 0,
                "rank_deficient_count": 0,
                "rank_not_certified_underflow_count": 0,
                "p_display_clipped_count": 0,
                "by_display_clipped_count": 0,
                "bh_display_clipped_count": 0,
            }
            for provider in ("cbase", "dig", "mutsig")
        }
        cohort_manifest = add_json(
            f"results/postprocess/{cohort}/{postprocess.COHORT_MANIFEST_NAME}",
            {
                "schema_version": postprocess.SCHEMA_VERSION,
                "contract": postprocess.DERIVATION_CONTRACT,
                "cohort": cohort,
                "pair_count": 0,
                "providers": ["cbase", "dig", "mutsig"],
                "family": "all-matched-unordered-pairs-excluding-same-base-M:N",
                "multiplicity": {
                    "primary": (
                        "provider-specific-BY-over-complete-within-cohort-family"
                    ),
                    "nominal_sensitivity": (
                        "provider-specific-BH-over-complete-within-cohort-family"
                    ),
                },
                "direction": "rho-sign-after-nondirectional-profile-LRT",
                "non_full_rank": (
                    "retain-in-family-with-p-one-and-no-directional-effect"
                ),
                "probability_representation": postprocess.PROBABILITY_REPRESENTATION,
                "diagnostics": diagnostics,
                "reporting_threshold_selected": False,
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
            "schema_version": release.provenance.runner.SCHEMA_VERSION,
            "contract": release.provenance.runner.COMPLETION_CONTRACT,
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
            "schema_version": postprocess.SCHEMA_VERSION,
            "contract": postprocess.ROOT_CONTRACT,
            "effective_p_policy": (
                "chi-square-one-df-for-full-affine-rank-otherwise-p-one"
            ),
            "probability_representation": postprocess.PROBABILITY_REPRESENTATION,
            "multiplicity": {
                "primary": "benjamini-yekutieli",
                "nominal_sensitivity": "benjamini-hochberg",
            },
            "run_completion": _record(
                completion_member,
                "completion_manifest.json",
            ),
            "cohorts": list(release.TCGA_COHORTS),
            "cohort_count": len(release.TCGA_COHORTS),
            "provider_family_count": len(release.TCGA_COHORTS) * 3,
            "pair_count_per_provider": 0,
            "reporting_threshold_selected": False,
            "cohort_manifests": postprocess_manifest_records,
        },
    )

    calibration_run_name = "results/calibration/run_manifest.json"
    calibration_config_value = calibration._load_config()  # noqa: SLF001
    calibration_protocol = calibration._protocol_cells(  # noqa: SLF001
        calibration_config_value,
    )
    calibration_cells = tuple(
        (cell.cohort, cell.provider) for cell in calibration_protocol
    )
    calibration_run_member = add_json(
        calibration_run_name,
        {
            "schema_version": calibration.SCHEMA_VERSION,
            "contract": calibration.RUN_CONTRACT,
            "config": _record(
                calibration_config,
                "analysis/tcga_revision_calibration_config.json",
            ),
            "run_completion": _record(completion_member, "completion_manifest.json"),
            "provider_manifest": _record(provider_member, "provider_manifest.json"),
            "cells": [
                {
                    "cohort": cell.cohort,
                    "provider": cell.provider,
                    "role": cell.role,
                }
                for cell in calibration_protocol
            ],
            "resource_contract": dict(calibration_config_value["resources"]),
            "runtime_resource_observation": (
                calibration._frozen_resource_observation(  # noqa: SLF001
                    calibration_config_value,
                )
            ),
            "thread_environment": dict(calibration.THREAD_ENV),
            "result_blindness": calibration._result_blindness_receipt(),  # noqa: SLF001
        },
    )
    sentinel_pairs = calibration._sentinel_pairs(features, 32)  # noqa: SLF001
    lrt = np.zeros((10_000, 32), dtype=np.float64)
    reportable = np.ones((10_000, 32), dtype=bool)
    fallback = np.zeros((10_000, 32), dtype=bool)
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        marginal_lrt=lrt,
        marginal_reportable=reportable,
        scalar_fallback=fallback,
        sentinel_pairs=sentinel_pairs,
    )
    calibration_array_bytes = buffer.getvalue()
    calibration_task_records = []
    calibration_rows = []
    for cell in calibration_protocol:
        cohort = cell.cohort
        provider = cell.provider
        cell_root = f"results/calibration/tasks/{cohort}/{provider}"
        cell_data = add(
            f"{cell_root}/{calibration.TASK_DATA_NAME}",
            calibration_array_bytes,
        )
        self_peak = calibration._normalized_peak_rss(  # noqa: SLF001
            1,
            source="resource.getrusage(resource.RUSAGE_SELF).ru_maxrss",
            semantics="task-process-maximum-resident-set-size",
        )
        child_peak = calibration._normalized_peak_rss(  # noqa: SLF001
            int(provider == "mutsig"),
            source="resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss",
            semantics="maximum-over-terminated-children-not-additive",
        )
        resource_usage = {
            "elapsed_seconds": 1.0,
            "peak_rss": self_peak,
            "user_cpu_seconds": 0.5,
            "system_cpu_seconds": 0.25,
            "self": {
                "user_cpu_seconds": 0.5,
                "system_cpu_seconds": 0.25,
                "peak_rss": self_peak,
            },
            "terminated_children": {
                "user_cpu_seconds": 0.1,
                "system_cpu_seconds": 0.05,
                "peak_rss": child_peak,
            },
        }
        fit_kernel = {
            "contract": calibration_config_value["marginal_lrt"]["fit_kernel"],
            "replicate_chunk_rule": calibration_config_value["marginal_lrt"][
                "replicate_chunk_rule"
            ],
            "replicate_chunk_size": calibration._replicate_chunk_size(  # noqa: SLF001
                sample_counts[cohort],
            ),
            "scalar_fallback_count": 0,
        }
        worker_topology = calibration._worker_topology(  # noqa: SLF001
            calibration_config_value,
            provider=provider,
        )
        raw_task_name = f"provenance/run/tasks/{cohort}/{provider}/task_manifest.json"
        raw_task_member = members[raw_task_name]
        single = raw_tasks[(cohort, provider)]["outputs"][
            "single_gene_results.csv"
        ]
        cell_manifest_name = f"{cell_root}/{calibration.TASK_MANIFEST_NAME}"
        cell_manifest = add_json(
            cell_manifest_name,
            {
                "schema_version": calibration.SCHEMA_VERSION,
                "contract": calibration.TASK_CONTRACT,
                "cohort": cohort,
                "provider": provider,
                "role": cell.role,
                "config_sha256": calibration_config.sha256,
                "run_completion_sha256": completion_member.sha256,
                "seed": calibration._seed(  # noqa: SLF001
                    int(calibration_config_value["seed"]),
                    cohort,
                    provider,
                ),
                "marginal_replicates": 10_000,
                "sentinel_pair_count": 32,
                "alphas": [0.01, 0.05],
                "replicate_rng": calibration_config_value["marginal_lrt"][
                    "replicate_rng"
                ],
                "fit_kernel": fit_kernel,
                "worker_topology": worker_topology,
                "marginal_reportable_count": int(reportable.sum()),
                "source_task_manifest": _record(
                    raw_task_member,
                    f"tasks/{cohort}/{provider}/task_manifest.json",
                ),
                "single_gene_input": {
                    "path": f"tasks/{cohort}/{provider}/single_gene_results.csv",
                    "bytes": single["bytes"],
                    "sha256": single["sha256"],
                },
                "resource_usage": resource_usage,
                "output": _record(cell_data, calibration.TASK_DATA_NAME),
            },
        )
        calibration_task_records.append(
            {
                **_record(
                    cell_manifest,
                    f"tasks/{cohort}/{provider}/{calibration.TASK_MANIFEST_NAME}",
                ),
                "cohort": cohort,
                "provider": provider,
                "role": cell.role,
                "fit_kernel": fit_kernel,
                "worker_topology": worker_topology,
                "resource_usage": resource_usage,
            },
        )
        for pair_index in range(32):
            for alpha in (0.01, 0.05):
                row: dict[str, object] = {
                    "cohort": cohort,
                    "provider": provider,
                    "role": cell.role,
                    "screen": calibration.MARGINAL_SCREEN,
                    "sentinel_pair_index": pair_index,
                    "threshold": alpha,
                    "events": 0,
                    "trials": 10_000,
                    "rate": 0.0,
                    "reportable_trials": 10_000,
                    "nonreportable_trials": 0,
                    "gate_endpoint": cell.role == calibration.PRIMARY_ROLE,
                    "exact_binomial_familywise_error": "",
                    "exact_binomial_endpoint_count": "",
                    "bonferroni_endpoint_error": "",
                    "clopper_pearson_upper_bound": "",
                    "acceptance_upper_bound": "",
                    "endpoint_gate_pass": calibration.GATE_NOT_APPLICABLE,
                }
                if cell.role == calibration.PRIMARY_ROLE:
                    row.update(
                        calibration._gate_fields(  # noqa: SLF001
                            successes=0,
                            trials=10_000,
                            alpha=alpha,
                            config=calibration_config_value,
                        ),
                    )
                calibration_rows.append(row)
    calibration_table_name = "results/calibration/calibration_cells.csv"
    calibration_frame = pd.DataFrame(
        calibration_rows,
        columns=calibration.SUMMARY_COLUMNS,
    )
    calibration_table = add(
        calibration_table_name,
        calibration._summary_csv_bytes(calibration_frame),  # noqa: SLF001
    )
    calibration_summary_name = "results/calibration/calibration_summary.json"
    calibration_gate_rows = calibration._validated_gate_rows(  # noqa: SLF001
        calibration_frame,
        calibration_config_value,
    )
    calibration_gate_passed = int(
        calibration_gate_rows["endpoint_gate_pass"].eq(
            calibration.ENDPOINT_ACCEPTED,
        ).sum(),
    )
    affirmative_gate = calibration_config_value["affirmative_gate"]
    reporting_candidates = calibration_config_value["reporting_candidates"]
    endpoint_error = float(affirmative_gate["familywise_error"]) / int(
        affirmative_gate["endpoint_count"],
    )
    calibration_summary_payload = {
        "schema_version": calibration.SCHEMA_VERSION,
        "contract": calibration.SUMMARY_CONTRACT,
        "config_sha256": calibration_config.sha256,
        "cell_count": len(calibration_cells),
        "primary_gate_cell_count": 32,
        "descriptive_cell_count": 10,
        "marginal_endpoint_count": len(calibration_frame),
        "primary_gate_endpoint_count": len(calibration_gate_rows),
        "primary_gate_passed_endpoint_count": calibration_gate_passed,
        "overall_gate_pass": calibration_gate_passed == len(calibration_gate_rows),
        "gate_provider": affirmative_gate["provider"],
        "gate_endpoint_unit": affirmative_gate["endpoint_unit"],
        "gate_method": affirmative_gate["method"],
        "exact_binomial_familywise_error": affirmative_gate["familywise_error"],
        "exact_binomial_endpoint_count": affirmative_gate["endpoint_count"],
        "bonferroni_endpoint_error": endpoint_error,
        "clopper_pearson_confidence_level": 1.0 - endpoint_error,
        "acceptance_upper_bounds": affirmative_gate["acceptance_upper_bounds"],
        "effective_p_policy": (
            "chi-square-one-df-for-full-affine-rank-otherwise-p-one"
        ),
        "nonreportable_fit_count": 0,
        "primary_adjustment": reporting_candidates["primary_adjustment"],
        "primary_q_candidate": reporting_candidates["primary_q_threshold"],
        "sensitivity_adjustment": reporting_candidates["sensitivity_adjustment"],
        "sensitivity_q_candidate": reporting_candidates["sensitivity_q_threshold"],
        "interpretation": reporting_candidates["interpretation"],
        "reporting_rule_selected": False,
        "resource_contract": dict(calibration_config_value["resources"]),
        "runtime_resource_observation": (
            calibration._frozen_resource_observation(  # noqa: SLF001
                calibration_config_value,
            )
        ),
        "thread_environment": dict(calibration.THREAD_ENV),
        "resource_usage_interpretation": {
            "self_and_terminated_child_cpu_seconds_reported_separately": True,
            "terminated_child_peak_rss": (
                "maximum-over-terminated-children-not-additive"
            ),
        },
        "result_blindness": calibration._result_blindness_receipt(),  # noqa: SLF001
        "run_completion_sha256": completion_member.sha256,
        "provider_manifest_sha256": provider_member.sha256,
        "run_manifest": _record(calibration_run_member, "run_manifest.json"),
        "task_manifests": calibration_task_records,
        "table": _record(calibration_table, calibration.SUMMARY_TABLE_NAME),
    }
    calibration_summary_payload["overall_gate_pass"] = calibration_gate
    if include_calibration_gate:
        pass
    else:
        calibration_summary_payload.pop("overall_gate_pass")
    calibration_summary = add_json(
        calibration_summary_name,
        calibration_summary_payload,
    )

    rule_name = "results/reporting_rule.json"
    rule_payload = {
        "schema_version": release.rule_module.SCHEMA_VERSION,
        "contract": release.rule_module.RULE_CONTRACT,
        "analysis_config_sha256": analysis_config.sha256,
        "calibration_config_sha256": calibration_config.sha256,
        "calibration_summary_sha256": calibration_summary.sha256,
        "postprocess_manifest_sha256": post_root_member.sha256,
        "scope": (
            "one-identical-rule-across-all-32-tcga-pan-cancer-atlas-cohorts"
        ),
        "test": "chi-square-one-df-profile-lrt",
        "effective_p_policy": (
            "chi-square-one-df-for-full-affine-rank-otherwise-p-one"
        ),
        "multiplicity": "provider-specific-complete-within-cohort-family",
        "primary_adjustment": "benjamini-yekutieli",
        "sensitivity_adjustment": "benjamini-hochberg",
        "primary_provider": "mutsig",
        "continuity_provider": "cbase",
        "supplementary_providers": ["dig"],
        "primary_q_threshold": 0.01,
        "sensitivity_q_threshold": 0.01,
        "threshold_comparison": "inclusive-less-than-or-equal",
        "direction": "primary-provider-rho-sign-after-nondirectional-rejection",
        "direction_unavailable": (
            "retain-nondirectional-rejection-exclude-from-me-co-lists"
        ),
        "provider_overlap": "descriptive-only-not-an-inferential-vote",
        "me_presentation": "primary-MutSig-with-CBaSE-continuity-comparison",
        "co_presentation": "primary-MutSig-with-CBaSE-and-DIG-sensitivity",
        "thresholds_selected_from_observed_pairs": False,
        "calibration_gate": {
            "provider": calibration_summary_payload["gate_provider"],
            "endpoint_unit": calibration_summary_payload["gate_endpoint_unit"],
            "method": calibration_summary_payload["gate_method"],
            "endpoint_count": calibration_summary_payload[
                "exact_binomial_endpoint_count"
            ],
            "familywise_error": calibration_summary_payload[
                "exact_binomial_familywise_error"
            ],
            "acceptance_upper_bounds": calibration_summary_payload[
                "acceptance_upper_bounds"
            ],
            "overall_gate_pass": rule_gate,
        },
        "inference_status": (
            release.rule_module.REPORTABLE_STATUS
            if rule_gate is True
            else release.rule_module.WITHHELD_STATUS
        ),
        "withheld_reason": (
            None
            if rule_gate is True
            else "prespecified-finite-scenario-calibration-gate-failed"
        ),
        "claim_scope": "finite-scenario-calibrated-nominal-inference",
        "calibration_interpretation": (
            "finite-scenario-stress-not-formal-uniform-FDR-proof"
        ),
    }
    if not include_rule_gate:
        rule_payload.pop("calibration_gate")
    rule_member = add_json(rule_name, rule_payload)

    empty_inference = pd.DataFrame(columns=postprocess.result_columns())
    summary_rows = []
    overlap_rows = []
    top_frames = []
    for cohort in release.TCGA_COHORTS:
        row = release._expected_summary_decisions(  # noqa: SLF001
            cohort=cohort,
            frame=empty_inference,
            features=features,
            sample_count=sample_counts[cohort],
        )
        row.update(
            {
                "burden_median": 0.0,
                "burden_q25": 0.0,
                "burden_q75": 0.0,
                "burden_p90": 0.0,
                "burden_p95": 0.0,
                "burden_max": 0.0,
                "high_burden_fraction": 1.0,
            },
        )
        summary_rows.append(row)
        overlap_rows.extend(
            release.reporting._overlap_rows(  # noqa: SLF001
                empty_inference,
                cohort=cohort,
                primary_adjustment="benjamini-yekutieli",
                primary_q=0.01,
            ),
        )
        top_frames.append(
            release.reporting._top_primary_pairs(  # noqa: SLF001
                empty_inference,
                cohort=cohort,
                primary_adjustment="benjamini-yekutieli",
                primary_q=0.01,
            ),
        )
    summary_frame = pd.DataFrame(
        summary_rows,
        columns=release.reporting.summary_columns(),
    )
    overlap_frame = pd.DataFrame(
        overlap_rows,
        columns=release.reporting.OVERLAP_COLUMNS,
    )
    top_frame = pd.concat(top_frames, ignore_index=True)
    runtime_frame = pd.DataFrame(
        [
            {
                "cohort": cohort,
                "provider": provider,
                "pairwise_rows": 0,
                "elapsed_seconds": 1.0,
                "user_cpu_seconds": 0.0,
                "system_cpu_seconds": 0.0,
                "peak_rss_bytes": 1,
            }
            for cohort in release.TCGA_COHORTS
            for provider in release.core.BMRS
        ],
        columns=release.reporting.RUNTIME_COLUMNS,
    )
    fit_frame = pd.DataFrame(
        [
            {
                "scope": scope,
                "pairwise_rows": 0,
                "converged_rows": 0,
                "nonconverged_rows": 0,
                "iterations_min": 0,
                "iterations_median": 0.0,
                "iterations_p95": 0.0,
                "iterations_max": 0,
                "minimum_last_ll_gain": 0.0,
                "maximum_last_ll_gain": 0.0,
                "maximum_fixed_point_residual": 0.0,
                "maximum_kkt_residual": 0.0,
                "full_affine_rank_rows": 0,
                "rank_deficient_rows": 0,
                "rank_not_certified_underflow_rows": 0,
            }
            for scope in ("all", *release.core.BMRS)
        ],
        columns=release.reporting.FIT_DIAGNOSTIC_COLUMNS,
    )
    first_edge = release.reporting.BURDEN_LOG1P_MAX / (
        release.reporting.BURDEN_BIN_COUNT
    )
    focal_count = sample_counts[release.reporting.FOCAL_BURDEN_COHORT]
    burden_frame = pd.DataFrame(
        [
            {
                "cohort": release.reporting.FOCAL_BURDEN_COHORT,
                "provider": provider,
                "observed_log1p_bin_lower": 0.0,
                "observed_log1p_bin_upper": first_edge,
                "expected_log1p_bin_lower": 0.0,
                "expected_log1p_bin_upper": first_edge,
                "tumor_count": focal_count,
            }
            for provider in release.core.BMRS
        ],
        columns=release.reporting.BURDEN_BIN_COLUMNS,
    )
    burden_histogram_frame = pd.DataFrame(
        [
            {
                "cohort": cohort,
                "total_nonsynonymous_snv_events": 0,
                "tumor_count": sample_counts[cohort],
            }
            for cohort in release.TCGA_COHORTS
        ],
        columns=release.reporting.COHORT_BURDEN_HISTOGRAM_COLUMNS,
    )
    report_csv_frames = {
        "cohort_burden_histogram.csv": burden_histogram_frame,
        "figure6_burden_bins.csv": burden_frame,
        "table_s5.csv": summary_frame,
        "provider_overlap.csv": overlap_frame,
        "top_primary_pairs.csv": top_frame,
        "runtime_summary.csv": runtime_frame,
        "fit_diagnostics_summary.csv": fit_frame,
    }
    table_tex = release.reporting._table_s5_tex(  # noqa: SLF001
        summary_frame,
        primary_adjustment="benjamini-yekutieli",
        primary_q=0.01,
        sensitivity_adjustment="benjamini-hochberg",
        sensitivity_q=0.01,
    ).encode()
    report_outputs = {}
    table_s5_content = b""
    for name in release.REQUIRED_REPORT_OUTPUTS:
        if name == "table_s5.csv" and sample_level_report:
            content = b"cohort,barcode\n"
        elif name.endswith(".csv"):
            content = release.reporting._csv_bytes(report_csv_frames[name])  # noqa: SLF001
        elif name.endswith(".pdf"):
            content = b"%PDF-test\n"
        else:
            content = table_tex
        if name == "table_s5.csv" and tampered_table_s5:
            forged = summary_frame.copy()
            forged.loc[0, "mutsig_primary_rejection_total"] = 1
            content = release.reporting._csv_bytes(forged)  # noqa: SLF001
        elif name == "table_s5.tex" and tampered_table_tex:
            content += b"% forged\n"
        elif name == "figure6.pdf" and tampered_figure:
            content = b"%PDF-forged\n"
        elif name == "fit_diagnostics_summary.csv" and tampered_fit_iterations:
            forged = fit_frame.copy()
            forged.loc[forged["scope"].eq("mutsig"), [
                "iterations_min",
                "iterations_median",
                "iterations_p95",
                "iterations_max",
            ]] = 1
            content = release.reporting._csv_bytes(forged)  # noqa: SLF001
        elif name == "fit_diagnostics_summary.csv" and tampered_fit_continuous:
            forged = fit_frame.copy()
            forged.loc[forged["scope"].eq("mutsig"), [
                "minimum_last_ll_gain",
                "maximum_last_ll_gain",
                "maximum_fixed_point_residual",
                "maximum_kkt_residual",
            ]] = 1e-12
            content = release.reporting._csv_bytes(forged)  # noqa: SLF001
        if name == "table_s5.csv":
            table_s5_content = content
        report_output = add(f"results/report/{name}", content)
        report_outputs[name] = _record(report_output, name)
    report_manifest_member = add_json(
        "results/report/report_manifest.json",
        {
            "schema_version": release.reporting.SCHEMA_VERSION,
            "contract": release.reporting.REPORT_CONTRACT,
            "cohorts": list(release.TCGA_COHORTS),
            "primary_provider": "mutsig",
            "inference_status": release.rule_module.REPORTABLE_STATUS,
            "effective_p_policy": (
                "chi-square-one-df-for-full-affine-rank-otherwise-p-one"
            ),
            "primary_adjustment": "benjamini-yekutieli",
            "primary_q_threshold": 0.01,
            "sensitivity_adjustment": "benjamini-hochberg",
            "sensitivity_q_threshold": 0.01,
            "provider_overlap": (
                "direction-concordant-descriptive-only-not-an-inferential-vote"
            ),
            "threshold_decision_scale": "natural-log-q-values",
            "probability_representation": postprocess.PROBABILITY_REPRESENTATION,
            "sample_level_rows_included": False,
            "burden_source_policy": release.reporting.BURDEN_SOURCE_POLICY,
            "high_burden_definition": {
                "measure": "pre-K total nonsynonymous SNV event count per tumor",
                "reference": "pooled 10,433-tumor 32-cohort analysis population",
                "pooled_tumor_count": release.reporting.EXPECTED_TUMOR_COUNT,
                "quantile": release.reporting.HIGH_BURDEN_QUANTILE,
                "threshold": 0.0,
                "source": "cohort_burden_histogram.csv",
                "comparison": "greater-than-or-equal",
                "interpretation": (
                    "descriptive high-burden fraction, not a clinical hypermutator "
                    "label"
                ),
            },
            "inputs": {
                "run_completion": _record(
                    completion_member,
                    "completion_manifest.json",
                ),
                "provider_manifest": _record(
                    provider_member,
                    "provider_manifest.json",
                ),
                "postprocess_manifest": _record(
                    post_root_member,
                    postprocess.ROOT_MANIFEST_NAME,
                ),
                "calibration_summary": _record(
                    calibration_summary,
                    calibration.SUMMARY_NAME,
                ),
                "reporting_rule": _record(rule_member, "reporting_rule.json"),
            },
            "outputs": report_outputs,
        },
    )

    document_outputs = {}
    for name in release.REQUIRED_DOCUMENTS:
        content = _submission_document_content(name, table_s5=table_s5_content)
        if name == "S1_Table.csv" and tampered_document_s1:
            content += b"ACC\n"
        elif name == "Fig1.tif" and invalid_portal_tiff:
            content = b"not-a-tiff"
        document = add(f"documents/{name}", content)
        document_outputs[name] = _record(document, name)
    add_json(
        f"documents/{release.DOCUMENT_MANIFEST_NAME}",
        {
            "schema_version": release.SCHEMA_VERSION,
            "contract": release.DOCUMENT_CONTRACT,
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
    fit_source_records = [
        {
            "path": path.as_posix(),
            "bytes": 1,
            "sha256": hashlib.sha256(path.as_posix().encode()).hexdigest(),
        }
        for path in sorted(
            provenance.FIT_SOURCE_FILES,
            key=lambda item: item.as_posix(),
        )
    ]
    release_source_records = [
        {
            "path": path.as_posix(),
            "bytes": 1,
            "sha256": hashlib.sha256(path.as_posix().encode()).hexdigest(),
        }
        for path in sorted(
            provenance.RELEASE_PIPELINE_FILES,
            key=lambda item: item.as_posix(),
        )
    ]
    add_json(
        release.FIT_ATTESTATION_MEMBER,
        {
            "schema_version": provenance.SCHEMA_VERSION,
            "contract": provenance.FIT_ATTESTATION_CONTRACT,
            "scope": (
                "post-run-source-runtime-and-receipt-reconstruction; "
                "not-loaded-process-memory-attestation"
            ),
            "source": {
                "fit_source_commit": fit_commit,
                "fit_source_tree": "a" * 40,
                "release_source_commit": release_commit,
                "release_source_tree": "c" * 40,
                "fit_is_ancestor_of_release": True,
                "fit_source_files": fit_source_records,
                "excluded_generated_fit_sources": [
                    {
                        "path": "src/dialect/_version.py",
                        "reason": (
                            "setuptools-scm generated module absent from both Git "
                            "commits; not independently attributable to the completed "
                            "task runtime"
                        ),
                    },
                ],
                "raw_fit_sources_unchanged_at_release": True,
                "release_pipeline_files": release_source_records,
                "repository": "raphael-group/dialect",
            },
            "runtime": {
                "scope": "post-run-runtime-readback-not-process-memory-attestation",
                "python": {
                    "basename": "python",
                    "bytes": 1,
                    "sha256": "d" * 64,
                    "version": "3.12.0",
                    "implementation": "CPython",
                    "cache_tag": "cpython-312",
                },
                "platform": {
                    "system": "Darwin",
                    "release": "test",
                    "machine": "arm64",
                    "byteorder": "little",
                },
                "packages": dict.fromkeys(provenance.RUNTIME_DISTRIBUTIONS, "test"),
                "thread_environment": dict(
                    sorted(provenance.runner.THREAD_ENV.items()),
                ),
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
            "schema_version": release.SCHEMA_VERSION,
            "contract": release.SOURCE_RECORD_CONTRACT,
            "repository": "raphael-group/dialect",
            "fit_source_commit": fit_commit,
            "release_source_commit": release_commit,
            "fit_is_ancestor_of_release": True,
            "raw_fit_sources_unchanged_at_release": True,
            "raw_tumor_level_inputs_included": False,
            "sample_identifiers_included": False,
            "restricted_mutsig_source_included": False,
        },
    )
    add(release.README_NAME, b"release\n")
    if extra_payload:
        add("provenance/provider/cohorts/ACC/unmanifested.csv", b"hidden\n")
    return list(members.values())


def _write_closure_archive(  # noqa: PLR0913
    path: Path,
    *,
    broken_postprocess_source: bool = False,
    broken_provider_artifact: bool = False,
    extra_payload: bool = False,
    sample_level_report: bool = False,
    calibration_gate: object = True,
    include_calibration_gate: bool = True,
    rule_gate: object = True,
    include_rule_gate: bool = True,
    tampered_table_s5: bool = False,
    tampered_table_tex: bool = False,
    tampered_figure: bool = False,
    tampered_fit_iterations: bool = False,
    tampered_fit_continuous: bool = False,
    tampered_document_s1: bool = False,
    invalid_portal_tiff: bool = False,
) -> bytes:
    members = _closure_members(
        broken_postprocess_source=broken_postprocess_source,
        broken_provider_artifact=broken_provider_artifact,
        extra_payload=extra_payload,
        sample_level_report=sample_level_report,
        calibration_gate=calibration_gate,
        include_calibration_gate=include_calibration_gate,
        rule_gate=rule_gate,
        include_rule_gate=include_rule_gate,
        tampered_table_s5=tampered_table_s5,
        tampered_table_tex=tampered_table_tex,
        tampered_figure=tampered_figure,
        tampered_fit_iterations=tampered_fit_iterations,
        tampered_fit_continuous=tampered_fit_continuous,
        tampered_document_s1=tampered_document_s1,
        invalid_portal_tiff=invalid_portal_tiff,
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

    canonical_receipt = release._canonical_json(receipt)  # noqa: SLF001
    receipt_path.write_bytes(canonical_receipt[:-1] + b',"member_count":0}\n')
    with pytest.raises(ValueError, match="duplicate key"):
        release.verify_release(first, receipt_path)

    receipt_path.write_bytes(canonical_receipt[:-1] + b',"unexpected":NaN}\n')
    with pytest.raises(ValueError, match="numeric constant"):
        release.verify_release(first, receipt_path)


def test_archive_routes_cover_letter_through_pdf_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scanned: list[str] = []

    def scan(_content: bytes, *, name: str) -> None:
        scanned.append(name)

    monkeypatch.setattr(release, "_scan_pdf_privacy", scan)
    archive = tmp_path / "cover-letter.tar.gz"
    _write_closure_archive(archive)

    release.verify_archive(archive)

    assert {
        "documents/S1_Table.pdf",
        "documents/cover_letter.pdf",
    }.issubset(scanned)


@pytest.mark.parametrize(
    ("fixture_flag", "message"),
    [
        ("tampered_document_s1", "S1 Table differs"),
        ("invalid_portal_tiff", "bounded classic TIFF"),
    ],
)
def test_archive_rejects_rehashed_invalid_portal_artifacts(
    tmp_path: Path,
    fixture_flag: str,
    message: str,
) -> None:
    archive = tmp_path / f"{fixture_flag}.tar.gz"
    _write_closure_archive(archive, **{fixture_flag: True})

    with pytest.raises(ValueError, match=message):
        release.verify_archive(archive)


@pytest.mark.parametrize(
    ("gate_kwargs", "error", "message"),
    [
        ({"calibration_gate": False}, RuntimeError, "gate failed"),
        (
            {"include_calibration_gate": False},
            TypeError,
            "exact boolean gate",
        ),
        ({"calibration_gate": "true"}, TypeError, "exact boolean gate"),
        ({"rule_gate": False}, RuntimeError, "withheld by the rule"),
        (
            {"include_rule_gate": False},
            TypeError,
            "calibration gate object",
        ),
        ({"rule_gate": "true"}, TypeError, "boolean calibration gate"),
    ],
)
def test_archive_gate_fails_before_association_member_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate_kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    archive_path = tmp_path / "withheld.tar.gz"
    _write_closure_archive(archive_path, **gate_kwargs)
    original_extractfile = release.tarfile.TarFile.extractfile
    opened: list[str] = []

    def guarded_extractfile(self, member):
        name = member.name if isinstance(member, release.tarfile.TarInfo) else member
        opened.append(str(name))
        if str(name).endswith(f"/{postprocess.RESULT_NAME}"):
            msg = f"association member was opened: {name}"
            raise AssertionError(msg)
        return original_extractfile(self, member)

    monkeypatch.setattr(
        release.tarfile.TarFile,
        "extractfile",
        guarded_extractfile,
    )

    with pytest.raises(error, match=message):
        release.verify_archive(archive_path)

    assert not any(
        name.endswith(f"/{postprocess.RESULT_NAME}") for name in opened
    )


@pytest.mark.parametrize(
    ("gate_value", "write_summary", "error"),
    [
        (False, True, RuntimeError),
        (None, False, FileNotFoundError),
        ("false", True, TypeError),
    ],
)
def test_local_release_paths_gate_before_raw_or_derived_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate_value: object,
    write_summary: bool,  # noqa: FBT001
    error: type[Exception],
) -> None:
    calibration_root = tmp_path / "calibration"
    calibration_root.mkdir()
    if write_summary:
        (calibration_root / calibration.SUMMARY_NAME).write_text(
            json.dumps({"overall_gate_pass": gate_value}),
            encoding="utf-8",
        )

    def fail_if_association_accessed(*_args, **_kwargs):
        msg = "raw or derived association table was accessed"
        raise AssertionError(msg)

    path_type = type(tmp_path)
    original_open = path_type.open

    def guarded_open(path, *args, **kwargs):
        if path.name in {
            postprocess.RESULT_NAME,
            "pairwise_interaction_results.csv",
        }:
            msg = f"association file was opened or hashed: {path}"
            raise AssertionError(msg)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(path_type, "open", guarded_open)
    monkeypatch.setattr(
        release.provenance,
        "validate_fit_attestation",
        fail_if_association_accessed,
    )
    monkeypatch.setattr(
        release.calibration,
        "validate_summary",
        fail_if_association_accessed,
    )
    monkeypatch.setattr(
        release.postprocess,
        "validate_derived_root",
        fail_if_association_accessed,
    )
    monkeypatch.setattr(release, "_file_member", fail_if_association_accessed)
    common = {
        "input_root": tmp_path / "input",
        "provider_root": tmp_path / "providers",
        "run_root": tmp_path / "run",
        "postprocess_root": tmp_path / "postprocess",
        "calibration_root": calibration_root,
        "report_root": tmp_path / "report",
        "rule_path": tmp_path / "rule.json",
        "fit_attestation_path": tmp_path / "fit-attestation.json",
    }

    with pytest.raises(error):
        release._validate_upstream(  # noqa: SLF001
            repository_root=tmp_path / "repo",
            fit_commit="a" * 40,
            release_commit="b" * 40,
            runtime_executable=tmp_path / "python",
            **common,
        )
    with pytest.raises(error):
        release._result_members(**common)  # noqa: SLF001


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


@pytest.mark.parametrize(
    ("fixture_flag", "message"),
    [
        ("tampered_table_s5", "Table S5 differs"),
        ("tampered_table_tex", "Table S5 LaTeX differs"),
        ("tampered_figure", "Figure 6 differs"),
        ("tampered_fit_iterations", "fit diagnostics differ"),
        ("tampered_fit_continuous", "fit diagnostics differ"),
    ],
)
def test_archive_verifier_recomputes_rehashed_report_outputs(
    tmp_path: Path,
    fixture_flag: str,
    message: str,
) -> None:
    archive = tmp_path / f"{fixture_flag}.tar.gz"
    _write_closure_archive(archive, **{fixture_flag: True})

    with pytest.raises(ValueError, match=message):
        release.verify_archive(archive)


def test_archived_inference_validator_recomputes_math_and_pair_axis() -> None:
    features = ["A_M", "B_M", "C_M"]
    pairs = list(release.core.iter_tested_pairs(features))
    frame = pd.DataFrame(pairs, columns=["gene_a", "gene_b"])
    statistics = np.asarray([0.0, 1.0, 4.0])
    rho = pd.Series([0.0, -0.5, 0.5], dtype=float)
    identifiability = pd.Series(["full-affine-rank"] * len(pairs), dtype="string")
    for provider in release.core.BMRS:
        derived = postprocess._provider_statistics(  # noqa: SLF001
            statistics,
            rho,
            identifiability,
        )
        for name, values in derived.items():
            frame[f"{provider}_{name}"] = values
        frame[f"{provider}_fit_converged"] = pd.Series(
            [True] * len(frame),
            dtype=bool,
        )
        frame[f"{provider}_fit_iterations"] = pd.Series(
            [1, 2, 3],
            dtype=np.int64,
        )
        frame[f"{provider}_fit_last_ll_gain"] = pd.Series(
            [1e-12, 2e-12, 3e-12],
            dtype=float,
        )
        frame[f"{provider}_fit_fixed_point_residual"] = pd.Series(
            [1e-12, 2e-12, 3e-12],
            dtype=float,
        )
        frame[f"{provider}_fit_kkt_residual"] = pd.Series(
            [3e-12, 2e-12, 1e-12],
            dtype=float,
        )
    frame = frame.loc[:, postprocess.result_columns()]

    _REAL_INFERENCE_VALIDATOR(frame, cohort="TEST", features=features)

    frame.loc[0, "mutsig_log_p_value"] = -1.0
    with pytest.raises(ValueError, match="complete-family policy"):
        _REAL_INFERENCE_VALIDATOR(frame, cohort="TEST", features=features)


def test_fit_diagnostic_summary_matches_exact_linear_quantiles() -> None:
    adversarial_counts = np.bincount([515, 918], minlength=1001)
    assert release._fit_iteration_quantile(  # noqa: SLF001
        adversarial_counts,
        0.95,
    ) == np.quantile([515, 918], 0.95)

    accumulator = release._new_fit_diagnostic_accumulator()  # noqa: SLF001
    frame = pd.DataFrame(
        {
            **{
                f"{provider}_effect_identifiability": [
                    "full-affine-rank",
                    "rank-deficient",
                    "rank-not-certified-underflow",
                ]
                for provider in release.core.BMRS
            },
            **{
                f"{provider}_fit_converged": [True, True, True]
                for provider in release.core.BMRS
            },
            **{
                f"{provider}_fit_iterations": [1, 2, 3]
                for provider in release.core.BMRS
            },
            **{
                f"{provider}_fit_last_ll_gain": [1e-12, 2e-12, 3e-12]
                for provider in release.core.BMRS
            },
            **{
                f"{provider}_fit_fixed_point_residual": [1e-13, 2e-13, 3e-13]
                for provider in release.core.BMRS
            },
            **{
                f"{provider}_fit_kkt_residual": [3e-13, 2e-13, 1e-13]
                for provider in release.core.BMRS
            },
        },
    )
    release._accumulate_fit_diagnostics(  # noqa: SLF001
        accumulator,
        frame,
        cohort="TEST",
    )

    summary = release._fit_diagnostic_summary(accumulator)  # noqa: SLF001
    mutsig = summary.loc[summary["scope"].eq("mutsig")].iloc[0]
    assert mutsig["iterations_median"] == np.quantile([1, 2, 3], 0.5)
    assert mutsig["iterations_p95"] == np.quantile([1, 2, 3], 0.95)
    assert mutsig["minimum_last_ll_gain"] == 1e-12
    assert mutsig["maximum_last_ll_gain"] == 3e-12
    assert mutsig["maximum_fixed_point_residual"] == 3e-13
    assert mutsig["maximum_kkt_residual"] == 3e-13
    assert mutsig["full_affine_rank_rows"] == 1
    assert mutsig["rank_deficient_rows"] == 1
    assert mutsig["rank_not_certified_underflow_rows"] == 1


def test_raw_task_validator_requires_exact_schema_and_resource_receipt() -> None:
    digest = "a" * 64
    task = {
        "schema_version": release.provenance.runner.SCHEMA_VERSION,
        "contract": release.provenance.runner.TASK_CONTRACT,
        "cohort": "ACC",
        "provider": "mutsig",
        "top_k": 500,
        "contract_sha256": digest,
        "config_sha256": digest,
        "single_gene_rows": 500,
        "pairwise_rows": 3,
        "resource_usage": {
            "elapsed_seconds": 1.0,
            "user_cpu_seconds": 0.5,
            "system_cpu_seconds": 0.25,
            "peak_rss": {
                "bytes": 1024,
                "native_value": 1,
                "native_unit": "KiB",
                "platform": "linux",
                "source": "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss",
            },
        },
        "outputs": {
            name: {"path": name, "bytes": 1, "sha256": digest}
            for name in (
                "pairwise_interaction_results.csv",
                "single_gene_results.csv",
            )
        },
    }
    validation = {
        "cohort": "ACC",
        "provider": "mutsig",
        "analysis_config_sha256": digest,
        "contract_sha256": digest,
        "pair_count": 3,
    }
    _REAL_RAW_TASK_VALIDATOR(task, **validation)

    task["untrusted"] = True
    with pytest.raises(ValueError, match="raw task manifest contract"):
        _REAL_RAW_TASK_VALIDATOR(task, **validation)


def test_document_plan_requires_exact_manifested_files(tmp_path: Path) -> None:
    table_s5 = (
        ",".join(release.reporting.report_csv_columns()["table_s5.csv"]) + "\n"
    ).encode()
    for name in release.REQUIRED_DOCUMENTS:
        (tmp_path / name).write_bytes(
            _submission_document_content(name, table_s5=table_s5),
        )
    outputs = {
        name: {
            "path": name,
            "bytes": (tmp_path / name).stat().st_size,
            "sha256": release._sha256_path(tmp_path / name),  # noqa: SLF001
        }
        for name in release.REQUIRED_DOCUMENTS
    }
    (tmp_path / release.DOCUMENT_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "schema_version": release.SCHEMA_VERSION,
                "contract": release.DOCUMENT_CONTRACT,
                "inputs": {
                    "report_manifest": {
                        "path": "report_manifest.json",
                        "bytes": 1,
                        "sha256": "a" * 64,
                    },
                },
                "outputs": outputs,
            },
        ),
        encoding="utf-8",
    )

    members = release._document_members(tmp_path)  # noqa: SLF001
    assert {member.name for member in members} == {
        *(f"documents/{name}" for name in release.REQUIRED_DOCUMENTS),
        f"documents/{release.DOCUMENT_MANIFEST_NAME}",
    }

    manifest_path = tmp_path / release.DOCUMENT_MANIFEST_NAME
    valid_manifest = manifest_path.read_bytes()
    manifest_path.write_bytes(
        b'{"schema_version":"1.0.0","schema_version":"1.0.0"}\n',
    )
    with pytest.raises(ValueError, match="duplicate key"):
        release._document_members(tmp_path)  # noqa: SLF001
    manifest_path.write_bytes(valid_manifest)

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


def _minimal_text_pdf(text: str, *, annotation_text: str | None = None) -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode()
    annotations = b" /Annots [6 0 R]" if annotation_text is not None else b""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R"
            + annotations
            + b" >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode()
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    ]
    if annotation_text is not None:
        annotation = annotation_text.encode("utf-16-be").hex().upper().encode()
        objects.append(
            b"<< /Type /Annot /Subtype /Text /Rect [50 50 70 70] "
            b"/Contents <FEFF" + annotation + b"> >>",
        )
    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{index} 0 obj\n".encode() + body + b"\nendobj\n")
    xref_offset = len(content)
    content.extend(
        f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n",
    )
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode())
    content.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode(),
    )
    return bytes(content)


def _minimal_image_pdf(text: str) -> bytes:
    image = Image.new("RGB", (1600, 400), "white")
    ImageDraw.Draw(image).text(
        (100, 100),
        text,
        fill="black",
        font=ImageFont.load_default(size=100),
    )
    output = BytesIO()
    image.save(output, "PDF", resolution=200)
    content = output.getvalue()
    assert text.encode() not in content
    return content


def test_public_text_requires_utf8_and_rejects_encoded_sample_ids() -> None:
    for content in (
        "safe text".encode("utf-16-le"),
        "TCGA-AB-1234".encode("utf-16"),
    ):
        with pytest.raises(
            ValueError,
            match=r"canonical UTF-8|forbidden control",
        ):
            release._hash_stream(  # noqa: SLF001
                BytesIO(content),
                public_name="results/report/test.csv",
            )
    for value in (
        "TCGA&#45;AB&#45;1234",
        "TCGA&hyphen;AB&#x2d;1234",
        "https://example.test/?sample=TCGA%2DAB%2D1234",
        "TCGA%2525252DAB%2525252D1234",
        "TCGA%252525252DAB%252525252D1234",
        "TCGA{-}AB{-}1234",
        r"TCGA\text{-}AB\text{-}1234",
        "\uff34\uff23\uff27\uff21\uff0d\uff21\uff22\uff0d"
        "\uff11\uff12\uff13\uff14",
        "TCGA-\u200bAB-1234",
    ):
        with pytest.raises(ValueError, match=r"TCGA sample barcode|forbidden control"):
            release._hash_stream(  # noqa: SLF001
                BytesIO(value.encode()),
                public_name="documents/rebuttal.md",
            )


def test_calibration_npz_rejects_unparsed_trailing_bytes() -> None:
    output = BytesIO()
    np.savez_compressed(
        output,
        marginal_lrt=np.zeros((2, 1), dtype=np.float64),
        marginal_reportable=np.ones((2, 1), dtype=bool),
        scalar_fallback=np.zeros((2, 1), dtype=bool),
        sentinel_pairs=np.asarray([[0, 1]], dtype=np.int32),
    )
    content = output.getvalue()
    assert release._calibration_arrays(content, name="test.npz")  # noqa: SLF001

    with pytest.raises(ValueError, match="not canonical"):
        release._calibration_arrays(  # noqa: SLF001
            content + b"TCGA-AB-1234",
            name="test.npz",
        )


def test_canonical_archive_stream_rejects_gzip_tail_and_second_member(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "release.tar.gz"
    release._write_archive(  # noqa: SLF001
        archive_path,
        [release._bytes_member("safe.txt", b"safe\n")],  # noqa: SLF001
        b"{}\n",
    )
    with tarfile.open(archive_path, mode="r:gz") as archive:
        infos = archive.getmembers()
    original = archive_path.read_bytes()
    release._verify_canonical_archive_stream(archive_path, infos)  # noqa: SLF001

    for trailer in (b"TCGA-AB-1234", gzip.compress(b"TCGA-AB-1234", mtime=0)):
        archive_path.write_bytes(original + trailer)
        with pytest.raises(ValueError, match="trailing bytes or members"):
            release._verify_canonical_archive_stream(  # noqa: SLF001
                archive_path,
                infos,
            )


def test_canonical_archive_stream_rejects_pax_metadata() -> None:
    info = tarfile.TarInfo("safe.txt")
    info.mode = 0o444
    info.offset = 0
    info.offset_data = tarfile.BLOCKSIZE
    info.pax_headers = {"comment": "TCGA-AB-1234"}

    with pytest.raises(ValueError, match="canonical USTAR"):
        release._canonical_tar_segments([info])  # noqa: SLF001


def test_canonical_archive_stream_rejects_device_metadata_on_regular_file() -> None:
    info = tarfile.TarInfo("safe.txt")
    info.mode = 0o444
    info.offset = 0
    info.offset_data = tarfile.BLOCKSIZE
    info.devmajor = 7

    with pytest.raises(ValueError, match="canonical USTAR"):
        release._canonical_tar_segments([info])  # noqa: SLF001


@pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in release._PDF_TOOLS),  # noqa: SLF001
    reason="Poppler PDF privacy tools are unavailable",
)
def test_pdf_privacy_scanner_extracts_text_and_rejects_malformed_pdf() -> None:
    _REAL_PDF_PRIVACY_SCANNER(
        _minimal_text_pdf("aggregate results"),
        name="safe.pdf",
    )
    with pytest.raises(ValueError, match="TCGA sample barcode"):
        _REAL_PDF_PRIVACY_SCANNER(
            _minimal_text_pdf("TCGA-AB-1234"),
            name="leak.pdf",
        )
    with pytest.raises(
        ValueError,
        match=r"TCGA sample barcode|non-link annotation",
    ):
        _REAL_PDF_PRIVACY_SCANNER(
            _minimal_text_pdf(
                "aggregate results",
                annotation_text="TCGA-AB-1234",
            ),
            name="annotation-leak.pdf",
        )
    with pytest.raises(ValueError, match="TCGA sample barcode"):
        _REAL_PDF_PRIVACY_SCANNER(
            _minimal_image_pdf("TCGA-AB-1234"),
            name="image-leak.pdf",
        )
    with pytest.raises(ValueError, match="inspection failed"):
        _REAL_PDF_PRIVACY_SCANNER(b"%PDF-invalid\n", name="invalid.pdf")
