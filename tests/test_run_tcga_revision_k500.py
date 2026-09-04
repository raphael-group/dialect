import copy
import csv
import hashlib
import json
import math
import os
import subprocess
import traceback
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from analysis import mutsig_lambda_co
from analysis import run_tcga_revision_k500 as runner
from analysis.mutsig_lambda_co import build_lambda_pmfs
from dialect.data.revision_fit_policy import (
    D3ImplementationBinding,
    MutSigEffectPagesPolicy,
    MutSigSupportPolicy,
)
from dialect.models.interaction import Interaction


def test_wald_recomputation_uses_model_tau_order() -> None:
    """CSV column order must not create a false bitwise Wald mismatch."""
    csv_taus = [
        0.8939152440541098,
        0.05246859089973929,
        0.026034157595259858,
        0.027582007450890997,
    ]
    interaction = Interaction.__new__(Interaction)

    log_odds, wald = runner._recompute_pair_effect_statistics_from_csv_taus(  # noqa: SLF001
        interaction,
        csv_taus,
    )

    assert log_odds == 2.893150470105523
    assert wald == 0.2970745343202113


def test_secure_runner_read_rejects_hardlinks_and_in_place_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = tmp_path / "authority.bin"
    authority.write_bytes(b"A" * 64)
    linked = tmp_path / "linked.bin"
    os.link(authority, linked)
    with pytest.raises(ValueError, match="single-link regular file"):
        runner._read_secure_regular_bytes(  # noqa: SLF001
            authority,
            label="hardlinked authority",
        )
    linked.unlink()

    original_read = runner.os.read
    mutated = False

    def mutate_after_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, count)
        if chunk and not mutated:
            mutated = True
            with authority.open("r+b") as handle:
                handle.write(b"B" * 64)
                handle.flush()
                os.fsync(handle.fileno())
        return chunk

    monkeypatch.setattr(runner.os, "read", mutate_after_read)
    with pytest.raises(ValueError, match=r"changed during.*descriptor read"):
        runner._read_secure_regular_bytes(  # noqa: SLF001
            authority,
            label="mutable authority",
        )


def test_secure_runner_hash_rejects_symlinked_file(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"authority\n")
    linked = tmp_path / "linked.bin"
    linked.symlink_to(target)

    with pytest.raises(OSError, match=r".+"):
        runner._sha256(linked)  # noqa: SLF001


_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "raw",
    [
        b'{"cohort":"CHOL","cohort":"BRCA"}\n',
        b'{"value":NaN}\n',
        b'{"value":Infinity}\n',
        b'{"value":-Infinity}\n',
        b'{"value":1e999}\n',
        b'{"value":"\\ud800"}\n',
        b'{ "value":1}\n',
        b'{"value":1}',
        b'{"value":1}\n\n',
    ],
)
def test_authoritative_json_parser_rejects_noncanonical_or_ambiguous_bytes(
    tmp_path: Path,
    raw: bytes,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(raw)

    with pytest.raises(ValueError, match="JSON"):
        runner._read_json(manifest)  # noqa: SLF001


def test_authoritative_json_writer_and_parser_share_one_canonical_contract(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    payload = {"z": [1, 0.5], "a": "exact"}
    runner._write_json_atomic(manifest, payload)  # noqa: SLF001

    assert runner._read_json(manifest) == payload  # noqa: SLF001
    with pytest.raises(ValueError, match="Out of range float values"):
        runner._canonical_json({"value": float("nan")})  # noqa: SLF001


def _write_mutsig_receipt(mutsig_dir: Path) -> None:
    artifacts = {
        "lambda_sha256": mutsig_dir / "persample_lambda.f32",
        "meta_sha256": mutsig_dir / "persample_meta.txt",
        "genes_sha256": mutsig_dir / "persample_genes.txt",
        "patients_sha256": mutsig_dir / "persample_patients.txt",
    }
    np_value = next(
        line.split("\t", 1)[1]
        for line in (mutsig_dir / "persample_meta.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.startswith("np\t")
    )
    fields = {
        "schema_version": runner.MUTSIG_RECEIPT_SCHEMA_VERSION,
        "cohort": mutsig_dir.name,
        "upstream_commit": runner.MUTSIG_UPSTREAM_COMMIT,
        "source_tree_sha256": hashlib.sha256(b"test-source-tree").hexdigest(),
        "source_file_count": "1",
        "patch_sha256": runner._sha256(  # noqa: SLF001
            _REPO_ROOT / runner.MUTSIG_PATCH_PATH,
        ),
        "runner_sha256": runner._sha256(  # noqa: SLF001
            _REPO_ROOT / runner.MUTSIG_RUNNER_PATH,
        ),
        "runtime_sha256": hashlib.sha256(b"test-runtime").hexdigest(),
        "maf_sha256": hashlib.sha256(b"test-maf").hexdigest(),
        "sample_axis_sha256": runner._sha256(  # noqa: SLF001
            mutsig_dir / "persample_patients.txt",
        ),
        "sample_axis_count": np_value,
        **{
            key: runner._sha256(path)  # noqa: SLF001
            for key, path in artifacts.items()
        },
    }
    (mutsig_dir / "persample_receipt.tsv").write_text(
        "".join(f"{key}\t{value}\n" for key, value in fields.items()),
        encoding="utf-8",
    )


def _write_inputs(root: Path) -> runner.RunPaths:
    source_root = root / "pancan"
    mutsig_root = root / "mutsig"
    output_root = root / "revision"
    cohort_dir = source_root / "CHOL"
    mutsig_dir = mutsig_root / "CHOL"
    cohort_dir.mkdir(parents=True)
    mutsig_dir.mkdir(parents=True)

    counts = pd.DataFrame(
        {
            "A_M": [1, 1, 1, 1],
            "A_N": [1, 1, 1, 0],
            "B_M": [1, 0, 1, 0],
            "C_N": [1, 0, 0, 0],
        },
        index=["s1", "s2", "s3", "s4"],
    )
    counts.rename_axis("sample").to_csv(cohort_dir / "count_matrix.csv")
    (cohort_dir / "sample_axis.txt").write_text(
        "\n".join(str(sample) for sample in counts.index) + "\n",
        encoding="utf-8",
    )
    pmfs = pd.DataFrame(
        [[0.7, 0.2, 0.08, 0.02]] * 4,
        index=counts.columns,
    )
    pmfs.index.name = "feature"
    pmfs.to_csv(cohort_dir / "bmr_pmfs.csv")
    pmfs.to_csv(cohort_dir / "bmr_pmfs.dig.csv")

    genes = ["A", "B", "C"]
    patients = list(counts.index)
    lambdas = np.full((len(genes), len(patients), 2), 0.1, dtype="<f4")
    (mutsig_dir / "persample_meta.txt").write_text(
        f"ng\t{len(genes)}\nnp\t{len(patients)}\nneff\t2\n",
        encoding="utf-8",
    )
    (mutsig_dir / "persample_genes.txt").write_text(
        "\n".join(genes) + "\n",
        encoding="utf-8",
    )
    (mutsig_dir / "persample_patients.txt").write_text(
        "\n".join(patients) + "\n",
        encoding="utf-8",
    )
    lambdas.ravel(order="F").tofile(mutsig_dir / "persample_lambda.f32")
    _write_mutsig_receipt(mutsig_dir)
    return runner.RunPaths(source_root, mutsig_root, output_root)


def _execute_tiny_cbase_task(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[runner.RunPaths, Path, dict[str, object]]:
    paths = _write_inputs(root)
    paths.output_root.mkdir()
    monkeypatch.setattr(
        runner.interaction_module,
        "LRT_CONTRACT",
        runner.REQUIRED_LRT_CONTRACT,
        raising=False,
    )
    monkeypatch.setattr(
        runner.gene_module,
        "OBSERVATION_SUPPORT_CONTRACT",
        runner.REQUIRED_GENE_SUPPORT_CONTRACT,
        raising=False,
    )
    runner.execute_task(
        paths,
        runner.Task("CHOL", "cbase"),
        nice_increment=0,
        top_k=3,
    )
    task_dir = paths.output_root / "tasks" / "CHOL" / "cbase"
    contract = runner._read_json(  # noqa: SLF001
        paths.output_root / "contracts" / "CHOL.json",
    )
    return paths, task_dir, contract


def _enable_tiny_fit_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner.interaction_module,
        "LRT_CONTRACT",
        runner.REQUIRED_LRT_CONTRACT,
        raising=False,
    )
    monkeypatch.setattr(
        runner.gene_module,
        "OBSERVATION_SUPPORT_CONTRACT",
        runner.REQUIRED_GENE_SUPPORT_CONTRACT,
        raising=False,
    )


def _with_revision_authority(
    paths: runner.RunPaths,
    root: Path,
) -> runner.RunPaths:
    provider_root = root / "provider"
    return runner.RunPaths(
        source_root=provider_root / "cohorts",
        mutsig_root=provider_root / "mutsig",
        output_root=paths.output_root,
        canonical_input_root=root / "canonical",
        input_approval_manifest=root / "input-approval.json",
        expected_input_approval_sha256="a" * 64,
        fit_approval_manifest=root / "fit-approval.json",
        expected_fit_approval_sha256="f" * 64,
        expected_canonical_input_sha256="b" * 64,
        provider_input_root=provider_root,
        expected_provider_input_manifest_sha256="c" * 64,
    )


def _write_completion_grid(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[runner.RunPaths, dict[str, object]]:
    cohorts = ("CHOL", "BRCA")
    monkeypatch.setattr(runner, "TCGA_COHORTS", cohorts)
    paths = _with_revision_authority(
        runner.RunPaths(root / "source", root / "mutsig", root / "run"),
        root,
    )
    paths.output_root.mkdir()
    git_executable = root / "provider-git"
    git_raw = b"synthetic provider-authorized git\n"
    git_executable.write_bytes(git_raw)
    provider_receipt = {
        "association_outputs_opened": False,
        "cohort_provider_receipts_sha256": "d" * 64,
        "contract": runner.PROVIDER_INPUT_CONTRACT,
        "expected_manifest_sha256": paths.expected_provider_input_manifest_sha256,
        "full_acceptance_receipt": {
            "association_outputs_opened": False,
            "authority_sha256": "a" * 64,
            "cohort_receipts_sha256": "d" * 64,
            "contract": "provider-full-acceptance-receipt-v1",
            "execution_snapshot": {"tree_sha256": "e" * 64},
            "full_inventory_validated": True,
            "provider_manifest_sha256": (paths.expected_provider_input_manifest_sha256),
            "schema_version": runner.SCHEMA_VERSION,
        },
        "git_executable": {
            "bytes": len(git_raw),
            "path": git_executable.as_posix(),
            "sha256": hashlib.sha256(git_raw).hexdigest(),
        },
        "manifest": {
            "bytes": 123,
            "path": (
                paths.provider_input_root / "provider_input_manifest.json"
            ).as_posix(),
            "sha256": paths.expected_provider_input_manifest_sha256,
        },
        "root": paths.provider_input_root.as_posix(),
    }
    provider_receipt["full_acceptance_receipt_sha256"] = (
        runner.full_acceptance_receipt_sha256(
            provider_receipt["full_acceptance_receipt"],
        )
    )
    authority = {
        "canonical_input_root": paths.canonical_input_root.as_posix(),
        "configured": True,
        "expected_canonical_input_sha256": (paths.expected_canonical_input_sha256),
        "expected_fit_approval_sha256": paths.expected_fit_approval_sha256,
        "expected_input_approval_sha256": paths.expected_input_approval_sha256,
        "fit_approval_manifest": paths.fit_approval_manifest.as_posix(),
        "input_approval_manifest": paths.input_approval_manifest.as_posix(),
        "provider_input": provider_receipt,
    }
    implementation = {"analysis/run_tcga_revision_k500.py": "e" * 64}
    runner._write_json_atomic(  # noqa: SLF001
        paths.output_root / "run_manifest.json",
        {
            "analysis": "tcga-revision-k500",
            "bmrs": list(runner.BMRS),
            "cohorts": list(cohorts),
            "git": {"dirty": False, "head": "synthetic"},
            "implementation_sha256": implementation,
            "revision_authority": authority,
            "schema_version": runner.SCHEMA_VERSION,
            "signed_tested_family": runner.asdict(runner.REQUIRED_TESTED_FAMILY),
            "tested_family_implementation": runner.asdict(
                runner.REQUIRED_TESTED_FAMILY,
            ),
            "top_k": runner.TOP_K,
        },
    )
    hashes = {
        name: str(index) * 64
        for index, name in enumerate(
            (
                "counts",
                "cbase",
                "dig",
                "lambda",
                "metadata",
                "genes",
                "patients",
                "receipt",
            ),
            start=1,
        )
    }
    mutsig_pmf_contract = runner.build_poisson_support_contract(0.0, 0)
    mutsig_pmf_storage_contract = runner.estimate_native_poisson_pmf_storage(
        1,
        1,
        mutsig_pmf_contract["inclusive_support_k"],
    )
    for cohort in cohorts:
        contract = {
            "schema_version": runner.SCHEMA_VERSION,
            "cohort": cohort,
            "features": ["A_M"],
            "inputs": {
                "cbase": {"sha256": hashes["cbase"]},
                "counts": {"sha256": hashes["counts"]},
                "dig": {"sha256": hashes["dig"]},
                "mutsig": {
                    "tensor_encoding": runner._mutsig_tensor_encoding_record(  # noqa: SLF001
                        read_only=True,
                    ),
                    "files": {
                        name: {"sha256": hashes[name]}
                        for name in (
                            "lambda",
                            "metadata",
                            "genes",
                            "patients",
                            "receipt",
                        )
                    },
                },
            },
            "pair_policy": {
                "ordered_pair_sha256": runner._sequence_sha256([]),  # noqa: SLF001
                "row_count": 0,
            },
            "mutsig_pmf_contract": mutsig_pmf_contract,
            "mutsig_pmf_storage_contract": mutsig_pmf_storage_contract,
            "provider_input_provenance": {"root_receipt": provider_receipt},
            "samples": {"count": 1},
            "top_k": runner.TOP_K,
        }
        runner._write_json_atomic(  # noqa: SLF001
            paths.output_root / "contracts" / f"{cohort}.json",
            contract,
        )
        for bmr in runner.BMRS:
            task_dir = paths.output_root / "tasks" / cohort / bmr
            task_dir.mkdir(parents=True)
            single_raw = f"opaque-single-{cohort}-{bmr}\n".encode()
            pair_raw = f"opaque-pair-{cohort}-{bmr}\n".encode()
            (task_dir / "single_gene_results.csv").write_bytes(single_raw)
            (task_dir / "pairwise_interaction_results.csv").write_bytes(pair_raw)
            validation = {
                "features": 1,
                "ordered_features_sha256": runner._sequence_sha256(["A_M"]),  # noqa: SLF001
                "ordered_pair_sha256": contract["pair_policy"]["ordered_pair_sha256"],
                "pairs": 0,
                "pairwise_sha256": hashlib.sha256(pair_raw).hexdigest(),
                "single_gene_sha256": hashlib.sha256(single_raw).hexdigest(),
            }
            task_manifest = {
                "bmr": bmr,
                "cohort": cohort,
                "completed_at_utc": "2026-08-28T00:00:00Z",
                "consumed_input_sha256": runner._consumed_input_hashes(  # noqa: SLF001
                    contract,
                    bmr,
                ),
                "contingency_table_contract": (
                    runner.REQUIRED_CONTINGENCY_TABLE_CONTRACT
                ),
                "contract_sha256": runner._json_sha256(contract),  # noqa: SLF001
                "exit_status": 0,
                "gene_support_contract": runner.REQUIRED_GENE_SUPPORT_CONTRACT,
                "implementation_sha256": implementation,
                "log_odds_ratio_contract": (runner.REQUIRED_LOG_ODDS_RATIO_CONTRACT),
                "lrt_contract": runner.REQUIRED_LRT_CONTRACT,
                "mutsig_cbase_feature_fallback": False,
                "mutsig_pmf_contract": mutsig_pmf_contract,
                "mutsig_pmf_storage_contract": mutsig_pmf_storage_contract,
                "native_support_only": True,
                "niceness": {
                    "requested_increment": runner.REQUIRED_NICE_INCREMENT,
                    "resulting_process_nice": runner.REQUIRED_NICE_INCREMENT,
                },
                "observation_support_universe": (runner.OBSERVATION_SUPPORT_UNIVERSE),
                "pair_effect_identifiability_contract": (
                    runner.REQUIRED_PAIR_EFFECT_IDENTIFIABILITY_CONTRACT
                ),
                "pair_fit_contract": runner.REQUIRED_PAIR_FIT_CONTRACT,
                "pair_fit_kkt_tolerance": runner.REQUIRED_PAIR_FIT_KKT_TOL,
                "pair_fit_max_iterations": runner.REQUIRED_PAIR_FIT_MAX_ITER,
                "pair_identifiability_relative_tolerance": (
                    runner.REQUIRED_PAIR_IDENTIFIABILITY_RTOL
                ),
                "pair_simplex_tolerance": runner.REQUIRED_PAIR_SIMPLEX_TOL,
                "lrt_nestedness_tolerance": (runner.REQUIRED_LRT_NESTEDNESS_TOL),
                "output_recomputation_atol": (
                    runner.REQUIRED_OUTPUT_RECOMPUTATION_ATOL
                ),
                "provider_input_root_receipt": provider_receipt,
                "resource_usage": {
                    "elapsed_seconds": 1.0,
                    "peak_rss": {
                        "bytes": 1024,
                        "native_unit": "KiB",
                        "native_value": 1,
                        "platform": "linux",
                        "source": (
                            "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss"
                        ),
                    },
                },
                "rho_contract": runner.REQUIRED_RHO_CONTRACT,
                "same_base_pairs_excluded_before_fit": True,
                "sample_axis_contract": runner.SAMPLE_AXIS_CONTRACT,
                "schema_version": runner.SCHEMA_VERSION,
                "top_k": runner.TOP_K,
                "undefined_rho_lrt_tolerance": (runner.REQUIRED_UNDEFINED_RHO_LRT_TOL),
                "validation": validation,
            }
            runner._write_json_atomic(  # noqa: SLF001
                task_dir / "task_manifest.json",
                task_manifest,
            )
    return paths, authority


def _fake_provider_bundle(paths: runner.RunPaths) -> dict[str, object]:
    def file_binding(path: Path, label: str) -> dict[str, object]:
        return {
            "path": path,
            "file": {
                "path": label,
                "bytes": 1,
                "sha256": hashlib.sha256(label.encode()).hexdigest(),
            },
        }

    cohort_bindings = {}
    for cohort in runner.TCGA_COHORTS:
        cohort_root = paths.source_root / cohort
        mutsig_root = paths.mutsig_root / cohort
        binding = {
            "cohort": cohort,
            "cohort_root": cohort_root,
            "mutsig_root": mutsig_root,
            "count_matrix": file_binding(
                cohort_root / "count_matrix.csv",
                f"cohorts/{cohort}/count_matrix.csv",
            ),
            "cbase_pmfs": file_binding(
                cohort_root / "bmr_pmfs.csv",
                f"cohorts/{cohort}/bmr_pmfs.csv",
            ),
            "dig_pmfs": file_binding(
                cohort_root / "bmr_pmfs.dig.csv",
                f"cohorts/{cohort}/bmr_pmfs.dig.csv",
            ),
            "sample_axis": file_binding(
                cohort_root / "sample_axis.txt",
                f"cohorts/{cohort}/sample_axis.txt",
            ),
            "mutsig_lambda": file_binding(
                mutsig_root / "persample_lambda.f32",
                f"mutsig/{cohort}/persample_lambda.f32",
            ),
            "mutsig_metadata": file_binding(
                mutsig_root / "persample_meta.txt",
                f"mutsig/{cohort}/persample_meta.txt",
            ),
            "mutsig_genes": file_binding(
                mutsig_root / "persample_genes.txt",
                f"mutsig/{cohort}/persample_genes.txt",
            ),
            "mutsig_patients": file_binding(
                mutsig_root / "persample_patients.txt",
                f"mutsig/{cohort}/persample_patients.txt",
            ),
            "mutsig_receipt": file_binding(
                mutsig_root / "persample_receipt.tsv",
                f"mutsig/{cohort}/persample_receipt.tsv",
            ),
            "canonical_inputs": {},
            "provider_receipt": {
                "cohort": cohort,
                "association_outputs_opened": False,
            },
        }
        cohort_bindings[cohort] = binding
    provider_root = paths.provider_input_root
    assert provider_root is not None
    expected_hash = paths.expected_provider_input_manifest_sha256
    assert expected_hash is not None
    manifest_path = provider_root / "provider_input_manifest.json"
    synthetic_git = Path(runner.sys.executable).resolve()
    synthetic_git_raw = runner._read_secure_regular_bytes(  # noqa: SLF001
        synthetic_git,
        label="synthetic provider Git",
    )
    cohort_receipts = [
        cohort_bindings[cohort]["provider_receipt"] for cohort in runner.TCGA_COHORTS
    ]
    full_acceptance = {
        "association_outputs_opened": False,
        "authority_sha256": "a" * 64,
        "cohort_receipts_sha256": runner._json_sha256(cohort_receipts),  # noqa: SLF001
        "contract": "provider-full-acceptance-receipt-v1",
        "execution_snapshot": {"tree_sha256": "e" * 64},
        "full_inventory_validated": True,
        "provider_manifest_sha256": expected_hash,
        "schema_version": runner.SCHEMA_VERSION,
    }
    return {
        "root": provider_root,
        "manifest": {
            "contract": runner.PROVIDER_INPUT_CONTRACT,
            "cohort_provider_receipts": cohort_receipts,
            "sources": {
                "git_executable": {
                    "bytes": len(synthetic_git_raw),
                    "path": synthetic_git.as_posix(),
                    "sha256": hashlib.sha256(synthetic_git_raw).hexdigest(),
                },
            },
        },
        "manifest_file": {
            "path": manifest_path,
            "file": {
                "path": "provider_input_manifest.json",
                "bytes": 1,
                "sha256": expected_hash,
            },
        },
        "roots": {"cohorts": paths.source_root, "mutsig": paths.mutsig_root},
        "cohorts": list(runner.TCGA_COHORTS),
        "cohort_bindings": cohort_bindings,
        "full_acceptance_receipt": full_acceptance,
        "full_acceptance_receipt_sha256": (
            runner.full_acceptance_receipt_sha256(full_acceptance)
        ),
        "association_outputs_opened": False,
    }


def _embedded_record(root: Path, path: Path) -> dict[str, object]:
    record = runner._file_record(path)  # noqa: SLF001
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": record["bytes"],
        "sha256": record["sha256"],
    }


def _write_local_authority_contract(  # noqa: PLR0915
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch | None = None,
    *,
    legacy_fit_v4: bool = False,
) -> tuple[runner.RunPaths, dict[str, object]]:
    canonical_root = tmp_path / "canonical"
    provider_root = tmp_path / "provider"
    canonical_root.mkdir()
    provider_root.mkdir()

    canonical_maf = canonical_root / "canonical_mafs" / "CHOL.maf"
    sample_axis = canonical_root / "cohorts" / "CHOL" / "sample_axis.txt"
    population_manifest = canonical_root / "population" / "population_manifest.json"
    canonical_maf.parent.mkdir(parents=True)
    sample_axis.parent.mkdir(parents=True)
    population_manifest.parent.mkdir(parents=True)
    canonical_maf.write_bytes(b"canonical maf\n")
    sample_axis.write_bytes(b"s1\n")
    runner._write_json_atomic(population_manifest, {"population": "signed"})  # noqa: SLF001
    child_manifest = canonical_root / "cohorts" / "CHOL.json"
    runner._write_json_atomic(  # noqa: SLF001
        child_manifest,
        {
            "output": {
                "canonical_maf": _embedded_record(canonical_root, canonical_maf),
            },
            "population": {
                "manifest": _embedded_record(canonical_root, population_manifest),
                "sample_axis": _embedded_record(canonical_root, sample_axis),
            },
        },
    )
    input_manifest = canonical_root / "input_manifest.json"
    runner._write_json_atomic(  # noqa: SLF001
        input_manifest,
        {
            "cohorts": ["CHOL"],
            "cohort_manifests": [
                {
                    "cohort": "CHOL",
                    "manifest": _embedded_record(canonical_root, child_manifest),
                },
            ],
        },
    )

    authority_dir = tmp_path / "approval-artifacts"
    authority_dir.mkdir()
    artifact_records = {}
    decisions = []
    for decision_id in runner.DECISION_IDS:
        artifact_path = authority_dir / f"{decision_id}.json"
        envelope = {
            "contract": f"contract-{decision_id}",
            "decision_id": decision_id,
            "payload": {"value": decision_id},
        }
        runner._write_json_atomic(artifact_path, envelope)  # noqa: SLF001
        artifact_records[decision_id] = runner._file_record(artifact_path)  # noqa: SLF001
        decisions.append(
            {
                "canonical_artifact": {
                    "path": f"approval-artifacts/{decision_id}.json",
                    "sha256": artifact_records[decision_id]["sha256"],
                },
                "decision_id": decision_id,
            },
        )
    input_approval = tmp_path / "input-approval.json"
    fit_approval = tmp_path / "fit-approval.json"
    runner._write_json_atomic(  # noqa: SLF001
        input_approval,
        {
            "schema": runner.STAGE_SCOPED_APPROVAL_SCHEMA,
            "allowed_stages": [runner.MATERIALIZE_FINAL_INPUTS_STAGE],
            "decisions": decisions[:2],
            "stage_bindings": {
                runner.MATERIALIZE_FINAL_INPUTS_STAGE: {
                    "d1_canonical_artifact_sha256": artifact_records["D1"]["sha256"],
                    "d2_canonical_artifact_sha256": artifact_records["D2"]["sha256"],
                },
            },
        },
    )
    input_approval_sha256 = runner._sha256(input_approval)  # noqa: SLF001
    decision_digests = {
        decision["decision_id"]: runner._json_sha256(decision)  # noqa: SLF001
        for decision in decisions
    }

    provider_paths = {
        "count_matrix": provider_root / "cohorts" / "CHOL" / "count_matrix.csv",
        "cbase_pmfs": provider_root / "cohorts" / "CHOL" / "bmr_pmfs.csv",
        "dig_pmfs": provider_root / "cohorts" / "CHOL" / "bmr_pmfs.dig.csv",
        "sample_axis": provider_root / "cohorts" / "CHOL" / "sample_axis.txt",
        "mutsig_lambda": provider_root / "mutsig" / "CHOL" / "persample_lambda.f32",
        "mutsig_metadata": provider_root / "mutsig" / "CHOL" / "persample_meta.txt",
        "mutsig_genes": provider_root / "mutsig" / "CHOL" / "persample_genes.txt",
        "mutsig_patients": provider_root / "mutsig" / "CHOL" / "persample_patients.txt",
        "mutsig_receipt": provider_root / "mutsig" / "CHOL" / "persample_receipt.tsv",
    }
    for name, path in provider_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{name}\n".encode())
    provider_files = {
        name: runner._file_record(path)  # noqa: SLF001
        for name, path in provider_paths.items()
    }
    canonical_records = {
        "child_manifest": runner._file_record(child_manifest),  # noqa: SLF001
        "canonical_maf": runner._file_record(canonical_maf),  # noqa: SLF001
        "sample_axis": runner._file_record(sample_axis),  # noqa: SLF001
        "population_manifest": runner._file_record(population_manifest),  # noqa: SLF001
    }
    provider_cohort_receipt = {
        "association_outputs_opened": False,
        "canonical_inputs": {
            name: {
                "bytes": record["bytes"],
                "path": Path(record["path"]).relative_to(canonical_root).as_posix(),
                "sha256": record["sha256"],
            }
            for name, record in canonical_records.items()
        },
        "cohort": "CHOL",
        "k500_support": {},
        "providers": {},
    }
    git_path = provider_root / "runtime" / "git"
    git_path.parent.mkdir()
    git_raw = b"synthetic provider Git\n"
    git_path.write_bytes(git_raw)
    git_record = {
        "bytes": len(git_raw),
        "path": git_path.as_posix(),
        "sha256": hashlib.sha256(git_raw).hexdigest(),
    }
    provider_manifest = provider_root / "provider_input_manifest.json"
    runner._write_json_atomic(  # noqa: SLF001
        provider_manifest,
        {
            "cohort_provider_receipts": [provider_cohort_receipt],
            "sources": {"git_executable": git_record},
        },
    )
    provider_manifest_sha256 = runner._sha256(provider_manifest)  # noqa: SLF001
    full_acceptance = {
        "association_outputs_opened": False,
        "authority_sha256": "a" * 64,
        "cohort_receipts_sha256": runner._json_sha256(  # noqa: SLF001
            [provider_cohort_receipt],
        ),
        "contract": "provider-full-acceptance-receipt-v1",
        "execution_snapshot": {"tree_sha256": "e" * 64},
        "full_inventory_validated": True,
        "provider_manifest_sha256": provider_manifest_sha256,
        "schema_version": runner.SCHEMA_VERSION,
    }
    input_manifest_sha256 = runner._sha256(input_manifest)  # noqa: SLF001
    full_validation_receipt = {
        "schema": "dialect-canonical-input-full-validation-receipt-v1",
        "validation_contract": "full-streaming-canonical-replay-v1",
        "input_manifest_sha256": input_manifest_sha256,
        "approval_manifest_sha256": input_approval_sha256,
        "population_manifest_sha256": canonical_records["population_manifest"][
            "sha256"
        ],
        "implementation_sha256": "1" * 64,
        "inventory_sha256": "2" * 64,
        "ordered_cohorts_sha256": runner._sequence_sha256(["CHOL"]),  # noqa: SLF001
        "validated_cohort_count": 1,
        "association_outputs_opened": False,
    }
    runner._write_json_atomic(  # noqa: SLF001
        fit_approval,
        {
            "schema": (
                runner.APPROVAL_SCHEMA
                if legacy_fit_v4
                else runner.STAGE_SCOPED_APPROVAL_SCHEMA
            ),
            "allowed_stages": [runner.FIT_SEALED_TCGA_K500_STAGE],
            "decisions": decisions if legacy_fit_v4 else decisions[:6],
            "stage_bindings": {
                runner.FIT_SEALED_TCGA_K500_STAGE: {
                    "canonical_input_manifest_sha256": input_manifest_sha256,
                    "provider_input_manifest_sha256": provider_manifest_sha256,
                },
            },
        },
    )
    fit_approval_sha256 = runner._sha256(fit_approval)  # noqa: SLF001
    paths = runner.RunPaths(
        source_root=provider_root / "cohorts",
        mutsig_root=provider_root / "mutsig",
        output_root=tmp_path / "run",
        canonical_input_root=canonical_root,
        input_approval_manifest=input_approval,
        expected_input_approval_sha256=input_approval_sha256,
        fit_approval_manifest=fit_approval,
        expected_fit_approval_sha256=fit_approval_sha256,
        expected_canonical_input_sha256=input_manifest_sha256,
        provider_input_root=provider_root,
        expected_provider_input_manifest_sha256=provider_manifest_sha256,
    )
    fit_policy_receipts = {}
    for decision_id in ("D3", "D4", "D5", "D6"):
        artifact = artifact_records[decision_id]
        fit_policy_receipts[decision_id] = {
            "canonical_artifact_path": f"approval-artifacts/{decision_id}.json",
            "canonical_artifact_sha256": artifact["sha256"],
            "canonical_artifact_size_bytes": artifact["bytes"],
            "contract": f"contract-{decision_id}",
            "decision_digest": decision_digests[decision_id],
            "decision_id": decision_id,
            "payload_sha256": runner._json_sha256({"value": decision_id}),  # noqa: SLF001
        }
    authority = {
        "status": "verified",
        "contract": runner.CANONICAL_INPUT_CONTRACT,
        "input_approval": {
            "manifest": runner._file_record(input_approval),  # noqa: SLF001
            "manifest_sha256": input_approval_sha256,
            "authorized_stage": runner.MATERIALIZE_FINAL_INPUTS_STAGE,
            "decision_digests": {
                decision_id: decision_digests[decision_id]
                for decision_id in ("D1", "D2")
            },
        },
        "fit_approval": {
            "manifest": runner._file_record(fit_approval),  # noqa: SLF001
            "manifest_sha256": fit_approval_sha256,
            "authorized_stage": runner.FIT_SEALED_TCGA_K500_STAGE,
            "decision_digests": {
                decision_id: decision_digests[decision_id]
                for decision_id in ("D1", "D2", "D3", "D4", "D5", "D6")
            },
        },
        "fit_policy": {
            "d3": {},
            "d4": {},
            "d5": {
                "tested_family": runner.asdict(runner.REQUIRED_TESTED_FAMILY),
            },
            "d6": {},
            "receipts": fit_policy_receipts,
        },
        "input_manifest": runner._file_record(input_manifest),  # noqa: SLF001
        "full_validation": {
            "receipt": full_validation_receipt,
            "receipt_sha256": runner.full_input_validation_receipt_sha256(
                full_validation_receipt,
            ),
        },
        "cohort_manifest": canonical_records["child_manifest"],
        "canonical_maf": canonical_records["canonical_maf"],
        "authoritative_sample_axis": canonical_records["sample_axis"],
        "population_manifest": canonical_records["population_manifest"],
    }
    provenance = {
        "association_outputs_opened": False,
        "cohort": "CHOL",
        "cohort_receipt": provider_cohort_receipt,
        "contract": runner.PROVIDER_INPUT_CONTRACT,
        "files": provider_files,
        "root_receipt": {
            "association_outputs_opened": False,
            "cohort_provider_receipts_sha256": runner._json_sha256(  # noqa: SLF001
                [provider_cohort_receipt],
            ),
            "contract": runner.PROVIDER_INPUT_CONTRACT,
            "expected_manifest_sha256": provider_manifest_sha256,
            "full_acceptance_receipt": full_acceptance,
            "full_acceptance_receipt_sha256": (
                runner.full_acceptance_receipt_sha256(full_acceptance)
            ),
            "git_executable": git_record,
            "manifest": runner._file_record(provider_manifest),  # noqa: SLF001
            "root": provider_root.as_posix(),
        },
    }
    features = [f"G{index}_M" for index in range(runner.TOP_K)]
    mutsig_pmf_contract = runner.build_poisson_support_contract(0.0, 0)
    contract = {
        "schema_version": runner.SCHEMA_VERSION,
        "cohort": "CHOL",
        "feature_policy": {
            "feature_ranking": runner.TESTED_FAMILY_FEATURE_RANKING,
            "mutsig_cbase_feature_fallback": False,
            "observation_support": runner.OBSERVATION_SUPPORT_UNIVERSE,
            "provider_support": runner.TESTED_FAMILY_PROVIDER_SUPPORT,
            "tie_break": runner.TESTED_FAMILY_TIE_BREAK,
        },
        "features": features,
        "inputs": {
            "counts": provider_files["count_matrix"],
            "cbase": provider_files["cbase_pmfs"],
            "dig": provider_files["dig_pmfs"],
            "sample_axis": provider_files["sample_axis"],
            "mutsig": {
                "tensor_encoding": runner._mutsig_tensor_encoding_record(  # noqa: SLF001
                    read_only=True,
                ),
                "files": {
                    "lambda": provider_files["mutsig_lambda"],
                    "metadata": provider_files["mutsig_metadata"],
                    "genes": provider_files["mutsig_genes"],
                    "patients": provider_files["mutsig_patients"],
                    "receipt": provider_files["mutsig_receipt"],
                },
            },
        },
        "provider_input_provenance": provenance,
        "mutsig_pmf_contract": mutsig_pmf_contract,
        "mutsig_pmf_storage_contract": runner.estimate_native_poisson_pmf_storage(
            len(features),
            1,
            mutsig_pmf_contract["inclusive_support_k"],
        ),
        "pair_policy": {
            "epsilon_pretest_filter": runner.TESTED_FAMILY_NO_PRETEST_FILTER,
            "marginal_effect_pretest_filter": (runner.TESTED_FAMILY_NO_PRETEST_FILTER),
            "pair_construction": runner.TESTED_FAMILY_PAIR_CONSTRUCTION,
            "same_base_missense_nonsense": runner.TESTED_FAMILY_SAME_BASE_POLICY,
            **runner._pair_contract(features),  # noqa: SLF001
        },
        "revision_input_authority": authority,
        "samples": {"count": 1},
        "tested_family": runner.asdict(runner.REQUIRED_TESTED_FAMILY),
        "top_k": runner.TOP_K,
    }
    if monkeypatch is not None:
        scoped_canonical_binding = {
            "cohort": "CHOL",
            **{
                name: {
                    "path": Path(record["path"]),
                    "file": {
                        "bytes": record["bytes"],
                        "path": Path(record["path"])
                        .relative_to(canonical_root)
                        .as_posix(),
                        "sha256": record["sha256"],
                    },
                }
                for name, record in canonical_records.items()
            },
        }
        binding = {
            "cohort": "CHOL",
            "cohort_root": provider_root / "cohorts" / "CHOL",
            "mutsig_root": provider_root / "mutsig" / "CHOL",
            "canonical_inputs": {
                name: scoped_canonical_binding[name]
                for name in ("canonical_maf", "sample_axis")
            },
            "canonical_input_receipts": {
                name: {
                    "bytes": canonical_records[name]["bytes"],
                    "path": Path(canonical_records[name]["path"])
                    .relative_to(canonical_root)
                    .as_posix(),
                    "sha256": canonical_records[name]["sha256"],
                }
                for name in ("child_manifest", "population_manifest")
            },
            "provider_receipt": provider_cohort_receipt,
            **{
                name: {
                    "path": Path(record["path"]),
                    "file": {
                        "bytes": record["bytes"],
                        "path": Path(record["path"])
                        .relative_to(provider_root)
                        .as_posix(),
                        "sha256": record["sha256"],
                    },
                }
                for name, record in provider_files.items()
            },
        }

        def validate_canonical_scope(*_args, **_kwargs):
            for record in canonical_records.values():
                runner._verify_file_record(record)  # noqa: SLF001
            return {
                "association_outputs_opened": False,
                "binding": scoped_canonical_binding,
                "full_validation_receipt": full_validation_receipt,
                "manifest": {},
            }

        monkeypatch.setattr(
            runner,
            "validate_materialized_input_cohort_binding",
            validate_canonical_scope,
        )
        monkeypatch.setattr(
            runner,
            "validate_materialized_provider_cohort_input",
            lambda *_args, **_kwargs: {
                "association_outputs_opened": False,
                "binding": binding,
                "cohort": "CHOL",
                "execution_snapshot": {
                    "root": "_orchestration/synthetic.ready",
                    "tree_sha256": "e" * 64,
                    "validation_scope": ("selected-cohort-and-exact-shared-closure"),
                },
                "full_acceptance_receipt": full_acceptance,
                "provider_receipt": provider_cohort_receipt,
                "root": provider_root,
            },
        )
    return paths, contract


def test_shared_feature_axis_preserves_count_column_order_for_ties():
    counts = pd.DataFrame(
        {
            "Z_M": [1, 0],
            "A_M": [0, 1],
            "B_N": [0, 1],
            "C_M": [0, 0],
        },
    )
    support = set(counts.columns)

    features, totals = runner.select_common_features(
        counts,
        cbase_features=support,
        dig_features=support,
        mutsig_genes={"Z", "A", "B", "C"},
        top_k=3,
    )

    assert features == ["Z_M", "A_M", "B_N"]
    assert totals == {"Z_M": 1, "A_M": 1, "B_N": 1}


def test_pair_universe_excludes_same_base_before_fit():
    features = ["A_M", "A_N", "B_M", "C_N"]

    pairs = list(runner.iter_tested_pairs(features))

    assert ("A_M", "A_N") not in pairs
    assert len(pairs) == 5
    assert (
        runner._pair_contract(features)["same_base_pairs_excluded"] == 1  # noqa: SLF001
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_k", 4),
        ("feature_ranking", "different-ranking"),
        ("tie_break", "different-tie-break"),
        ("provider_support", "different-provider-support"),
        ("pair_construction", "different-pair-construction"),
        ("same_base_missense_nonsense", "retain"),
        ("epsilon_pretest_filter", "epsilon-positive"),
        ("marginal_effect_pretest_filter", "nonzero-only"),
        ("family", "partial-family"),
    ],
)
def test_runner_rejects_self_authorized_tested_family_metadata(
    tmp_path,
    field,
    value,
):
    paths = _write_inputs(tmp_path)
    contract = runner.build_cohort_contract(paths, "CHOL", top_k=3)
    contract["tested_family"][field] = value

    with pytest.raises(ValueError, match="exact tested family"):
        runner._require_tested_family_contract(  # noqa: SLF001
            contract,
            require_signed_k500=False,
        )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("feature_policy", "feature_ranking", "different-ranking"),
        ("feature_policy", "tie_break", "different-tie-break"),
        ("feature_policy", "provider_support", "different-provider-support"),
        ("pair_policy", "pair_construction", "different-pair-construction"),
        ("pair_policy", "same_base_missense_nonsense", "retain"),
        ("pair_policy", "epsilon_pretest_filter", "epsilon-positive"),
        ("pair_policy", "marginal_effect_pretest_filter", "nonzero-only"),
        ("pair_policy", "row_count", 1),
    ],
)
def test_runner_cross_checks_family_labels_against_actual_contract(
    tmp_path,
    section,
    field,
    value,
):
    paths = _write_inputs(tmp_path)
    contract = runner.build_cohort_contract(paths, "CHOL", top_k=3)
    contract[section][field] = value

    with pytest.raises(ValueError, match=r"selection|pair construction"):
        runner._require_tested_family_contract(  # noqa: SLF001
            contract,
            require_signed_k500=False,
        )


def test_strict_pmf_loader_preserves_noncontiguous_integer_count_keys(tmp_path):
    path = tmp_path / "pmfs.csv"
    frame = pd.DataFrame([[0.75, 0.25]], index=["A_M"], columns=[0, 2])
    frame.to_csv(path)

    pmfs = runner._load_strict_pmfs(path)  # noqa: SLF001

    assert pmfs == {"A_M": {0: 0.75, 2: 0.25}}
    assert runner._shared_pmf_has_observation_support(  # noqa: SLF001
        np.array([2]),
        pmfs["A_M"],
    )


def test_snapshot_count_parser_returns_read_only_numeric_storage() -> None:
    counts = runner._parse_counts_csv(  # noqa: SLF001
        b"sample,A_M,B_N\ns1,1,0\ns2,0,1\n",
        label="synthetic counts",
    )

    assert not counts.to_numpy(copy=False).flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        counts.iloc[0, 0] = 2


def test_support_audit_reports_pair_specific_effective_masks():
    counts = pd.DataFrame(
        {"A_M": [2, 0], "B_M": [0, 0], "C_M": [0, 0]},
        index=["s1", "s2"],
    )
    pmfs = {feature: {0: 1.0} for feature in counts.columns}

    audit = runner.audit_background_support(counts, list(counts.columns), pmfs)

    assert audit["zero_support_feature_samples"] == 1
    assert audit["zero_support_by_feature"] == {"A_M": 1}
    assert audit["pairs"]["excluded_sample_count_histogram"] == {"0": 1, "1": 2}
    assert audit["pairs"]["with_excluded_samples"] == 2
    assert audit["inference_implication"]["pair_specific_effective_mask_refit_required"]


def test_strict_mutsig_builder_rejects_sample_and_feature_fallback(tmp_path):
    paths = _write_inputs(tmp_path)
    mutsig_dir = paths.mutsig_root / "CHOL"

    with pytest.raises(ValueError, match="patient axis"):
        build_lambda_pmfs(
            ["A_M"],
            pd.Index(["s1", "absent"]),
            mutsig_dir,
            None,
            2,
            allow_cbase_fallback=False,
            require_all_features=True,
            require_all_samples=True,
        )

    with pytest.raises(ValueError, match="natively cover"):
        build_lambda_pmfs(
            ["MISSING_M"],
            pd.Index(["s1"]),
            mutsig_dir,
            None,
            2,
            allow_cbase_fallback=False,
            require_all_features=True,
            require_all_samples=True,
        )


def test_frozen_mutsig_builder_preserves_exact_zero_lambda(tmp_path):
    paths = _write_inputs(tmp_path)
    mutsig_dir = paths.mutsig_root / "CHOL"
    np.zeros((3, 4, 2), dtype="<f4").ravel(order="F").tofile(
        mutsig_dir / "persample_lambda.f32",
    )

    pmfs = build_lambda_pmfs(
        ["A_M"],
        pd.Index(["s1"]),
        mutsig_dir,
        None,
        2,
        allow_cbase_fallback=False,
        require_all_features=True,
        require_all_samples=True,
        lambda_floor=None,
    )

    assert pmfs["A_M"][0] == {0: 1.0, 1: 0.0, 2: 0.0}


def test_production_mutsig_support_extends_material_observed_tail(tmp_path):
    paths = _write_inputs(tmp_path)
    mutsig_dir = paths.mutsig_root / "CHOL"
    native_rate = np.float32(12.0)
    np.full((3, 4, 2), native_rate, dtype="<f4").ravel(order="F").tofile(
        mutsig_dir / "persample_lambda.f32",
    )
    contract = runner.build_poisson_support_contract(float(native_rate), 1)

    pmfs = build_lambda_pmfs(
        ["A_M", "A_N"],
        pd.Index(["s1", "s2", "s3", "s4"]),
        mutsig_dir,
        None,
        1,
        allow_cbase_fallback=False,
        require_all_features=True,
        require_all_samples=True,
        lambda_floor=None,
        production_contract=contract,
    )

    assert float(runner.poisson.sf(1, native_rate)) > 0.99
    assert contract["inclusive_support_k"] > 1
    assert contract["worst_discarded_tail_probability"] <= 1e-12
    assert contract["predecessor_tail_probability"] > 1e-12
    for feature_pmfs in pmfs.values():
        for row in feature_pmfs:
            assert not isinstance(row, dict)
            assert max(row) == contract["inclusive_support_k"]
            assert math.isclose(math.fsum(row.values()), 1.0, abs_tol=1e-15)


def test_production_mutsig_support_is_exact_minimum_or_observed_maximum():
    tail_bound = runner.build_poisson_support_contract(float(np.float32(5.5)), 0)
    observed_bound = runner.build_poisson_support_contract(
        float(np.float32(0.1)),
        100,
    )

    assert tail_bound["tail_criterion_binds"] is True
    assert tail_bound["worst_discarded_tail_probability"] <= 1e-12
    assert tail_bound["predecessor_tail_probability"] > 1e-12
    assert observed_bound["tail_criterion_binds"] is False
    assert observed_bound["inclusive_support_k"] == 100
    assert observed_bound["predecessor_tail_probability"] <= 1e-12


def test_production_mutsig_zero_lambda_has_exact_minimal_support():
    contract = runner.build_poisson_support_contract(0.0, 0)
    pmfs = runner.build_native_poisson_pmfs(
        {"A_M": np.zeros(2, dtype="<f4")},
        contract,
    )

    assert contract["inclusive_support_k"] == 0
    assert contract["worst_discarded_tail_probability"] == 0.0
    assert contract["predecessor_tail_probability"] == 1.0
    assert [dict(row) for row in pmfs["A_M"]] == [{0: 1.0}, {0: 1.0}]


def test_production_mutsig_rejects_axis_dtype_and_policy_drift(tmp_path):
    paths = _write_inputs(tmp_path)
    mutsig_dir = paths.mutsig_root / "CHOL"
    contract = runner.build_poisson_support_contract(float(np.float32(0.1)), 1)

    with pytest.raises(ValueError, match="exactly binary32-representable"):
        runner.build_poisson_support_contract(0.1, 1)
    with pytest.raises(ValueError, match="equally sample-aligned"):
        runner.build_native_poisson_pmfs(
            {
                "A_M": np.full(2, 0.1, dtype="<f4"),
                "B_M": np.full(3, 0.1, dtype="<f4"),
            },
            contract,
        )
    for invalid in (
        np.full(2, 0.1, dtype=np.float64),
        np.full(2, 0.1, dtype=">f4"),
    ):
        with pytest.raises(TypeError, match="little-endian float32"):
            runner.build_native_poisson_pmfs({"A_M": invalid}, contract)
    with pytest.raises(ValueError, match="no lambda floor or fallback"):
        build_lambda_pmfs(
            ["A_M"],
            pd.Index(["s1"]),
            mutsig_dir,
            None,
            1,
            production_contract=contract,
        )
    production_kwargs = {
        "allow_cbase_fallback": False,
        "require_all_features": True,
        "require_all_samples": True,
        "lambda_floor": None,
        "production_contract": contract,
    }
    with pytest.raises(ValueError, match="observed support drifted"):
        build_lambda_pmfs(
            ["A_M"],
            pd.Index(["s1"]),
            mutsig_dir,
            None,
            0,
            **production_kwargs,
        )
    with pytest.raises(ValueError, match="patient axis"):
        build_lambda_pmfs(
            ["A_M"],
            pd.Index(["absent"]),
            mutsig_dir,
            None,
            1,
            **production_kwargs,
        )
    with pytest.raises(ValueError, match="natively cover"):
        build_lambda_pmfs(
            ["MISSING_M"],
            pd.Index(["s1"]),
            mutsig_dir,
            None,
            1,
            **production_kwargs,
        )


def test_frozen_mutsig_replay_is_byte_contract_exact_and_rejects_drift(tmp_path):
    paths = _write_inputs(tmp_path)
    contract = runner.build_cohort_contract(paths, "CHOL", top_k=3)
    counts = runner._read_counts(  # noqa: SLF001
        paths.source_root / "CHOL" / "count_matrix.csv",
    )
    features = contract["features"]
    live = runner._task_pmfs(  # noqa: SLF001
        paths,
        runner.Task("CHOL", "mutsig"),
        counts.loc[:, features],
        features,
        contract=contract,
    )
    frozen_counts, frozen = runner._load_frozen_scientific_inputs(  # noqa: SLF001
        contract,
        "mutsig",
    )

    assert frozen_counts.equals(counts)
    assert {
        feature: [list(row.items()) for row in rows] for feature, rows in frozen.items()
    } == {
        feature: [list(row.items()) for row in rows] for feature, rows in live.items()
    }

    drifted = copy.deepcopy(contract)
    drifted["mutsig_pmf_contract"]["inclusive_support_k"] += 1
    with pytest.raises(ValueError, match="support contract drifted"):
        runner._load_frozen_scientific_inputs(drifted, "mutsig")  # noqa: SLF001

    page_drifted = copy.deepcopy(contract)
    page_drifted["mutsig_pmf_contract"]["effect_pages"] = {"M": 1, "N": 0}
    with pytest.raises(ValueError, match="support contract drifted"):
        runner._load_frozen_scientific_inputs(  # noqa: SLF001
            page_drifted,
            "mutsig",
        )


def test_v3_runner_rejects_v2_observed_max_mutsig_contract(tmp_path):
    paths = _write_inputs(tmp_path)
    contract = runner.build_cohort_contract(paths, "CHOL", top_k=3)
    contract["schema_version"] = "2.0.0"
    contract["mutsig_pmf_contract"] = {
        "lambda_floor": None,
        "native_lambda_only": True,
        "poisson_count_keys": [0, 1],
        "selected_observed_count_max": 1,
        "selected_observed_count_min": 0,
        "truncated_pmf_renormalized": True,
    }
    contract.pop("mutsig_pmf_storage_contract")

    with pytest.raises(ValueError, match="schema version"):
        runner._require_tested_family_contract(  # noqa: SLF001
            contract,
            require_signed_k500=False,
        )


def test_v3_runner_rejects_v2_task_manifest_without_tail_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _paths, task_dir, contract = _execute_tiny_cbase_task(tmp_path, monkeypatch)
    manifest_path = task_dir / "task_manifest.json"
    current_manifest = runner._read_json(manifest_path)  # noqa: SLF001
    manifest = copy.deepcopy(current_manifest)
    manifest["schema_version"] = "2.0.0"
    manifest["mutsig_pmf_contract"] = {
        "lambda_floor": None,
        "native_lambda_only": True,
        "poisson_count_keys": [0, 0],
        "selected_observed_count_max": 0,
        "selected_observed_count_min": 0,
        "truncated_pmf_renormalized": True,
    }
    manifest.pop("mutsig_pmf_storage_contract")
    manifest_path.write_bytes(runner._canonical_json(manifest) + b"\n")  # noqa: SLF001

    with pytest.raises(ValueError, match="schema version"):
        runner.validate_task_output(task_dir, contract, bmr="cbase")

    storage_drift = copy.deepcopy(current_manifest)
    storage_drift["mutsig_pmf_storage_contract"]["dense_probability_bytes"] += 8
    manifest_path.write_bytes(
        runner._canonical_json(storage_drift) + b"\n",  # noqa: SLF001
    )
    with pytest.raises(ValueError, match="PMF storage provenance"):
        runner.validate_task_output(task_dir, contract, bmr="cbase")


def test_production_mutsig_storage_receipt_rejects_axis_or_estimate_drift(tmp_path):
    paths = _write_inputs(tmp_path)
    contract = runner.build_cohort_contract(paths, "CHOL", top_k=3)
    contract["mutsig_pmf_storage_contract"]["dense_probability_bytes"] += 8

    with pytest.raises(ValueError, match="storage estimate drifted"):
        runner._require_tested_family_contract(  # noqa: SLF001
            contract,
            require_signed_k500=False,
        )


def test_production_mutsig_storage_estimate_is_safe_for_ucec_sized_axes():
    storage = runner.estimate_native_poisson_pmf_storage(500, 515, 107)

    assert storage["dense_probability_bytes"] == 222_480_000
    assert (
        storage["estimated_peak_numeric_array_bytes"] < runner.PRIOR_TASK_PEAK_RSS_BYTES
    )
    assert storage["legacy_per_probability_dict_entries_materialized"] is False
    assert storage["container_overhead"].startswith("excluded-platform-dependent")
    assert set(storage) == {
        "container_overhead",
        "contract",
        "dense_probability_bytes",
        "estimated_peak_numeric_array_bytes",
        "estimated_persistent_numeric_array_bytes",
        "feature_count",
        "feature_list_count",
        "inclusive_support_k",
        "legacy_per_probability_dict_entries_materialized",
        "normalizer_array_bytes",
        "per_feature_matrix_bytes",
        "probability_dtype",
        "probability_value_count",
        "row_mapping_view_count",
        "sample_count",
        "scope",
        "selected_native_rate_bytes",
        "support_vector_bytes",
    }


def test_cohort_contract_hashes_exact_native_universe_and_mapping(tmp_path):
    paths = _write_inputs(tmp_path)

    contract = runner.build_cohort_contract(paths, "CHOL", top_k=3)

    assert contract["features"] == ["A_M", "A_N", "B_M"]
    assert contract["tested_family"] == runner.asdict(
        runner._implemented_tested_family(3),  # noqa: SLF001
    )
    assert contract["feature_policy"] == {
        "feature_ranking": runner.TESTED_FAMILY_FEATURE_RANKING,
        "mutsig_cbase_feature_fallback": False,
        "observation_support": runner.OBSERVATION_SUPPORT_UNIVERSE,
        "provider_support": runner.TESTED_FAMILY_PROVIDER_SUPPORT,
        "tie_break": runner.TESTED_FAMILY_TIE_BREAK,
    }
    assert contract["pair_policy"]["row_count"] == 2
    assert contract["pair_policy"]["same_base_pairs_excluded"] == 1
    assert contract["pair_policy"]["pair_construction"] == (
        runner.TESTED_FAMILY_PAIR_CONSTRUCTION
    )
    assert contract["pair_policy"]["same_base_missense_nonsense"] == (
        runner.TESTED_FAMILY_SAME_BASE_POLICY
    )
    assert contract["pair_policy"]["epsilon_pretest_filter"] == "none"
    assert contract["pair_policy"]["marginal_effect_pretest_filter"] == "none"
    assert contract["samples"]["matched_samples"] == 4
    assert contract["samples"]["authoritative_samples"] == 4
    assert contract["samples"]["extra_mutsig_samples"] == 0
    assert contract["samples"]["exact_order_match"]
    assert contract["samples"]["contract"] == runner.SAMPLE_AXIS_CONTRACT
    assert contract["samples"]["cohort_mean_fallback_samples"] == 0
    assert contract["feature_policy"]["mutsig_cbase_feature_fallback"] is False
    mutsig_pmf = contract["mutsig_pmf_contract"]
    assert mutsig_pmf["observed_kmax"] == 1
    assert mutsig_pmf["lambda_floor"] is None
    assert mutsig_pmf["max_selected_native_lambda"] == float(np.float32(0.1))
    assert mutsig_pmf["max_selected_native_lambda_float32_le_hex"] == "cdcccc3d"
    assert mutsig_pmf["inclusive_support_k"] > mutsig_pmf["observed_kmax"]
    assert mutsig_pmf["support_rule"] == runner.PRODUCTION_POISSON_SUPPORT_RULE
    assert mutsig_pmf["tail_tolerance"] == 1e-12
    assert mutsig_pmf["normalization"] == runner.PRODUCTION_POISSON_NORMALIZATION
    assert mutsig_pmf["feature_fallback"] is False
    assert mutsig_pmf["sample_fallback"] is False
    assert mutsig_pmf["worst_discarded_tail_probability"] <= 1e-12
    assert mutsig_pmf["predecessor_tail_probability"] > 1e-12
    assert mutsig_pmf["tail_criterion_binds"] is True
    assert mutsig_pmf["tensor_dtype"] == "little-endian-float32"
    assert mutsig_pmf["tensor_order"] == "Fortran-(gene,patient,effect)"
    assert mutsig_pmf["effect_pages"] == {"M": 0, "N": 1}
    storage = contract["mutsig_pmf_storage_contract"]
    assert storage["feature_count"] == 3
    assert storage["sample_count"] == 4
    assert storage["legacy_per_probability_dict_entries_materialized"] is False
    assert len(contract["inputs"]["mutsig"]["files"]["lambda"]["sha256"]) == 64
    assert contract["inputs"]["mutsig"]["receipt"]["upstream_commit"] == (
        runner.MUTSIG_UPSTREAM_COMMIT
    )
    assert "receipt" in contract["inputs"]["mutsig"]["files"]
    assert contract["inputs"]["mutsig"]["tensor_encoding"] == {
        "contract": runner.MUTSIG_NATIVE_FWRITE_ENDIAN_CONTRACT,
        "dtype": "<f4",
        "effect_pages": {"M": 0, "N": 1},
        "layout_canary": runner.MUTSIG_TENSOR_LAYOUT_CANARY,
        "observed_consumer_sys_byteorder": "little",
        "order": "Fortran-(gene,patient,effect)",
        "producer_fwrite_byte_order": "native",
        "read_only": True,
        "required_consumer_sys_byteorder": "little",
    }
    assert (
        contract["inputs"]["sample_axis"]["sha256"]
        == (contract["inputs"]["mutsig"]["receipt"]["sample_axis_sha256"])
    )
    assert contract["inputs"]["mutsig"]["receipt"]["canonical_maf_binding"] == {
        "status": runner.MUTSIG_MAF_BINDING_STATUS,
        "required_before_production": runner.MUTSIG_MAF_BINDING_REQUIREMENT,
    }
    for bmr in runner.BMRS:
        support = contract["observed_count_support_audit"][bmr]
        assert support["zero_support_feature_samples"] == 0
        assert support["pairs"]["full_sample_support"] == 2


def _replacement_scientific_bytes(
    snapshot: runner.CohortScientificSnapshot,
    name: str,
) -> bytes:
    original = snapshot.files[name].content
    if name == "mutsig_lambda":
        values = np.frombuffer(original, dtype="<f4").copy()
        assert len(values) > 1
        assert float(values.max()) == float(np.float32(0.1))
        values[0] = np.float32(0.05)
        replacement = values.astype("<f4", copy=False).tobytes()
        assert float(np.frombuffer(replacement, dtype="<f4").max()) == float(
            np.float32(0.1),
        )
        return replacement
    if name == "counts":
        return original.replace(b",1,", b",0,", 1)
    if name in {"cbase", "dig"}:
        return original.replace(b"0.7", b"0.6", 1)
    if name == "mutsig_patients":
        return original.replace(b"s1\ns2\n", b"s2\ns1\n", 1)
    if name == "mutsig_receipt":
        marker = b"runtime_sha256\t"
        start = original.index(marker) + len(marker)
        replacement = bytearray(original)
        replacement[start : start + 64] = b"f" * 64
        return bytes(replacement)
    raise AssertionError(name)


@pytest.mark.parametrize(
    "name",
    [
        "counts",
        "cbase",
        "dig",
        "mutsig_patients",
        "mutsig_receipt",
        "mutsig_lambda",
    ],
)
def test_contract_rejects_parse_vs_record_scientific_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    """Every computation and record uses one snapshot; final paths must persist."""
    paths = _write_inputs(tmp_path)
    original_builder = runner._build_cohort_scientific_snapshot  # noqa: SLF001

    def snapshot_then_replace(**kwargs):
        snapshot = original_builder(**kwargs)
        target = snapshot.files[name].path
        replacement = _replacement_scientific_bytes(snapshot, name)
        assert len(replacement) == len(snapshot.files[name].content)
        candidate = target.with_name(f".{target.name}.replacement")
        candidate.write_bytes(replacement)
        candidate.replace(target)
        return snapshot

    monkeypatch.setattr(
        runner,
        "_build_cohort_scientific_snapshot",
        snapshot_then_replace,
    )

    with pytest.raises(ValueError, match="immutable snapshot"):
        runner._ensure_contract(paths, "CHOL", top_k=3)  # noqa: SLF001

    assert not (paths.output_root / "contracts" / "CHOL.json").exists()


def test_contract_rejects_same_bytes_persistent_inode_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_inputs(tmp_path)
    original_builder = runner._build_cohort_scientific_snapshot  # noqa: SLF001

    def snapshot_then_replace(**kwargs):
        snapshot = original_builder(**kwargs)
        frozen = snapshot.files["counts"]
        candidate = frozen.path.with_name(".count_matrix.same-bytes")
        candidate.write_bytes(frozen.content)
        candidate.replace(frozen.path)
        return snapshot

    monkeypatch.setattr(
        runner,
        "_build_cohort_scientific_snapshot",
        snapshot_then_replace,
    )

    with pytest.raises(ValueError, match="changed or was replaced"):
        runner.build_cohort_contract(paths, "CHOL", top_k=3)


def test_contract_publication_rechecks_snapshot_after_build_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_inputs(tmp_path)
    original_builder = runner.build_cohort_contract

    def build_then_replace(*args, **kwargs):
        contract = original_builder(*args, **kwargs)
        count_path = Path(contract["inputs"]["counts"]["path"])
        candidate = count_path.with_name(".count_matrix.prepublication")
        candidate.write_bytes(count_path.read_bytes())
        candidate.replace(count_path)
        return contract

    monkeypatch.setattr(runner, "build_cohort_contract", build_then_replace)

    with pytest.raises(ValueError, match="path identity changed"):
        runner._ensure_contract(paths, "CHOL", top_k=3)  # noqa: SLF001

    assert not (paths.output_root / "contracts" / "CHOL.json").exists()


def test_contract_computation_never_reopens_or_reparses_snapshot_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_inputs(tmp_path)
    scientific_paths = {
        paths.source_root / "CHOL" / "count_matrix.csv",
        paths.source_root / "CHOL" / "sample_axis.txt",
        paths.source_root / "CHOL" / "bmr_pmfs.csv",
        paths.source_root / "CHOL" / "bmr_pmfs.dig.csv",
        paths.mutsig_root / "CHOL" / "persample_meta.txt",
        paths.mutsig_root / "CHOL" / "persample_genes.txt",
        paths.mutsig_root / "CHOL" / "persample_patients.txt",
        paths.mutsig_root / "CHOL" / "persample_lambda.f32",
        paths.mutsig_root / "CHOL" / "persample_receipt.tsv",
    }
    original_read = runner._read_secure_regular_with_stat  # noqa: SLF001
    original_visible_read = runner._read_visible_regular_with_stat  # noqa: SLF001
    read_counts = dict.fromkeys(scientific_paths, 0)

    def counted_read(path, *, label):
        if path in read_counts:
            read_counts[path] += 1
        return original_read(path, label=label)

    def counted_visible_read(path, *, label):
        if path in read_counts:
            read_counts[path] += 1
        return original_visible_read(path, label=label)

    def forbidden_path_parser(*_args, **_kwargs):
        msg = "path-based parser reopened a scientific snapshot"
        raise AssertionError(msg)

    monkeypatch.setattr(runner, "_read_secure_regular_with_stat", counted_read)
    monkeypatch.setattr(runner, "_read_visible_regular_with_stat", counted_visible_read)
    for name in (
        "_file_record",
        "_load_strict_pmfs",
        "_read_authoritative_sample_axis",
        "_read_axis",
        "_read_counts",
        "_read_mutsig_metadata",
        "_read_mutsig_receipt",
    ):
        monkeypatch.setattr(runner, name, forbidden_path_parser)

    contract = runner.build_cohort_contract(paths, "CHOL", top_k=3)

    assert contract["features"] == ["A_M", "A_N", "B_M"]
    assert set(read_counts.values()) == {2}


def test_cohort_snapshot_contract_replays_deterministically(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)

    first = runner.build_cohort_contract(paths, "CHOL", top_k=3)
    second = runner.build_cohort_contract(paths, "CHOL", top_k=3)

    assert runner._canonical_json(first) == runner._canonical_json(second)  # noqa: SLF001


def test_mutsig_native_fwrite_requires_little_endian_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_inputs(tmp_path)
    monkeypatch.setattr(runner.sys, "byteorder", "big")

    with pytest.raises(RuntimeError, match=r"sys\.byteorder == 'little'"):
        runner.build_cohort_contract(paths, "CHOL", top_k=3)


def test_nonuniform_tensor_canary_binds_fortran_order_and_effect_pages() -> None:
    tensor = np.empty((2, 3, 2), dtype="<f4", order="F")
    for gene_position in range(2):
        for patient_position in range(3):
            for effect_position in range(2):
                tensor[gene_position, patient_position, effect_position] = (
                    100 * gene_position + 10 * patient_position + effect_position + 0.25
                )
    parsed = runner._reshape_mutsig_lambda_bytes(  # noqa: SLF001
        tensor.tobytes(order="F"),
        {"ng": 2, "np": 3, "neff": 2},
        path=Path("canary.f32"),
    )
    rates = runner._selected_mutsig_native_rates(  # noqa: SLF001
        parsed,
        ("G0", "G1"),
        ("P0", "P1", "P2"),
        ("P2", "P0"),
        ("G1_M", "G0_N"),
    )

    assert parsed.flags.f_contiguous
    assert not parsed.flags.writeable
    assert rates["G1_M"].tolist() == [120.25, 100.25]
    assert rates["G0_N"].tolist() == [21.25, 1.25]
    assert not rates["G1_M"].flags.writeable
    assert runner._require_mutsig_tensor_layout_canary() == (  # noqa: SLF001
        runner.MUTSIG_TENSOR_LAYOUT_CANARY
    )


def test_nonuniform_tensor_canary_rejects_m_n_page_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "MUTSIG_EFFECT_INDEX", {"M": 1, "N": 0})

    with pytest.raises(RuntimeError, match="layout canary failed"):
        runner._require_mutsig_tensor_layout_canary()  # noqa: SLF001


def test_standalone_mutsig_entrypoint_fails_before_legacy_output(
    tmp_path: Path,
) -> None:
    root = tmp_path / "output"

    with pytest.raises(RuntimeError, match=r"deprecated.*production"):
        mutsig_lambda_co.run(
            "CHOL",
            root,
            tmp_path / "mutsig",
            100,
            "legacy",
        )

    assert not root.exists()


def test_cohort_contract_rejects_extra_or_reordered_mutsig_samples(tmp_path):
    paths = _write_inputs(tmp_path)
    mutsig_dir = paths.mutsig_root / "CHOL"
    patients = ["s1", "s2", "s3", "s4", "s5"]
    (mutsig_dir / "persample_patients.txt").write_text(
        "\n".join(patients) + "\n",
        encoding="utf-8",
    )
    (mutsig_dir / "persample_meta.txt").write_text(
        "ng\t3\nnp\t5\nneff\t2\n",
        encoding="utf-8",
    )
    np.full((3, 5, 2), 0.1, dtype="<f4").ravel(order="F").tofile(
        mutsig_dir / "persample_lambda.f32",
    )

    with pytest.raises(ValueError, match="exactly equal"):
        runner.build_cohort_contract(paths, "CHOL", top_k=3)

    patients = ["s2", "s1", "s3", "s4"]
    (mutsig_dir / "persample_patients.txt").write_text(
        "\n".join(patients) + "\n",
        encoding="utf-8",
    )
    (mutsig_dir / "persample_meta.txt").write_text(
        "ng\t3\nnp\t4\nneff\t2\n",
        encoding="utf-8",
    )
    np.full((3, 4, 2), 0.1, dtype="<f4").ravel(order="F").tofile(
        mutsig_dir / "persample_lambda.f32",
    )

    with pytest.raises(ValueError, match="same_set=True"):
        runner.build_cohort_contract(paths, "CHOL", top_k=3)


def test_cohort_contract_requires_current_mutsig_receipt_and_all_sidecars(
    tmp_path,
) -> None:
    paths = _write_inputs(tmp_path)
    mutsig_dir = paths.mutsig_root / "CHOL"
    receipt_path = mutsig_dir / "persample_receipt.tsv"
    receipt_path.unlink()

    with pytest.raises(FileNotFoundError):
        runner.build_cohort_contract(paths, "CHOL", top_k=3)

    _write_mutsig_receipt(mutsig_dir)
    lambda_path = mutsig_dir / "persample_lambda.f32"
    lambda_bytes = lambda_path.read_bytes()
    tampered_lambda = bytearray(lambda_bytes)
    tampered_lambda[0] ^= 1
    lambda_path.write_bytes(tampered_lambda)
    with pytest.raises(ValueError, match=r"hash does not match lambda"):
        runner.build_cohort_contract(paths, "CHOL", top_k=3)

    lambda_path.write_bytes(lambda_bytes)
    _write_mutsig_receipt(mutsig_dir)
    receipt = receipt_path.read_text(encoding="utf-8")
    receipt_path.write_text(
        receipt.replace(
            "lambda_sha256\t",
            f"lambda_sha256\t{'0' * 64}\nignored\t",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="wrong fields"):
        runner.build_cohort_contract(paths, "CHOL", top_k=3)

    _write_mutsig_receipt(mutsig_dir)
    receipt_path.write_text(
        receipt_path.read_text(encoding="utf-8").replace(
            next(
                line
                for line in receipt_path.read_text(encoding="utf-8").splitlines()
                if line.startswith("patch_sha256\t")
            ),
            f"patch_sha256\t{'0' * 64}",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"patch_sha256.*tracked source"):
        runner.build_cohort_contract(paths, "CHOL", top_k=3)


def test_cohort_contract_binds_canonical_authoritative_sample_axis(tmp_path) -> None:
    paths = _write_inputs(tmp_path)
    cohort_dir = paths.source_root / "CHOL"
    axis_path = cohort_dir / "sample_axis.txt"
    axis_path.unlink()
    with pytest.raises(FileNotFoundError):
        runner.build_cohort_contract(paths, "CHOL", top_k=3)

    axis_path.write_bytes(b"s1\r\ns2\r\ns3\r\ns4\r\n")
    with pytest.raises(ValueError, match="LF separators"):
        runner.build_cohort_contract(paths, "CHOL", top_k=3)

    axis_path.write_text("s2\ns1\ns3\ns4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lexicographically ordered"):
        runner.build_cohort_contract(paths, "CHOL", top_k=3)

    axis_path.write_text("s1\ns2\ns3\ns4\n", encoding="utf-8")
    receipt_path = paths.mutsig_root / "CHOL" / "persample_receipt.tsv"
    receipt_lines = receipt_path.read_text(encoding="utf-8").splitlines()
    receipt_path.write_text(
        "\n".join(
            f"sample_axis_sha256\t{'0' * 64}"
            if line.startswith("sample_axis_sha256\t")
            else line
            for line in receipt_lines
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"authoritative sample_axis\.txt"):
        runner.build_cohort_contract(paths, "CHOL", top_k=3)


def test_unbound_canonical_maf_receipt_is_an_explicit_production_stop_ship(
    tmp_path,
) -> None:
    paths = _write_inputs(tmp_path)
    contract = runner.build_cohort_contract(paths, "CHOL", top_k=3)

    with pytest.raises(RuntimeError, match=r"MAF provenance stop-ship"):
        runner._require_canonical_mutsig_maf_binding(contract)  # noqa: SLF001


def test_signed_canonical_binding_is_derived_not_manually_toggled(
    tmp_path,
    monkeypatch,
) -> None:
    paths = _write_inputs(tmp_path)
    cohort_dir = paths.source_root / "CHOL"
    mutsig_receipt = runner._read_mutsig_receipt(  # noqa: SLF001
        paths.mutsig_root / "CHOL" / "persample_receipt.tsv",
    )
    binding = {
        "status": "verified",
        "contract": "test-canonical-input-contract",
        "input_approval": {
            "manifest_sha256": "a" * 64,
            "decision_digests": {
                decision_id: decision_id.lower() * 32 for decision_id in ("D1", "D2")
            },
        },
        "fit_approval": {
            "manifest_sha256": "f" * 64,
            "decision_digests": {
                decision_id: decision_id.lower() * 32
                for decision_id in ("D1", "D2", "D3", "D4", "D5", "D6")
            },
        },
        "input_manifest": {"sha256": "b" * 64},
        "cohort_manifest": {"sha256": "e" * 64},
        "canonical_maf": {"sha256": mutsig_receipt["maf_sha256"]},
        "authoritative_sample_axis": {
            "sha256": runner._sha256(cohort_dir / "sample_axis.txt"),  # noqa: SLF001
        },
    }
    provider_provenance = {
        "contract": "test-provider-receipts",
    }
    monkeypatch.setattr(
        runner,
        "_canonical_input_binding",
        lambda _paths, _cohort: binding,
    )
    monkeypatch.setattr(
        runner,
        "_verify_provider_stage_authority",
        lambda _paths, _cohort, _binding: provider_provenance,
    )

    contract = runner.build_cohort_contract(paths, "CHOL", top_k=3)

    assert contract["revision_input_authority"] == binding
    assert contract["provider_input_provenance"] == provider_provenance
    assert contract["inputs"]["mutsig"]["receipt"]["canonical_maf_binding"] == {
        "status": "verified",
        "contract": "test-canonical-input-contract",
        "input_approval_manifest_sha256": "a" * 64,
        "fit_approval_manifest_sha256": "f" * 64,
        "input_manifest_sha256": "b" * 64,
        "cohort_manifest_sha256": "e" * 64,
        "canonical_maf_sha256": mutsig_receipt["maf_sha256"],
    }


def test_signed_canonical_binding_rejects_mutsig_maf_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    paths = _write_inputs(tmp_path)
    binding = {
        "contract": "test",
        "canonical_maf": {"sha256": "0" * 64},
        "authoritative_sample_axis": {
            "sha256": runner._sha256(  # noqa: SLF001
                paths.source_root / "CHOL" / "sample_axis.txt",
            ),
        },
    }
    monkeypatch.setattr(
        runner,
        "_canonical_input_binding",
        lambda _paths, _cohort: binding,
    )

    with pytest.raises(ValueError, match="does not bind the signed canonical MAF"):
        runner.build_cohort_contract(paths, "CHOL", top_k=3)


@pytest.mark.parametrize(
    ("receipt_key", "artifact"),
    [
        ("lambda_sha256", "lambda"),
        ("meta_sha256", "metadata"),
        ("genes_sha256", "genes"),
        ("patients_sha256", "patients"),
    ],
)
def test_mutsig_receipt_binds_each_sidecar(
    tmp_path,
    receipt_key,
    artifact,
) -> None:
    paths = _write_inputs(tmp_path)
    receipt_path = paths.mutsig_root / "CHOL" / "persample_receipt.tsv"
    lines = receipt_path.read_text(encoding="utf-8").splitlines()
    receipt_path.write_text(
        "\n".join(
            f"{receipt_key}\t{'0' * 64}"
            if line.startswith(f"{receipt_key}\t")
            else line
            for line in lines
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=rf"hash does not match {artifact}"):
        runner.build_cohort_contract(paths, "CHOL", top_k=3)


def test_full_support_universe_skips_unsupported_high_count_feature(tmp_path):
    paths = _write_inputs(tmp_path)
    cohort_dir = paths.source_root / "CHOL"
    counts = pd.read_csv(cohort_dir / "count_matrix.csv", index_col=0)
    counts.loc["s1", "A_M"] = 2
    counts.to_csv(cohort_dir / "count_matrix.csv")
    cbase = pd.read_csv(cohort_dir / "bmr_pmfs.csv", index_col=0)
    cbase.loc["A_M"] = [1.0, 0.0, 0.0, 0.0]
    cbase.to_csv(cohort_dir / "bmr_pmfs.csv")

    contract = runner.build_cohort_contract(paths, "CHOL", top_k=3)
    exclusion = next(
        item
        for item in contract["full_support_universe"]["excluded_features"]
        if item["feature"] == "A_M"
    )

    assert contract["features"] == ["A_N", "B_M", "C_N"]
    assert exclusion["reasons"] == [
        {"provider": "cbase", "reason": "zero_observation_support"},
    ]
    assert (
        contract["observed_count_support_audit"]["cbase"][
            "zero_support_feature_samples"
        ]
        == 0
    )


def test_full_support_universe_excludes_exact_zero_mutsig_rate(tmp_path):
    paths = _write_inputs(tmp_path)
    cohort_dir = paths.source_root / "CHOL"
    mutsig_dir = paths.mutsig_root / "CHOL"
    counts = pd.read_csv(cohort_dir / "count_matrix.csv", index_col=0)
    counts.loc["s1", "A_M"] = 2
    counts.to_csv(cohort_dir / "count_matrix.csv")
    lambdas = np.full((3, 4, 2), 0.1, dtype="<f4")
    lambdas[0, :, :] = 0
    lambdas.ravel(order="F").tofile(mutsig_dir / "persample_lambda.f32")
    _write_mutsig_receipt(mutsig_dir)

    contract = runner.build_cohort_contract(paths, "CHOL", top_k=3)
    exclusion = next(
        item
        for item in contract["full_support_universe"]["excluded_features"]
        if item["feature"] == "A_M"
    )

    assert contract["features"] == ["A_N", "B_M", "C_N"]
    assert exclusion["reasons"] == [
        {"provider": "mutsig", "reason": "zero_observation_support"},
    ]


def test_corrected_lrt_contract_is_a_hard_launch_gate(monkeypatch):
    monkeypatch.delattr(runner.interaction_module, "LRT_CONTRACT", raising=False)

    with pytest.raises(RuntimeError, match="K=500 launch blocked"):
        runner._require_corrected_lrt()  # noqa: SLF001


def test_pair_fit_contract_is_a_hard_launch_gate(monkeypatch):
    monkeypatch.setattr(
        runner.interaction_module,
        "PAIR_FIT_CONTRACT",
        "wrong-fit-contract",
    )

    with pytest.raises(RuntimeError, match="PAIR_FIT_CONTRACT"):
        runner._require_corrected_lrt()  # noqa: SLF001


def test_pair_fit_kkt_tolerance_is_a_hard_launch_gate(monkeypatch):
    monkeypatch.setattr(
        runner.interaction_module,
        "PAIR_FIT_KKT_TOL",
        runner.REQUIRED_PAIR_FIT_KKT_TOL * 10,
    )

    with pytest.raises(RuntimeError, match="PAIR_FIT_KKT_TOL"):
        runner._require_corrected_lrt()  # noqa: SLF001


@pytest.mark.parametrize(
    ("attribute", "required"),
    [
        ("PAIR_FIT_MAX_ITER", runner.REQUIRED_PAIR_FIT_MAX_ITER),
        (
            "PAIR_IDENTIFIABILITY_RTOL",
            runner.REQUIRED_PAIR_IDENTIFIABILITY_RTOL,
        ),
        ("PAIR_SIMPLEX_TOL", runner.REQUIRED_PAIR_SIMPLEX_TOL),
        ("LRT_NESTEDNESS_TOL", runner.REQUIRED_LRT_NESTEDNESS_TOL),
    ],
)
def test_pair_numeric_tolerances_are_hard_launch_gates(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    required: float,
) -> None:
    monkeypatch.setattr(runner.interaction_module, attribute, required * 10)

    with pytest.raises(RuntimeError, match=attribute):
        runner._require_corrected_lrt()  # noqa: SLF001


def test_effect_identifiability_contract_is_a_hard_launch_gate(monkeypatch) -> None:
    monkeypatch.setattr(
        runner.interaction_module,
        "PAIR_EFFECT_IDENTIFIABILITY_CONTRACT",
        "wrong-effect-identifiability-contract",
    )

    with pytest.raises(RuntimeError, match="PAIR_EFFECT_IDENTIFIABILITY_CONTRACT"):
        runner._require_corrected_lrt()  # noqa: SLF001


def test_effect_identifiability_statuses_are_hard_launch_gates(monkeypatch) -> None:
    monkeypatch.setattr(
        runner.interaction_module,
        "PAIR_EFFECT_RANK_DEFICIENT_STATUS",
        "wrong-rank-status",
    )

    with pytest.raises(RuntimeError, match="identifiability statuses"):
        runner._require_corrected_lrt()  # noqa: SLF001


def test_pair_simplex_boundary_is_exact() -> None:
    inside_simplex = [
        0.25,
        0.25,
        0.25,
        0.25 + (runner.REQUIRED_PAIR_SIMPLEX_TOL / 2),
    ]
    outside_simplex = [
        0.25,
        0.25,
        0.25,
        0.25 + (runner.REQUIRED_PAIR_SIMPLEX_TOL * 2),
    ]
    assert runner._pair_simplex_is_valid(inside_simplex)  # noqa: SLF001
    assert not runner._pair_simplex_is_valid(outside_simplex)  # noqa: SLF001


def test_rho_contract_and_tolerance_are_hard_launch_gates(monkeypatch):
    monkeypatch.setattr(
        runner.interaction_module,
        "RHO_CONTRACT",
        "wrong-rho-contract",
    )

    with pytest.raises(RuntimeError, match="RHO_CONTRACT"):
        runner._require_corrected_lrt()  # noqa: SLF001

    monkeypatch.setattr(
        runner.interaction_module,
        "RHO_CONTRACT",
        runner.REQUIRED_RHO_CONTRACT,
    )
    monkeypatch.setattr(
        runner.interaction_module,
        "UNDEFINED_RHO_LRT_TOL",
        runner.REQUIRED_UNDEFINED_RHO_LRT_TOL * 10,
    )

    with pytest.raises(RuntimeError, match="UNDEFINED_RHO_LRT_TOL"):
        runner._require_corrected_lrt()  # noqa: SLF001


def test_output_semantic_contracts_are_hard_launch_gates(monkeypatch):
    monkeypatch.setattr(
        runner.interaction_module,
        "CONTINGENCY_TABLE_CONTRACT",
        "wrong-contingency-contract",
    )
    with pytest.raises(RuntimeError, match="CONTINGENCY_TABLE_CONTRACT"):
        runner._require_corrected_lrt()  # noqa: SLF001

    monkeypatch.setattr(
        runner.interaction_module,
        "CONTINGENCY_TABLE_CONTRACT",
        runner.REQUIRED_CONTINGENCY_TABLE_CONTRACT,
    )
    monkeypatch.setattr(
        runner.interaction_module,
        "LOG_ODDS_RATIO_CONTRACT",
        "wrong-log-odds-contract",
    )
    with pytest.raises(RuntimeError, match="LOG_ODDS_RATIO_CONTRACT"):
        runner._require_corrected_lrt()  # noqa: SLF001


def test_pairwise_rho_validation_is_fail_closed():
    pair = ("A_M", "B_M")
    independent_taus = [0.25, 0.25, 0.25, 0.25]
    runner._validate_pairwise_rho(  # noqa: SLF001
        "0.0",
        independent_taus,
        0.0,
        pair,
        effect_identifiable=True,
    )

    for invalid_rho in ("", "nan", "inf", "0.5"):
        with pytest.raises(ValueError, match="rho"):
            runner._validate_pairwise_rho(  # noqa: SLF001
                invalid_rho,
                independent_taus,
                0.0,
                pair,
                effect_identifiable=True,
            )

    tiny_taus = [1.0, 1e-200, 1e-200, 0.0]
    tiny_rho = runner.interaction_module.compute_marshall_olkin_rho(tiny_taus)
    assert tiny_rho is not None
    runner._validate_pairwise_rho(  # noqa: SLF001
        str(tiny_rho),
        tiny_taus,
        0.0,
        pair,
        effect_identifiable=True,
    )
    for corrupted_tiny_rho in ("0.0", str(-tiny_rho)):
        with pytest.raises(ValueError, match="rho"):
            runner._validate_pairwise_rho(  # noqa: SLF001
                corrupted_tiny_rho,
                tiny_taus,
                0.0,
                pair,
                effect_identifiable=True,
            )

    degenerate_taus = [1.0, 0.0, 0.0, 0.0]
    runner._validate_pairwise_rho(  # noqa: SLF001
        "",
        degenerate_taus,
        0.0,
        pair,
        effect_identifiable=True,
    )
    with pytest.raises(ValueError, match="undefined-rho boundary"):
        runner._validate_pairwise_rho(  # noqa: SLF001
            "",
            degenerate_taus,
            runner.REQUIRED_UNDEFINED_RHO_LRT_TOL * 2,
            pair,
            effect_identifiable=True,
        )
    runner._validate_pairwise_rho(  # noqa: SLF001
        "",
        independent_taus,
        1.0,
        pair,
        effect_identifiable=False,
    )
    with pytest.raises(ValueError, match="must not report rho"):
        runner._validate_pairwise_rho(  # noqa: SLF001
            "0.0",
            independent_taus,
            1.0,
            pair,
            effect_identifiable=False,
        )


def test_gene_observation_support_contract_is_a_hard_launch_gate(monkeypatch):
    monkeypatch.setattr(
        runner.interaction_module,
        "LRT_CONTRACT",
        runner.REQUIRED_LRT_CONTRACT,
        raising=False,
    )
    monkeypatch.setattr(
        runner.gene_module,
        "OBSERVATION_SUPPORT_CONTRACT",
        "wrong-contract",
    )

    with pytest.raises(RuntimeError, match="OBSERVATION_SUPPORT_CONTRACT"):
        runner._require_corrected_lrt()  # noqa: SLF001


def test_task_completion_is_atomic_validated_and_resumable(tmp_path, monkeypatch):
    paths = _write_inputs(tmp_path)
    paths.output_root.mkdir()
    monkeypatch.setattr(
        runner.interaction_module,
        "LRT_CONTRACT",
        runner.REQUIRED_LRT_CONTRACT,
        raising=False,
    )
    monkeypatch.setattr(
        runner.gene_module,
        "OBSERVATION_SUPPORT_CONTRACT",
        runner.REQUIRED_GENE_SUPPORT_CONTRACT,
        raising=False,
    )
    task = runner.Task("CHOL", "cbase")

    state = runner.execute_task(paths, task, nice_increment=0, top_k=3)
    final_dir = paths.output_root / "tasks" / "CHOL" / "cbase"
    contract = runner._read_json(  # noqa: SLF001
        paths.output_root / "contracts" / "CHOL.json",
    )

    assert state == "completed"
    assert not any((paths.output_root / "work").glob("CHOL/cbase.*"))
    assert runner.validate_task_output(final_dir, contract)["pairs"] == 2
    manifest = runner._read_json(final_dir / "task_manifest.json")  # noqa: SLF001
    usage = manifest["resource_usage"]
    assert usage["elapsed_seconds"] > 0
    assert usage["peak_rss"]["bytes"] > 0
    assert usage["peak_rss"]["platform"] in {"darwin", "linux"}
    assert usage["peak_rss"]["source"] == (
        "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss"
    )
    assert manifest["sample_axis_contract"] == runner.SAMPLE_AXIS_CONTRACT
    assert (
        manifest["contingency_table_contract"]
        == runner.REQUIRED_CONTINGENCY_TABLE_CONTRACT
    )
    pairwise = pd.read_csv(final_dir / "pairwise_interaction_results.csv")
    asymmetric = pairwise.loc[
        (pairwise["Gene A"] == "A_N") & (pairwise["Gene B"] == "B_M")
    ].iloc[0]
    assert [
        asymmetric["_00_"],
        asymmetric["_10_"],
        asymmetric["_01_"],
        asymmetric["_11_"],
    ] == [1, 1, 0, 2]
    manifest["rho_contract"] = "wrong-rho-contract"
    manifest_path = final_dir / "task_manifest.json"
    manifest_path.write_text(runner._canonical_json(manifest).decode() + "\n")  # noqa: SLF001
    with pytest.raises(ValueError, match="statistical-contract provenance"):
        runner.validate_task_output(final_dir, contract)
    manifest["rho_contract"] = runner.REQUIRED_RHO_CONTRACT
    manifest_path.write_text(runner._canonical_json(manifest).decode() + "\n")  # noqa: SLF001
    assert runner.execute_task(paths, task, nice_increment=0, top_k=3) == (
        "already-complete"
    )

    pairwise_path = final_dir / "pairwise_interaction_results.csv"
    with pairwise_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["A_M", "B_M", *([0] * (len(runner.PAIRWISE_COLUMNS) - 2))])
    with pytest.raises(runner.SealedFitError, match="pair-output-invalid"):
        runner.validate_task_output(final_dir, contract)


def test_production_mutsig_tail_contract_runs_end_to_end_on_tiny_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_inputs(tmp_path)
    paths.output_root.mkdir()
    _enable_tiny_fit_contracts(monkeypatch)
    task = runner.Task("CHOL", "mutsig")

    assert runner.execute_task(paths, task, nice_increment=0, top_k=3) == "completed"
    task_dir = paths.output_root / "tasks" / "CHOL" / "mutsig"
    contract = runner._read_json(  # noqa: SLF001
        paths.output_root / "contracts" / "CHOL.json",
    )
    manifest = runner._read_json(task_dir / "task_manifest.json")  # noqa: SLF001

    assert runner.validate_task_output(task_dir, contract, bmr="mutsig")["pairs"] == 2
    assert manifest["schema_version"] == "3.0.0"
    assert manifest["mutsig_pmf_contract"] == contract["mutsig_pmf_contract"]
    assert (
        manifest["mutsig_pmf_storage_contract"]
        == contract["mutsig_pmf_storage_contract"]
    )


def test_execute_task_detects_staged_directory_inode_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_inputs(tmp_path)
    paths.output_root.mkdir()
    _enable_tiny_fit_contracts(monkeypatch)
    original_writer = runner._write_pairwise_results_at  # noqa: SLF001
    moved_staging = tmp_path / "moved-staging"

    def write_then_swap(directory_fd, name, genes, features):
        rows = original_writer(directory_fd, name, genes, features)
        staging_parent = paths.output_root / "work" / "CHOL"
        staged = next(staging_parent.iterdir())
        staged.rename(moved_staging)
        staged.mkdir()
        return rows

    monkeypatch.setattr(runner, "_write_pairwise_results_at", write_then_swap)

    with pytest.raises(runner.SealedFitError, match="task-execution-failed"):
        runner.execute_task(
            paths,
            runner.Task("CHOL", "cbase"),
            nice_increment=0,
            top_k=3,
        )

    assert not (paths.output_root / "tasks" / "CHOL" / "cbase").exists()
    replacement = next((paths.output_root / "work" / "CHOL").iterdir())
    assert list(replacement.iterdir()) == []
    assert (moved_staging / "single_gene_results.csv").is_file()
    assert (moved_staging / "pairwise_interaction_results.csv").is_file()


def test_execute_task_detects_output_root_ancestor_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_inputs(tmp_path)
    paths.output_root.mkdir()
    _enable_tiny_fit_contracts(monkeypatch)
    original_writer = runner._write_pairwise_results_at  # noqa: SLF001
    moved_root = tmp_path / "moved-run"

    def write_then_swap(directory_fd, name, genes, features):
        rows = original_writer(directory_fd, name, genes, features)
        paths.output_root.rename(moved_root)
        paths.output_root.mkdir()
        return rows

    monkeypatch.setattr(runner, "_write_pairwise_results_at", write_then_swap)

    with pytest.raises(runner.SealedFitError, match="task-execution-failed"):
        runner.execute_task(
            paths,
            runner.Task("CHOL", "cbase"),
            nice_increment=0,
            top_k=3,
        )

    assert not (paths.output_root / "tasks" / "CHOL" / "cbase").exists()
    failed_attempts = list((moved_root / "work" / "CHOL").iterdir())
    assert len(failed_attempts) == 1
    assert {path.name for path in failed_attempts[0].iterdir()} == {
        "failure_manifest.json",
    }


def test_execute_task_detects_visible_destination_swap_after_final_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A validated dirfd cannot authorize a different visible task directory."""
    paths = _write_inputs(tmp_path)
    paths.output_root.mkdir()
    _enable_tiny_fit_contracts(monkeypatch)
    final_dir = paths.output_root / "tasks" / "CHOL" / "cbase"
    moved_validated = tmp_path / "moved-validated-task"
    original_validate = runner.validate_task_output
    swapped = False

    def validate_then_swap(task_dir, contract, **kwargs):
        nonlocal swapped
        result = original_validate(task_dir, contract, **kwargs)
        if Path(task_dir) == final_dir and not swapped:
            swapped = True
            final_dir.rename(moved_validated)
            final_dir.mkdir()
            (final_dir / "attacker-marker").write_text("attacker\n", encoding="utf-8")
        return result

    monkeypatch.setattr(runner, "validate_task_output", validate_then_swap)

    with pytest.raises(runner.SealedFitError, match="task-execution-failed"):
        runner.execute_task(
            paths,
            runner.Task("CHOL", "cbase"),
            nice_increment=0,
            top_k=3,
        )

    assert swapped
    assert (final_dir / "attacker-marker").read_text(encoding="utf-8") == "attacker\n"
    assert (moved_validated / "task_manifest.json").is_file()


def test_execute_task_detects_in_place_file_change_after_final_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final success must replay the exact pairwise inode validated for return."""
    paths = _write_inputs(tmp_path)
    paths.output_root.mkdir()
    _enable_tiny_fit_contracts(monkeypatch)
    final_dir = paths.output_root / "tasks" / "CHOL" / "cbase"
    pairwise_path = final_dir / "pairwise_interaction_results.csv"
    original_validate = runner.validate_task_output
    mutated = False
    original_inode: int | None = None

    def validate_then_mutate(task_dir, contract, **kwargs):
        nonlocal mutated, original_inode
        result = original_validate(task_dir, contract, **kwargs)
        if Path(task_dir) == final_dir and not mutated:
            mutated = True
            original_inode = pairwise_path.stat().st_ino
            attacked = bytearray(pairwise_path.read_bytes())
            attacked[0] ^= 1
            with pairwise_path.open("r+b") as handle:
                handle.write(attacked)
                handle.flush()
                os.fsync(handle.fileno())
        return result

    monkeypatch.setattr(runner, "validate_task_output", validate_then_mutate)

    with pytest.raises(runner.SealedFitError, match="task-execution-failed"):
        runner.execute_task(
            paths,
            runner.Task("CHOL", "cbase"),
            nice_increment=0,
            top_k=3,
        )

    assert mutated
    assert pairwise_path.stat().st_ino == original_inode


def test_execute_task_already_complete_detects_in_place_change_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume success must remain bound to the exact validated task-file bytes."""
    paths = _write_inputs(tmp_path)
    paths.output_root.mkdir()
    _enable_tiny_fit_contracts(monkeypatch)
    task = runner.Task("CHOL", "cbase")
    assert runner.execute_task(paths, task, nice_increment=0, top_k=3) == "completed"
    final_dir = paths.output_root / "tasks" / "CHOL" / "cbase"
    pairwise_path = final_dir / "pairwise_interaction_results.csv"
    original_validate = runner.validate_task_output
    mutated = False
    original_inode = pairwise_path.stat().st_ino

    def validate_then_mutate(task_dir, contract, **kwargs):
        nonlocal mutated
        result = original_validate(task_dir, contract, **kwargs)
        if Path(task_dir) == final_dir and not mutated:
            mutated = True
            attacked = bytearray(pairwise_path.read_bytes())
            attacked[0] ^= 1
            with pairwise_path.open("r+b") as handle:
                handle.write(attacked)
                handle.flush()
                os.fsync(handle.fileno())
        return result

    monkeypatch.setattr(runner, "validate_task_output", validate_then_mutate)

    with pytest.raises(ValueError, match=r"pinned descriptor replay|bytes changed"):
        runner.execute_task(paths, task, nice_increment=0, top_k=3)

    assert mutated
    assert pairwise_path.stat().st_ino == original_inode


def test_execute_task_destination_race_never_replaces_racing_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_inputs(tmp_path)
    paths.output_root.mkdir()
    _enable_tiny_fit_contracts(monkeypatch)
    original_rename = runner._rename_exclusive_at  # noqa: SLF001

    def race_destination(
        source_parent_fd,
        source_name,
        destination_parent_fd,
        destination_name,
    ):
        if destination_name == "cbase":
            os.mkdir(destination_name, dir_fd=destination_parent_fd)
        original_rename(
            source_parent_fd,
            source_name,
            destination_parent_fd,
            destination_name,
        )

    monkeypatch.setattr(runner, "_rename_exclusive_at", race_destination)

    with pytest.raises(runner.SealedFitError, match="task-execution-failed"):
        runner.execute_task(
            paths,
            runner.Task("CHOL", "cbase"),
            nice_increment=0,
            top_k=3,
        )

    racing = paths.output_root / "tasks" / "CHOL" / "cbase"
    assert racing.is_dir()
    assert list(racing.iterdir()) == []
    failed_attempts = list((paths.output_root / "work" / "CHOL").iterdir())
    assert len(failed_attempts) == 1
    assert {path.name for path in failed_attempts[0].iterdir()} == {
        "failure_manifest.json",
    }


@pytest.mark.parametrize(
    ("column", "attack"),
    [
        ("Tau_1X", "margin"),
        ("Tau_X1", "margin"),
        ("Likelihood Ratio", "negative-lrt"),
        ("Alternative Log Likelihood", "negative-raw-delta"),
    ],
)
def test_pair_seal_rejects_tau_margin_and_signed_lrt_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    column: str,
    attack: str,
) -> None:
    _paths, task_dir, contract = _execute_tiny_cbase_task(tmp_path, monkeypatch)
    pairwise_path = task_dir / "pairwise_interaction_results.csv"
    pairwise = pd.read_csv(pairwise_path)
    if attack == "margin":
        original = pairwise.loc[0, column]
        pairwise.loc[0, column] = 0.01 if pd.isna(original) else float(original) + 0.01
    elif attack == "negative-lrt":
        pairwise.loc[0, column] = -1e-12
    else:
        pairwise.loc[0, "Likelihood Ratio"] = 0.0
        pairwise.loc[0, column] = (
            float(pairwise.loc[0, "Null Log Likelihood"])
            - runner.REQUIRED_LRT_NESTEDNESS_TOL
        )
    pairwise.to_csv(pairwise_path, index=False)

    with pytest.raises(runner.SealedFitError, match="pair-output-invalid"):
        runner.validate_task_output(
            task_dir,
            contract,
            bmr="cbase",
        )


def test_pair_seal_rejects_internally_consistent_likelihood_shift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frozen inputs, not self-consistent output fields, determine pair LLs."""
    _paths, task_dir, contract = _execute_tiny_cbase_task(tmp_path, monkeypatch)
    pairwise_path = task_dir / "pairwise_interaction_results.csv"
    pairwise = pd.read_csv(pairwise_path)
    pairwise.loc[0, "Null Log Likelihood"] += 1.0
    pairwise.loc[0, "Alternative Log Likelihood"] += 1.0
    pairwise.to_csv(pairwise_path, index=False)

    with pytest.raises(runner.SealedFitError, match="pair-output-invalid"):
        runner.validate_task_output(task_dir, contract, bmr="cbase")


def test_pair_seal_never_surfaces_extra_result_row_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _paths, task_dir, contract = _execute_tiny_cbase_task(tmp_path, monkeypatch)
    sentinel = "SENTINEL_EXTRA_PAIR_VALUE"
    pairwise_path = task_dir / "pairwise_interaction_results.csv"
    pairwise = pd.read_csv(pairwise_path)
    extra = pairwise.iloc[0].copy()
    extra["Gene A"] = sentinel
    pairwise.loc[len(pairwise)] = extra
    pairwise.to_csv(pairwise_path, index=False)

    with pytest.raises(runner.SealedFitError) as caught:
        runner.validate_task_output(task_dir, contract, bmr="cbase")

    surfaced = str(caught.value) + "".join(
        traceback.format_exception(
            type(caught.value),
            caught.value,
            caught.value.__traceback__,
        ),
    )
    assert caught.value.code == "pair-output-invalid"
    assert sentinel not in surfaced


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("Fit Iterations", runner.REQUIRED_PAIR_FIT_MAX_ITER + 1),
        ("Effect Identifiability", "sentinel-identifiability-status"),
    ],
)
def test_pair_seal_rejects_impossible_fit_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    column: str,
    value: object,
) -> None:
    _paths, task_dir, contract = _execute_tiny_cbase_task(tmp_path, monkeypatch)
    pairwise_path = task_dir / "pairwise_interaction_results.csv"
    pairwise = pd.read_csv(pairwise_path)
    pairwise.loc[0, column] = value
    pairwise.to_csv(pairwise_path, index=False)

    with pytest.raises(runner.SealedFitError, match="pair-output-invalid"):
        runner.validate_task_output(task_dir, contract, bmr="cbase")


def test_pair_seal_rejects_nonzero_gain_at_zero_iterations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _paths, task_dir, contract = _execute_tiny_cbase_task(tmp_path, monkeypatch)
    pairwise_path = task_dir / "pairwise_interaction_results.csv"
    pairwise = pd.read_csv(pairwise_path)
    pairwise.loc[0, "Fit Iterations"] = 0
    pairwise.loc[0, "Fit Last LL Gain"] = 1e-15
    pairwise.to_csv(pairwise_path, index=False)

    with pytest.raises(runner.SealedFitError, match="pair-output-invalid"):
        runner.validate_task_output(task_dir, contract, bmr="cbase")


@pytest.mark.parametrize(
    "column",
    ["Tau_1X", "Tau_X1", "Rho", "Log Odds Ratio", "Wald Statistic"],
)
def test_rank_deficient_pair_seal_requires_every_effect_field_to_be_blank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    column: str,
) -> None:
    _paths, task_dir, contract = _execute_tiny_cbase_task(tmp_path, monkeypatch)
    pairwise_path = task_dir / "pairwise_interaction_results.csv"
    pairwise = pd.read_csv(pairwise_path)
    row_index = pairwise.index[
        pairwise["Effect Identifiability"]
        == runner.REQUIRED_PAIR_EFFECT_RANK_DEFICIENT_STATUS
    ][0]
    assert pd.isna(pairwise.loc[row_index, column])
    pairwise.loc[row_index, column] = 0.0
    pairwise.to_csv(pairwise_path, index=False)

    with pytest.raises(runner.SealedFitError, match="pair-output-invalid"):
        runner.validate_task_output(task_dir, contract, bmr="cbase")


def test_rank_deficient_pair_seal_enforces_deterministic_tau_tie_break(
    tmp_path: Path,
) -> None:
    features = ["A_M", "B_M"]
    counts = pd.DataFrame(
        {feature: [1, 1] for feature in features},
        index=["s1", "s2"],
    )
    pmfs = {feature: {0: 0.5, 1: 0.5} for feature in features}
    genes = runner._build_genes(counts, features, pmfs)  # noqa: SLF001
    runner.estimate_pi_for_each_gene(list(genes.values()))
    pairwise_path = tmp_path / "rank-deficient-pairwise.csv"
    assert (
        runner._write_pairwise_results(  # noqa: SLF001
            pairwise_path,
            genes,
            features,
        )
        == 1
    )
    pairwise = pd.read_csv(pairwise_path)
    assert (
        pairwise.loc[0, "Effect Identifiability"]
        == runner.REQUIRED_PAIR_EFFECT_RANK_DEFICIENT_STATUS
    )
    assert pairwise.loc[0, ["Tau_00", "Tau_10", "Tau_01", "Tau_11"]].tolist() == [
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    pairwise.loc[0, ["Tau_00", "Tau_10"]] = [0.0, 1.0]
    pairwise.to_csv(pairwise_path, index=False)
    contract = {
        "features": features,
        "samples": {
            "count": len(counts),
            "ordered_ids_sha256": runner._sequence_sha256(counts.index),  # noqa: SLF001
        },
        "pair_policy": runner._pair_contract(features),  # noqa: SLF001
    }

    with pytest.raises(ValueError, match="deterministic tie-break"):
        runner._validate_pairwise_output_impl(  # noqa: SLF001
            pairwise_path.read_bytes(),
            contract,
            counts,
            genes,
        )


def test_underflow_pair_output_is_rank_unverified_and_effect_blind(
    tmp_path: Path,
) -> None:
    features = ["A_M", "B_M"]
    counts = pd.DataFrame({"A_M": [1], "B_M": [1]}, index=["s1"])
    pmfs = {feature: {0: 1e-200, 1: 1.0} for feature in features}
    genes = runner._build_genes(counts, features, pmfs)  # noqa: SLF001
    runner.estimate_pi_for_each_gene(list(genes.values()))
    pairwise_path = tmp_path / "underflow-pairwise.csv"
    assert (
        runner._write_pairwise_results(  # noqa: SLF001
            pairwise_path,
            genes,
            features,
        )
        == 1
    )
    pairwise = pd.read_csv(pairwise_path)
    row = pairwise.iloc[0]
    assert row["Effect Identifiability"] == runner.REQUIRED_PAIR_EFFECT_UNDERFLOW_STATUS
    for column in ("Tau_1X", "Tau_X1", "Rho", "Log Odds Ratio", "Wald Statistic"):
        assert pd.isna(row[column])
    contract = {
        "features": features,
        "samples": {
            "count": 1,
            "ordered_ids_sha256": runner._sequence_sha256(counts.index),  # noqa: SLF001
        },
        "pair_policy": runner._pair_contract(features),  # noqa: SLF001
    }
    assert (
        runner._validate_pairwise_output_impl(  # noqa: SLF001
            pairwise_path.read_bytes(),
            contract,
            counts,
            genes,
        )
        == 1
    )


@pytest.mark.parametrize(
    ("column", "attack"),
    [
        ("Gene Name", "text"),
        ("Pi", "numeric"),
        ("Log Odds Ratio", "text"),
        ("Likelihood Ratio", "numeric"),
        ("Observed Mutations", "numeric"),
        ("Expected Mutations", "numeric"),
        ("Obs. - Exp. Mutations", "numeric"),
        ("MLE Algorithm", "text"),
        ("MLE Iterations", "numeric"),
        ("MLE Bracket Width", "numeric"),
        ("MLE Fixed-Point Residual", "numeric"),
        ("MLE KKT Residual", "numeric"),
        ("MLE Log Likelihood", "numeric"),
        ("MLE Converged", "boolean"),
        ("Single-Gene LRT Status", "text"),
        ("LRT Contract", "text"),
        ("Single-Gene Count Contract", "text"),
    ],
)
def test_single_gene_seal_recomputes_every_scientific_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    column: str,
    attack: str,
) -> None:
    _paths, task_dir, contract = _execute_tiny_cbase_task(tmp_path, monkeypatch)
    single_path = task_dir / "single_gene_results.csv"
    single = pd.read_csv(single_path)
    if attack == "numeric":
        single[column] = single[column].astype(float)
        single.loc[0, column] = float(single.loc[0, column]) + 0.125
    elif attack == "boolean":
        single.loc[0, column] = False
    else:
        single[column] = single[column].astype(object)
        single.loc[0, column] = "sentinel-result-value"
    single.to_csv(single_path, index=False)

    with pytest.raises(runner.SealedFitError, match="single-row-invalid"):
        runner.validate_task_output(
            task_dir,
            contract,
            bmr="cbase",
        )


@pytest.mark.parametrize("column", ["Observed Mutations", "MLE Iterations"])
def test_single_gene_seal_rejects_fractional_discrete_values_inside_float_atol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    column: str,
) -> None:
    _paths, task_dir, contract = _execute_tiny_cbase_task(tmp_path, monkeypatch)
    single_path = task_dir / "single_gene_results.csv"
    single = pd.read_csv(single_path)
    single[column] = single[column].astype(float)
    single.loc[0, column] = float(single.loc[0, column]) + (
        runner.REQUIRED_OUTPUT_RECOMPUTATION_ATOL / 2
    )
    single.to_csv(single_path, index=False)

    with pytest.raises(runner.SealedFitError, match="single-row-invalid"):
        runner.validate_task_output(task_dir, contract, bmr="cbase")


@pytest.mark.parametrize(
    "column",
    [
        "MLE Bracket Width",
        "MLE Fixed-Point Residual",
        "MLE KKT Residual",
    ],
)
def test_single_gene_seal_rejects_sub_atol_mle_certificate_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    column: str,
) -> None:
    _paths, task_dir, contract = _execute_tiny_cbase_task(tmp_path, monkeypatch)
    single_path = task_dir / "single_gene_results.csv"
    single = pd.read_csv(single_path)
    row_index = 0
    if column == "MLE Bracket Width":
        boundary_rows = single.index[single[column].astype(float) == 0]
        assert not boundary_rows.empty
        row_index = int(boundary_rows[0])
    original = float(single.loc[row_index, column])
    attacked = original + runner.REQUIRED_OUTPUT_RECOMPUTATION_ATOL / 2
    assert attacked != original
    assert abs(attacked - original) < runner.REQUIRED_OUTPUT_RECOMPUTATION_ATOL
    single.loc[row_index, column] = attacked
    single.to_csv(single_path, index=False)

    with pytest.raises(runner.SealedFitError, match="single-row-invalid"):
        runner.validate_task_output(task_dir, contract, bmr="cbase")


@pytest.mark.parametrize("attack", ["missing", "extra", "reordered"])
def test_single_gene_seal_requires_exact_closed_column_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    _paths, task_dir, contract = _execute_tiny_cbase_task(tmp_path, monkeypatch)
    single_path = task_dir / "single_gene_results.csv"
    single = pd.read_csv(single_path)
    if attack == "missing":
        single = single.drop(columns=[runner.SINGLE_GENE_RESULT_COLUMNS[-1]])
    elif attack == "extra":
        single["Unexpected Column"] = "sentinel"
    else:
        columns = list(single.columns)
        columns[0], columns[1] = columns[1], columns[0]
        single = single.loc[:, columns]
    single.to_csv(single_path, index=False)

    with pytest.raises(runner.SealedFitError, match="single-schema-invalid"):
        runner.validate_task_output(
            task_dir,
            contract,
            bmr="cbase",
        )


def test_single_gene_seal_accepts_and_recomputes_heterogeneous_sample_pmfs(
    tmp_path: Path,
) -> None:
    counts = pd.DataFrame({"G_M": [1, 2]}, index=["s0", "s1"])
    sample_pmfs = [
        {0: 0.5, 1: 0.5},
        {0: 0.25, 1: 0.25, 2: 0.5},
    ]
    gene = runner.Gene(
        name="G_M",
        samples=counts.index,
        counts=counts["G_M"].to_numpy(),
        bmr_pmf=sample_pmfs,
    )
    gene.estimate_pi_with_mle()
    output = tmp_path / "single.csv"
    runner.create_single_gene_results(
        [gene],
        output.as_posix(),
        cbase_phi_vals_present=False,
    )
    raw = output.read_bytes()
    row = pd.read_csv(output).iloc[0]
    assert row["Observed Mutations"] == 3
    assert row["Expected Mutations"] == pytest.approx(1.75)
    assert row["Obs. - Exp. Mutations"] == pytest.approx(1.25)
    assert row["Single-Gene Count Contract"] == runner.SINGLE_GENE_COUNT_CONTRACT

    assert (
        runner._validate_single_gene_output(  # noqa: SLF001
            raw,
            {"features": ["G_M"]},
            counts,
            {"G_M": sample_pmfs},
        )
        == 1
    )
    corrupted = pd.read_csv(output)
    corrupted.loc[0, "Expected Mutations"] = 0.875
    corrupted.to_csv(output, index=False)
    with pytest.raises(runner.SealedFitError, match="single-row-invalid"):
        runner._validate_single_gene_output(  # noqa: SLF001
            output.read_bytes(),
            {"features": ["G_M"]},
            counts,
            {"G_M": sample_pmfs},
        )


def test_sealed_fit_failure_never_leaks_scientific_exception_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_inputs(tmp_path)
    paths.output_root.mkdir()
    sentinel_gene = "SENTINEL_SECRET_GENE"
    sentinel_value = "SENTINEL_FITTED_VALUE_0.987654321"
    sentinel_error = f"{sentinel_gene} {sentinel_value}"

    def fail_with_scientific_values(*_args, **_kwargs):
        raise ValueError(sentinel_error)

    monkeypatch.setattr(
        runner,
        "create_single_gene_results",
        fail_with_scientific_values,
    )

    with pytest.raises(runner.SealedFitError) as caught:
        runner.execute_task(
            paths,
            runner.Task("CHOL", "cbase"),
            nice_increment=0,
            top_k=3,
        )
    traceback.print_exception(caught.value)
    captured = capsys.readouterr()
    formatted = "".join(
        traceback.format_exception(
            type(caught.value),
            caught.value,
            caught.value.__traceback__,
        ),
    )
    failure_paths = list(
        (paths.output_root / "work" / "CHOL").glob(
            "cbase.*/failure_manifest.json",
        ),
    )
    assert len(failure_paths) == 1
    failure_raw = failure_paths[0].read_text(encoding="utf-8")
    surfaced = "\n".join(
        (
            str(caught.value),
            formatted,
            captured.out,
            captured.err,
            failure_raw,
        ),
    )
    assert sentinel_gene not in surfaced
    assert sentinel_value not in surfaced
    assert caught.value.__context__ is None
    failure = runner._read_json(failure_paths[0])  # noqa: SLF001
    assert failure["failure"] == {
        "code": "task-execution-failed",
        "phase": "fit-single-gene",
        "row_index": None,
    }
    assert "traceback" not in failure
    assert {path.name for path in failure_paths[0].parent.iterdir()} == {
        "failure_manifest.json",
    }


def test_frozen_input_read_rejects_persistent_path_swap_after_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scientific_input = tmp_path / "scientific.csv"
    original = b"original-scientific-bytes\n"
    replacement = b"replaced-scientific-bytes\n"
    assert len(original) == len(replacement)
    scientific_input.write_bytes(original)
    record = runner._file_record(scientific_input)  # noqa: SLF001
    original_fstat = runner.os.fstat
    switched = False

    def swap_after_open(descriptor):
        nonlocal switched
        observed = original_fstat(descriptor)
        if runner.stat_module.S_ISREG(observed.st_mode) and not switched:
            switched = True
            scientific_input.rename(tmp_path / "opened-original.csv")
            scientific_input.write_bytes(replacement)
        return observed

    monkeypatch.setattr(runner.os, "fstat", swap_after_open)

    with pytest.raises(ValueError, match=r"visible-entry readback|must remain"):
        runner._read_frozen_record_bytes(  # noqa: SLF001
            record,
            label="synthetic scientific input",
        )

    assert scientific_input.read_bytes() == replacement


def test_output_read_parses_and_hashes_one_descriptor_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output.csv"
    original = b"original-output-bytes\n"
    replacement = b"swapped--output-bytes\n"
    assert len(original) == len(replacement)
    output.write_bytes(original)
    directory_fd = runner._open_secure_directory(tmp_path, label="test output")  # noqa: SLF001
    original_fstat = runner.os.fstat
    switched = False

    def swap_after_open(descriptor):
        nonlocal switched
        observed = original_fstat(descriptor)
        if runner.stat_module.S_ISREG(observed.st_mode) and not switched:
            switched = True
            output.rename(tmp_path / "opened-output.csv")
            output.write_bytes(replacement)
        return observed

    monkeypatch.setattr(runner.os, "fstat", swap_after_open)
    try:
        consumed = runner._read_regular_at(  # noqa: SLF001
            directory_fd,
            output.name,
            label="synthetic output",
        )
    finally:
        runner.os.close(directory_fd)

    assert consumed == original
    assert output.read_bytes() == replacement


def test_status_is_strictly_metadata_hash_only(tmp_path, monkeypatch) -> None:
    paths = _write_inputs(tmp_path)
    paths.output_root.mkdir()
    monkeypatch.setattr(
        runner.interaction_module,
        "LRT_CONTRACT",
        runner.REQUIRED_LRT_CONTRACT,
        raising=False,
    )
    monkeypatch.setattr(
        runner.gene_module,
        "OBSERVATION_SUPPORT_CONTRACT",
        runner.REQUIRED_GENE_SUPPORT_CONTRACT,
        raising=False,
    )
    runner.execute_task(
        paths,
        runner.Task("CHOL", "cbase"),
        nice_increment=0,
        top_k=3,
    )

    def forbid_row_inspection(*_args, **_kwargs):
        msg = "status inspected scientific rows"
        raise AssertionError(msg)

    monkeypatch.setattr(runner, "validate_task_output", forbid_row_inspection)
    monkeypatch.setattr(runner, "_validate_pairwise_output", forbid_row_inspection)
    monkeypatch.setattr(runner.pd, "read_csv", forbid_row_inspection)

    status = runner._status(paths, ["CHOL"])  # noqa: SLF001

    assert status == {
        "counts": {"complete": 1, "invalid": 0, "pending": 2},
        "tasks": {
            "CHOL": {"cbase": "complete", "dig": "pending", "mutsig": "pending"},
        },
    }


def test_sealed_completion_is_whole_grid_metadata_only_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, authority = _write_completion_grid(tmp_path, monkeypatch)

    def forbid_row_inspection(*_args, **_kwargs):
        msg = "completion inspected scientific rows"
        raise AssertionError(msg)

    monkeypatch.setattr(runner, "validate_task_output", forbid_row_inspection)
    monkeypatch.setattr(runner, "_validate_pairwise_output", forbid_row_inspection)
    monkeypatch.setattr(runner.pd, "read_csv", forbid_row_inspection)

    first = runner._finalize_sealed_completion(  # noqa: SLF001
        paths,
        ["BRCA", "CHOL"],
    )
    second = runner._finalize_sealed_completion(  # noqa: SLF001
        paths,
        ["CHOL", "BRCA"],
    )

    assert first == second
    assert first["manifest"]["authority"] == authority
    assert first["manifest"]["grid"] == {
        "ordered_coordinates_sha256": runner._sequence_sha256(  # noqa: SLF001
            [
                f"{cohort}/{bmr}"
                for cohort in runner.TCGA_COHORTS
                for bmr in runner.BMRS
            ],
        ),
        "task_count": 6,
    }
    assert first["manifest"]["result_rows_opened"] is False
    assert first["manifest"]["downstream_binding"] == {
        "field": "upstream_result_manifest_sha256",
        "stage": "inspect-tcga-k500",
    }
    completion_path = paths.output_root / runner.SEALED_COMPLETION_NAME
    assert (
        first["manifest_file"]["sha256"]
        == hashlib.sha256(
            completion_path.read_bytes(),
        ).hexdigest()
    )


def test_sealed_completion_rejects_subset_and_incomplete_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _authority = _write_completion_grid(tmp_path, monkeypatch)
    completion_path = paths.output_root / runner.SEALED_COMPLETION_NAME

    with pytest.raises(ValueError, match="subset"):
        runner._finalize_sealed_completion(paths, ["CHOL"])  # noqa: SLF001
    assert not completion_path.exists()

    missing = paths.output_root / "tasks" / "BRCA" / "dig" / "task_manifest.json"
    missing.unlink()
    with pytest.raises(ValueError, match="inventory"):
        runner._finalize_sealed_completion(  # noqa: SLF001
            paths,
            list(runner.TCGA_COHORTS),
        )
    assert not completion_path.exists()


def test_sealed_completion_detects_post_seal_task_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _authority = _write_completion_grid(tmp_path, monkeypatch)
    cohorts = list(runner.TCGA_COHORTS)
    completion = runner._finalize_sealed_completion(paths, cohorts)  # noqa: SLF001
    completion_path = paths.output_root / runner.SEALED_COMPLETION_NAME
    frozen = completion_path.read_bytes()
    pairwise = (
        paths.output_root
        / "tasks"
        / "CHOL"
        / "cbase"
        / "pairwise_interaction_results.csv"
    )
    pairwise.write_bytes(pairwise.read_bytes() + b"tamper\n")

    with pytest.raises(ValueError, match="metadata/hash receipt"):
        runner._finalize_sealed_completion(paths, cohorts)  # noqa: SLF001

    assert completion_path.read_bytes() == frozen
    assert completion["manifest_file"]["sha256"] == hashlib.sha256(frozen).hexdigest()


def test_sealed_completion_never_replaces_a_racing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _authority = _write_completion_grid(tmp_path, monkeypatch)
    original_write = runner._write_bytes_atomic_at  # noqa: SLF001
    racing_payload = {"raced": True}

    def race_write(parent_fd, name, content, *, label):
        if name == runner.SEALED_COMPLETION_NAME:
            original_write(
                parent_fd,
                name,
                runner._canonical_json(racing_payload) + b"\n",  # noqa: SLF001
                label=label,
            )
        original_write(parent_fd, name, content, label=label)

    monkeypatch.setattr(runner, "_write_bytes_atomic_at", race_write)

    with pytest.raises(ValueError, match="differs from the whole grid"):
        runner._finalize_sealed_completion(  # noqa: SLF001
            paths,
            list(runner.TCGA_COHORTS),
        )

    assert (
        runner._read_json(  # noqa: SLF001
            paths.output_root / runner.SEALED_COMPLETION_NAME,
        )
        == racing_payload
    )


def test_sealed_completion_detects_output_root_ancestor_swap_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "run"
    output_root.mkdir()
    paths = runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", output_root)
    moved_root = tmp_path / "moved-run"
    original_write = runner._write_bytes_atomic_at  # noqa: SLF001

    def publish_then_swap(parent_fd, name, content, *, label):
        original_write(parent_fd, name, content, label=label)
        if name == runner.SEALED_COMPLETION_NAME:
            output_root.rename(moved_root)
            output_root.mkdir()

    monkeypatch.setattr(runner, "_write_bytes_atomic_at", publish_then_swap)

    with pytest.raises(ValueError, match="path identity changed"):
        runner._publish_sealed_completion(  # noqa: SLF001
            paths,
            {"sealed": True},
        )

    assert not (output_root / runner.SEALED_COMPLETION_NAME).exists()
    assert (moved_root / runner.SEALED_COMPLETION_NAME).is_file()


def test_sealed_completion_detects_entry_swap_after_final_readback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A readback receipt cannot authorize a replacement visible manifest inode."""
    output_root = tmp_path / "run"
    output_root.mkdir()
    paths = runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", output_root)
    completion_path = output_root / runner.SEALED_COMPLETION_NAME
    moved_completion = tmp_path / "opened-sealed-completion.json"
    original_read = runner._read_regular_at  # noqa: SLF001
    completion_reads = 0

    def read_then_swap(directory_fd, name, *, label):
        nonlocal completion_reads
        observed = original_read(directory_fd, name, label=label)
        if name == runner.SEALED_COMPLETION_NAME:
            completion_reads += 1
            if completion_reads == 2:
                completion_path.rename(moved_completion)
                completion_path.write_bytes(b'{"attacker":true}\n')
        return observed

    monkeypatch.setattr(runner, "_read_regular_at", read_then_swap)

    with pytest.raises(ValueError, match="single-link regular file"):
        runner._publish_sealed_completion(  # noqa: SLF001
            paths,
            {"sealed": True},
        )

    assert completion_reads == 2
    assert completion_path.read_bytes() == b'{"attacker":true}\n'
    assert moved_completion.read_bytes() == b'{"sealed":true}\n'


def test_sealed_completion_detects_in_place_change_after_visible_entry_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A same-inode change after first readback cannot retain its old receipt."""
    output_root = tmp_path / "run"
    output_root.mkdir()
    paths = runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", output_root)
    completion_path = output_root / runner.SEALED_COMPLETION_NAME
    original_identity = runner._require_regular_entry_identity  # noqa: SLF001
    mutated = False
    original_inode: int | None = None

    def bind_then_mutate(parent_fd, name, descriptor, *, label):
        nonlocal mutated, original_inode
        original_identity(parent_fd, name, descriptor, label=label)
        if label == "sealed completion manifest after final readback":
            mutated = True
            original_inode = completion_path.stat().st_ino
            attacked = bytearray(completion_path.read_bytes())
            attacked[2] ^= 1
            with completion_path.open("r+b") as handle:
                handle.write(attacked)
                handle.flush()
                os.fsync(handle.fileno())

    monkeypatch.setattr(runner, "_require_regular_entry_identity", bind_then_mutate)

    with pytest.raises(ValueError, match=r"pinned descriptor replay|bytes changed"):
        runner._publish_sealed_completion(  # noqa: SLF001
            paths,
            {"sealed": True},
        )

    assert mutated
    assert completion_path.stat().st_ino == original_inode


@pytest.mark.parametrize("attack", ["file-symlink", "file-hardlink", "dir-symlink"])
def test_metadata_only_task_receipt_rejects_link_attacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    paths, _authority = _write_completion_grid(tmp_path, monkeypatch)
    task = runner.Task("CHOL", "cbase")
    task_dir = paths.output_root / "tasks" / task.cohort / task.bmr
    contract = runner._read_json(  # noqa: SLF001
        paths.output_root / "contracts" / "CHOL.json",
    )
    pairwise = task_dir / "pairwise_interaction_results.csv"
    if attack == "file-symlink":
        external = tmp_path / "external-pairwise.csv"
        external.write_bytes(pairwise.read_bytes())
        pairwise.unlink()
        pairwise.symlink_to(external)
    elif attack == "file-hardlink":
        (tmp_path / "second-link.csv").hardlink_to(pairwise)
    else:
        moved = tmp_path / "moved-task"
        task_dir.rename(moved)
        task_dir.symlink_to(moved, target_is_directory=True)

    with pytest.raises((OSError, RuntimeError, ValueError)):
        runner._metadata_task_receipt(task_dir, contract, task)  # noqa: SLF001


def test_exclusive_task_publish_never_replaces_racing_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "source.txt").write_bytes(b"source\n")
    (destination / "destination.txt").write_bytes(b"destination\n")

    with pytest.raises(FileExistsError, match="target already exists"):
        runner._rename_exclusive(source, destination)  # noqa: SLF001

    assert (source / "source.txt").read_bytes() == b"source\n"
    assert (destination / "destination.txt").read_bytes() == b"destination\n"


def test_exclusive_task_publish_atomically_moves_absent_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "source.txt").write_bytes(b"source\n")

    runner._rename_exclusive(source, destination)  # noqa: SLF001

    assert not source.exists()
    assert (destination / "source.txt").read_bytes() == b"source\n"


def test_atomic_json_detects_output_root_ancestor_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "run"
    artifact_parent = output_root / "contracts"
    artifact_parent.mkdir(parents=True)
    moved_root = tmp_path / "moved-run"
    original_rename = runner._rename_exclusive_at  # noqa: SLF001

    def swap_then_publish(
        source_parent,
        source_name,
        destination_parent,
        destination_name,
    ):
        if destination_name == "CHOL.json":
            output_root.rename(moved_root)
            (output_root / "contracts").mkdir(parents=True)
        original_rename(
            source_parent,
            source_name,
            destination_parent,
            destination_name,
        )

    monkeypatch.setattr(runner, "_rename_exclusive_at", swap_then_publish)

    with pytest.raises(ValueError, match="path identity changed"):
        runner._write_json_atomic(  # noqa: SLF001
            artifact_parent / "CHOL.json",
            {"cohort": "CHOL"},
        )

    assert not (artifact_parent / "CHOL.json").exists()
    assert (moved_root / "contracts" / "CHOL.json").is_file()


def test_exclusive_directory_publish_detects_parent_inode_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_parent = tmp_path / "work"
    destination_parent = tmp_path / "tasks"
    source = source_parent / "attempt"
    destination = destination_parent / "cbase"
    source.mkdir(parents=True)
    destination_parent.mkdir()
    (source / "result.csv").write_bytes(b"sealed\n")
    moved_parent = tmp_path / "moved-tasks"
    original_rename = runner._rename_exclusive_at  # noqa: SLF001

    def swap_then_publish(source_fd, source_name, destination_fd, destination_name):
        destination_parent.rename(moved_parent)
        destination_parent.mkdir()
        original_rename(source_fd, source_name, destination_fd, destination_name)

    monkeypatch.setattr(runner, "_rename_exclusive_at", swap_then_publish)

    with pytest.raises(ValueError, match="path identity changed"):
        runner._rename_exclusive(source, destination)  # noqa: SLF001

    assert not destination.exists()
    assert (moved_parent / "cbase" / "result.csv").read_bytes() == b"sealed\n"


@pytest.mark.parametrize("attack", ["symlink", "hardlink"])
def test_sealed_completion_rejects_linked_destination_without_overwrite(
    tmp_path: Path,
    attack: str,
) -> None:
    output_root = tmp_path / "run"
    output_root.mkdir()
    paths = runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", output_root)
    payload = {"sealed": True}
    expected = runner._canonical_json(payload) + b"\n"  # noqa: SLF001
    external = tmp_path / "external.json"
    external.write_bytes(expected)
    destination = output_root / runner.SEALED_COMPLETION_NAME
    if attack == "symlink":
        destination.symlink_to(external)
    else:
        destination.hardlink_to(external)

    with pytest.raises((OSError, ValueError)):
        runner._publish_sealed_completion(paths, payload)  # noqa: SLF001

    assert external.read_bytes() == expected


def test_atomic_file_publication_fsyncs_file_and_parent_then_reads_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "artifacts"
    parent.mkdir()
    parent_fd = runner._open_secure_directory(parent, label="test parent")  # noqa: SLF001
    fsync_modes = []
    original_fsync = runner.os.fsync

    def record_fsync(descriptor):
        fsync_modes.append(os.fstat(descriptor).st_mode)
        original_fsync(descriptor)

    monkeypatch.setattr(runner.os, "fsync", record_fsync)
    try:
        runner._write_bytes_atomic_at(  # noqa: SLF001
            parent_fd,
            "receipt.json",
            b"sealed\n",
            label="test receipt",
        )
        observed = runner._read_regular_at(  # noqa: SLF001
            parent_fd,
            "receipt.json",
            label="test receipt",
        )
    finally:
        os.close(parent_fd)

    assert observed == b"sealed\n"
    assert any(runner.stat_module.S_ISREG(mode) for mode in fsync_modes)
    assert any(runner.stat_module.S_ISDIR(mode) for mode in fsync_modes)


def test_atomic_file_publication_rejects_failed_final_readback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "artifacts"
    parent.mkdir()
    parent_fd = runner._open_secure_directory(parent, label="test parent")  # noqa: SLF001
    original_read = runner._read_regular_at  # noqa: SLF001

    def corrupt_readback(directory_fd, name, *, label):
        observed = original_read(directory_fd, name, label=label)
        return b"corrupt\n" if name == "receipt.json" else observed

    monkeypatch.setattr(runner, "_read_regular_at", corrupt_readback)
    try:
        with pytest.raises(ValueError, match="exact descriptor readback"):
            runner._write_bytes_atomic_at(  # noqa: SLF001
                parent_fd,
                "receipt.json",
                b"sealed\n",
                label="test receipt",
            )
    finally:
        os.close(parent_fd)


def test_task_validation_rejects_resource_provenance_drift(tmp_path, monkeypatch):
    paths = _write_inputs(tmp_path)
    paths.output_root.mkdir()
    monkeypatch.setattr(
        runner.interaction_module,
        "LRT_CONTRACT",
        runner.REQUIRED_LRT_CONTRACT,
        raising=False,
    )
    monkeypatch.setattr(
        runner.gene_module,
        "OBSERVATION_SUPPORT_CONTRACT",
        runner.REQUIRED_GENE_SUPPORT_CONTRACT,
        raising=False,
    )
    task = runner.Task("CHOL", "cbase")
    runner.execute_task(paths, task, nice_increment=0, top_k=3)
    final_dir = paths.output_root / "tasks" / "CHOL" / "cbase"
    contract = runner._read_json(  # noqa: SLF001
        paths.output_root / "contracts" / "CHOL.json",
    )
    manifest_path = final_dir / "task_manifest.json"
    manifest = runner._read_json(manifest_path)  # noqa: SLF001
    resource_usage = manifest.pop("resource_usage")
    manifest_path.write_text(runner._canonical_json(manifest).decode() + "\n")  # noqa: SLF001

    with pytest.raises(TypeError, match="lacks resource usage provenance"):
        runner.validate_task_output(final_dir, contract)

    manifest["resource_usage"] = resource_usage
    elapsed_seconds = resource_usage["elapsed_seconds"]
    resource_usage["elapsed_seconds"] = 0
    manifest_path.write_text(runner._canonical_json(manifest).decode() + "\n")  # noqa: SLF001

    with pytest.raises(ValueError, match="elapsed-time/RSS provenance"):
        runner.validate_task_output(final_dir, contract)

    resource_usage["elapsed_seconds"] = elapsed_seconds
    resource_usage["peak_rss"]["native_unit"] = "MB"
    manifest_path.write_text(runner._canonical_json(manifest).decode() + "\n")  # noqa: SLF001

    with pytest.raises(ValueError, match="peak-RSS provenance"):
        runner.validate_task_output(final_dir, contract)


def test_default_concurrency_is_strictly_below_half_of_fourteen_cores():
    assert runner.safe_default_jobs(14) == 3
    assert runner.safe_default_jobs(14) < 14 / 2


def test_same_uid_machine_lease_blocks_a_second_orchestrator(tmp_path):
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    with runner._same_uid_machine_execution_lease(  # noqa: SLF001
        first_output,
    ) as lease_path:
        assert lease_path.is_file()
        assert lease_path.parent == runner.SAME_UID_MACHINE_LEASE_DIRECTORY
        with (
            pytest.raises(RuntimeError, match="this UID already holds"),
            runner._same_uid_machine_execution_lease(  # noqa: SLF001
                second_output,
            ),
        ):
            pytest.fail("a second same-UID machine lease was acquired")

    with runner._same_uid_machine_execution_lease(  # noqa: SLF001
        second_output,
    ):
        pass


def test_same_uid_machine_lease_identity_ignores_tmpdir(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path / "different-session-temp"))

    with runner._same_uid_machine_execution_lease(tmp_path / "run") as lease_path:  # noqa: SLF001
        assert lease_path == runner.SAME_UID_MACHINE_LEASE_DIRECTORY / (
            f"dialect-k500-{runner.os.getuid()}.lock"
        )


def test_same_uid_machine_lease_rejects_insecure_or_replaced_file(tmp_path):
    lease_path = tmp_path / "lease"
    lease_path.write_text("lease", encoding="utf-8")
    lease_path.chmod(0o644)
    descriptor = runner.os.open(lease_path, runner.os.O_RDWR)
    try:
        with pytest.raises(RuntimeError, match="not a stable private file"):
            runner._require_secure_lease_file(descriptor, lease_path)  # noqa: SLF001

        lease_path.chmod(0o600)
        replacement = tmp_path / "replacement"
        replacement.write_text("replacement", encoding="utf-8")
        replacement.chmod(0o600)
        replacement.replace(lease_path)
        with pytest.raises(RuntimeError, match="not a stable private file"):
            runner._require_secure_lease_file(descriptor, lease_path)  # noqa: SLF001
    finally:
        runner.os.close(descriptor)


def test_darwin_memory_pressure_parser_uses_aggregate_free_percentage():
    total, available = runner._parse_darwin_memory_pressure(  # noqa: SLF001
        "The system has 25769803776 (1572864 pages).\n"
        "System-wide memory free percentage: 39%\n",
    )

    assert total == 25769803776
    assert available == total * 39 // 100


def test_linux_meminfo_parser_uses_memavailable():
    total, available = runner._parse_linux_meminfo(  # noqa: SLF001
        "MemTotal:       24576000 kB\n"
        "MemFree:         1000000 kB\n"
        "MemAvailable:   10000000 kB\n",
    )

    assert total == 24576000 * 1024
    assert available == 10000000 * 1024


@pytest.mark.parametrize(
    "output",
    [
        "System-wide memory free percentage: 39%\n",
        "The system has 25769803776 bytes.\nSystem-wide memory free percentage: 101%\n",
    ],
)
def test_darwin_memory_pressure_parser_rejects_incomplete_or_invalid_output(output):
    with pytest.raises(RuntimeError, match="macOS memory_pressure"):
        runner._parse_darwin_memory_pressure(output)  # noqa: SLF001


@pytest.mark.parametrize(
    "content",
    [
        "MemTotal: 100 kB\nMemFree: 50 kB\n",
        "MemTotal: 100 kB\nMemAvailable: 101 kB\n",
        "MemTotal: 100 bytes\nMemAvailable: 50 bytes\n",
    ],
)
def test_linux_meminfo_parser_rejects_missing_invalid_or_wrong_unit_fields(content):
    with pytest.raises(RuntimeError, match="Linux /proc/meminfo"):
        runner._parse_linux_meminfo(content)  # noqa: SLF001


def test_live_resource_gate_requires_cpu_memory_and_disk_headroom():
    snapshot = runner.HostResourceSnapshot(
        measured_at_utc="2026-08-28T00:00:00+00:00",
        logical_cores=14,
        load_average_1m=1.0,
        total_memory_bytes=24 * 1024**3,
        available_memory_bytes=10 * 1024**3,
        free_disk_bytes=100 * 1024**3,
        cpu_source="test",
        memory_source="test",
    )

    assert runner.evaluate_host_resource_gate(snapshot, jobs=3)["passed"] is True

    low_memory = runner.HostResourceSnapshot(
        **{
            **snapshot.__dict__,
            "available_memory_bytes": 7 * 1024**3,
        },
    )
    evaluation = runner.evaluate_host_resource_gate(low_memory, jobs=3)
    assert evaluation["passed"] is False
    assert "available memory" in evaluation["reasons"][0]

    evaluation = runner.evaluate_host_resource_gate(snapshot, jobs=4)
    assert evaluation["passed"] is False
    assert "safe live cap" in evaluation["reasons"][0]


def test_live_resource_gate_reserves_planned_jobs_below_half_host() -> None:
    snapshot = runner.HostResourceSnapshot(
        measured_at_utc="2026-08-28T00:00:00+00:00",
        logical_cores=14,
        load_average_1m=4.0,
        total_memory_bytes=24 * 1024**3,
        available_memory_bytes=10 * 1024**3,
        free_disk_bytes=100 * 1024**3,
        cpu_source="test",
        memory_source="test",
    )

    boundary = runner.evaluate_host_resource_gate(snapshot, jobs=3)
    assert boundary["passed"] is False
    assert boundary["strict_half_core_limit"] == 7.0
    assert boundary["projected_load_with_planned_jobs"] == 7.0
    assert "not below half" in boundary["reasons"][0]

    below = runner.evaluate_host_resource_gate(
        runner.HostResourceSnapshot(
            **{**snapshot.__dict__, "load_average_1m": 3.999},
        ),
        jobs=3,
    )
    assert below["passed"] is True


@pytest.mark.parametrize("load_average", [float("nan"), float("inf"), -0.1])
def test_live_resource_gate_rejects_invalid_aggregate_cpu_load(
    load_average: float,
) -> None:
    snapshot = runner.HostResourceSnapshot(
        measured_at_utc="2026-08-28T00:00:00+00:00",
        logical_cores=14,
        load_average_1m=load_average,
        total_memory_bytes=24 * 1024**3,
        available_memory_bytes=10 * 1024**3,
        free_disk_bytes=100 * 1024**3,
        cpu_source="test",
        memory_source="test",
    )

    evaluation = runner.evaluate_host_resource_gate(snapshot, jobs=1)

    assert evaluation["passed"] is False
    assert "aggregate CPU load" in evaluation["reasons"][0]


def test_live_resource_gate_rejects_malformed_aggregate_readback():
    snapshot = runner.HostResourceSnapshot(
        measured_at_utc="",
        logical_cores=0,
        load_average_1m=-0.1,
        total_memory_bytes=0,
        available_memory_bytes=1,
        free_disk_bytes=-1,
        cpu_source="",
        memory_source="",
    )

    evaluation = runner.evaluate_host_resource_gate(snapshot, jobs=1)

    assert evaluation["passed"] is False
    assert evaluation["reasons"] == [
        "logical core count must be positive",
        "one-minute aggregate CPU load plus planned jobs is not below half the host",
        "total memory must be positive",
        "available memory is outside the physical-memory range",
        "free disk cannot be negative",
        "resource readback provenance is incomplete",
        "free disk is below the 2x historical-output gate",
    ]


def test_live_resource_gate_records_invalid_snapshot_before_failing(
    tmp_path,
    monkeypatch,
):
    paths = runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run")
    paths.output_root.mkdir()
    snapshot = runner.HostResourceSnapshot(
        measured_at_utc="not-a-timestamp",
        logical_cores=0,
        load_average_1m=-0.1,
        total_memory_bytes=0,
        available_memory_bytes=1,
        free_disk_bytes=-1,
        cpu_source="test",
        memory_source="test",
    )
    monkeypatch.setattr(runner, "read_host_resources", lambda _root: snapshot)

    with pytest.raises(RuntimeError, match="Live resource gate failed"):
        runner._require_live_resource_gate(  # noqa: SLF001
            paths,
            jobs=1,
            label="invalid-snapshot",
        )

    records = list((paths.output_root / "resource_readbacks").glob("*.json"))
    assert len(records) == 1
    record = runner._read_json(records[0])  # noqa: SLF001
    assert record["evaluation"]["passed"] is False
    assert (
        "resource readback provenance is incomplete" in record["evaluation"]["reasons"]
    )


@pytest.mark.parametrize(
    ("load_average", "load_state"),
    [
        (float("nan"), "nan"),
        (float("inf"), "positive-infinity"),
        (float("-inf"), "negative-infinity"),
    ],
)
def test_live_resource_gate_canonically_records_nonfinite_load_before_failing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_average: float,
    load_state: str,
) -> None:
    paths = runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run")
    paths.output_root.mkdir()
    snapshot = runner.HostResourceSnapshot(
        measured_at_utc="2026-08-28T00:00:00+00:00",
        logical_cores=14,
        load_average_1m=load_average,
        total_memory_bytes=24 * 1024**3,
        available_memory_bytes=10 * 1024**3,
        free_disk_bytes=100 * 1024**3,
        cpu_source="test",
        memory_source="test",
    )
    monkeypatch.setattr(runner, "read_host_resources", lambda _root: snapshot)

    with pytest.raises(RuntimeError, match="Live resource gate failed"):
        runner._require_live_resource_gate(  # noqa: SLF001
            paths,
            jobs=1,
            label="nonfinite-load",
        )

    records = list((paths.output_root / "resource_readbacks").glob("*.json"))
    assert len(records) == 1
    record = runner._read_json(records[0])  # noqa: SLF001
    assert record["snapshot"]["load_average_1m"] is None
    assert record["snapshot"]["load_average_1m_state"] == load_state
    assert record["evaluation"]["projected_load_with_planned_jobs"] is None


def test_resource_readback_record_cannot_be_overwritten(tmp_path, monkeypatch):
    paths = runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run")
    paths.output_root.mkdir()
    snapshot = runner.HostResourceSnapshot(
        measured_at_utc="2026-08-28T00:00:00+00:00",
        logical_cores=14,
        load_average_1m=1.0,
        total_memory_bytes=24 * 1024**3,
        available_memory_bytes=10 * 1024**3,
        free_disk_bytes=100 * 1024**3,
        cpu_source="test",
        memory_source="test",
    )

    class FixedUuid:
        hex = "fixed"

    monkeypatch.setattr(runner, "read_host_resources", lambda _root: snapshot)
    monkeypatch.setattr(runner.uuid, "uuid4", FixedUuid)
    runner._require_live_resource_gate(  # noqa: SLF001
        paths,
        jobs=1,
        label="first",
    )
    record_path = paths.output_root / "resource_readbacks" / "fixed.json"
    original = record_path.read_bytes()

    with pytest.raises(FileExistsError):
        runner._require_live_resource_gate(  # noqa: SLF001
            paths,
            jobs=1,
            label="replacement",
        )

    assert record_path.read_bytes() == original


def test_internal_task_environment_requires_exact_sealed_mapping(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(runner.os, "environ", dict(runner.SEALED_TASK_ENVIRONMENT))
    runner._require_internal_task_environment()  # noqa: SLF001

    runner.os.environ["PYTHONPATH"] = (tmp_path / "poison").as_posix()
    with pytest.raises(RuntimeError, match="PYTHONPATH"):
        runner._require_internal_task_environment()  # noqa: SLF001

    runner.os.environ["PYTHONPATH"] = runner.SEALED_TASK_ENVIRONMENT["PYTHONPATH"]
    runner.os.environ["OMP_NUM_THREADS"] = "2"
    with pytest.raises(RuntimeError, match="OMP_NUM_THREADS"):
        runner._require_internal_task_environment()  # noqa: SLF001


def test_sealed_task_environment_survives_real_python_startup(tmp_path: Path) -> None:
    completed = runner.subprocess.run(
        [
            runner.sys.executable,
            "-P",
            "-s",
            "-c",
            (
                "from analysis.run_tcga_revision_k500 import "
                "_require_internal_task_environment; "
                "_require_internal_task_environment()"
            ),
        ],
        cwd=tmp_path,
        env=dict(runner.SEALED_TASK_ENVIRONMENT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_production_python_must_match_provider_pinned_interpreter(
    tmp_path,
    monkeypatch,
) -> None:
    substituted = tmp_path / "substituted-python"
    substituted.write_bytes(b"not the pinned interpreter\n")
    substituted.chmod(0o700)
    monkeypatch.setattr(runner.sys, "executable", substituted.as_posix())

    with pytest.raises(RuntimeError, match="provider-pinned Python executable"):
        runner._frozen_python_executable()  # noqa: SLF001


def test_task_subprocess_uses_exact_sealed_environment(
    tmp_path,
    monkeypatch,
):
    paths = runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run")
    runner._write_json_atomic(  # noqa: SLF001
        paths.output_root / "contracts" / "CHOL.json",
        {"cohort": "CHOL"},
    )
    captured = {}

    def record_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return runner.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", record_run)
    monkeypatch.setenv("PYTHONPATH", (tmp_path / "poison").as_posix())
    monkeypatch.setenv(
        "DYLD_INSERT_LIBRARIES",
        (tmp_path / "poison.dylib").as_posix(),
    )

    task, return_code = runner._invoke_task(  # noqa: SLF001
        paths,
        runner.Task("CHOL", "cbase"),
        10,
    )

    assert task == runner.Task("CHOL", "cbase")
    assert return_code == 0
    assert captured["command"][0] == (
        runner.PROVIDER_CHILD_PYTHON_EXECUTABLE.resolve().as_posix()
    )
    assert captured["command"][1:4] == ["-P", "-s", "-m"]
    assert captured["env"] == runner.SEALED_TASK_ENVIRONMENT
    assert captured["env"]["PYTHONPATH"] == os.pathsep.join(
        (runner.RUNNER_REPO_ROOT.as_posix(), runner.RUNNER_SOURCE_ROOT.as_posix()),
    )
    assert captured["env"]["PYTHONPATH"] != (tmp_path / "poison").as_posix()
    assert "DYLD_INSERT_LIBRARIES" not in captured["env"]


def test_import_roots_and_hashes_are_frozen_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implementation = runner._source_snapshot(runner.RUNNER_REPO_ROOT)  # noqa: SLF001
    runner._require_pinned_import_roots(implementation)  # noqa: SLF001

    drifted = dict(implementation)
    drifted["analysis/run_tcga_revision_k500.py"] = "0" * 64
    with pytest.raises(RuntimeError, match="frozen implementation receipt"):
        runner._require_pinned_import_roots(drifted)  # noqa: SLF001

    poison = tmp_path / "dialect" / "__init__.py"
    poison.parent.mkdir()
    poison.write_text("# stale editable import\n", encoding="utf-8")
    monkeypatch.setattr(
        runner.sys.modules["dialect"],
        "__file__",
        poison.as_posix(),
    )
    with pytest.raises(RuntimeError, match="outside the pinned repository tree"):
        runner._require_pinned_import_roots(implementation)  # noqa: SLF001


def test_source_manifest_exactly_covers_pristine_recursive_local_imports() -> None:
    script = """
import json
import sys
from pathlib import Path

from analysis import run_tcga_revision_k500 as runner

root = runner.RUNNER_REPO_ROOT.resolve()
observed = {}
for module_name, module in tuple(sys.modules.items()):
    if module_name.partition(".")[0] not in {"analysis", "dialect"}:
        continue
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str):
        continue
    path = Path(raw_path).resolve()
    if path.suffix != ".py" or not path.is_relative_to(root):
        continue
    specification = getattr(module, "__spec__", None)
    canonical_name = getattr(specification, "name", None) or module_name
    observed[canonical_name] = path.relative_to(root).as_posix()
print(json.dumps(observed, sort_keys=True))
"""
    completed = subprocess.run(  # noqa: S603
        [
            runner.PROVIDER_CHILD_PYTHON_EXECUTABLE.resolve().as_posix(),
            "-P",
            "-s",
            "-c",
            script,
        ],
        cwd=runner.RUNNER_REPO_ROOT,
        env=dict(runner.SEALED_TASK_ENVIRONMENT),
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(completed.stdout)
    expected = {
        module_name: path.as_posix()
        for module_name, path in runner.EXECUTED_LOCAL_PYTHON_MODULES
    }
    expected_paths = tuple(path for _, path in runner.EXECUTED_LOCAL_PYTHON_MODULES)

    assert observed == expected
    assert (
        *expected_paths,
        *runner.NON_PYTHON_EXECUTION_SOURCE_FILES,
    ) == runner.SOURCE_FILES


def test_import_root_symlink_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linked_repo = tmp_path / "linked-repo"
    linked_repo.symlink_to(runner.RUNNER_REPO_ROOT, target_is_directory=True)
    monkeypatch.setattr(runner, "RUNNER_REPO_ROOT", linked_repo)
    monkeypatch.setattr(runner, "RUNNER_SOURCE_ROOT", linked_repo / "src")

    with pytest.raises(OSError, match=r".+"):
        runner._require_pinned_import_roots({})  # noqa: SLF001


def test_task_subprocess_propagates_every_revision_authority_argument(
    tmp_path,
    monkeypatch,
) -> None:
    paths = _with_revision_authority(
        runner.RunPaths(
            tmp_path / "source",
            tmp_path / "mutsig",
            tmp_path / "run",
        ),
        tmp_path,
    )
    runner._write_json_atomic(  # noqa: SLF001
        paths.output_root / "contracts" / "CHOL.json",
        {"cohort": "CHOL"},
    )
    captured = {}

    def record_run(command, **_kwargs):
        captured["command"] = command
        return runner.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", record_run)

    runner._invoke_task(paths, runner.Task("CHOL", "cbase"), 10)  # noqa: SLF001

    command = captured["command"]
    for flag, expected in (
        ("--provider-input-root", paths.provider_input_root.as_posix()),
        ("--expected-provider-input-manifest-sha256", "c" * 64),
        ("--canonical-input-root", paths.canonical_input_root.as_posix()),
        (
            "--input-approval-manifest",
            paths.input_approval_manifest.as_posix(),
        ),
        ("--expected-input-approval-sha256", "a" * 64),
        ("--fit-approval-manifest", paths.fit_approval_manifest.as_posix()),
        ("--expected-fit-approval-sha256", "f" * 64),
        ("--expected-canonical-input-sha256", "b" * 64),
    ):
        assert command[command.index(flag) + 1] == expected
    assert "--source-root" not in command
    assert "--mutsig-root" not in command


def test_provider_bundle_validation_uses_every_independent_anchor_and_current_env(
    tmp_path,
    monkeypatch,
) -> None:
    paths = _with_revision_authority(
        runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run"),
        tmp_path,
    )
    expected = _fake_provider_bundle(paths)
    captured = {}

    def validate(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(runner, "validate_materialized_provider_input_bundle", validate)
    runner._validated_provider_bundle_cached.cache_clear()  # noqa: SLF001

    assert runner._validated_provider_bundle(paths, fresh=True) == expected  # noqa: SLF001
    assert captured == {
        "args": (
            paths.provider_input_root,
            "c" * 64,
            paths.canonical_input_root,
            "b" * 64,
            paths.input_approval_manifest,
            "a" * 64,
        ),
        "kwargs": {"require_current_execution_environment": True},
    }


def test_provider_bundle_validation_rejects_root_substitution(
    tmp_path,
    monkeypatch,
) -> None:
    paths = _with_revision_authority(
        runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run"),
        tmp_path,
    )
    substituted = _fake_provider_bundle(paths)
    substituted["roots"] = {
        "cohorts": tmp_path / "attacker" / "cohorts",
        "mutsig": tmp_path / "attacker" / "mutsig",
    }
    monkeypatch.setattr(
        runner,
        "validate_materialized_provider_input_bundle",
        lambda *_args, **_kwargs: substituted,
    )
    runner._validated_provider_bundle_cached.cache_clear()  # noqa: SLF001

    with pytest.raises(ValueError, match="does not match the derived runner paths"):
        runner._validated_provider_bundle(paths, fresh=True)  # noqa: SLF001


def test_parent_revision_prime_replays_one_provider_bundle(
    tmp_path,
    monkeypatch,
) -> None:
    paths = _with_revision_authority(
        runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run"),
        tmp_path,
    )
    provider = _fake_provider_bundle(paths)
    canonical = {"status": "verified"}
    calls = []
    monkeypatch.setattr(
        runner,
        "_validated_provider_bundle",
        lambda _paths, *, fresh=False: calls.append(("provider", fresh)) or provider,
    )
    monkeypatch.setattr(
        runner,
        "_canonical_input_binding",
        lambda _paths, cohort: calls.append(("canonical", cohort)) or canonical,
    )
    monkeypatch.setattr(
        runner,
        "_verify_provider_stage_authority",
        lambda _paths, cohort, binding, *, bundle=None: {
            "cohort": cohort,
            "binding": binding,
            "same_bundle": bundle is provider,
        },
    )

    receipt = runner._prime_parent_revision_authority(paths, "CHOL")  # noqa: SLF001

    assert calls[0] == ("provider", False)
    assert receipt == {
        "cohort": "CHOL",
        "binding": canonical,
        "same_bundle": True,
    }


def test_runner_pins_signed_family_parser_to_actual_k500_implementation(
    tmp_path,
    monkeypatch,
):
    paths = _with_revision_authority(
        runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run"),
        tmp_path,
    )
    approval = object()
    captured = {}
    d3_checks = []
    monkeypatch.setattr(runner, "_validated_fit_approval", lambda *_args: approval)
    monkeypatch.setattr(runner, "_require_fit_stage_binding", lambda *_args: {})
    monkeypatch.setattr(
        runner,
        "_require_d3_runtime_contract",
        lambda policy, actual_paths: d3_checks.append((policy, actual_paths)),
    )

    def parse_policy(
        actual_approval,
        *,
        expected_d4_implementation,
        expected_tested_family,
    ):
        captured["approval"] = actual_approval
        captured["d4"] = expected_d4_implementation
        captured["family"] = expected_tested_family
        return SimpleNamespace(
            d5=SimpleNamespace(tested_family=runner.REQUIRED_TESTED_FAMILY),
        )

    monkeypatch.setattr(runner, "validate_revision_fit_policy", parse_policy)

    record = runner._signed_tested_family_record(paths)  # noqa: SLF001

    assert captured == {
        "approval": approval,
        "d4": runner.REQUIRED_D4_IMPLEMENTATION,
        "family": runner.REQUIRED_TESTED_FAMILY,
    }
    assert len(d3_checks) == 1
    assert d3_checks[0][1] is paths
    assert record == runner.asdict(runner.REQUIRED_TESTED_FAMILY)


def test_runner_rejects_parser_return_that_drifts_from_implemented_family(
    tmp_path,
    monkeypatch,
):
    paths = _with_revision_authority(
        runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run"),
        tmp_path,
    )
    monkeypatch.setattr(runner, "_validated_fit_approval", lambda *_args: object())
    monkeypatch.setattr(runner, "_require_fit_stage_binding", lambda *_args: {})
    monkeypatch.setattr(
        runner,
        "_require_d3_runtime_contract",
        lambda *_args: pytest.fail("D3 must not mask an invalid signed D5 family"),
    )
    monkeypatch.setattr(
        runner,
        "validate_revision_fit_policy",
        lambda *_args, **_kwargs: SimpleNamespace(
            d5=SimpleNamespace(
                tested_family=runner._implemented_tested_family(499),  # noqa: SLF001
            ),
        ),
    )

    with pytest.raises(ValueError, match="does not match the runner implementation"):
        runner._signed_tested_family_record(paths)  # noqa: SLF001


def _d3_runtime_policy_fixture(
    source_files: dict[str, str],
    reviewed_commit: str,
) -> SimpleNamespace:
    support = MutSigSupportPolicy(
        dtype="<f4",
        effect_pages=MutSigEffectPagesPolicy(M=0, N=1),
        fallback_or_floor="none",
        lambda_dtype="native-binary32",
        layout_canary=runner.MUTSIG_TENSOR_LAYOUT_CANARY,
        normalization=runner.PRODUCTION_POISSON_NORMALIZATION,
        order="Fortran-(gene,patient,effect)",
        predecessor_proof="required-when-tail-endpoint-binds",
        read_only=True,
        storage_contract=runner.PRODUCTION_POISSON_STORAGE_CONTRACT,
        support_contract=runner.PRODUCTION_POISSON_SUPPORT_CONTRACT,
        support_rule=runner.PRODUCTION_POISSON_SUPPORT_RULE,
        tail_tolerance=runner.PRODUCTION_POISSON_TAIL_TOLERANCE,
    )
    runner_path = "analysis/run_tcga_revision_k500.py"
    source_snapshot_sha256 = hashlib.sha256(
        json.dumps(
            source_files,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
    ).hexdigest()
    receipts = {
        "D4": SimpleNamespace(canonical_artifact_sha256="4" * 64),
        "D5": SimpleNamespace(canonical_artifact_sha256="5" * 64),
    }
    source_contract = {
        "association_results_read": False,
        "d4_canonical_artifact_sha256": (receipts["D4"].canonical_artifact_sha256),
        "d5_canonical_artifact_sha256": (receipts["D5"].canonical_artifact_sha256),
        "mutsig_minimal_tail_contract": runner.asdict(support),
        "reviewed_scientific_commit": reviewed_commit,
        "runner": {
            "path": runner_path,
            "sha256": source_files[runner_path],
        },
        "schema": "dialect-revision-k500-scientific-source-v1",
        "source_file_count": len(source_files),
        "source_files": source_files,
        "source_snapshot_sha256": source_snapshot_sha256,
    }
    source_contract_bytes = (
        json.dumps(
            source_contract,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    binding = D3ImplementationBinding(
        reviewed_scientific_commit=reviewed_commit,
        runner_path=runner_path,
        runner_sha256=source_files[runner_path],
        source_contract_sha256=hashlib.sha256(source_contract_bytes).hexdigest(),
        source_file_count=len(source_files),
        source_snapshot_sha256=source_snapshot_sha256,
    )
    return SimpleNamespace(
        d3=SimpleNamespace(
            mutsig_support=support,
            implementation_binding=binding,
        ),
        d5=SimpleNamespace(tested_family=runner.REQUIRED_TESTED_FAMILY),
        receipts=receipts,
    )


def _install_d3_runtime_source_fixture(
    monkeypatch: pytest.MonkeyPatch,
    source_files: dict[str, str],
    reviewed_commit: str,
) -> None:
    monkeypatch.setattr(runner, "_source_snapshot", lambda _root: source_files)
    monkeypatch.setattr(runner, "_validated_provider_bundle", lambda _paths: {})
    monkeypatch.setattr(
        runner,
        "_provider_root_receipt",
        lambda _paths, _provider: {"git_executable": {}},
    )
    monkeypatch.setattr(
        runner,
        "_git_snapshot",
        lambda _root, _git: {
            "dirty": False,
            "head": reviewed_commit,
            "status": [],
        },
    )


def test_runner_accepts_exact_d3_v2_native_support_and_source_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_files = {
        "analysis/run_tcga_revision_k500.py": "a" * 64,
        "src/dialect/data/revision_fit_policy.py": "b" * 64,
    }
    reviewed_commit = "c" * 40
    policy = _d3_runtime_policy_fixture(source_files, reviewed_commit)
    _install_d3_runtime_source_fixture(monkeypatch, source_files, reviewed_commit)

    record = runner._require_d3_runtime_contract(  # noqa: SLF001
        policy,
        _with_revision_authority(
            runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run"),
            tmp_path,
        ),
    )

    assert record["mutsig_support"] == runner.asdict(policy.d3.mutsig_support)
    assert record["source_contract_sha256"] == (
        policy.d3.implementation_binding.source_contract_sha256
    )
    assert record["tensor_encoding"]["dtype"] == "<f4"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dtype", ">f4"),
        ("effect_pages", {"M": 1, "N": 0}),
        ("layout_canary", "different-canary"),
        ("order", "C-(gene,patient,effect)"),
        ("read_only", False),
    ],
)
def test_runner_rejects_each_live_native_tensor_semantic_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    source_files = {"analysis/run_tcga_revision_k500.py": "a" * 64}
    reviewed_commit = "c" * 40
    policy = _d3_runtime_policy_fixture(source_files, reviewed_commit)
    tensor = runner._mutsig_tensor_encoding_record(read_only=True)  # noqa: SLF001
    tensor[field] = value

    def tensor_record(*, read_only: bool) -> dict[str, object]:
        assert read_only
        return tensor

    monkeypatch.setattr(
        runner,
        "_mutsig_tensor_encoding_record",
        tensor_record,
    )

    with pytest.raises(RuntimeError, match="tensor semantics"):
        runner._require_d3_runtime_contract(  # noqa: SLF001
            policy,
            _with_revision_authority(
                runner.RunPaths(
                    tmp_path / "source",
                    tmp_path / "mutsig",
                    tmp_path / "run",
                ),
                tmp_path,
            ),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dtype", ">f4"),
        ("effect_pages", MutSigEffectPagesPolicy(M=1, N=0)),
        ("fallback_or_floor", "floor"),
        ("lambda_dtype", "binary64"),
        ("layout_canary", "different-canary"),
        ("normalization", "ordinary-sum"),
        ("order", "C-(gene,patient,effect)"),
        ("predecessor_proof", "not-required"),
        ("read_only", False),
        ("storage_contract", "different-storage"),
        ("support_contract", "different-support"),
        ("support_rule", "different-rule"),
        ("tail_tolerance", 1e-9),
    ],
)
def test_runner_rejects_each_d3_v2_native_support_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    source_files = {"analysis/run_tcga_revision_k500.py": "a" * 64}
    reviewed_commit = "c" * 40
    policy = _d3_runtime_policy_fixture(source_files, reviewed_commit)
    policy.d3.mutsig_support = replace(
        policy.d3.mutsig_support,
        **{field: value},
    )
    _install_d3_runtime_source_fixture(monkeypatch, source_files, reviewed_commit)

    with pytest.raises(ValueError, match="native-tail contract"):
        runner._require_d3_runtime_contract(  # noqa: SLF001
            policy,
            _with_revision_authority(
                runner.RunPaths(
                    tmp_path / "source",
                    tmp_path / "mutsig",
                    tmp_path / "run",
                ),
                tmp_path,
            ),
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("reviewed_scientific_commit", "d" * 40, "reviewed_scientific_commit"),
        ("runner_path", "analysis/other.py", "runner_path"),
        ("runner_sha256", "d" * 64, "runner_sha256"),
        ("source_file_count", 3, "source_file_count"),
        ("source_snapshot_sha256", "d" * 64, "source_snapshot_sha256"),
        ("source_contract_sha256", "d" * 64, "source-contract digest"),
    ],
)
def test_runner_rejects_each_d3_v2_implementation_binding_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    error: str,
) -> None:
    source_files = {"analysis/run_tcga_revision_k500.py": "a" * 64}
    reviewed_commit = "c" * 40
    policy = _d3_runtime_policy_fixture(source_files, reviewed_commit)
    policy.d3.implementation_binding = replace(
        policy.d3.implementation_binding,
        **{field: value},
    )
    _install_d3_runtime_source_fixture(monkeypatch, source_files, reviewed_commit)

    with pytest.raises(ValueError, match=error):
        runner._require_d3_runtime_contract(  # noqa: SLF001
            policy,
            _with_revision_authority(
                runner.RunPaths(
                    tmp_path / "source",
                    tmp_path / "mutsig",
                    tmp_path / "run",
                ),
                tmp_path,
            ),
        )


def test_production_runner_rejects_legacy_d3_before_source_or_input_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _with_revision_authority(
        runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run"),
        tmp_path,
    )
    legacy_policy = SimpleNamespace(
        d3=SimpleNamespace(mutsig_support=None, implementation_binding=None),
        d5=SimpleNamespace(tested_family=runner.REQUIRED_TESTED_FAMILY),
    )
    monkeypatch.setattr(runner, "_validated_input_approval", lambda *_args: object())
    monkeypatch.setattr(runner, "_validated_fit_approval", lambda *_args: object())
    monkeypatch.setattr(runner, "_require_fit_stage_binding", lambda *_args: {})
    monkeypatch.setattr(
        runner,
        "validate_revision_fit_policy",
        lambda *_args, **_kwargs: legacy_policy,
    )

    def forbidden_read(*_args, **_kwargs):
        pytest.fail("legacy D3 reached scientific source or canonical input reads")

    monkeypatch.setattr(runner, "_source_snapshot", forbidden_read)
    monkeypatch.setattr(runner, "_validated_input_bundle", forbidden_read)

    with pytest.raises(ValueError, match="signed D3-v2 MutSig contract"):
        runner._canonical_input_binding(paths, "CHOL")  # noqa: SLF001


def test_frozen_cohort_authority_accepts_stage_scoped_v5_input_and_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, contract = _write_local_authority_contract(tmp_path, monkeypatch)
    input_manifest = runner._read_json(paths.input_approval_manifest)  # noqa: SLF001
    fit_manifest = runner._read_json(paths.fit_approval_manifest)  # noqa: SLF001
    assert input_manifest["schema"] == runner.STAGE_SCOPED_APPROVAL_SCHEMA
    assert [item["decision_id"] for item in input_manifest["decisions"]] == [
        "D1",
        "D2",
    ]
    assert fit_manifest["schema"] == runner.STAGE_SCOPED_APPROVAL_SCHEMA
    assert [item["decision_id"] for item in fit_manifest["decisions"]] == [
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
    ]
    canonical_validator = runner.validate_materialized_input_cohort_binding
    provider_validator = runner.validate_materialized_provider_cohort_input
    narrow_calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        runner,
        "validate_materialized_input_cohort_binding",
        lambda *args, **kwargs: (
            narrow_calls.append(("canonical", args, kwargs))
            or canonical_validator(*args, **kwargs)
        ),
    )
    monkeypatch.setattr(
        runner,
        "validate_materialized_provider_cohort_input",
        lambda *args, **kwargs: (
            narrow_calls.append(("provider", args, kwargs))
            or provider_validator(*args, **kwargs)
        ),
    )

    runner._verify_frozen_cohort_authority(paths, contract)  # noqa: SLF001

    assert [call[0] for call in narrow_calls] == ["canonical", "provider"]
    assert narrow_calls[0][2] == {"require_current_execution_environment": False}
    assert narrow_calls[1][2] == {"require_current_execution_environment": True}
    canonical_receipt = contract["revision_input_authority"]["full_validation"]
    provider_receipt = contract["provider_input_provenance"]["root_receipt"]
    assert narrow_calls[0][1][5:] == (
        canonical_receipt["receipt"],
        canonical_receipt["receipt_sha256"],
    )
    assert narrow_calls[1][1][3:5] == (
        provider_receipt["full_acceptance_receipt"],
        provider_receipt["full_acceptance_receipt_sha256"],
    )


def test_production_frozen_cohort_authority_rejects_v4_fit_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, contract = _write_local_authority_contract(
        tmp_path,
        monkeypatch,
        legacy_fit_v4=True,
    )
    input_manifest = runner._read_json(paths.input_approval_manifest)  # noqa: SLF001
    fit_manifest = runner._read_json(paths.fit_approval_manifest)  # noqa: SLF001

    assert input_manifest["schema"] == runner.STAGE_SCOPED_APPROVAL_SCHEMA
    assert fit_manifest["schema"] == runner.APPROVAL_SCHEMA
    assert [item["decision_id"] for item in fit_manifest["decisions"]] == list(
        runner.DECISION_IDS,
    )
    with pytest.raises(ValueError, match="invalid schema"):
        runner._verify_frozen_cohort_authority(paths, contract)  # noqa: SLF001


@pytest.mark.parametrize(
    (
        "schema",
        "manifest_stage",
        "decision_ids",
        "expected_stage",
        "expected_decision_ids",
        "allowed_schemas",
        "error",
    ),
    [
        (
            runner.APPROVAL_SCHEMA,
            runner.MATERIALIZE_FINAL_INPUTS_STAGE,
            runner.DECISION_IDS,
            runner.MATERIALIZE_FINAL_INPUTS_STAGE,
            ("D1", "D2"),
            (runner.STAGE_SCOPED_APPROVAL_SCHEMA,),
            "invalid schema",
        ),
        (
            runner.STAGE_SCOPED_APPROVAL_SCHEMA,
            runner.FIT_SEALED_TCGA_K500_STAGE,
            ("D1", "D2"),
            runner.MATERIALIZE_FINAL_INPUTS_STAGE,
            ("D1", "D2"),
            (runner.STAGE_SCOPED_APPROVAL_SCHEMA,),
            "stage envelope",
        ),
        (
            runner.STAGE_SCOPED_APPROVAL_SCHEMA,
            runner.MATERIALIZE_FINAL_INPUTS_STAGE,
            ("D1", "D2", "D3"),
            runner.MATERIALIZE_FINAL_INPUTS_STAGE,
            ("D1", "D2"),
            (runner.STAGE_SCOPED_APPROVAL_SCHEMA,),
            "decision sequence",
        ),
        (
            runner.APPROVAL_SCHEMA,
            runner.FIT_SEALED_TCGA_K500_STAGE,
            ("D1", "D2", "D3", "D4", "D5", "D6"),
            runner.FIT_SEALED_TCGA_K500_STAGE,
            ("D1", "D2", "D3", "D4", "D5", "D6"),
            (runner.STAGE_SCOPED_APPROVAL_SCHEMA, runner.APPROVAL_SCHEMA),
            "decision sequence",
        ),
    ],
)
def test_frozen_approval_receipt_rejects_schema_stage_and_decision_scope_cross_use(  # noqa: PLR0913
    tmp_path: Path,
    schema: str,
    manifest_stage: str,
    decision_ids: tuple[str, ...],
    expected_stage: str,
    expected_decision_ids: tuple[str, ...],
    allowed_schemas: tuple[str, ...],
    error: str,
) -> None:
    manifest_path = tmp_path / "approval.json"
    decisions = [{"decision_id": decision_id} for decision_id in decision_ids]
    runner._write_json_atomic(  # noqa: SLF001
        manifest_path,
        {
            "allowed_stages": [manifest_stage],
            "decisions": decisions,
            "schema": schema,
        },
    )
    manifest_sha256 = runner._sha256(manifest_path)  # noqa: SLF001
    decision_by_id = {decision["decision_id"]: decision for decision in decisions}
    authority = {
        "authorized_stage": expected_stage,
        "decision_digests": {
            decision_id: runner._json_sha256(decision_by_id[decision_id])  # noqa: SLF001
            for decision_id in expected_decision_ids
            if decision_id in decision_by_id
        },
        "manifest": runner._file_record(manifest_path),  # noqa: SLF001
        "manifest_sha256": manifest_sha256,
    }

    with pytest.raises(ValueError, match=error):
        runner._approval_manifest_decisions(  # noqa: SLF001
            authority,
            expected_path=manifest_path,
            expected_sha256=manifest_sha256,
            expected_stage=expected_stage,
            expected_decision_ids=expected_decision_ids,
            allowed_schemas=allowed_schemas,
            label="test",
        )


def test_production_family_contract_rejects_signed_record_drift(
    tmp_path,
    monkeypatch,
):
    _paths, contract = _write_local_authority_contract(tmp_path, monkeypatch)
    signed = contract["revision_input_authority"]["fit_policy"]["d5"]["tested_family"]
    signed["epsilon_pretest_filter"] = "artifact-self-authorized-filter"

    with pytest.raises(ValueError, match="bind signed D5"):
        runner._require_tested_family_contract(  # noqa: SLF001
            contract,
            require_signed_k500=True,
        )


@pytest.mark.parametrize(
    ("record_group", "record_name"),
    [
        ("provider", "cbase_pmfs"),
        ("canonical", "canonical_maf"),
    ],
)
def test_frozen_cohort_authority_rejects_local_byte_substitution(
    tmp_path,
    monkeypatch,
    record_group,
    record_name,
) -> None:
    paths, contract = _write_local_authority_contract(tmp_path, monkeypatch)
    if record_group == "provider":
        record = contract["provider_input_provenance"]["files"][record_name]
    else:
        record = contract["revision_input_authority"][record_name]
    Path(record["path"]).write_bytes(b"substituted\n")

    with pytest.raises(ValueError, match="Frozen input"):
        runner._verify_frozen_cohort_authority(paths, contract)  # noqa: SLF001


@pytest.mark.parametrize("attack", ["path", "extra-key"])
def test_frozen_cohort_authority_rejects_scoped_canonical_receipt_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    paths, contract = _write_local_authority_contract(tmp_path, monkeypatch)
    scoped_validator = runner.validate_materialized_provider_cohort_input

    def substituted_scope(*args, **kwargs):
        result = copy.deepcopy(scoped_validator(*args, **kwargs))
        receipts = result["binding"]["canonical_input_receipts"]
        if attack == "path":
            receipts["child_manifest"]["path"] = "cohorts/BRCA.json"
        else:
            receipts["unexpected"] = receipts["child_manifest"]
        return result

    monkeypatch.setattr(
        runner,
        "validate_materialized_provider_cohort_input",
        substituted_scope,
    )

    with pytest.raises((TypeError, ValueError)):
        runner._verify_frozen_cohort_authority(paths, contract)  # noqa: SLF001


def test_child_contract_load_never_calls_whole_family_validators(
    tmp_path,
    monkeypatch,
) -> None:
    paths, contract = _write_local_authority_contract(tmp_path, monkeypatch)
    canonical_validator = runner.validate_materialized_input_cohort_binding
    provider_validator = runner.validate_materialized_provider_cohort_input
    narrow_calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        runner,
        "validate_materialized_input_cohort_binding",
        lambda *args, **kwargs: (
            narrow_calls.append(("canonical", args, kwargs))
            or canonical_validator(*args, **kwargs)
        ),
    )
    monkeypatch.setattr(
        runner,
        "validate_materialized_provider_cohort_input",
        lambda *args, **kwargs: (
            narrow_calls.append(("provider", args, kwargs))
            or provider_validator(*args, **kwargs)
        ),
    )
    contract_path = runner._contract_path(paths, "CHOL")  # noqa: SLF001
    contract_path.parent.mkdir(parents=True)
    runner._write_json_atomic(contract_path, contract)  # noqa: SLF001
    monkeypatch.setattr(runner, "_require_exact_sample_axis", lambda _contract: None)
    monkeypatch.setattr(
        runner,
        "_require_full_observation_support",
        lambda _contract: None,
    )

    def forbid_whole_family(*_args, **_kwargs):
        msg = "whole-family validator entered child task"
        raise AssertionError(msg)

    monkeypatch.setattr(
        runner,
        "validate_materialized_provider_input_bundle",
        forbid_whole_family,
    )
    monkeypatch.setattr(
        runner,
        "validate_materialized_input_bundle",
        forbid_whole_family,
    )

    loaded = runner._load_verified_contract(  # noqa: SLF001
        paths,
        "CHOL",
        top_k=runner.TOP_K,
    )

    assert loaded == contract
    assert [call[0] for call in narrow_calls] == ["canonical", "provider"]
    assert narrow_calls[0][2] == {"require_current_execution_environment": False}
    assert narrow_calls[1][2] == {"require_current_execution_environment": True}
    canonical_receipt = contract["revision_input_authority"]["full_validation"]
    provider_receipt = contract["provider_input_provenance"]["root_receipt"]
    assert narrow_calls[0][1][5:] == (
        canonical_receipt["receipt"],
        canonical_receipt["receipt_sha256"],
    )
    assert narrow_calls[1][1][3:5] == (
        provider_receipt["full_acceptance_receipt"],
        provider_receipt["full_acceptance_receipt_sha256"],
    )


def test_cohort_authority_propagates_provider_receipt_substitution_failure(
    tmp_path,
    monkeypatch,
) -> None:
    paths = _with_revision_authority(
        runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run"),
        tmp_path,
    )
    contract = {
        "cohort": "CHOL",
        "revision_input_authority": {},
        "provider_input_provenance": {},
        "inputs": {},
    }
    monkeypatch.setattr(
        runner,
        "_approval_manifest_decisions",
        lambda *_args, **_kwargs: ({"decisions": []}, {}),
    )
    monkeypatch.setattr(
        runner,
        "_verify_fit_policy_receipts",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runner,
        "_require_materialize_stage_binding",
        lambda *_args: {},
    )
    monkeypatch.setattr(runner, "_require_fit_stage_binding", lambda *_args: {})
    monkeypatch.setattr(runner, "_verify_canonical_cohort_receipts", lambda *_a: None)

    def reject_provider(*_args):
        msg = "provider receipt changed"
        raise ValueError(msg)

    monkeypatch.setattr(runner, "_verify_provider_cohort_receipts", reject_provider)

    with pytest.raises(ValueError, match="provider receipt changed"):
        runner._verify_frozen_cohort_authority(paths, contract)  # noqa: SLF001


def test_production_parser_rejects_legacy_manual_provider_roots() -> None:
    parser = runner._parser()  # noqa: SLF001

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--output-root",
                "run",
                "--source-root",
                "manual-source",
                "--mutsig-root",
                "manual-mutsig",
            ],
        )


def test_revision_authority_arguments_are_all_or_none(tmp_path) -> None:
    paths = runner.RunPaths(
        tmp_path / "source",
        tmp_path / "mutsig",
        tmp_path / "run",
        canonical_input_root=tmp_path / "canonical",
    )

    with pytest.raises(ValueError, match="must be supplied together"):
        runner._revision_authority_is_configured(paths)  # noqa: SLF001


@pytest.mark.parametrize("attack", ["extra-stage", "extra-binding"])
def test_raw_materialize_authority_rejects_any_additional_stage(
    attack: str,
) -> None:
    expected = {
        "d1_canonical_artifact_sha256": "a" * 64,
        "d2_canonical_artifact_sha256": "b" * 64,
    }
    approval = {
        "allowed_stages": [runner.MATERIALIZE_FINAL_INPUTS_STAGE],
        "decisions": [
            {
                "canonical_artifact": {"path": "D1.json", "sha256": "a" * 64},
                "decision_id": "D1",
            },
            {
                "canonical_artifact": {"path": "D2.json", "sha256": "b" * 64},
                "decision_id": "D2",
            },
        ],
        "stage_bindings": {runner.MATERIALIZE_FINAL_INPUTS_STAGE: expected},
    }
    if attack == "extra-stage":
        approval["allowed_stages"].append(runner.FIT_SEALED_TCGA_K500_STAGE)
    else:
        approval["stage_bindings"][runner.FIT_SEALED_TCGA_K500_STAGE] = {
            "canonical_input_manifest_sha256": "c" * 64,
            "provider_input_manifest_sha256": "d" * 64,
        }

    with pytest.raises(ValueError, match="only the materialize-final-inputs stage"):
        runner._require_materialize_stage_binding(approval)  # noqa: SLF001


def test_raw_materialize_authority_rejects_artifact_binding_replay() -> None:
    approval = {
        "allowed_stages": [runner.MATERIALIZE_FINAL_INPUTS_STAGE],
        "decisions": [
            {
                "canonical_artifact": {"path": "D1.json", "sha256": "a" * 64},
                "decision_id": "D1",
            },
            {
                "canonical_artifact": {"path": "D2.json", "sha256": "b" * 64},
                "decision_id": "D2",
            },
        ],
        "stage_bindings": {
            runner.MATERIALIZE_FINAL_INPUTS_STAGE: {
                "d1_canonical_artifact_sha256": "a" * 64,
                "d2_canonical_artifact_sha256": "e" * 64,
            },
        },
    }

    with pytest.raises(ValueError, match="exact D1/D2 artifacts"):
        runner._require_materialize_stage_binding(approval)  # noqa: SLF001


def test_validated_materialize_authority_requires_exact_singleton_envelope() -> None:
    expected = {
        "d1_canonical_artifact_sha256": "a" * 64,
        "d2_canonical_artifact_sha256": "b" * 64,
    }
    decisions = {
        "D1": SimpleNamespace(canonical_artifact=SimpleNamespace(sha256="a" * 64)),
        "D2": SimpleNamespace(canonical_artifact=SimpleNamespace(sha256="b" * 64)),
    }
    approval = runner.RevisionApproval(
        schema="synthetic",
        source_notice=SimpleNamespace(),
        allowed_stages=(runner.MATERIALIZE_FINAL_INPUTS_STAGE,),
        stage_bindings={runner.MATERIALIZE_FINAL_INPUTS_STAGE: expected},
        decisions=decisions,
        manifest_sha256="f" * 64,
        decision_digests={},
    )

    assert runner._require_materialize_stage_binding(approval) == expected  # noqa: SLF001
    expanded = runner.RevisionApproval(
        schema=approval.schema,
        source_notice=approval.source_notice,
        allowed_stages=(
            runner.MATERIALIZE_FINAL_INPUTS_STAGE,
            runner.FIT_SEALED_TCGA_K500_STAGE,
        ),
        stage_bindings={
            runner.MATERIALIZE_FINAL_INPUTS_STAGE: expected,
            runner.FIT_SEALED_TCGA_K500_STAGE: {
                "canonical_input_manifest_sha256": "c" * 64,
                "provider_input_manifest_sha256": "d" * 64,
            },
        },
        decisions=decisions,
        manifest_sha256=approval.manifest_sha256,
        decision_digests=approval.decision_digests,
    )
    with pytest.raises(ValueError, match="only the materialize-final-inputs stage"):
        runner._require_materialize_stage_binding(expanded)  # noqa: SLF001


@pytest.mark.parametrize("attack", ["extra-stage", "extra-binding"])
def test_raw_fit_authority_rejects_any_additional_stage(
    tmp_path: Path,
    attack: str,
) -> None:
    paths = _with_revision_authority(
        runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run"),
        tmp_path,
    )
    expected = {
        "canonical_input_manifest_sha256": (paths.expected_canonical_input_sha256),
        "provider_input_manifest_sha256": (
            paths.expected_provider_input_manifest_sha256
        ),
    }
    approval = {
        "allowed_stages": [runner.FIT_SEALED_TCGA_K500_STAGE],
        "stage_bindings": {runner.FIT_SEALED_TCGA_K500_STAGE: expected},
    }
    if attack == "extra-stage":
        approval["allowed_stages"].append("inspect-tcga-k500")
    else:
        approval["stage_bindings"]["inspect-tcga-k500"] = {
            "upstream_result_manifest_sha256": "d" * 64,
        }

    with pytest.raises(ValueError, match="only the sealed-fit stage"):
        runner._require_fit_stage_binding(approval, paths)  # noqa: SLF001


def test_validated_fit_authority_requires_exact_singleton_envelope(
    tmp_path: Path,
) -> None:
    paths = _with_revision_authority(
        runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run"),
        tmp_path,
    )
    expected = {
        "canonical_input_manifest_sha256": (paths.expected_canonical_input_sha256),
        "provider_input_manifest_sha256": (
            paths.expected_provider_input_manifest_sha256
        ),
    }
    approval = runner.RevisionApproval(
        schema=runner.STAGE_SCOPED_APPROVAL_SCHEMA,
        source_notice=SimpleNamespace(),
        allowed_stages=(runner.FIT_SEALED_TCGA_K500_STAGE,),
        stage_bindings={runner.FIT_SEALED_TCGA_K500_STAGE: expected},
        decisions={
            decision_id: SimpleNamespace()
            for decision_id in ("D1", "D2", "D3", "D4", "D5", "D6")
        },
        manifest_sha256="f" * 64,
        decision_digests={},
    )

    assert runner._require_fit_stage_binding(approval, paths) == expected  # noqa: SLF001
    historical = runner.RevisionApproval(
        schema=runner.APPROVAL_SCHEMA,
        source_notice=approval.source_notice,
        allowed_stages=approval.allowed_stages,
        stage_bindings=approval.stage_bindings,
        decisions=approval.decisions,
        manifest_sha256=approval.manifest_sha256,
        decision_digests=approval.decision_digests,
    )
    with pytest.raises(ValueError, match=r"stage-scoped v5.*exactly D1-D6"):
        runner._require_fit_stage_binding(historical, paths)  # noqa: SLF001
    expanded = runner.RevisionApproval(
        schema=approval.schema,
        source_notice=approval.source_notice,
        allowed_stages=(
            runner.FIT_SEALED_TCGA_K500_STAGE,
            "inspect-tcga-k500",
        ),
        stage_bindings={
            runner.FIT_SEALED_TCGA_K500_STAGE: expected,
            "inspect-tcga-k500": {"upstream_result_manifest_sha256": "d" * 64},
        },
        decisions=approval.decisions,
        manifest_sha256=approval.manifest_sha256,
        decision_digests=approval.decision_digests,
    )
    with pytest.raises(ValueError, match="only the sealed-fit stage"):
        runner._require_fit_stage_binding(expanded, paths)  # noqa: SLF001


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("exact_resolution", "scientifically substituted resolution"),
        ("execution_owner", "substituted executor"),
        ("claim_owner", "attacker"),
        ("rerun_or_reuse_consequence", "substituted consequence"),
        ("permitted_claims", ("substituted claim",)),
        (
            "canonical_artifact",
            SimpleNamespace(
                sha256="b" * 64,
                size_bytes=11,
                content=b"substituted",
            ),
        ),
    ],
)
def test_two_stage_reauthorization_allows_stage_scope_only_and_rejects_semantic_drift(
    field: str,
    replacement: object,
) -> None:
    artifact = SimpleNamespace(
        sha256="a" * 64,
        size_bytes=10,
        content=b"canonical",
    )
    base = {
        "decision_id": "D1",
        "disposition": "go",
        "exact_resolution": "exact",
        "canonical_artifact": artifact,
        "execution_owner": "owner",
        "claim_owner": "claim-owner",
        "rerun_or_reuse_consequence": "rerun",
        "permitted_claims": ("claim",),
        "forbidden_claims": ("overclaim",),
        "allowed_stages": (runner.MATERIALIZE_FINAL_INPUTS_STAGE,),
    }
    input_decision = SimpleNamespace(**base)
    fit_decision = SimpleNamespace(
        **{**base, "allowed_stages": (runner.FIT_SEALED_TCGA_K500_STAGE,)},
    )
    substituted = SimpleNamespace(
        **{
            **base,
            "allowed_stages": (runner.FIT_SEALED_TCGA_K500_STAGE,),
            field: replacement,
        },
    )

    input_record = runner._decision_reauthorization_record(input_decision)  # noqa: SLF001
    assert input_record == runner._decision_reauthorization_record(  # noqa: SLF001
        fit_decision,
    )
    assert input_record != runner._decision_reauthorization_record(  # noqa: SLF001
        substituted,
    )


def test_production_task_requires_child_gate_before_any_inference(
    tmp_path,
    monkeypatch,
):
    paths = _with_revision_authority(
        runner.RunPaths(
            tmp_path / "source",
            tmp_path / "mutsig",
            tmp_path / "run",
        ),
        tmp_path,
    )
    monkeypatch.setattr(runner.os, "environ", dict(runner.SEALED_TASK_ENVIRONMENT))
    monkeypatch.setattr(
        runner,
        "_read_json",
        lambda _path: {"implementation_sha256": {}},
    )
    monkeypatch.setattr(runner, "_require_pinned_import_roots", lambda _value: None)

    def reject_gate(_paths, *, jobs, label):
        assert jobs == 1
        assert label == "task-start-CHOL-cbase"
        msg = "blocked by child gate"
        raise RuntimeError(msg)

    monkeypatch.setattr(runner, "_require_live_resource_gate", reject_gate)

    with pytest.raises(RuntimeError, match="blocked by child gate"):
        runner.execute_task(
            paths,
            runner.Task("CHOL", "cbase"),
            nice_increment=runner.REQUIRED_NICE_INCREMENT,
            expected_contract_sha256="frozen",
        )


def test_production_task_revalidates_provider_after_fit_before_publication(
    tmp_path,
    monkeypatch,
) -> None:
    paths = _with_revision_authority(
        runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run"),
        tmp_path,
    )
    paths.output_root.mkdir()
    task = runner.Task("CHOL", "cbase")
    provider_receipt = {"expected_manifest_sha256": "c" * 64}
    mutsig_pmf_contract = runner.build_poisson_support_contract(0.0, 0)
    contract = {
        "features": ["A_M"],
        "inputs": {
            "counts": {"path": (tmp_path / "count_matrix.csv").as_posix()},
            "mutsig": {
                "receipt": {"canonical_maf_binding": {"status": "verified"}},
            },
        },
        "mutsig_pmf_contract": mutsig_pmf_contract,
        "mutsig_pmf_storage_contract": runner.estimate_native_poisson_pmf_storage(
            1,
            1,
            mutsig_pmf_contract["inclusive_support_k"],
        ),
        "pair_policy": {"row_count": 0},
        "provider_input_provenance": {"root_receipt": provider_receipt},
    }
    events = []
    monkeypatch.setattr(runner.os, "environ", dict(runner.SEALED_TASK_ENVIRONMENT))
    original_read_json = runner._read_json  # noqa: SLF001
    monkeypatch.setattr(
        runner,
        "_read_json",
        lambda path: (
            {"implementation_sha256": {}}
            if Path(path).name == "run_manifest.json"
            else original_read_json(path)
        ),
    )
    monkeypatch.setattr(runner, "_require_pinned_import_roots", lambda _value: None)
    monkeypatch.setattr(runner.os, "nice", lambda increment: increment)
    monkeypatch.setattr(runner, "_require_live_resource_gate", lambda *_a, **_k: None)
    monkeypatch.setattr(
        runner,
        "_require_corrected_lrt",
        lambda: (
            "lrt",
            "fit",
            "effect-identifiability",
            "rho",
            "support",
            "cells",
            "lor",
        ),
    )
    monkeypatch.setattr(runner, "_verify_run_implementation", lambda _paths: {})
    monkeypatch.setattr(
        runner,
        "_load_verified_contract",
        lambda _paths, _cohort, **_kwargs: contract,
    )
    counts = pd.DataFrame({"A_M": [0]}, index=["s1"])
    monkeypatch.setattr(
        runner,
        "_load_frozen_scientific_inputs",
        lambda *_args, **_kwargs: (counts, {"A_M": {0: 1.0}}),
    )
    monkeypatch.setattr(
        runner,
        "_consumed_input_hashes",
        lambda *_args, **_kwargs: {"synthetic": "0" * 64},
    )
    monkeypatch.setattr(runner, "_build_genes", lambda *_args: {"A_M": object()})
    monkeypatch.setattr(runner, "estimate_pi_for_each_gene", lambda _genes: None)
    monkeypatch.setattr(runner, "create_single_gene_results", lambda *_a, **_k: None)

    def write_empty_pairwise(directory_fd, name, *_args):
        runner._write_bytes_atomic_at(  # noqa: SLF001
            directory_fd,
            name,
            b"",
            label="synthetic pairwise output",
        )
        return 0

    monkeypatch.setattr(runner, "_write_pairwise_results_at", write_empty_pairwise)
    monkeypatch.setattr(
        runner,
        "validate_task_output",
        lambda *_args, **_kwargs: {"synthetic": True},
    )
    monkeypatch.setattr(
        runner,
        "_task_resource_usage",
        lambda _started: {
            "elapsed_seconds": 1.0,
            "peak_rss": {
                "bytes": 1024,
                "native_value": 1,
                "native_unit": "KiB",
                "platform": "linux",
                "source": "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss",
            },
        },
    )
    monkeypatch.setattr(
        runner,
        "_verify_frozen_cohort_authority",
        lambda *_args: events.append("post-fit-authority"),
    )

    original_publish = runner._rename_exclusive_at  # noqa: SLF001

    def publish(source_parent, source_name, destination_parent, destination_name):
        if destination_name == task.bmr:
            events.append("publish")
        original_publish(
            source_parent,
            source_name,
            destination_parent,
            destination_name,
        )

    monkeypatch.setattr(runner, "_rename_exclusive_at", publish)

    state = runner.execute_task(
        paths,
        task,
        expected_contract_sha256=runner._json_sha256(contract),  # noqa: SLF001
    )

    assert state == "completed"
    assert events == ["post-fit-authority", "publish"]
    manifest = runner._read_json(  # noqa: SLF001
        paths.output_root / "tasks" / "CHOL" / "cbase" / "task_manifest.json",
    )
    assert manifest["provider_input_root_receipt"] == provider_receipt
    assert manifest["niceness"] == {
        "requested_increment": runner.REQUIRED_NICE_INCREMENT,
        "resulting_process_nice": runner.REQUIRED_NICE_INCREMENT,
    }


def test_production_task_rejects_niceness_below_frozen_increment_before_gate(
    tmp_path,
    monkeypatch,
):
    paths = runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run")
    gated = []
    monkeypatch.setattr(
        runner,
        "_require_live_resource_gate",
        lambda *_args, **_kwargs: gated.append(1),
    )

    with pytest.raises(ValueError, match="frozen niceness increment 10"):
        runner.execute_task(
            paths,
            runner.Task("CHOL", "cbase"),
            nice_increment=0,
            expected_contract_sha256="frozen",
        )

    assert gated == []


def test_execute_task_rejects_negative_niceness_before_reading_inputs(tmp_path):
    paths = runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run")

    with pytest.raises(ValueError, match="niceness increment"):
        runner.execute_task(
            paths,
            runner.Task("CHOL", "cbase"),
            nice_increment=-1,
            top_k=3,
        )


def test_cli_resource_overrides_share_the_computed_host_cap():
    with pytest.raises(ValueError, match="--jobs 4 exceeds"):
        runner._validate_cli_resource_options(  # noqa: SLF001
            jobs=4,
            mutsig_jobs=1,
            nice_increment=10,
            logical_cores=14,
        )

    with pytest.raises(ValueError, match=r"--mutsig-jobs.*between 1 and 1"):
        runner._validate_cli_resource_options(  # noqa: SLF001
            jobs=1,
            mutsig_jobs=2,
            nice_increment=10,
            logical_cores=2,
        )

    with pytest.raises(ValueError, match=r"--nice must equal.*10"):
        runner._validate_cli_resource_options(  # noqa: SLF001
            jobs=1,
            mutsig_jobs=1,
            nice_increment=0,
            logical_cores=14,
        )


def test_internal_cli_rejects_negative_niceness_before_execution(
    tmp_path,
    monkeypatch,
):
    invoked = []
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [
            "runner",
            "--output-root",
            str(tmp_path / "run"),
            "--internal-cohort",
            "CHOL",
            "--internal-bmr",
            "cbase",
            "--internal-contract-sha256",
            "frozen",
            "--nice",
            "-1",
        ],
    )

    def record_execution(*_args, **_kwargs):
        invoked.append(1)

    monkeypatch.setattr(runner, "execute_task", record_execution)

    with pytest.raises(ValueError, match="--nice must be nonnegative"):
        runner.main()

    assert invoked == []


def test_task_batch_rechecks_resources_before_each_bounded_wave(
    tmp_path,
    monkeypatch,
):
    paths = runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run")
    tasks = [runner.Task(f"C{i}", "cbase") for i in range(5)]
    gates = []
    invoked = []

    def record_gate(_paths, *, jobs, label):
        gates.append((jobs, label))

    def record_invocation(_paths, task, _nice_increment):
        invoked.append(task)
        return task, 0

    monkeypatch.setattr(runner, "_require_live_resource_gate", record_gate)
    monkeypatch.setattr(runner, "_invoke_task", record_invocation)

    failures = runner._run_task_batch(  # noqa: SLF001
        paths,
        tasks,
        jobs=2,
        nice_increment=10,
    )

    assert failures == 0
    assert [jobs for jobs, _label in gates] == [2, 2, 1]
    assert len({label for _jobs, label in gates}) == 3
    assert set(invoked) == set(tasks)


def test_task_batch_never_invokes_a_task_after_a_failed_live_gate(
    tmp_path,
    monkeypatch,
):
    paths = runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run")
    invoked = []

    def reject_gate(_paths, *, jobs, label):
        del jobs, label
        msg = "unsafe host"
        raise RuntimeError(msg)

    def record_invocation(_paths, task, _nice_increment):
        invoked.append(task)
        return task, 0

    monkeypatch.setattr(runner, "_require_live_resource_gate", reject_gate)
    monkeypatch.setattr(runner, "_invoke_task", record_invocation)

    with pytest.raises(RuntimeError, match="unsafe host"):
        runner._run_task_batch(  # noqa: SLF001
            paths,
            [runner.Task("CHOL", "cbase")],
            jobs=1,
            nice_increment=10,
        )

    assert invoked == []


def test_task_batch_rejects_nonpositive_concurrency_before_scheduling(tmp_path):
    paths = runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run")

    with pytest.raises(ValueError, match="concurrency must be positive"):
        runner._run_task_batch(  # noqa: SLF001
            paths,
            [runner.Task("CHOL", "cbase")],
            jobs=0,
            nice_increment=10,
        )


def test_noncanary_subset_cannot_bypass_missing_chol_canaries(
    tmp_path,
    monkeypatch,
):
    paths = runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run")
    contract_path = paths.output_root / "contracts" / "BRCA.json"
    runner._write_json_atomic(contract_path, {})  # noqa: SLF001
    batches = []
    monkeypatch.setattr(runner, "_initialize_run", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        runner,
        "_prime_parent_revision_authority",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(runner, "_ensure_contract", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "_require_corrected_lrt", lambda: ())
    monkeypatch.setattr(
        runner,
        "_require_canonical_mutsig_maf_binding",
        lambda _contract: None,
    )
    monkeypatch.setattr(
        runner,
        "_run_task_batch",
        lambda *_args, **_kwargs: batches.append(1),
    )

    with pytest.raises(RuntimeError, match="Validated CHOL canaries"):
        runner._orchestrate(  # noqa: SLF001
            paths,
            ["BRCA"],
            jobs=1,
            mutsig_jobs=1,
            nice_increment=10,
            preflight_only=False,
        )

    assert batches == []


def test_canary_gate_revalidates_all_three_background_outputs(
    tmp_path,
    monkeypatch,
):
    paths = runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run")
    for bmr in runner.BMRS:
        (paths.output_root / "tasks" / "CHOL" / bmr).mkdir(parents=True)
    validated = []
    monkeypatch.setattr(
        runner,
        "_load_verified_contract",
        lambda _paths, cohort, *, top_k: {"cohort": cohort, "top_k": top_k},
    )
    monkeypatch.setattr(
        runner,
        "validate_task_output",
        lambda task_dir, _contract: validated.append(task_dir.name),
    )

    runner._require_validated_canary_outputs(paths)  # noqa: SLF001

    assert validated == list(runner.BMRS)


def test_production_initialization_rejects_dirty_git_tree(tmp_path, monkeypatch):
    paths = runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run")
    dirty = {"head": "abc123", "dirty": True, "status": [" M file.py"]}
    monkeypatch.setattr(runner, "_git_snapshot", lambda _root, _record: dirty)

    with pytest.raises(RuntimeError, match="clean Git tree"):
        runner._initialize_run(paths, allow_dirty=False)  # noqa: SLF001

    assert not paths.output_root.exists()
    manifest = runner._initialize_run(paths, allow_dirty=True)  # noqa: SLF001
    assert manifest["git"] == dirty
    assert manifest["tested_family_implementation"] == runner.asdict(
        runner.REQUIRED_TESTED_FAMILY,
    )
    assert manifest["signed_tested_family"] is None
    with pytest.raises(RuntimeError, match="clean Git tree"):
        runner._initialize_run(paths, allow_dirty=False)  # noqa: SLF001


def test_run_resume_rejects_resource_policy_drift(tmp_path, monkeypatch):
    paths = runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run")
    clean = {"head": "abc123", "dirty": False, "status": []}
    monkeypatch.setattr(runner, "_git_snapshot", lambda _root, _record: clean)
    monkeypatch.setattr(
        runner,
        "_source_snapshot",
        lambda _root: {"runner.py": "frozen"},
    )
    monkeypatch.setattr(runner, "_require_pinned_import_roots", lambda _value: None)
    manifest = runner._initialize_run(paths, allow_dirty=False)  # noqa: SLF001
    manifest["resource_policy"]["maximum_general_jobs"] = 99
    manifest_path = paths.output_root / "run_manifest.json"
    manifest_path.write_bytes(runner._canonical_json(manifest) + b"\n")  # noqa: SLF001

    with pytest.raises(ValueError, match="resource_policy"):
        runner._initialize_run(paths, allow_dirty=False)  # noqa: SLF001


def test_run_resume_rejects_tested_family_manifest_drift(tmp_path, monkeypatch):
    paths = runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run")
    clean = {"head": "abc123", "dirty": False, "status": []}
    monkeypatch.setattr(runner, "_git_snapshot", lambda _root, _record: clean)
    monkeypatch.setattr(
        runner,
        "_source_snapshot",
        lambda _root: {"runner.py": "frozen"},
    )
    monkeypatch.setattr(runner, "_require_pinned_import_roots", lambda _value: None)
    manifest = runner._initialize_run(paths, allow_dirty=False)  # noqa: SLF001
    manifest["tested_family_implementation"]["top_k"] = 499
    manifest_path = paths.output_root / "run_manifest.json"
    manifest_path.write_bytes(runner._canonical_json(manifest) + b"\n")  # noqa: SLF001

    with pytest.raises(ValueError, match="tested_family_implementation"):
        runner._initialize_run(paths, allow_dirty=False)  # noqa: SLF001
