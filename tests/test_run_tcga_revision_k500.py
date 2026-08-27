import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import run_tcga_revision_k500 as runner
from analysis.mutsig_lambda_co import build_lambda_pmfs


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
    return runner.RunPaths(source_root, mutsig_root, output_root)


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


def test_cohort_contract_hashes_exact_native_universe_and_mapping(tmp_path):
    paths = _write_inputs(tmp_path)

    contract = runner.build_cohort_contract(paths, "CHOL", top_k=3)

    assert contract["features"] == ["A_M", "A_N", "B_M"]
    assert contract["pair_policy"]["row_count"] == 2
    assert contract["pair_policy"]["same_base_pairs_excluded"] == 1
    assert contract["samples"]["matched_samples"] == 4
    assert contract["samples"]["cohort_mean_fallback_samples"] == 0
    assert contract["feature_policy"]["mutsig_cbase_feature_fallback"] is False
    assert contract["mutsig_pmf_contract"]["selected_observed_count_max"] == 1
    assert contract["mutsig_pmf_contract"]["lambda_floor"] is None
    assert len(contract["inputs"]["mutsig"]["files"]["lambda"]["sha256"]) == 64
    for bmr in runner.BMRS:
        support = contract["observed_count_support_audit"][bmr]
        assert support["zero_support_feature_samples"] == 0
        assert support["pairs"]["full_sample_support"] == 2


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
    assert contract["observed_count_support_audit"]["cbase"][
        "zero_support_feature_samples"
    ] == 0


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


def test_pairwise_rho_validation_is_fail_closed():
    pair = ("A_M", "B_M")
    independent_taus = [0.25, 0.25, 0.25, 0.25]
    runner._validate_pairwise_rho("0.0", independent_taus, 0.0, pair)  # noqa: SLF001

    for invalid_rho in ("", "nan", "inf", "0.5"):
        with pytest.raises(ValueError, match="rho"):
            runner._validate_pairwise_rho(  # noqa: SLF001
                invalid_rho,
                independent_taus,
                0.0,
                pair,
            )

    tiny_taus = [1.0, 1e-200, 1e-200, 0.0]
    tiny_rho = runner.interaction_module.compute_marshall_olkin_rho(tiny_taus)
    assert tiny_rho is not None
    runner._validate_pairwise_rho(  # noqa: SLF001
        str(tiny_rho),
        tiny_taus,
        0.0,
        pair,
    )
    for corrupted_tiny_rho in ("0.0", str(-tiny_rho)):
        with pytest.raises(ValueError, match="rho"):
            runner._validate_pairwise_rho(  # noqa: SLF001
                corrupted_tiny_rho,
                tiny_taus,
                0.0,
                pair,
            )

    degenerate_taus = [1.0, 0.0, 0.0, 0.0]
    runner._validate_pairwise_rho("", degenerate_taus, 0.0, pair)  # noqa: SLF001
    with pytest.raises(ValueError, match="undefined-rho boundary"):
        runner._validate_pairwise_rho(  # noqa: SLF001
            "",
            degenerate_taus,
            runner.REQUIRED_UNDEFINED_RHO_LRT_TOL * 2,
            pair,
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
    with pytest.raises(ValueError, match="extra row"):
        runner.validate_task_output(final_dir, contract)


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
    assert runner.safe_default_jobs(14) == 5
    assert runner.safe_default_jobs(14) < 14 / 2


def test_production_initialization_rejects_dirty_git_tree(tmp_path, monkeypatch):
    paths = runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run")
    dirty = {"head": "abc123", "dirty": True, "status": [" M file.py"]}
    monkeypatch.setattr(runner, "_git_snapshot", lambda _root: dirty)

    with pytest.raises(RuntimeError, match="clean Git tree"):
        runner._initialize_run(paths, allow_dirty=False)  # noqa: SLF001

    assert not paths.output_root.exists()
    manifest = runner._initialize_run(paths, allow_dirty=True)  # noqa: SLF001
    assert manifest["git"] == dirty
    with pytest.raises(RuntimeError, match="clean Git tree"):
        runner._initialize_run(paths, allow_dirty=False)  # noqa: SLF001
