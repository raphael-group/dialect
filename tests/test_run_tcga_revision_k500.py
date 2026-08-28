import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import run_tcga_revision_k500 as runner
from analysis.mutsig_lambda_co import build_lambda_pmfs

_REPO_ROOT = Path(__file__).resolve().parents[1]


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
    assert contract["samples"]["authoritative_samples"] == 4
    assert contract["samples"]["extra_mutsig_samples"] == 0
    assert contract["samples"]["exact_order_match"]
    assert contract["samples"]["contract"] == runner.SAMPLE_AXIS_CONTRACT
    assert contract["samples"]["cohort_mean_fallback_samples"] == 0
    assert contract["feature_policy"]["mutsig_cbase_feature_fallback"] is False
    assert contract["mutsig_pmf_contract"]["selected_observed_count_max"] == 1
    assert contract["mutsig_pmf_contract"]["lambda_floor"] is None
    assert len(contract["inputs"]["mutsig"]["files"]["lambda"]["sha256"]) == 64
    assert contract["inputs"]["mutsig"]["receipt"]["upstream_commit"] == (
        runner.MUTSIG_UPSTREAM_COMMIT
    )
    assert "receipt" in contract["inputs"]["mutsig"]["files"]
    assert contract["inputs"]["sample_axis"]["sha256"] == (
        contract["inputs"]["mutsig"]["receipt"]["sample_axis_sha256"]
    )
    assert contract["inputs"]["mutsig"]["receipt"]["canonical_maf_binding"] == {
        "status": runner.MUTSIG_MAF_BINDING_STATUS,
        "required_before_production": runner.MUTSIG_MAF_BINDING_REQUIREMENT,
    }
    for bmr in runner.BMRS:
        support = contract["observed_count_support_audit"][bmr]
        assert support["zero_support_feature_samples"] == 0
        assert support["pairs"]["full_sample_support"] == 2


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
    assert runner.safe_default_jobs(14) == 3
    assert runner.safe_default_jobs(14) < 14 / 2


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
        "The system has 25769803776 bytes.\n"
        "System-wide memory free percentage: 101%\n",
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
        total_memory_bytes=24 * 1024**3,
        available_memory_bytes=10 * 1024**3,
        free_disk_bytes=100 * 1024**3,
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


def test_live_resource_gate_rejects_malformed_aggregate_readback():
    snapshot = runner.HostResourceSnapshot(
        measured_at_utc="",
        logical_cores=0,
        total_memory_bytes=0,
        available_memory_bytes=1,
        free_disk_bytes=-1,
        memory_source="",
    )

    evaluation = runner.evaluate_host_resource_gate(snapshot, jobs=1)

    assert evaluation["passed"] is False
    assert evaluation["reasons"] == [
        "logical core count must be positive",
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
        total_memory_bytes=0,
        available_memory_bytes=1,
        free_disk_bytes=-1,
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
    assert "resource readback provenance is incomplete" in record["evaluation"][
        "reasons"
    ]


def test_resource_readback_record_cannot_be_overwritten(tmp_path, monkeypatch):
    paths = runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run")
    paths.output_root.mkdir()
    snapshot = runner.HostResourceSnapshot(
        measured_at_utc="2026-08-28T00:00:00+00:00",
        logical_cores=14,
        total_memory_bytes=24 * 1024**3,
        available_memory_bytes=10 * 1024**3,
        free_disk_bytes=100 * 1024**3,
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


def test_internal_task_environment_requires_every_exact_limit(monkeypatch):
    required = {
        **runner.THREAD_LIMIT_ENV,
        runner.INTERNAL_TASK_ENV: "1",
        "PYTHONHASHSEED": "0",
    }
    for name in required:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="single-thread environment"):
        runner._require_internal_task_environment()  # noqa: SLF001

    for name, value in required.items():
        monkeypatch.setenv(name, value)
    runner._require_internal_task_environment()  # noqa: SLF001

    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(RuntimeError, match="OMP_NUM_THREADS"):
        runner._require_internal_task_environment()  # noqa: SLF001


def test_task_subprocess_inherits_exact_single_thread_environment(
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

    task, return_code = runner._invoke_task(  # noqa: SLF001
        paths,
        runner.Task("CHOL", "cbase"),
        10,
    )

    assert task == runner.Task("CHOL", "cbase")
    assert return_code == 0
    assert {
        name: captured["env"][name]
        for name in runner.THREAD_LIMIT_ENV
    } == runner.THREAD_LIMIT_ENV
    assert captured["env"][runner.INTERNAL_TASK_ENV] == "1"
    assert captured["env"]["PYTHONHASHSEED"] == "0"


def test_production_task_requires_child_gate_before_any_inference(
    tmp_path,
    monkeypatch,
):
    paths = runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run")
    for name, value in runner.THREAD_LIMIT_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(runner.INTERNAL_TASK_ENV, "1")
    monkeypatch.setenv("PYTHONHASHSEED", "0")

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
            nice_increment=0,
            expected_contract_sha256="frozen",
        )


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
    monkeypatch.setattr(runner, "_git_snapshot", lambda _root: dirty)

    with pytest.raises(RuntimeError, match="clean Git tree"):
        runner._initialize_run(paths, allow_dirty=False)  # noqa: SLF001

    assert not paths.output_root.exists()
    manifest = runner._initialize_run(paths, allow_dirty=True)  # noqa: SLF001
    assert manifest["git"] == dirty
    with pytest.raises(RuntimeError, match="clean Git tree"):
        runner._initialize_run(paths, allow_dirty=False)  # noqa: SLF001


def test_run_resume_rejects_resource_policy_drift(tmp_path, monkeypatch):
    paths = runner.RunPaths(tmp_path / "source", tmp_path / "mutsig", tmp_path / "run")
    clean = {"head": "abc123", "dirty": False, "status": []}
    monkeypatch.setattr(runner, "_git_snapshot", lambda _root: clean)
    monkeypatch.setattr(
        runner,
        "_source_snapshot",
        lambda _root: {"runner.py": "frozen"},
    )
    manifest = runner._initialize_run(paths, allow_dirty=False)  # noqa: SLF001
    manifest["resource_policy"]["maximum_general_jobs"] = 99
    manifest_path = paths.output_root / "run_manifest.json"
    manifest_path.write_bytes(runner._canonical_json(manifest) + b"\n")  # noqa: SLF001

    with pytest.raises(ValueError, match="resource_policy"):
        runner._initialize_run(paths, allow_dirty=False)  # noqa: SLF001
