from __future__ import annotations

import copy
import hashlib
import inspect
import json
import re
import stat
from pathlib import Path

import pytest

from analysis import build_tcga_revision_artifact_registry as registry

_POSTPROCESS_SHA256 = "1" * 64
_SOURCE_MANIFEST_SHA256 = "2" * 64


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical(value))


def _receipt(gate: str, *, digest: str | None = None) -> dict[str, str]:
    return {
        "gate": gate,
        "receipt_id": f"{gate.lower()}-receipt",
        "sha256": digest or _sha256(f"receipt:{gate}".encode()),
    }


def _omitted(
    spec: registry.ArtifactSpec,
    *,
    satisfied: tuple[str, ...] = (),
    reason_code: str = "required_gate_not_satisfied",
) -> dict[str, object]:
    receipts = [_receipt(gate) for gate in registry.GATE_ORDER if gate in satisfied]
    unsatisfied = [
        gate
        for gate in registry.GATE_ORDER
        if gate in spec.required_gates and gate not in satisfied
    ]
    return {
        "semantic_id": spec.semantic_id,
        "status": "omitted",
        "gate_receipts": receipts,
        "omission": {
            "reason_code": reason_code,
            "unsatisfied_gates": unsatisfied,
        },
    }


def _write_ready_files(
    spec: registry.ArtifactSpec,
    renderer_root: Path,
    output_root: Path,
    *,
    output_count: int = 1,
) -> tuple[dict[str, object], list[bytes]]:
    script_member = f"analysis/render_{spec.semantic_id}.py"
    script = renderer_root / script_member
    script.parent.mkdir(parents=True, exist_ok=True)
    script_bytes = f'"""Synthetic renderer for {spec.semantic_id}."""\n'.encode()
    script.write_bytes(script_bytes)

    media_type, suffix = {
        "figure": ("application/pdf", ".pdf"),
        "table": ("application/pdf", ".pdf"),
        "supplementary-data": ("text/csv", ".csv"),
        "provenance-record": ("application/json", ".json"),
    }[spec.kind]
    outputs: list[dict[str, object]] = []
    output_bytes: list[bytes] = []
    for index in range(output_count):
        output_id = f"{spec.semantic_id}-output-{index}"
        output_member = f"rendered/{spec.semantic_id}/{output_id}{suffix}"
        output = output_root / output_member
        output.parent.mkdir(parents=True, exist_ok=True)
        raw = f"%PDF-1.7 synthetic {spec.semantic_id} {index}\n".encode()
        output.write_bytes(raw)
        output_bytes.append(raw)
        outputs.append(
            {
                "output_id": output_id,
                "release_member": output_member,
                "media_type": media_type,
                "sha256": _sha256(raw),
                "bytes": len(raw),
            },
        )

    if spec.source_requirement == "none":
        sources: list[dict[str, object]] = []
    elif spec.source_requirement == "upstream-manifest":
        sources = [
            {
                "source_id": "postprocess-release-manifest",
                "release_member": "postprocess/release_manifest.json",
                "role": "provenance",
                "sha256": _POSTPROCESS_SHA256,
                "bytes": 321,
            },
            {
                "source_id": "source-data-manifest",
                "release_member": "source-data/source_data_manifest.json",
                "role": "provenance",
                "sha256": _SOURCE_MANIFEST_SHA256,
                "bytes": 123,
            },
        ]
    else:
        sources = []
        for role in spec.required_source_roles:
            sources.append(
                {
                    "source_id": f"{spec.semantic_id}-{role}",
                    "release_member": (
                        f"source-data/{spec.semantic_id}.{role}.csv"
                    ),
                    "role": role,
                    "sha256": _sha256(
                        f"source:{spec.semantic_id}:{role}".encode(),
                    ),
                    "bytes": 123,
                },
            )

    return {
        "semantic_id": spec.semantic_id,
        "status": "ready",
        "gate_receipts": [_receipt(gate) for gate in spec.required_gates],
        "source_data": sources,
        "renderer": {
            "script": script_member,
            "sha256": _sha256(script_bytes),
        },
        "outputs": outputs,
    }, output_bytes


def _reconciliation(
    renderer_root: Path,
    output_root: Path,
    *,
    ready_ids: tuple[str, ...] = ("cross_cancer_bmr_co_sensitivity",),
    output_count: int = 1,
) -> dict[str, object]:
    global_gates = {
        gate
        for spec in registry.ARTIFACT_SPECS
        if spec.semantic_id in ready_ids
        for gate in spec.required_gates
    }
    gate_ledger = [
        _receipt(gate) for gate in registry.GATE_ORDER if gate in global_gates
    ]
    artifacts: list[dict[str, object]] = []
    for spec in registry.ARTIFACT_SPECS:
        if spec.semantic_id in ready_ids:
            artifact, _ = _write_ready_files(
                spec,
                renderer_root,
                output_root,
                output_count=output_count,
            )
            artifacts.append(artifact)
        else:
            satisfied = tuple(
                gate for gate in spec.required_gates if gate in global_gates
            )
            if set(satisfied) != set(spec.required_gates):
                reason_code = "required_gate_not_satisfied"
            elif "COAUTH" in spec.required_gates:
                reason_code = "coauthor_decision_to_omit"
            else:
                reason_code = "release_scope_exclusion"
            artifacts.append(
                _omitted(
                    spec,
                    satisfied=satisfied,
                    reason_code=reason_code,
                ),
            )
    return {
        "schema": registry.RECONCILIATION_INPUT_SCHEMA,
        "release": {
            "release_id": "tcga-k500-revision-release",
            "postprocess_release_sha256": _POSTPROCESS_SHA256,
            "source_data_manifest_sha256": _SOURCE_MANIFEST_SHA256,
        },
        "gate_ledger": gate_ledger,
        "artifacts": artifacts,
    }


@pytest.fixture
def roots(tmp_path: Path) -> tuple[Path, Path]:
    renderer_root = tmp_path / "repo"
    output_root = tmp_path / "release"
    renderer_root.mkdir()
    output_root.mkdir()
    builder = renderer_root / "analysis/build_tcga_revision_artifact_registry.py"
    builder.parent.mkdir()
    builder.write_bytes(Path(registry.__file__).read_bytes())
    return renderer_root, output_root


def _build(
    tmp_path: Path,
    roots: tuple[Path, Path],
    reconciliation: dict[str, object],
    *,
    stem: str = "artifact_registry",
) -> tuple[registry.ArtifactRegistryReceipt, Path, Path]:
    source = tmp_path / f"{stem}_input.json"
    destination = tmp_path / f"{stem}.json"
    _write_json(source, reconciliation)
    receipt = _invoke_build(source, roots, destination)
    return receipt, source, destination


def _invoke_build(
    source: Path,
    roots: tuple[Path, Path],
    destination: Path,
    *,
    expected_sha256: str | None = None,
) -> registry.ArtifactRegistryReceipt:
    return registry.build_artifact_registry(
        source,
        *roots,
        destination,
        expected_reconciliation_sha256=(
            expected_sha256 or _sha256(source.read_bytes())
        ),
    )


def test_semantic_catalog_is_complete_gate_exact_and_number_free() -> None:
    expected = {
        "cross_cancer_bmr_co_sensitivity": (
            ("K500", "CAL", "COAUTH"),
            ("primary", "calibration"),
        ),
        "interaction_model_diagnostic_panels": (
            ("K500", "CAL", "COAUTH"),
            ("calibration",),
        ),
        "selected_pair_biological_validation": (
            ("K500", "CAL", "COAUTH", "MSK"),
            ("primary", "calibration", "validation"),
        ),
        "simulation_method_comparison": (
            ("CAL", "COAUTH", "COMP", "SIM"),
            ("calibration", "comparison", "simulation"),
        ),
        "interaction_summary": (
            ("K500", "CAL", "COAUTH"),
            ("primary", "calibration"),
        ),
        "raw_supplementary_inventory": (
            ("K500", "COAUTH"),
            ("primary",),
        ),
        "provider_conjunction_summary": (
            ("K500", "CAL", "COAUTH"),
            ("primary", "calibration"),
        ),
        "comparator_benchmark": (
            ("K500", "COAUTH", "COMP"),
            ("comparison",),
        ),
        "calibration_diagnostics": (
            ("CAL", "COAUTH"),
            ("calibration",),
        ),
        "runtime_failure_summary": (("K500",), ("runtime",)),
        "msk_validation": (
            ("K500", "CAL", "COAUTH", "MSK"),
            ("primary", "calibration", "validation"),
        ),
        "method_overview": (("COAUTH",), ()),
        "release_provenance": (("COAUTH",), ("provenance",)),
    }
    assert registry.GATE_ORDER == ("K500", "CAL", "COAUTH", "COMP", "MSK", "SIM")
    assert {
        spec.semantic_id: (spec.required_gates, spec.required_source_roles)
        for spec in registry.ARTIFACT_SPECS
    } == expected
    for spec in registry.ARTIFACT_SPECS:
        semantic_text = f"{spec.semantic_id} {spec.title}"
        assert (
            re.search(
                r"\b(?:fig(?:ure)?|table)\s*[0-9]",
                semantic_text,
                re.IGNORECASE,
            )
            is None
        )
        assert spec.claims


def test_gate_catalog_and_trust_model_do_not_overstate_digest_references() -> None:
    assert all(
        "declared digest reference" in record["meaning"]
        for record in registry.GATE_CATALOG
    )
    emitted_text = json.dumps(
        {
            "gates": registry.GATE_CATALOG,
            "trust": registry.TRUST_MODEL,
        },
    ).lower()
    assert "approved" not in emitted_text
    assert "payloads are not opened" in emitted_text
    assert "row-bearing sources are not opened" in emitted_text


@pytest.mark.parametrize(
    "spec",
    registry.ARTIFACT_SPECS,
    ids=lambda spec: spec.semantic_id,
)
def test_every_artifact_declares_gate_specific_source_roles(
    spec: registry.ArtifactSpec,
) -> None:
    required_by_gate = {
        "CAL": "calibration",
        "COMP": "comparison",
        "MSK": "validation",
        "SIM": "simulation",
    }
    for gate, role in required_by_gate.items():
        if gate in spec.required_gates:
            assert role in spec.required_source_roles
    if spec.semantic_id == "runtime_failure_summary":
        assert "runtime" in spec.required_source_roles
    if spec.source_requirement == "none":
        assert spec.required_source_roles == ()
    if spec.source_requirement == "upstream-manifest":
        assert spec.required_source_roles == ("provenance",)


@pytest.mark.parametrize(
    "spec",
    registry.ARTIFACT_SPECS,
    ids=lambda spec: spec.semantic_id,
)
def test_every_artifact_ready_branch_accepts_its_required_source_roles(
    tmp_path: Path,
    roots: tuple[Path, Path],
    spec: registry.ArtifactSpec,
) -> None:
    value = _reconciliation(*roots, ready_ids=(spec.semantic_id,))
    receipt, _, _ = _build(tmp_path, roots, value, stem=spec.semantic_id)
    assert receipt.ready_count == 1


@pytest.mark.parametrize(
    "spec",
    [
        spec
        for spec in registry.ARTIFACT_SPECS
        if spec.source_requirement == "required"
    ],
    ids=lambda spec: spec.semantic_id,
)
def test_every_data_artifact_rejects_a_missing_required_source_role(
    tmp_path: Path,
    roots: tuple[Path, Path],
    spec: registry.ArtifactSpec,
) -> None:
    value = _reconciliation(*roots, ready_ids=(spec.semantic_id,))
    artifact = next(  # type: ignore[arg-type]
        record
        for record in value["artifacts"]
        if record["semantic_id"] == spec.semantic_id
    )
    missing_role = spec.required_source_roles[0]
    replacement_role = next(
        role
        for role in sorted(registry._SOURCE_ROLES)  # noqa: SLF001
        if role not in spec.required_source_roles
    )
    source = next(  # type: ignore[arg-type]
        source
        for source in artifact["source_data"]
        if source["role"] == missing_role
    )
    source["role"] = replacement_role
    reconciliation = tmp_path / "input.json"
    _write_json(reconciliation, value)

    with pytest.raises(registry.ArtifactRegistryError, match="source role"):
        _invoke_build(reconciliation, roots, tmp_path / "registry.json")


def test_build_and_validate_ready_and_omitted_branches(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    value = _reconciliation(*roots)
    receipt, _, destination = _build(tmp_path, roots, value)

    assert receipt.ready_count == 1
    assert receipt.omitted_count == 12
    assert stat.S_IMODE(destination.stat().st_mode) == 0o400
    raw = destination.read_bytes()
    assert receipt.manifest_sha256 == _sha256(raw)
    manifest = json.loads(raw)
    assert manifest["trust_model"] == registry.TRUST_MODEL
    assert manifest["builder"] == {
        "script": "analysis/build_tcga_revision_artifact_registry.py",
        "sha256": _sha256(Path(registry.__file__).read_bytes()),
        "bytes": len(Path(registry.__file__).read_bytes()),
    }
    assert set(manifest) == {
        "schema",
        "contract",
        "trust_model",
        "release",
        "builder",
        "gate_catalog",
        "gate_ledger",
        "artifact_catalog_sha256",
        "artifacts",
        "registry_payload_sha256",
    }
    ready = manifest["artifacts"][0]
    assert set(ready) == {
        "semantic_id",
        "title",
        "kind",
        "required_gates",
        "required_source_roles",
        "source_requirement",
        "status",
        "gate_receipts",
        "source_data",
        "renderer",
        "outputs",
        "claims",
    }
    omitted = manifest["artifacts"][1]
    assert set(omitted) == {
        "semantic_id",
        "title",
        "kind",
        "required_gates",
        "required_source_roles",
        "source_requirement",
        "status",
        "gate_receipts",
        "omission",
        "planned_claims",
    }
    validated = registry.validate_artifact_registry(
        destination,
        *roots,
        expected_manifest_sha256=receipt.manifest_sha256,
    )
    assert validated == receipt


def test_all_artifacts_may_remain_explicitly_omitted_pending_gates(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    value = _reconciliation(*roots, ready_ids=())
    receipt, _, destination = _build(tmp_path, roots, value)

    assert receipt.ready_count == 0
    assert receipt.omitted_count == len(registry.ARTIFACT_SPECS)
    manifest = json.loads(destination.read_bytes())
    assert all(record["status"] == "omitted" for record in manifest["artifacts"])
    assert all(
        record["omission"]["unsatisfied_gates"]
        for record in manifest["artifacts"]
    )


def test_complete_coauthor_decision_may_omit_a_digest_bound_artifact(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    value = _reconciliation(*roots)
    spec = registry.ARTIFACT_SPECS[0]
    value["artifacts"][0] = _omitted(  # type: ignore[index]
        spec,
        satisfied=spec.required_gates,
        reason_code="coauthor_decision_to_omit",
    )

    _, _, destination = _build(tmp_path, roots, value)
    record = json.loads(destination.read_bytes())["artifacts"][0]
    assert record["omission"] == {
        "reason_code": "coauthor_decision_to_omit",
        "reason": (
            "All required gates have declared receipts and the coauthors elected to "
            "omit this artifact from the release."
        ),
        "unsatisfied_gates": [],
    }


def test_runtime_summary_has_explicit_release_scope_omission_after_k500(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    value = _reconciliation(*roots)
    _, _, destination = _build(tmp_path, roots, value)
    runtime = next(
        artifact
        for artifact in json.loads(destination.read_bytes())["artifacts"]
        if artifact["semantic_id"] == "runtime_failure_summary"
    )

    assert runtime["gate_receipts"] == [_receipt("K500")]
    assert runtime["omission"]["reason_code"] == "release_scope_exclusion"
    assert runtime["omission"]["unsatisfied_gates"] == []


def test_ready_branch_requires_every_exact_gate_receipt(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    value = _reconciliation(*roots)
    value["artifacts"][0]["gate_receipts"].pop()  # type: ignore[index,union-attr]
    source = tmp_path / "input.json"
    _write_json(source, value)

    with pytest.raises(registry.ArtifactRegistryError, match="missing required gates"):
        _invoke_build(source, roots, tmp_path / "registry.json")


def test_omission_requires_exact_receipt_complement(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    value = _reconciliation(*roots, ready_ids=())
    value["artifacts"][0]["omission"]["unsatisfied_gates"].pop()  # type: ignore[index,union-attr]
    source = tmp_path / "input.json"
    _write_json(source, value)

    with pytest.raises(registry.ArtifactRegistryError, match="exactly complement"):
        _invoke_build(source, roots, tmp_path / "registry.json")


def test_coauthor_omission_requires_complete_receipts(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    value = _reconciliation(*roots, ready_ids=())
    value["artifacts"][0]["omission"]["reason_code"] = (  # type: ignore[index]
        "coauthor_decision_to_omit"
    )
    source = tmp_path / "input.json"
    _write_json(source, value)

    with pytest.raises(registry.ArtifactRegistryError, match="complete gates"):
        _invoke_build(source, roots, tmp_path / "registry.json")


def test_closed_schema_rejects_row_bearing_payload_before_outputs_open(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    value = _reconciliation(*roots)
    artifact = value["artifacts"][0]  # type: ignore[index]
    artifact["rows"] = [{"gene_a": "A", "gene_b": "B"}]  # type: ignore[index]
    output = roots[1] / artifact["outputs"][0]["release_member"]  # type: ignore[index]
    output.unlink()
    source = tmp_path / "input.json"
    _write_json(source, value)

    with pytest.raises(registry.ArtifactRegistryError, match="non-closed schema"):
        _invoke_build(source, roots, tmp_path / "registry.json")


def test_inventory_rejects_missing_and_unknown_semantic_artifacts(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    value = _reconciliation(*roots, ready_ids=())
    artifacts = value["artifacts"]  # type: ignore[assignment]
    artifacts.pop()  # type: ignore[union-attr]
    artifacts.append(  # type: ignore[union-attr]
        {
            "semantic_id": "unsupported_numbered_figure",
            "status": "omitted",
            "gate_receipts": [],
            "omission": {
                "reason_code": "required_gate_not_satisfied",
                "unsatisfied_gates": ["K500"],
            },
        },
    )
    source = tmp_path / "input.json"
    _write_json(source, value)

    with pytest.raises(registry.ArtifactRegistryError, match="inventory mismatch"):
        _invoke_build(source, roots, tmp_path / "registry.json")


def test_registry_is_deterministic_across_input_orderings(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    value = _reconciliation(*roots, output_count=2)
    first = value["artifacts"][0]  # type: ignore[index]
    first["source_data"].append(  # type: ignore[union-attr]
        {
            "source_id": "aaa-additional-source",
            "release_member": "source-data/additional.json",
            "role": "provenance",
            "sha256": "3" * 64,
            "bytes": 7,
        },
    )
    value_reordered = copy.deepcopy(value)
    value_reordered["artifacts"].reverse()  # type: ignore[union-attr]
    value_reordered["gate_ledger"].reverse()  # type: ignore[union-attr]
    ready = next(  # type: ignore[arg-type]
        record
        for record in value_reordered["artifacts"]
        if record["status"] == "ready"
    )
    ready["gate_receipts"].reverse()
    ready["source_data"].reverse()
    ready["outputs"].reverse()

    _, _, destination_a = _build(tmp_path, roots, value, stem="registry_a")
    _, _, destination_b = _build(
        tmp_path,
        roots,
        value_reordered,
        stem="registry_b",
    )
    assert destination_a.read_bytes() == destination_b.read_bytes()


def test_publication_is_no_replace_and_preserves_existing_bytes(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    value = _reconciliation(*roots)
    _, source, destination = _build(tmp_path, roots, value)
    before = destination.read_bytes()

    with pytest.raises(registry.ArtifactRegistryError, match="already exists"):
        _invoke_build(source, roots, destination)
    assert destination.read_bytes() == before


def test_renderer_digest_must_match_opaque_script_bytes(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    value = _reconciliation(*roots)
    value["artifacts"][0]["renderer"]["sha256"] = "f" * 64  # type: ignore[index]
    source = tmp_path / "input.json"
    _write_json(source, value)

    with pytest.raises(
        registry.ArtifactRegistryError,
        match="does not match the script",
    ):
        _invoke_build(source, roots, tmp_path / "registry.json")


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [("sha256", "f" * 64), ("bytes", 999)],
)
def test_rendered_output_identity_must_match_opaque_bytes(
    tmp_path: Path,
    roots: tuple[Path, Path],
    field: str,
    bad_value: object,
) -> None:
    value = _reconciliation(*roots)
    value["artifacts"][0]["outputs"][0][field] = bad_value  # type: ignore[index]
    source = tmp_path / "input.json"
    _write_json(source, value)

    with pytest.raises(registry.ArtifactRegistryError, match="rendered output"):
        _invoke_build(source, roots, tmp_path / "registry.json")


def test_rendered_output_is_confined_to_its_semantic_artifact(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    value = _reconciliation(*roots)
    value["artifacts"][0]["outputs"][0]["release_member"] = (  # type: ignore[index]
        "rendered/another_artifact/output.pdf"
    )
    source = tmp_path / "input.json"
    _write_json(source, value)

    with pytest.raises(registry.ArtifactRegistryError, match="must start with"):
        _invoke_build(source, roots, tmp_path / "registry.json")


@pytest.mark.parametrize(
    ("semantic_id", "media_type", "suffix"),
    [
        ("release_provenance", "application/json", ".json"),
        ("raw_supplementary_inventory", "text/csv", ".csv"),
        ("cross_cancer_bmr_co_sensitivity", "image/tiff", ".tif"),
        ("provider_conjunction_summary", "application/x-tex", ".tex"),
    ],
)
def test_release_compatible_rendered_media_are_supported(
    tmp_path: Path,
    roots: tuple[Path, Path],
    semantic_id: str,
    media_type: str,
    suffix: str,
) -> None:
    value = _reconciliation(*roots, ready_ids=(semantic_id,))
    artifact = next(  # type: ignore[arg-type]
        record for record in value["artifacts"] if record["semantic_id"] == semantic_id
    )
    output = artifact["outputs"][0]
    original_path = roots[1] / output["release_member"]  # type: ignore[arg-type]
    raw = original_path.read_bytes()
    replacement_member = f"rendered/{semantic_id}/compatible-output{suffix}"
    replacement_path = roots[1] / replacement_member
    replacement_path.write_bytes(raw)
    output["release_member"] = replacement_member  # type: ignore[index]
    output["media_type"] = media_type  # type: ignore[index]

    receipt, _, _ = _build(
        tmp_path,
        roots,
        value,
        stem=f"media_{suffix.removeprefix('.')}",
    )
    assert receipt.ready_count == 1


def test_figure_cannot_be_ready_with_only_data_sidecars(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    value = _reconciliation(*roots)
    output = value["artifacts"][0]["outputs"][0]  # type: ignore[index]
    original_path = roots[1] / output["release_member"]  # type: ignore[arg-type]
    raw = original_path.read_bytes()
    replacement_member = (
        "rendered/cross_cancer_bmr_co_sensitivity/data-only.json"
    )
    (roots[1] / replacement_member).write_bytes(raw)
    output["release_member"] = replacement_member  # type: ignore[index]
    output["media_type"] = "application/json"  # type: ignore[index]
    source = tmp_path / "input.json"
    _write_json(source, value)

    with pytest.raises(
        registry.ArtifactRegistryError,
        match="compatible with 'figure'",
    ):
        _invoke_build(source, roots, tmp_path / "registry.json")


def test_source_reference_rejects_path_traversal_without_opening_source_rows(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    value = _reconciliation(*roots)
    value["artifacts"][0]["source_data"][0]["release_member"] = (  # type: ignore[index]
        "../private/results.csv"
    )
    source = tmp_path / "input.json"
    _write_json(source, value)

    with pytest.raises(registry.ArtifactRegistryError, match="declared release root"):
        _invoke_build(source, roots, tmp_path / "registry.json")


@pytest.mark.parametrize(
    "unsafe_member",
    [
        "source-data/control\ncharacter.csv",
        "source-data/surrogate-\ud800.csv",
        "source-data/bidi-\u202e.csv",
    ],
)
def test_source_reference_rejects_control_and_surrogate_characters(
    tmp_path: Path,
    roots: tuple[Path, Path],
    unsafe_member: str,
) -> None:
    value = _reconciliation(*roots)
    value["artifacts"][0]["source_data"][0]["release_member"] = unsafe_member  # type: ignore[index]
    source = tmp_path / "input.json"
    _write_json(source, value)

    with pytest.raises(registry.ArtifactRegistryError, match="canonical POSIX"):
        _invoke_build(source, roots, tmp_path / "registry.json")


def test_gate_receipt_identity_is_globally_consistent(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    ready_ids = (
        "cross_cancer_bmr_co_sensitivity",
        "interaction_model_diagnostic_panels",
    )
    value = _reconciliation(*roots, ready_ids=ready_ids)
    second = value["artifacts"][1]  # type: ignore[index]
    second["gate_receipts"][0]["sha256"] = "f" * 64  # type: ignore[index]
    source = tmp_path / "input.json"
    _write_json(source, value)

    with pytest.raises(registry.ArtifactRegistryError, match="global ledger"):
        _invoke_build(source, roots, tmp_path / "registry.json")


def test_omitted_artifact_cannot_call_a_global_gate_unsatisfied(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    value = _reconciliation(*roots)
    artifact = value["artifacts"][1]  # type: ignore[index]
    artifact["gate_receipts"] = [  # type: ignore[index]
        receipt
        for receipt in artifact["gate_receipts"]  # type: ignore[union-attr]
        if receipt["gate"] != "K500"
    ]
    artifact["omission"] = {  # type: ignore[index]
        "reason_code": "required_gate_not_satisfied",
        "unsatisfied_gates": ["K500"],
    }
    source = tmp_path / "input.json"
    _write_json(source, value)

    with pytest.raises(registry.ArtifactRegistryError, match="global ledger"):
        _invoke_build(source, roots, tmp_path / "registry.json")


def test_source_identity_is_globally_consistent(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    ready_ids = (
        "cross_cancer_bmr_co_sensitivity",
        "interaction_model_diagnostic_panels",
    )
    value = _reconciliation(*roots, ready_ids=ready_ids)
    first_source = value["artifacts"][0]["source_data"][0]  # type: ignore[index]
    second_source = value["artifacts"][1]["source_data"][0]  # type: ignore[index]
    second_source["source_id"] = first_source["source_id"]  # type: ignore[index]
    source = tmp_path / "input.json"
    _write_json(source, value)

    with pytest.raises(registry.ArtifactRegistryError, match="source_id"):
        _invoke_build(source, roots, tmp_path / "registry.json")


def test_output_identity_cannot_be_reused_across_artifacts(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    ready_ids = (
        "cross_cancer_bmr_co_sensitivity",
        "interaction_model_diagnostic_panels",
    )
    value = _reconciliation(*roots, ready_ids=ready_ids)
    first_id = value["artifacts"][0]["outputs"][0]["output_id"]  # type: ignore[index]
    value["artifacts"][1]["outputs"][0]["output_id"] = first_id  # type: ignore[index]
    source = tmp_path / "input.json"
    _write_json(source, value)

    with pytest.raises(
        registry.ArtifactRegistryError,
        match="multiple semantic artifacts",
    ):
        _invoke_build(source, roots, tmp_path / "registry.json")


def test_validator_rejects_payload_tampering(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    value = _reconciliation(*roots)
    _, _, destination = _build(tmp_path, roots, value)
    manifest = json.loads(destination.read_bytes())
    manifest["release"]["release_id"] = "tampered-release"
    destination.chmod(0o600)
    _write_json(destination, manifest)

    with pytest.raises(registry.ArtifactRegistryError, match="payload digest"):
        registry.validate_artifact_registry(
            destination,
            *roots,
            expected_manifest_sha256=_sha256(destination.read_bytes()),
        )


def test_validator_rejects_noncanonical_serialization(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    value = _reconciliation(*roots)
    _, _, destination = _build(tmp_path, roots, value)
    manifest = json.loads(destination.read_bytes())
    destination.chmod(0o600)
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(registry.ArtifactRegistryError, match="not canonical JSON"):
        registry.validate_artifact_registry(
            destination,
            *roots,
            expected_manifest_sha256=_sha256(destination.read_bytes()),
        )


def test_validator_rejects_renderer_or_output_mutation_after_publication(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    value = _reconciliation(*roots)
    receipt, _, destination = _build(tmp_path, roots, value)
    output_member = value["artifacts"][0]["outputs"][0]["release_member"]  # type: ignore[index]
    (roots[1] / output_member).write_bytes(b"%PDF-1.7 mutated\n")  # type: ignore[arg-type]

    with pytest.raises(registry.ArtifactRegistryError, match="rendered output"):
        registry.validate_artifact_registry(
            destination,
            *roots,
            expected_manifest_sha256=receipt.manifest_sha256,
        )


def test_symlinked_renderer_is_rejected(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    value = _reconciliation(*roots)
    script_member = value["artifacts"][0]["renderer"]["script"]  # type: ignore[index]
    script = roots[0] / script_member  # type: ignore[arg-type]
    raw = script.read_bytes()
    target = tmp_path / "other_renderer.py"
    target.write_bytes(raw)
    script.unlink()
    script.symlink_to(target)
    source = tmp_path / "input.json"
    _write_json(source, value)

    with pytest.raises(registry.ArtifactRegistryError, match="cannot open"):
        _invoke_build(source, roots, tmp_path / "registry.json")


def test_builder_binds_the_exact_live_implementation(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    builder = roots[0] / "analysis/build_tcga_revision_artifact_registry.py"
    builder.write_bytes(b'"""Different synthetic builder."""\n')
    value = _reconciliation(*roots)
    source = tmp_path / "input.json"
    _write_json(source, value)

    with pytest.raises(registry.ArtifactRegistryError, match="live artifact registry"):
        _invoke_build(source, roots, tmp_path / "registry.json")


def test_validator_rechecks_bound_builder_implementation(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    value = _reconciliation(*roots)
    receipt, _, destination = _build(tmp_path, roots, value)
    builder = roots[0] / "analysis/build_tcga_revision_artifact_registry.py"
    builder.write_bytes(b'"""Mutated builder archive."""\n')

    with pytest.raises(registry.ArtifactRegistryError, match="builder"):
        registry.validate_artifact_registry(
            destination,
            *roots,
            expected_manifest_sha256=receipt.manifest_sha256,
        )


def test_api_has_no_row_bearing_source_root_and_missing_sources_are_not_opened(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    parameters = inspect.signature(registry.build_artifact_registry).parameters
    assert set(parameters) == {
        "reconciliation_manifest",
        "renderer_root",
        "rendered_output_root",
        "destination",
        "expected_reconciliation_sha256",
    }
    value = _reconciliation(*roots)
    value["artifacts"][0]["source_data"][0]["release_member"] = (  # type: ignore[index]
        "not-present/complete-family.csv"
    )
    receipt, _, _ = _build(tmp_path, roots, value)
    assert receipt.ready_count == 1


def test_conceptual_method_overview_requires_no_fabricated_source_data(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    value = _reconciliation(*roots, ready_ids=("method_overview",))
    method = next(  # type: ignore[arg-type]
        artifact
        for artifact in value["artifacts"]
        if artifact["semantic_id"] == "method_overview"
    )
    assert method["source_data"] == []

    receipt, _, destination = _build(tmp_path, roots, value)
    published = next(
        artifact
        for artifact in json.loads(destination.read_bytes())["artifacts"]
        if artifact["semantic_id"] == "method_overview"
    )
    assert receipt.ready_count == 1
    assert published["source_requirement"] == "none"
    assert published["source_data"] == []


def test_conceptual_method_overview_rejects_fabricated_source_data(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    value = _reconciliation(*roots, ready_ids=("method_overview",))
    method = next(  # type: ignore[arg-type]
        artifact
        for artifact in value["artifacts"]
        if artifact["semantic_id"] == "method_overview"
    )
    method["source_data"] = [  # type: ignore[index]
        {
            "source_id": "fake-method-data",
            "release_member": "source-data/fake.csv",
            "role": "primary",
            "sha256": "a" * 64,
            "bytes": 1,
        },
    ]
    source = tmp_path / "input.json"
    _write_json(source, value)

    with pytest.raises(registry.ArtifactRegistryError, match="conceptual artifact"):
        _invoke_build(source, roots, tmp_path / "registry.json")


def test_release_provenance_requires_an_upstream_manifest_source(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    value = _reconciliation(*roots, ready_ids=("release_provenance",))
    provenance = next(  # type: ignore[arg-type]
        artifact
        for artifact in value["artifacts"]
        if artifact["semantic_id"] == "release_provenance"
    )
    provenance["source_data"] = []  # type: ignore[index]
    source = tmp_path / "input.json"
    _write_json(source, value)

    with pytest.raises(registry.ArtifactRegistryError, match="at least one source"):
        _invoke_build(source, roots, tmp_path / "registry.json")


@pytest.mark.parametrize(
    "missing_digest",
    [_POSTPROCESS_SHA256, _SOURCE_MANIFEST_SHA256],
)
def test_release_provenance_exactly_binds_both_release_manifest_digests(
    tmp_path: Path,
    roots: tuple[Path, Path],
    missing_digest: str,
) -> None:
    value = _reconciliation(*roots, ready_ids=("release_provenance",))
    provenance = next(  # type: ignore[arg-type]
        artifact
        for artifact in value["artifacts"]
        if artifact["semantic_id"] == "release_provenance"
    )
    source = next(  # type: ignore[arg-type]
        record
        for record in provenance["source_data"]
        if record["sha256"] == missing_digest
    )
    source["sha256"] = "a" * 64
    reconciliation = tmp_path / "input.json"
    _write_json(reconciliation, value)

    with pytest.raises(
        registry.ArtifactRegistryError,
        match="must exactly bind",
    ):
        _invoke_build(reconciliation, roots, tmp_path / "registry.json")


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("source_id", "a-renamed-postprocess-manifest"),
        ("release_member", "postprocess/renamed_manifest.json"),
    ],
)
def test_release_provenance_freezes_upstream_manifest_identities(
    tmp_path: Path,
    roots: tuple[Path, Path],
    field: str,
    replacement: str,
) -> None:
    value = _reconciliation(*roots, ready_ids=("release_provenance",))
    provenance = next(  # type: ignore[arg-type]
        artifact
        for artifact in value["artifacts"]
        if artifact["semantic_id"] == "release_provenance"
    )
    provenance["source_data"][0][field] = replacement  # type: ignore[index]
    reconciliation = tmp_path / "input.json"
    _write_json(reconciliation, value)

    with pytest.raises(registry.ArtifactRegistryError, match="canonical"):
        _invoke_build(reconciliation, roots, tmp_path / "registry.json")


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("sha256", "a" * 64),
        ("source_id", "a-renamed-postprocess-manifest"),
        ("release_member", "postprocess/renamed_manifest.json"),
    ],
)
def test_validator_rechecks_exact_release_provenance_manifest_bindings(
    tmp_path: Path,
    roots: tuple[Path, Path],
    field: str,
    replacement: str,
) -> None:
    value = _reconciliation(*roots, ready_ids=("release_provenance",))
    _, _, destination = _build(tmp_path, roots, value)
    manifest = json.loads(destination.read_bytes())
    provenance = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["semantic_id"] == "release_provenance"
    )
    provenance["source_data"][0][field] = replacement
    payload = dict(manifest)
    payload.pop("registry_payload_sha256")
    manifest["registry_payload_sha256"] = _sha256(
        _canonical(payload).removesuffix(b"\n"),
    )
    destination.chmod(0o600)
    _write_json(destination, manifest)

    with pytest.raises(registry.ArtifactRegistryError, match="must exactly bind"):
        registry.validate_artifact_registry(
            destination,
            *roots,
            expected_manifest_sha256=_sha256(destination.read_bytes()),
        )


def test_reconciliation_itself_must_be_canonical_and_small_metadata(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    value = _reconciliation(*roots, ready_ids=())
    source = tmp_path / "input.json"
    source.write_text(json.dumps(value, indent=2), encoding="utf-8")

    with pytest.raises(registry.ArtifactRegistryError, match="not canonical JSON"):
        _invoke_build(source, roots, tmp_path / "registry.json")


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b'{"schema":"first","schema":"second"}\n', "duplicate JSON key"),
        (b'{"artifacts":[],"release":NaN,"schema":"x"}\n', "non-finite"),
    ],
)
def test_json_decoder_rejects_duplicate_keys_and_nonfinite_constants(
    tmp_path: Path,
    roots: tuple[Path, Path],
    raw: bytes,
    message: str,
) -> None:
    source = tmp_path / "input.json"
    source.write_bytes(raw)

    with pytest.raises(registry.ArtifactRegistryError, match=message):
        _invoke_build(source, roots, tmp_path / "registry.json")


def test_independent_reconciliation_anchor_precedes_render_opening(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    value = _reconciliation(*roots)
    output_member = value["artifacts"][0]["outputs"][0]["release_member"]  # type: ignore[index]
    (roots[1] / output_member).unlink()  # type: ignore[arg-type]
    source = tmp_path / "input.json"
    _write_json(source, value)

    with pytest.raises(registry.ArtifactRegistryError, match="independent anchor"):
        _invoke_build(
            source,
            roots,
            tmp_path / "registry.json",
            expected_sha256="f" * 64,
        )


def test_builder_rejects_canonical_replacement_before_opening_render_roots(
    tmp_path: Path,
    roots: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.json"
    value = _reconciliation(*roots)
    _write_json(source, value)
    expected_sha256 = _sha256(source.read_bytes())
    value["release"]["release_id"] = "canonical-malicious-replacement"  # type: ignore[index]
    _write_json(source, value)
    root_opened = False

    def reject_root_open(*_args: object, **_kwargs: object) -> None:
        nonlocal root_opened
        root_opened = True
        raise AssertionError

    monkeypatch.setattr(registry, "_pin_root", reject_root_open)
    with pytest.raises(registry.ArtifactRegistryError, match="independent anchor"):
        registry.build_artifact_registry(
            source,
            *roots,
            tmp_path / "registry.json",
            expected_reconciliation_sha256=expected_sha256,
        )
    assert not root_opened


def test_api_requires_independent_build_and_validation_anchors() -> None:
    build_parameter = inspect.signature(
        registry.build_artifact_registry,
    ).parameters["expected_reconciliation_sha256"]
    validate_parameter = inspect.signature(
        registry.validate_artifact_registry,
    ).parameters["expected_manifest_sha256"]
    assert build_parameter.default is inspect.Parameter.empty
    assert validate_parameter.default is inspect.Parameter.empty


@pytest.mark.parametrize(
    "arguments",
    [
        [
            "build",
            "--reconciliation",
            "input.json",
            "--renderer-root",
            "repo",
            "--rendered-output-root",
            "release",
            "--out",
            "registry.json",
        ],
        [
            "validate",
            "--manifest",
            "registry.json",
            "--renderer-root",
            "repo",
            "--rendered-output-root",
            "release",
        ],
    ],
)
def test_cli_requires_independent_build_and_validation_anchors(
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit):
        registry._parser().parse_args(arguments)  # noqa: SLF001


def test_validator_rejects_canonical_replacement_before_opening_render_roots(
    tmp_path: Path,
    roots: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _reconciliation(*roots)
    receipt, _, destination = _build(tmp_path, roots, value)
    malicious = json.loads(destination.read_bytes())
    malicious["release"]["release_id"] = "canonical-malicious-replacement"
    payload = dict(malicious)
    payload.pop("registry_payload_sha256")
    malicious["registry_payload_sha256"] = _sha256(
        _canonical(payload).removesuffix(b"\n"),
    )
    destination.chmod(0o600)
    _write_json(destination, malicious)
    root_opened = False

    def reject_root_open(*_args: object, **_kwargs: object) -> None:
        nonlocal root_opened
        root_opened = True
        raise AssertionError

    monkeypatch.setattr(registry, "_pin_root", reject_root_open)
    with pytest.raises(registry.ArtifactRegistryError, match="independent anchor"):
        registry.validate_artifact_registry(
            destination,
            *roots,
            expected_manifest_sha256=receipt.manifest_sha256,
        )
    assert not root_opened


def test_staged_registry_hash_readback_prevents_corrupt_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "registry.json"
    original_write = registry.os.write

    def corrupt_write(descriptor: int, raw: bytes) -> int:
        corrupt = (b"X" + raw[1:]) if raw else raw
        return original_write(descriptor, corrupt)

    monkeypatch.setattr(registry.os, "write", corrupt_write)
    with pytest.raises(registry.ArtifactRegistryError, match="readback accounting"):
        registry._publish_no_replace(destination, b'{"valid":true}\n')  # noqa: SLF001
    assert not destination.exists()


@pytest.mark.parametrize("replacement_fsync_call", [3, 4])
def test_publication_detects_destination_replacement_at_each_directory_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_fsync_call: int,
) -> None:
    destination = tmp_path / "registry.json"
    attacker = b'{"attacker":true}\n'
    fsync_calls = 0
    original_fsync = registry.os.fsync

    def replacing_fsync(descriptor: int) -> None:
        nonlocal fsync_calls
        original_fsync(descriptor)
        fsync_calls += 1
        if fsync_calls == replacement_fsync_call:
            destination.unlink()
            destination.write_bytes(attacker)

    monkeypatch.setattr(registry.os, "fsync", replacing_fsync)
    with pytest.raises(
        registry.ArtifactRegistryError,
        match="does not match the staged file",
    ):
        registry._publish_no_replace(  # noqa: SLF001
            destination,
            b'{"valid":true}\n',
        )
    assert destination.read_bytes() == attacker


def test_publication_final_check_detects_callback_destination_replacement(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "registry.json"
    attacker = b'{"attacker":true}\n'
    callback_calls = 0

    def replace_during_final_callback() -> None:
        nonlocal callback_calls
        callback_calls += 1
        if callback_calls == 3:
            destination.unlink()
            destination.write_bytes(attacker)

    with pytest.raises(
        registry.ArtifactRegistryError,
        match="does not match the staged file",
    ):
        registry._publish_no_replace(  # noqa: SLF001
            destination,
            b'{"valid":true}\n',
            link_boundary_check=replace_during_final_callback,
        )
    assert callback_calls == 3
    assert destination.read_bytes() == attacker


def test_publication_detects_directory_entry_swap_during_final_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "registry.json"
    moved = tmp_path / "moved.json"
    attacker = b'{"attacker":true}\n'
    digest_calls = 0
    original_digest = registry._digest_descriptor  # noqa: SLF001

    def swapping_digest(descriptor: int):
        nonlocal digest_calls
        result = original_digest(descriptor)
        digest_calls += 1
        if digest_calls == 4:
            destination.rename(moved)
            destination.write_bytes(attacker)
        return result

    monkeypatch.setattr(registry, "_digest_descriptor", swapping_digest)
    with pytest.raises(
        registry.ArtifactRegistryError,
        match="changed during readback",
    ):
        registry._publish_no_replace(  # noqa: SLF001
            destination,
            b'{"valid":true}\n',
        )
    assert destination.read_bytes() == attacker
    assert moved.read_bytes() == b'{"valid":true}\n'


def test_publication_detects_parent_swap_during_final_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "publication"
    moved_parent = tmp_path / "publication-moved"
    parent.mkdir()
    destination = parent / "registry.json"
    attacker = b'{"attacker":true}\n'
    digest_calls = 0
    original_digest = registry._digest_descriptor  # noqa: SLF001

    def swapping_digest(descriptor: int):
        nonlocal digest_calls
        result = original_digest(descriptor)
        digest_calls += 1
        if digest_calls == 4:
            parent.rename(moved_parent)
            parent.mkdir()
            destination.write_bytes(attacker)
        return result

    monkeypatch.setattr(registry, "_digest_descriptor", swapping_digest)
    with pytest.raises(registry.ArtifactRegistryError, match="parent changed"):
        registry._publish_no_replace(  # noqa: SLF001
            destination,
            b'{"valid":true}\n',
        )
    assert destination.read_bytes() == attacker
    assert not (moved_parent / "registry.json").exists()


@pytest.mark.parametrize("failure", ["unlink", "fsync"])
def test_publication_cleanup_attempts_all_steps_and_closes_parent_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    parent = tmp_path / "publication"
    parent.mkdir()
    destination = parent / "registry.json"
    captured_parent_descriptors: list[int] = []
    unlink_attempts: list[str] = []
    boundary_calls = 0
    fsync_calls = 0
    original_ensure = registry._ensure_destination_parent  # noqa: SLF001
    original_fstat = registry.os.fstat
    original_fsync = registry.os.fsync
    original_unlink = registry.os.unlink

    def tracking_ensure(path: Path) -> tuple[Path, int]:
        result = original_ensure(path)
        captured_parent_descriptors.append(result[1])
        return result

    def injected_unlink(member: str, **kwargs: object) -> None:
        unlink_attempts.append(member)
        if failure == "unlink" and member == destination.name:
            message = "injected cleanup unlink failure"
            raise OSError(message)
        original_unlink(member, **kwargs)

    def injected_fsync(descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if failure == "fsync" and fsync_calls == 4:
            message = "injected cleanup fsync failure"
            raise OSError(message)
        original_fsync(descriptor)

    def fail_after_link() -> None:
        nonlocal boundary_calls
        boundary_calls += 1
        if boundary_calls == 2:
            message = "injected boundary failure"
            raise registry.ArtifactRegistryError(message)

    monkeypatch.setattr(registry, "_ensure_destination_parent", tracking_ensure)
    monkeypatch.setattr(registry.os, "unlink", injected_unlink)
    monkeypatch.setattr(registry.os, "fsync", injected_fsync)
    with pytest.raises(OSError, match="injected cleanup"):
        registry._publish_no_replace(  # noqa: SLF001
            destination,
            b'{"valid":true}\n',
            link_boundary_check=fail_after_link,
        )

    assert captured_parent_descriptors
    assert destination.name in unlink_attempts
    assert any(
        member.startswith(".registry.json.staging-")
        for member in unlink_attempts
    )
    with pytest.raises(OSError, match=r"[Bb]ad file descriptor"):
        original_fstat(captured_parent_descriptors[0])


def test_publication_rejects_symlinked_destination_ancestors(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    nested = actual / "nested"
    nested.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)

    with pytest.raises(registry.ArtifactRegistryError, match="symlinked ancestors"):
        registry._publish_no_replace(  # noqa: SLF001
            alias / "nested" / "registry.json",
            b'{"valid":true}\n',
        )


def test_publication_rejects_destination_parent_rename_after_pin(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "publication"
    parent.mkdir()
    moved_parent = tmp_path / "publication-moved"
    calls = 0

    def rename_parent() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            parent.rename(moved_parent)
            parent.mkdir()

    with pytest.raises(registry.ArtifactRegistryError, match="parent changed"):
        registry._publish_no_replace(  # noqa: SLF001
            parent / "registry.json",
            b'{"valid":true}\n',
            link_boundary_check=rename_parent,
        )
    assert not (parent / "registry.json").exists()
    assert not (moved_parent / "registry.json").exists()


def test_member_mutation_at_link_boundary_rolls_back_registry(
    tmp_path: Path,
    roots: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _reconciliation(*roots)
    output_member = value["artifacts"][0]["outputs"][0]["release_member"]  # type: ignore[index]
    output = roots[1] / output_member  # type: ignore[arg-type]
    source = tmp_path / "input.json"
    destination = tmp_path / "registry.json"
    _write_json(source, value)
    original_link = registry.os.link

    def mutating_link(*args: object, **kwargs: object) -> None:
        output.write_bytes(b"%PDF-1.7 mutated at link boundary\n")
        original_link(*args, **kwargs)

    monkeypatch.setattr(registry.os, "link", mutating_link)
    with pytest.raises(
        registry.ArtifactRegistryError,
        match="changed during validation",
    ):
        _invoke_build(source, roots, destination)
    assert not destination.exists()


def test_builder_mutation_at_link_boundary_rolls_back_registry(
    tmp_path: Path,
    roots: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _reconciliation(*roots)
    source = tmp_path / "input.json"
    destination = tmp_path / "registry.json"
    _write_json(source, value)
    builder = roots[0] / "analysis/build_tcga_revision_artifact_registry.py"
    original_link = registry.os.link

    def mutating_link(*args: object, **kwargs: object) -> None:
        builder.write_bytes(b'"""Mutated at publication boundary."""\n')
        original_link(*args, **kwargs)

    monkeypatch.setattr(registry.os, "link", mutating_link)
    with pytest.raises(registry.ArtifactRegistryError, match="changed during"):
        _invoke_build(source, roots, destination)
    assert not destination.exists()


@pytest.mark.parametrize("target", ["root", "destination"])
def test_immediate_fstat_failure_closes_new_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    opened: list[int] = []
    original_open = registry.os.open
    original_fstat = registry.os.fstat

    def tracking_open(*args: object, **kwargs: object) -> int:
        descriptor = original_open(*args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def failing_fstat(_descriptor: int) -> stat.stat_result:
        message = "injected fstat failure"
        raise OSError(message)

    monkeypatch.setattr(registry.os, "open", tracking_open)
    monkeypatch.setattr(registry.os, "fstat", failing_fstat)

    def invoke_root() -> object:
        return registry._pin_root(  # noqa: SLF001
            directory,
            context="synthetic root",
        )

    def invoke_destination() -> object:
        return registry._ensure_destination_parent(  # noqa: SLF001
            directory / "registry.json",
        )

    invoke = invoke_root if target == "root" else invoke_destination
    with pytest.raises(registry.ArtifactRegistryError, match="cannot inspect pinned"):
        invoke()
    assert len(opened) == 1
    with pytest.raises(OSError, match=r"[Bb]ad file descriptor"):
        original_fstat(opened[0])


def test_noncanonical_output_rejection_closes_every_pinned_member(
    tmp_path: Path,
    roots: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _reconciliation(*roots, output_count=2)
    _, _, destination = _build(tmp_path, roots, value)
    manifest = json.loads(destination.read_bytes())
    manifest["artifacts"][0]["outputs"].reverse()
    payload = dict(manifest)
    payload.pop("registry_payload_sha256")
    manifest["registry_payload_sha256"] = _sha256(
        _canonical(payload).removesuffix(b"\n"),
    )
    destination.chmod(0o600)
    _write_json(destination, manifest)

    opened: list[int] = []
    closed: list[int] = []
    original_pin = registry._pin_member  # noqa: SLF001
    original_close = registry._PinnedMember.close  # noqa: SLF001

    def tracking_pin(*args: object, **kwargs: object):
        pinned = original_pin(*args, **kwargs)
        opened.append(pinned.descriptor)
        return pinned

    def tracking_close(pinned) -> None:
        closed.append(pinned.descriptor)
        original_close(pinned)

    monkeypatch.setattr(registry, "_pin_member", tracking_pin)
    monkeypatch.setattr(registry._PinnedMember, "close", tracking_close)  # noqa: SLF001
    with pytest.raises(registry.ArtifactRegistryError, match="canonically ordered"):
        registry.validate_artifact_registry(
            destination,
            *roots,
            expected_manifest_sha256=_sha256(destination.read_bytes()),
        )
    assert sorted(opened) == sorted(closed)


def test_expected_registry_digest_is_an_independent_validation_anchor(
    tmp_path: Path,
    roots: tuple[Path, Path],
) -> None:
    value = _reconciliation(*roots)
    receipt, _, destination = _build(tmp_path, roots, value)

    with pytest.raises(registry.ArtifactRegistryError, match="SHA-256 does not match"):
        registry.validate_artifact_registry(
            destination,
            *roots,
            expected_manifest_sha256="f" * 64,
        )
    assert registry.validate_artifact_registry(
        destination,
        *roots,
        expected_manifest_sha256=receipt.manifest_sha256,
    ).manifest_sha256 == receipt.manifest_sha256
