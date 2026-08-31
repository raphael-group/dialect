"""Clean-checkout guard for the public dependency-provenance boundary."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

PROVENANCE_ROOT = (
    Path(__file__).resolve().parents[1] / "provenance" / "dependencies"
)
EXTERNAL_README = PROVENANCE_ROOT.parents[1] / "external" / "README.md"
PROVENANCE_README = PROVENANCE_ROOT / "README.md"
SCHEMA_PATH = PROVENANCE_ROOT / "record.schema.json"

EXPECTED_RECORDS = {
    "atlas-code-v2.3.1-563ae0f": "atlas_code",
    "atlas-k100-v1.0.0-0ef212a": "atlas_k100_data",
    "cbase-v1.2-dialect-fork": "cbase_source",
    "dig-pancan-artifact-4402b76e": "dig_artifact",
    "digdriver-source-5bb565a": "dig_source",
    "discover-0.9.6-a46d99f": "discover_source",
    "megsa-source-9e75152f": "megsa_source",
    "msk-chord-2024-eb53cc4": "msk_raw_source",
    "msk-impact-50k-2026-eb53cc4": "msk_raw_source",
    "mutsig2cv-patch-1e0aa209": "mutsig_patch",
    "mutsig2cv-source-0109e27": "mutsig_source",
    "oncokb-cancer-gene-list-2024-12-19-56cea460": "oncokb_snapshot",
    "tcga-datahub-64392ef-32-study": "tcga_raw_source",
}
EXPECTED_BOUNDARIES: dict[str, dict[str, Any]] = {
    "atlas-code-v2.3.1-563ae0f": {
        "license_id": "BSD-3-Clause",
        "license_status": "permitted",
        "redistribution": "exclude",
        "included_in_public_release": False,
        "unresolved": [],
    },
    "atlas-k100-v1.0.0-0ef212a": {
        "license_id": "NOASSERTION",
        "license_status": "unknown",
        "redistribution": "exclude",
        "included_in_public_release": False,
        "unresolved": [
            "Publish a data license and third-party notice after inherited source "
            "terms are resolved.",
        ],
    },
    "cbase-v1.2-dialect-fork": {
        "license_id": "LicenseRef-CBaSE-Public-Domain AND BSD-3-Clause",
        "license_status": "permitted",
        "redistribution": "include",
        "included_in_public_release": True,
        "unresolved": [],
    },
    "dig-pancan-artifact-4402b76e": {
        "license_id": "NOASSERTION",
        "license_status": "unknown",
        "redistribution": "exclude",
        "included_in_public_release": False,
        "unresolved": [
            "Record authoritative acquisition URLs, versions, timestamps, and terms "
            "for all three artifacts.",
        ],
    },
    "digdriver-source-5bb565a": {
        "license_id": "BSD-3-Clause",
        "license_status": "permitted",
        "redistribution": "exclude",
        "included_in_public_release": False,
        "unresolved": [],
    },
    "discover-0.9.6-a46d99f": {
        "license_id": "Apache-2.0",
        "license_status": "permitted",
        "redistribution": "exclude",
        "included_in_public_release": False,
        "unresolved": [],
    },
    "megsa-source-9e75152f": {
        "license_id": "NOASSERTION",
        "license_status": "unknown",
        "redistribution": "exclude",
        "included_in_public_release": False,
        "unresolved": [
            "Identify the authoritative upstream artifact, version, source URL, and "
            "license or obtain written permission.",
        ],
    },
    "msk-chord-2024-eb53cc4": {
        "license_id": "CC-BY-NC-ND-4.0",
        "license_status": "restricted",
        "redistribution": "exclude",
        "included_in_public_release": False,
        "unresolved": [
            "Freeze exact file bytes, access times, and attribution if MSK "
            "sensitivity analysis is retained.",
            "Approve internal analysis and derived-output treatment.",
        ],
    },
    "msk-impact-50k-2026-eb53cc4": {
        "license_id": "CC-BY-NC-ND-4.0",
        "license_status": "restricted",
        "redistribution": "exclude",
        "included_in_public_release": False,
        "unresolved": [
            "Freeze exact file bytes, access times, and attribution if MSK "
            "sensitivity analysis is retained.",
            "Approve internal analysis and derived-output treatment.",
        ],
    },
    "mutsig2cv-patch-1e0aa209": {
        "license_id": "LicenseRef-Broad-MutSig2CV",
        "license_status": "restricted",
        "redistribution": "exclude",
        "included_in_public_release": False,
        "unresolved": [
            "Obtain a written redistribution decision for the source-derived patch.",
        ],
    },
    "mutsig2cv-source-0109e27": {
        "license_id": "LicenseRef-Broad-MutSig2CV",
        "license_status": "restricted",
        "redistribution": "exclude",
        "included_in_public_release": False,
        "unresolved": [
            "Close the license obligation concerning modifications and any required "
            "notice to Broad.",
        ],
    },
    "oncokb-cancer-gene-list-2024-12-19-56cea460": {
        "license_id": "NOASSERTION",
        "license_status": "unknown",
        "redistribution": "exclude",
        "included_in_public_release": False,
        "unresolved": [
            "Record the authoritative URL, access time, version identifier, terms, "
            "and derived-classification attribution decision.",
        ],
    },
    "tcga-datahub-64392ef-32-study": {
        "license_id": "NOASSERTION",
        "license_status": "unknown",
        "redistribution": "exclude",
        "included_in_public_release": False,
        "unresolved": [
            "Fetch and hash each retained study license at the pinned commit and "
            "record the raw-data redistribution decision.",
        ],
    },
}
BOUNDARY_KEYS = frozenset(next(iter(EXPECTED_BOUNDARIES.values())))
EXPECTED_FILES = {
    "README.md",
    "record.schema.json",
    *(f"{dependency_id}.json" for dependency_id in EXPECTED_RECORDS),
}
EXPECTED_RECORD_KEYS = {
    "record_schema_version",
    "dependency_id",
    "dependency_class",
    "recorded_on",
    "source",
    "version",
    "acquisition",
    "identity",
    "license_id",
    "license_status",
    "redistribution",
    "included_in_public_release",
    "source_artifacts",
    "scope",
    "unresolved",
}
EXPECTED_SOURCE_ARTIFACT_KEYS = {
    "role",
    "source_uri",
    "bytes",
    "sha256",
    "accessed_at",
}
EXPECTED_SOURCE_ARTIFACT_ROLES = {"clinical", "mutations", "panel_matrix"}
SCHEMA_ID = (
    "https://dialectcanceratlas.com/provenance/dependency-record-v1.schema.json"
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            msg = f"duplicate JSON object key: {key}"
            raise ValueError(msg)
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    assert isinstance(value, dict), f"{path} must contain a JSON object"
    return value


def _matches_schema(value: Any, schema: dict[str, Any], context: str) -> bool:
    try:
        _assert_matches_schema(value, schema, context)
    except AssertionError:
        return False
    return True


def _assert_matches_schema(  # noqa: C901, PLR0912, PLR0915
    value: Any,
    schema: dict[str, Any],
    context: str,
) -> None:
    """Validate the JSON-Schema keywords used by ``record.schema.json``."""
    if "oneOf" in schema:
        branches = schema["oneOf"]
        assert isinstance(branches, list), f"{context}: oneOf must be an array"
        match_count = sum(
            _matches_schema(value, branch, f"{context}.oneOf[{index}]")
            for index, branch in enumerate(branches)
        )
        assert match_count == 1, f"{context}: expected exactly one oneOf match"

    expected_type = schema.get("type")
    type_matches = {
        "array": isinstance(value, list),
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "object": isinstance(value, dict),
        "string": isinstance(value, str),
    }
    if expected_type is not None:
        assert expected_type in type_matches, (
            f"{context}: unsupported schema type {expected_type!r}"
        )
        assert type_matches[expected_type], (
            f"{context}: expected {expected_type}, got {type(value).__name__}"
        )

    if "const" in schema:
        assert value == schema["const"], f"{context}: const mismatch"
    if "enum" in schema:
        assert value in schema["enum"], f"{context}: value is outside enum"

    if isinstance(value, str):
        if "minLength" in schema:
            assert len(value) >= schema["minLength"], f"{context}: string is too short"
        if "pattern" in schema:
            assert re.search(schema["pattern"], value), (
                f"{context}: value does not match pattern"
            )

    if isinstance(value, int) and not isinstance(value, bool) and "minimum" in schema:
        assert value >= schema["minimum"], f"{context}: integer is below minimum"

    has_object_contract = expected_type == "object" or any(
        key in schema
        for key in ("properties", "required", "additionalProperties", "minProperties")
    )
    if has_object_contract:
        assert isinstance(value, dict), f"{context}: expected object contract"
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        assert required <= set(value), f"{context}: missing required keys"
        if schema.get("additionalProperties") is False:
            assert set(value) <= set(properties), f"{context}: unexpected object keys"
        if "minProperties" in schema:
            assert len(value) >= schema["minProperties"], (
                f"{context}: object has too few properties"
            )
        for key, property_schema in properties.items():
            if key in value:
                _assert_matches_schema(value[key], property_schema, f"{context}.{key}")

    has_array_contract = expected_type == "array" or any(
        key in schema
        for key in ("items", "minItems", "maxItems", "uniqueItems", "contains")
    )
    if has_array_contract:
        assert isinstance(value, list), f"{context}: expected array contract"
        if "minItems" in schema:
            assert len(value) >= schema["minItems"], f"{context}: array is too short"
        if "maxItems" in schema:
            assert len(value) <= schema["maxItems"], f"{context}: array is too long"
        if schema.get("uniqueItems") is True:
            canonical_items = {
                json.dumps(item, sort_keys=True, separators=(",", ":"))
                for item in value
            }
            assert len(canonical_items) == len(value), (
                f"{context}: array items are not unique"
            )
        if "items" in schema:
            for index, item in enumerate(value):
                _assert_matches_schema(item, schema["items"], f"{context}[{index}]")
        if "contains" in schema:
            contains_count = sum(
                _matches_schema(item, schema["contains"], f"{context}[{index}]")
                for index, item in enumerate(value)
            )
            assert contains_count >= schema.get("minContains", 1), (
                f"{context}: contains match count is below minimum"
            )
            if "maxContains" in schema:
                assert contains_count <= schema["maxContains"], (
                    f"{context}: contains match count is above maximum"
                )

    for index, clause in enumerate(schema.get("allOf", [])):
        _assert_matches_schema(value, clause, f"{context}.allOf[{index}]")

    if "if" in schema and _matches_schema(value, schema["if"], f"{context}.if"):
        _assert_matches_schema(value, schema["then"], f"{context}.then")


def test_dependency_provenance_inventory_is_exact() -> None:
    actual = {path.name for path in PROVENANCE_ROOT.iterdir()}
    assert actual == EXPECTED_FILES
    assert len(EXPECTED_FILES) == 15
    assert len(EXPECTED_RECORDS) == 13
    for filename in EXPECTED_FILES:
        path = PROVENANCE_ROOT / filename
        assert path.is_file(), f"{path} must be a regular file"
        assert not path.is_symlink(), f"{path} must not be a symlink"


def test_dependency_provenance_schema_is_fail_closed() -> None:
    schema = _load_json(SCHEMA_PATH)
    assert set(schema) == {
        "$id",
        "$schema",
        "additionalProperties",
        "allOf",
        "oneOf",
        "properties",
        "required",
        "title",
        "type",
    }
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == SCHEMA_ID
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == EXPECTED_RECORD_KEYS
    assert set(schema["properties"]) == EXPECTED_RECORD_KEYS
    assert schema["properties"]["record_schema_version"] == {"const": "1.0.0"}
    assert set(schema["properties"]["dependency_id"]["enum"]) == set(
        EXPECTED_RECORDS,
    )
    assert set(schema["properties"]["dependency_class"]["enum"]) == set(
        EXPECTED_RECORDS.values(),
    )

    source_artifact_schema = schema["properties"]["source_artifacts"]["items"]
    assert source_artifact_schema["type"] == "object"
    assert source_artifact_schema["additionalProperties"] is False
    assert set(source_artifact_schema["required"]) == EXPECTED_SOURCE_ARTIFACT_KEYS
    assert set(source_artifact_schema["properties"]) == EXPECTED_SOURCE_ARTIFACT_KEYS
    assert set(source_artifact_schema["properties"]["role"]["enum"]) == (
        EXPECTED_SOURCE_ARTIFACT_ROLES
    )

    conditional_schemas = {
        clause["if"]["properties"]["dependency_id"]["const"]: clause
        for clause in schema["allOf"]
    }
    assert set(conditional_schemas) == set(EXPECTED_RECORDS)
    assert len(conditional_schemas) == len(schema["allOf"])
    for dependency_id, dependency_class in EXPECTED_RECORDS.items():
        clause = conditional_schemas[dependency_id]
        assert clause["if"]["required"] == ["dependency_id"]
        then_properties = clause["then"]["properties"]
        assert then_properties["dependency_class"]["const"] == dependency_class
        identity_schema = then_properties["identity"]
        assert identity_schema["type"] == "object"
        assert identity_schema["additionalProperties"] is False
        assert set(identity_schema["required"]) == set(identity_schema["properties"])

    boundary_schemas = {
        branch["properties"]["dependency_id"]["const"]: branch
        for branch in schema["oneOf"]
    }
    assert set(boundary_schemas) == set(EXPECTED_BOUNDARIES)
    assert len(boundary_schemas) == len(schema["oneOf"])
    for dependency_id, expected_boundary in EXPECTED_BOUNDARIES.items():
        branch = boundary_schemas[dependency_id]
        assert set(branch) == {"properties", "required"}
        assert set(branch["required"]) == {"dependency_id", *BOUNDARY_KEYS}
        expected_properties = {
            "dependency_id": {"const": dependency_id},
            **{
                key: {"const": value}
                for key, value in expected_boundary.items()
            },
        }
        assert branch["properties"] == expected_properties


def test_unselected_wesme_boundary_remains_conditional() -> None:
    """Do not turn an open comparator decision into a false code or release claim."""
    readme = " ".join(EXTERNAL_README.read_text(encoding="utf-8").split())
    required = (
        "public comparison API retains optional WeSME support",
        "no current corrected-revision dependency record or coauthor-approved "
        "comparator scope selects WeSME or WeSCO",
        "remain excluded from the prepared public release unless a future "
        "stage-scoped decision explicitly selects them",
        "provenance, license, acquisition, and redistribution review passes",
    )
    for phrase in required:
        assert phrase in readme
    assert "omitted from the corrected revision analysis" not in readme
    assert "hard-blocked from the public release" not in readme


def test_provenance_readme_requires_coordinated_boundary_change() -> None:
    """One edited record must never be described as sufficient for promotion."""
    readme = " ".join(PROVENANCE_README.read_text(encoding="utf-8").split())
    assert (
        "schema pins the exact current license, redistribution, inclusion, and "
        "unresolved-gate disposition for every dependency"
    ) in readme
    assert (
        "requires a reviewed, coordinated record/schema/test/manifest change, not "
        "an edit to one record"
    ) in readme


@pytest.mark.parametrize(
    ("dependency_id", "expected_boundary"),
    sorted(EXPECTED_BOUNDARIES.items()),
)
def test_dependency_release_boundary_is_exact(
    dependency_id: str,
    expected_boundary: dict[str, Any],
) -> None:
    """Freeze every license and redistribution disposition, not only its shape."""
    record = _load_json(PROVENANCE_ROOT / f"{dependency_id}.json")
    observed = {key: record[key] for key in BOUNDARY_KEYS}
    assert observed == expected_boundary
    assert record["included_in_public_release"] is (
        record["redistribution"] == "include"
    )
    if record["redistribution"] == "include":
        assert record["license_status"] == "permitted"
        assert record["unresolved"] == []
    if record["license_status"] in {"restricted", "unknown"}:
        assert record["redistribution"] == "exclude"
        assert record["included_in_public_release"] is False
        assert record["unresolved"]


def test_schema_rejects_atlas_release_boundary_escalation() -> None:
    """An excluded unknown dataset cannot become includable by editing five fields."""
    schema = _load_json(SCHEMA_PATH)
    record = _load_json(PROVENANCE_ROOT / "atlas-k100-v1.0.0-0ef212a.json")
    record.update(
        {
            "license_id": "BSD-3-Clause",
            "license_status": "permitted",
            "redistribution": "include",
            "included_in_public_release": True,
            "unresolved": [],
        },
    )
    assert not _matches_schema(record, schema, "mutated-atlas-k100")


@pytest.mark.parametrize(
    ("dependency_id", "dependency_class"),
    sorted(EXPECTED_RECORDS.items()),
)
def test_dependency_record_matches_public_schema(
    dependency_id: str,
    dependency_class: str,
) -> None:
    schema = _load_json(SCHEMA_PATH)
    record_path = PROVENANCE_ROOT / f"{dependency_id}.json"
    record = _load_json(record_path)

    assert record["dependency_id"] == dependency_id
    assert record["dependency_class"] == dependency_class
    _assert_matches_schema(record, schema, record_path.name)
