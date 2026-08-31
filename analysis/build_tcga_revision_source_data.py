"""Publish immutable source data from the sealed TCGA K=500 postprocess release.

The production entrypoint has one scientific input: the complete postprocess
release produced by :mod:`analysis.postprocess_tcga_revision_k500`.  It does not
accept caller-supplied tables, rows, predicates, or ranking functions.  A pinned
singleton release-stage approval is validated before any association row is
parsed.  The output is an exact-copy, machine-readable source-data package for
the complete within-cohort conjunction families; raw fit, runtime, calibration,
comparator, and replication datasets remain separate gated artifacts.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import io
import json
import math
import os
import re
import stat
import sys
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

from dialect.data import revision_approval as revision_approval_module
from dialect.data.revision_approval import (
    DECISION_IDS,
    RELEASE_STAGE,
    STAGE_SCOPED_APPROVAL_SCHEMA_V6,
    validate_revision_approval,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

SOURCE_DATA_SCHEMA: Final = "dialect-tcga-k500-source-data-release-v1"
SOURCE_DATA_CONTRACT: Final = (
    "exact-copy-of-authenticated-complete-conjunction-families-v1"
)
SOURCE_DATA_MANIFEST_NAME: Final = "source_data_manifest.json"
DATA_DICTIONARY_NAME: Final = "data_dictionary.json"
README_NAME: Final = "README.md"
COHORT_DIRECTORY_NAME: Final = "cohorts"

POSTPROCESS_RELEASE_SCHEMA: Final = "dialect-tcga-k500-d5-postprocess-release-v1"
POSTPROCESS_RELEASE_CONTRACT: Final = "sealed-whole-grid-no-replace-publication-v1"
POSTPROCESS_AUTHORITY_SCHEMA: Final = "dialect-tcga-k500-postprocess-authority-v4"
POSTPROCESS_AUTHORITY_CONTRACT: Final = (
    "pinned-roots-sealed-grid-signed-d1-d6-evidence-v4"
)
POSTPROCESS_COHORT_SCHEMA: Final = "dialect-tcga-k500-d5-postprocess-v1"
DERIVATION_CONTRACT: Final = "complete-family-max-p-by-bh-derivation-v1"
D5_CONTRACT: Final = "conjunction-multiplicity-family-policy-v3"
POSTPROCESS_RELEASE_MANIFEST_NAME: Final = "postprocess_release_manifest.json"
POSTPROCESS_AUTHORITY_NAME: Final = "postprocess_authority.json"
SEALED_COMPLETION_NAME: Final = "sealed_completion_manifest.json"
POSTPROCESS_CSV_NAME: Final = "conjunction_interaction_results.csv"
POSTPROCESS_COHORT_MANIFEST_NAME: Final = "postprocess_manifest.json"

TOP_K: Final = 500
BMRS: Final = ("cbase", "dig", "mutsig")
TCGA_COHORTS: Final = (
    "ACC",
    "BLCA",
    "BRCA",
    "CESC",
    "CHOL",
    "CRAD",
    "DLBC",
    "ESCA",
    "GBM",
    "HNSC",
    "KICH",
    "KIRC",
    "KIRP",
    "LAML",
    "LGG",
    "LIHC",
    "LUAD",
    "LUSC",
    "MESO",
    "OV",
    "PAAD",
    "PCPG",
    "PRAD",
    "SARC",
    "SKCM",
    "STAD",
    "TGCT",
    "THCA",
    "THYM",
    "UCEC",
    "UCS",
    "UVM",
)

OUTPUT_COLUMNS: Final = (
    "schema",
    "derivation_contract",
    "d5_contract",
    "cohort",
    "gene_a",
    "gene_b",
    "cbase_component_status",
    "cbase_p_value",
    "cbase_direction",
    "cbase_effect_identifiability",
    "dig_component_status",
    "dig_p_value",
    "dig_direction",
    "dig_effect_identifiability",
    "mutsig_component_status",
    "mutsig_p_value",
    "mutsig_direction",
    "mutsig_effect_identifiability",
    "conjunction_p_value",
    "consensus_direction",
    "d3_conjunction_role",
    "by_q_value",
    "bh_q_value",
    "marginal_validity_status",
    "conditional_by_inferential_eligible",
    "by_q_le_0_01",
    "conditional_by_q_le_0_01_reportable",
    "bh_q_le_0_01_nominal",
    "by_q_le_0_05_descriptive",
    "bh_q_le_0_05_descriptive",
    "cbase_source_sha256",
    "dig_source_sha256",
    "mutsig_source_sha256",
    "cohort_contract_sha256",
    "ordered_features_sha256",
    "ordered_pair_sha256",
)
RAW_PAIRWISE_COLUMNS: Final = (
    "Gene A",
    "Gene B",
    "Tau_00",
    "Tau_10",
    "Tau_01",
    "Tau_11",
    "_00_",
    "_10_",
    "_01_",
    "_11_",
    "Tau_1X",
    "Tau_X1",
    "Effect Identifiability",
    "Rho",
    "Log Odds Ratio",
    "Likelihood Ratio",
    "Wald Statistic",
    "Fit Algorithm",
    "Fit Converged",
    "Fit Iterations",
    "Fit Last LL Gain",
    "Fit Fixed-Point Residual",
    "Fit KKT Residual",
    "Pair Fit Contract",
    "Null Log Likelihood",
    "Alternative Log Likelihood",
    "LRT Contract",
)

VALID_COMPONENT_STATUSES: Final = (
    "valid-profile-lrt",
    "valid-degenerate-null-p-one",
)
COMPONENT_DIRECTIONS: Final = frozenset({"me", "co", "neutral", "unavailable"})
CONSENSUS_DIRECTIONS: Final = frozenset(
    {"unanimous-me", "unanimous-co", "discordant", "unavailable"},
)
EFFECT_IDENTIFIABILITY_STATUSES: Final = frozenset(
    {"full-affine-rank", "rank-deficient", "rank-not-certified-underflow"},
)

_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_FEATURE_PATTERN: Final = re.compile(r".+_[MN]")
_MAX_POSTPROCESS_METADATA_BYTES: Final = 16 * 1024 * 1024
_MAX_SOURCE_DATA_MANIFEST_BYTES: Final = 16 * 1024 * 1024
# K=500 caps a cohort at 124,750 rows; 512 MiB gives the fixed 36-column
# source-data contract >4 KiB/row of headroom while bounding sparse-file hashes.
_MAX_SOURCE_DATA_COHORT_CSV_BYTES: Final = 512 * 1024 * 1024
_PRODUCTION_SEAL: Final = object()


class SourceDataBuildError(ValueError):
    """Raised when a source-data package cannot be authenticated or published."""


@dataclass(frozen=True, slots=True)
class SourceDataBuildConfig:
    """Independent trust anchors for one production source-data publication."""

    postprocess_root: Path
    release_approval_manifest: Path
    expected_postprocess_release_sha256: str
    expected_postprocess_authority_sha256: str
    expected_postprocess_implementation_sha256: str
    expected_sealed_completion_sha256: str
    expected_canonical_input_sha256: str
    expected_provider_input_sha256: str
    expected_release_approval_sha256: str
    expected_marginal_validity_evidence_sha256: str | None


@dataclass(frozen=True, slots=True)
class SourceDataReleaseReceipt:
    """Digest-only receipt returned after atomic publication succeeds."""

    output_root: str
    manifest_sha256: str
    total_rows: int
    cohort_count: int


@dataclass(frozen=True, slots=True)
class SourceDataValidationReceipt:
    """Opaque-byte receipt returned after a frozen release validates."""

    source_data_root: str
    manifest_sha256: str
    file_count: int
    cohort_count: int
    total_bytes: int
    total_rows: int


@dataclass(frozen=True, slots=True)
class _PinnedFile:
    name: str
    descriptor: int = field(repr=False, compare=False)
    identity: os.stat_result = field(repr=False)
    sha256: str
    size_bytes: int
    content: bytes | None = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _PinnedCohort:
    cohort: str
    descriptor: int = field(repr=False, compare=False)
    identity: os.stat_result = field(repr=False)
    csv_file: _PinnedFile = field(repr=False)
    manifest_file: _PinnedFile = field(repr=False)


@dataclass(frozen=True, slots=True)
class _PinnedOutputFile:
    name: str
    descriptor: int = field(repr=False, compare=False)
    device: int
    inode: int
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class _SourceDataValidationPlan:
    """Closed manifest-derived byte plan; scientific member bytes stay opaque."""

    builder_implementation: Mapping[str, object]
    supporting_files: tuple[Mapping[str, object], ...]
    cohort_files: tuple[Mapping[str, object], ...]
    total_rows: int


@dataclass(frozen=True, slots=True, init=False)
class _ValidatedSourceData:
    approval_sha256: str
    decision_digests: Mapping[str, str]
    release_manifest_sha256: str
    authority_sha256: str
    grid_authority_sha256: str
    postprocess_implementation_sha256: str
    sealed_completion_sha256: str
    canonical_input_sha256: str
    provider_input_sha256: str
    marginal_validity_evidence_sha256: str | None
    builder_implementation: Mapping[str, object]
    cohorts: tuple[dict[str, object], ...]
    total_rows: int
    _seal: object = field(repr=False, compare=False)


def build_source_data_release(
    config: SourceDataBuildConfig,
    output_root: Path,
) -> SourceDataReleaseReceipt:
    """Validate and atomically publish the K=500 conjunction source data.

    The release approval is validated before the postprocess root is opened.
    Scientific rows are then read only through descriptor-pinned cohort files
    named by the authenticated postprocess release manifest.
    """
    _validate_config(config, output_root)
    builder_implementation = _builder_implementation()
    approval = _validate_release_approval(config)
    root_path, root_fd, root_identity = _open_frozen_directory(
        config.postprocess_root,
        label="postprocess release root",
    )
    root_files: list[_PinnedFile] = []
    cohort_pins: list[_PinnedCohort] = []
    try:
        _require_inventory(
            root_fd,
            {
                *TCGA_COHORTS,
                POSTPROCESS_AUTHORITY_NAME,
                POSTPROCESS_RELEASE_MANIFEST_NAME,
            },
            label="postprocess release root",
        )
        release_file = _open_frozen_file(
            root_fd,
            POSTPROCESS_RELEASE_MANIFEST_NAME,
            label="postprocess release manifest",
            max_bytes=_MAX_POSTPROCESS_METADATA_BYTES,
        )
        root_files.append(release_file)
        release_bytes = _retained_content(
            release_file,
            label="postprocess release manifest",
        )
        if release_file.sha256 != config.expected_postprocess_release_sha256:
            msg = "Postprocess release manifest does not match its independent anchor."
            raise SourceDataBuildError(msg)
        release = _parse_canonical_json(
            release_bytes,
            label="postprocess release manifest",
        )
        authority_file = _open_frozen_file(
            root_fd,
            POSTPROCESS_AUTHORITY_NAME,
            label="postprocess authority",
            max_bytes=_MAX_POSTPROCESS_METADATA_BYTES,
        )
        root_files.append(authority_file)
        authority_bytes = _retained_content(
            authority_file,
            label="postprocess authority",
        )
        if authority_file.sha256 != config.expected_postprocess_authority_sha256:
            msg = "Postprocess authority does not match its independent anchor."
            raise SourceDataBuildError(msg)
        authority = _parse_canonical_json(
            authority_bytes,
            label="postprocess authority",
        )
        release_outputs = _validate_release_metadata(config, release, authority)
        _validate_release_fit_decision_binding(approval, authority)

        cohort_records: list[dict[str, object]] = []
        total_rows = 0
        for cohort, output_record in zip(TCGA_COHORTS, release_outputs, strict=True):
            pin = _open_cohort(root_fd, cohort)
            cohort_pins.append(pin)
            record = _validate_cohort(
                config,
                cohort,
                output_record,
                pin,
                grid_authority_sha256=config.expected_postprocess_authority_sha256,
                root_authority=authority,
            )
            cohort_records.append(record)
            total_rows += int(record["rows"])

        validated = _mint_validated_source_data(
            config,
            approval=approval,
            release=release,
            cohorts=tuple(cohort_records),
            total_rows=total_rows,
            builder_implementation=builder_implementation,
        )
        _revalidate_input_tree(
            root_path,
            root_fd,
            root_identity,
            root_files,
            cohort_pins,
        )
        return _publish(
            validated,
            output_root,
            cohort_pins=cohort_pins,
            source_root=root_path,
            input_revalidator=lambda: _revalidate_input_tree(
                root_path,
                root_fd,
                root_identity,
                root_files,
                cohort_pins,
            ),
        )
    finally:
        for pin in cohort_pins:
            _close_cohort(pin)
        for pin in root_files:
            os.close(pin.descriptor)
        os.close(root_fd)


def validate_source_data_release(
    source_data_root: Path,
    expected_manifest_sha256: str,
) -> SourceDataValidationReceipt:
    """Validate one frozen source-data release without interpreting CSV bytes.

    The caller-supplied manifest digest is checked before the cohort directory or
    any cohort member is opened. The canonical manifest supplies the closed byte
    inventory; README, dictionary, and cohort CSV members are only descriptor-
    pinned, sized, and hashed. Scientific rows are never decoded or parsed.

    Args:
        source_data_root: Normalized absolute path to the frozen release root.
        expected_manifest_sha256: Independent lowercase SHA-256 of the canonical
            source-data manifest.

    Returns:
        A digest/count-only validation receipt.

    Raises:
        SourceDataBuildError: If any manifest, path, identity, inventory, mode,
            size, digest, or live-builder invariant fails closed.
    """
    _validate_source_data_validation_inputs(
        source_data_root,
        expected_manifest_sha256,
    )
    root_path, root_fd, root_identity = _open_frozen_directory(
        source_data_root,
        label="source-data release root",
    )
    root_pins: list[_PinnedFile] = []
    cohort_pins: list[_PinnedFile] = []
    cohorts_fd: int | None = None
    cohorts_identity: os.stat_result | None = None
    try:
        manifest_pin = _open_frozen_file(
            root_fd,
            SOURCE_DATA_MANIFEST_NAME,
            label="source-data release manifest",
            max_bytes=_MAX_SOURCE_DATA_MANIFEST_BYTES,
        )
        root_pins.append(manifest_pin)
        if manifest_pin.sha256 != expected_manifest_sha256:
            msg = "Source-data manifest does not match its independent SHA-256 anchor."
            raise SourceDataBuildError(msg)

        manifest = _parse_canonical_json(
            _retained_content(
                manifest_pin,
                label="source-data release manifest",
            ),
            label="source-data release manifest",
        )
        plan = _validate_source_data_release_manifest(manifest)
        _require_live_builder_implementation(plan.builder_implementation)

        _require_inventory(
            root_fd,
            {
                COHORT_DIRECTORY_NAME,
                DATA_DICTIONARY_NAME,
                README_NAME,
                SOURCE_DATA_MANIFEST_NAME,
            },
            label="source-data release root",
        )
        for record in plan.supporting_files:
            name = str(record["path"])
            pin = _open_frozen_file(
                root_fd,
                name,
                label=f"source-data support file {name}",
                retain_content=False,
                expected_size_bytes=int(record["bytes"]),
            )
            root_pins.append(pin)
            _require_pinned_release_member(pin, record, label=name)

        cohorts_fd, cohorts_identity = _open_frozen_child_directory(
            root_fd,
            COHORT_DIRECTORY_NAME,
            label="source-data cohorts directory",
        )
        _require_inventory(
            cohorts_fd,
            {f"{cohort}.csv" for cohort in TCGA_COHORTS},
            label="source-data cohort files",
        )
        for record in plan.cohort_files:
            cohort = str(record["cohort"])
            name = f"{cohort}.csv"
            pin = _open_frozen_file(
                cohorts_fd,
                name,
                label=f"source-data cohort file {name}",
                retain_content=False,
                expected_size_bytes=int(record["bytes"]),
                max_bytes=_MAX_SOURCE_DATA_COHORT_CSV_BYTES,
            )
            cohort_pins.append(pin)
            _require_pinned_release_member(pin, record, label=name)

        _revalidate_source_data_release_tree(
            root_path,
            root_fd,
            root_identity,
            root_pins=root_pins,
            cohorts_fd=cohorts_fd,
            cohorts_identity=cohorts_identity,
            cohort_pins=cohort_pins,
        )
        _require_live_builder_implementation(plan.builder_implementation)
        _revalidate_source_data_release_tree(
            root_path,
            root_fd,
            root_identity,
            root_pins=root_pins,
            cohorts_fd=cohorts_fd,
            cohorts_identity=cohorts_identity,
            cohort_pins=cohort_pins,
        )
        all_pins = (*root_pins, *cohort_pins)
        return SourceDataValidationReceipt(
            source_data_root=root_path.as_posix(),
            manifest_sha256=manifest_pin.sha256,
            file_count=len(all_pins),
            cohort_count=len(cohort_pins),
            total_bytes=sum(pin.size_bytes for pin in all_pins),
            total_rows=plan.total_rows,
        )
    finally:
        for pin in cohort_pins:
            os.close(pin.descriptor)
        if cohorts_fd is not None:
            os.close(cohorts_fd)
        for pin in root_pins:
            os.close(pin.descriptor)
        os.close(root_fd)


def _validate_source_data_validation_inputs(
    source_data_root: Path,
    expected_manifest_sha256: str,
) -> None:
    _require_sha256(expected_manifest_sha256, label="source-data manifest")
    if not isinstance(source_data_root, Path) or not source_data_root.is_absolute():
        msg = "Source-data root must be an absolute pathlib.Path."
        raise SourceDataBuildError(msg)
    if (
        source_data_root.name in {"", ".", ".."}
        or ".." in source_data_root.parts
        or Path(os.path.normpath(os.fspath(source_data_root))) != source_data_root
    ):
        msg = "Source-data root must be a normalized traversal-free path."
        raise SourceDataBuildError(msg)


def _validate_config(config: SourceDataBuildConfig, output_root: Path) -> None:
    if not isinstance(config, SourceDataBuildConfig):
        msg = "Production source data requires SourceDataBuildConfig."
        raise TypeError(msg)
    for label, path in (
        ("postprocess root", config.postprocess_root),
        ("release approval", config.release_approval_manifest),
    ):
        if not isinstance(path, Path) or not path.is_absolute():
            msg = f"{label} must be an absolute pathlib.Path."
            raise SourceDataBuildError(msg)
    if not isinstance(output_root, Path) or not output_root.is_absolute():
        msg = "Output root must be an absolute pathlib.Path."
        raise SourceDataBuildError(msg)
    if output_root.name in {"", ".", ".."}:
        msg = "Output root must have a safe final component."
        raise SourceDataBuildError(msg)
    for label, digest in (
        ("postprocess release", config.expected_postprocess_release_sha256),
        ("postprocess authority", config.expected_postprocess_authority_sha256),
        (
            "postprocess implementation",
            config.expected_postprocess_implementation_sha256,
        ),
        ("sealed completion", config.expected_sealed_completion_sha256),
        ("canonical input", config.expected_canonical_input_sha256),
        ("provider input", config.expected_provider_input_sha256),
        ("release approval", config.expected_release_approval_sha256),
    ):
        _require_sha256(digest, label=label)
    if config.expected_marginal_validity_evidence_sha256 is not None:
        _require_sha256(
            config.expected_marginal_validity_evidence_sha256,
            label="marginal-validity evidence",
        )


def _validate_release_approval(config: SourceDataBuildConfig) -> object:
    try:
        approval = validate_revision_approval(
            config.release_approval_manifest,
            config.expected_release_approval_sha256,
            RELEASE_STAGE,
        )
    except (OSError, ValueError) as error:
        msg = "Release-stage approval failed closed before source-data row access."
        raise SourceDataBuildError(msg) from error
    expected_binding = {
        "canonical_input_manifest_sha256": config.expected_canonical_input_sha256,
        "provider_input_manifest_sha256": config.expected_provider_input_sha256,
        "upstream_result_manifest_sha256": (config.expected_postprocess_release_sha256),
    }
    if (
        approval.schema != STAGE_SCOPED_APPROVAL_SCHEMA_V6
        or tuple(approval.allowed_stages) != (RELEASE_STAGE,)
        or tuple(approval.decisions) != DECISION_IDS
        or tuple(approval.decision_digests) != DECISION_IDS
        or set(approval.stage_bindings) != {RELEASE_STAGE}
        or dict(approval.stage_bindings[RELEASE_STAGE]) != expected_binding
    ):
        msg = "Release approval must be singleton v6 D1-D10 with exact roots."
        raise SourceDataBuildError(msg)
    return approval


def _validate_release_metadata(
    config: SourceDataBuildConfig,
    release: Mapping[str, object],
    authority: Mapping[str, object],
) -> list[Mapping[str, object]]:
    _require_exact_keys(
        release,
        {
            "analysis",
            "authority_receipt",
            "bmrs",
            "cohorts",
            "contract",
            "grid_authority_sha256",
            "marginal_validity_evidence_sha256",
            "outputs",
            "schema",
            "sealed_completion_sha256",
            "top_k",
        },
        label="postprocess release manifest",
    )
    expected_marginal = config.expected_marginal_validity_evidence_sha256
    if (
        release.get("schema") != POSTPROCESS_RELEASE_SCHEMA
        or release.get("contract") != POSTPROCESS_RELEASE_CONTRACT
        or release.get("analysis") != "tcga-revision-k500"
        or release.get("top_k") != TOP_K
        or release.get("cohorts") != list(TCGA_COHORTS)
        or release.get("bmrs") != list(BMRS)
        or release.get("grid_authority_sha256")
        != config.expected_postprocess_authority_sha256
        or release.get("sealed_completion_sha256")
        != config.expected_sealed_completion_sha256
        or release.get("marginal_validity_evidence_sha256") != expected_marginal
        or release.get("authority_receipt")
        != {
            "name": POSTPROCESS_AUTHORITY_NAME,
            "sha256": config.expected_postprocess_authority_sha256,
        }
    ):
        msg = "Postprocess release metadata does not match the pinned K=500 grid."
        raise SourceDataBuildError(msg)
    outputs = release.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != len(TCGA_COHORTS):
        msg = "Postprocess outputs do not cover the exact 32-cohort grid."
        raise SourceDataBuildError(msg)
    typed_outputs: list[Mapping[str, object]] = []
    for cohort, raw in zip(TCGA_COHORTS, outputs, strict=True):
        record = _require_mapping(raw, label=f"release output {cohort}")
        _require_exact_keys(
            record,
            {
                "cohort",
                "csv_sha256",
                "directory",
                "manifest_sha256",
                "publication_binding_sha256",
                "rows",
            },
            label=f"release output {cohort}",
        )
        if record.get("cohort") != cohort or record.get("directory") != cohort:
            msg = f"Release output order or directory is invalid for {cohort}."
            raise SourceDataBuildError(msg)
        for key in ("csv_sha256", "manifest_sha256", "publication_binding_sha256"):
            _require_sha256(record.get(key), label=f"{cohort} {key}")
        _require_positive_integer(record.get("rows"), label=f"{cohort} rows")
        typed_outputs.append(record)

    _validate_authority(config, authority)
    return typed_outputs


def _validate_authority(
    config: SourceDataBuildConfig,
    authority: Mapping[str, object],
) -> None:
    _require_exact_keys(
        authority,
        {
            "analysis",
            "approvals",
            "bmrs",
            "cohorts",
            "contract",
            "contracts",
            "d3_conjunction_role",
            "fit_decisions",
            "fit_policy",
            "marginal_validity_evidence",
            "roots",
            "schema",
            "sealed_completion",
            "top_k",
        },
        label="postprocess authority",
    )
    roots = _require_mapping(authority.get("roots"), label="authority roots")
    sealed = _require_mapping(
        authority.get("sealed_completion"),
        label="authority sealed completion",
    )
    marginal = _require_mapping(
        authority.get("marginal_validity_evidence"),
        label="authority marginal validity",
    )
    _require_exact_keys(
        roots,
        {
            "canonical_input_manifest_sha256",
            "canonical_input_root",
            "provider_input_manifest_sha256",
            "provider_input_root",
            "run_output_root",
        },
        label="authority roots",
    )
    _require_exact_keys(
        sealed,
        {"name", "sha256", "task_count"},
        label="authority sealed completion",
    )
    _require_exact_keys(
        marginal,
        {"evidence_id", "path", "sha256", "status"},
        label="authority marginal validity",
    )
    for key in ("run_output_root", "canonical_input_root", "provider_input_root"):
        path = roots.get(key)
        if (
            not isinstance(path, str)
            or not path
            or not Path(path).is_absolute()
            or Path(path).as_posix() != path
        ):
            msg = f"Postprocess authority {key} is not a normalized absolute path."
            raise SourceDataBuildError(msg)
    expected_evidence = config.expected_marginal_validity_evidence_sha256
    if expected_evidence is None:
        marginal_valid = (
            marginal.get("path") is None
            and marginal.get("sha256") is None
            and marginal.get("evidence_id") is None
            and marginal.get("status") in {"absent", "invalid", "inconclusive"}
        )
    else:
        evidence_path = marginal.get("path")
        marginal_valid = (
            isinstance(evidence_path, str)
            and bool(evidence_path)
            and Path(evidence_path).is_absolute()
            and Path(evidence_path).as_posix() == evidence_path
            and marginal.get("sha256") == expected_evidence
            and isinstance(marginal.get("evidence_id"), str)
            and bool(marginal["evidence_id"])
            and marginal.get("status") == "certified"
        )
    if (
        authority.get("schema") != POSTPROCESS_AUTHORITY_SCHEMA
        or authority.get("contract") != POSTPROCESS_AUTHORITY_CONTRACT
        or authority.get("analysis") != "tcga-revision-k500"
        or authority.get("top_k") != TOP_K
        or authority.get("cohorts") != list(TCGA_COHORTS)
        or authority.get("bmrs") != list(BMRS)
        or authority.get("d3_conjunction_role") != "secondary"
        or roots.get("canonical_input_manifest_sha256")
        != config.expected_canonical_input_sha256
        or roots.get("provider_input_manifest_sha256")
        != config.expected_provider_input_sha256
        or sealed.get("name") != SEALED_COMPLETION_NAME
        or sealed.get("sha256") != config.expected_sealed_completion_sha256
        or sealed.get("task_count") != len(TCGA_COHORTS) * len(BMRS)
        or not marginal_valid
    ):
        msg = "Postprocess authority does not bind the exact production roots."
        raise SourceDataBuildError(msg)
    contracts = authority.get("contracts")
    if not isinstance(contracts, list) or len(contracts) != len(TCGA_COHORTS):
        msg = "Postprocess authority lacks the exact cohort contract inventory."
        raise SourceDataBuildError(msg)
    for cohort, raw in zip(TCGA_COHORTS, contracts, strict=True):
        record = _require_mapping(raw, label=f"authority contract {cohort}")
        if (
            set(record) != {"cohort", "contract_sha256", "file_sha256"}
            or record.get("cohort") != cohort
        ):
            msg = f"Authority contract order is invalid for {cohort}."
            raise SourceDataBuildError(msg)
        _require_sha256(
            record.get("contract_sha256"),
            label=f"{cohort} semantic contract",
        )
        _require_sha256(record.get("file_sha256"), label=f"{cohort} contract")
    _validate_fit_decision_records(authority)
    _validate_root_approvals(authority, config=config)
    _validate_root_fit_policy(authority)


def _validate_fit_decision_records(
    authority: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    raw_records = authority.get("fit_decisions")
    decision_ids = tuple(f"D{index}" for index in range(1, 7))
    if not isinstance(raw_records, list) or len(raw_records) != len(decision_ids):
        msg = "Postprocess authority lacks the exact D1-D6 artifact inventory."
        raise SourceDataBuildError(msg)
    records: list[Mapping[str, object]] = []
    for decision_id, raw in zip(decision_ids, raw_records, strict=True):
        record = _require_mapping(raw, label=f"postprocess {decision_id} artifact")
        _require_exact_keys(
            record,
            {
                "canonical_artifact_sha256",
                "canonical_artifact_size_bytes",
                "contract",
                "decision_id",
                "payload_sha256",
            },
            label=f"postprocess {decision_id} artifact",
        )
        if (
            record.get("decision_id") != decision_id
            or not isinstance(record.get("contract"), str)
            or not record["contract"]
        ):
            msg = f"Postprocess {decision_id} artifact binding is invalid."
            raise SourceDataBuildError(msg)
        _require_positive_integer(
            record.get("canonical_artifact_size_bytes"),
            label=f"postprocess {decision_id} artifact size",
        )
        _require_sha256(
            record.get("canonical_artifact_sha256"),
            label=f"postprocess {decision_id} artifact",
        )
        _require_sha256(
            record.get("payload_sha256"),
            label=f"postprocess {decision_id} payload",
        )
        records.append(record)
    return tuple(records)


def _validate_release_fit_decision_binding(
    approval: object,
    authority: Mapping[str, object],
) -> None:
    records = _validate_fit_decision_records(authority)
    decisions = getattr(approval, "decisions", None)
    if not isinstance(decisions, Mapping):
        msg = "Release approval decisions are not an immutable decision mapping."
        raise SourceDataBuildError(msg)
    for record in records:
        decision_id = str(record["decision_id"])
        decision = decisions.get(decision_id)
        artifact = getattr(decision, "canonical_artifact", None)
        content = getattr(artifact, "content", None)
        if not isinstance(content, bytes):
            msg = f"Release approval lacks immutable {decision_id} artifact bytes."
            raise SourceDataBuildError(msg)
        envelope = _parse_canonical_json(
            content,
            label=f"release {decision_id} artifact",
        )
        payload = envelope.get("payload")
        if (
            set(envelope) != {"contract", "decision_id", "payload", "schema"}
            or envelope.get("decision_id") != decision_id
            or envelope.get("contract") != record["contract"]
            or not isinstance(payload, dict)
            or getattr(artifact, "sha256", None) != record["canonical_artifact_sha256"]
            or getattr(artifact, "size_bytes", None)
            != record["canonical_artifact_size_bytes"]
            or _sha256(content) != record["canonical_artifact_sha256"]
            or _sha256(_canonical_json(payload)) != record["payload_sha256"]
        ):
            msg = (
                f"Release approval does not exactly reauthorize {decision_id} "
                "from the sealed fit."
            )
            raise SourceDataBuildError(msg)


def _open_cohort(root_fd: int, cohort: str) -> _PinnedCohort:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(cohort, flags, dir_fd=root_fd)
    except OSError as error:
        msg = f"Cannot securely open postprocess cohort {cohort}."
        raise SourceDataBuildError(msg) from error
    try:
        identity = os.fstat(descriptor)
        _require_frozen_directory_identity(
            identity,
            label=f"postprocess cohort {cohort}",
        )
        _require_directory_entry_identity(
            root_fd,
            cohort,
            identity,
            label=f"postprocess cohort {cohort}",
        )
        _require_inventory(
            descriptor,
            {POSTPROCESS_CSV_NAME, POSTPROCESS_COHORT_MANIFEST_NAME},
            label=f"postprocess cohort {cohort}",
        )
        csv_file = _open_frozen_file(
            descriptor,
            POSTPROCESS_CSV_NAME,
            label=f"{cohort} source CSV",
            retain_content=False,
            max_bytes=_MAX_SOURCE_DATA_COHORT_CSV_BYTES,
        )
        try:
            manifest_file = _open_frozen_file(
                descriptor,
                POSTPROCESS_COHORT_MANIFEST_NAME,
                label=f"{cohort} postprocess manifest",
                max_bytes=_MAX_POSTPROCESS_METADATA_BYTES,
            )
        except Exception:
            os.close(csv_file.descriptor)
            raise
        return _PinnedCohort(
            cohort=cohort,
            descriptor=descriptor,
            identity=identity,
            csv_file=csv_file,
            manifest_file=manifest_file,
        )
    except OSError as error:
        os.close(descriptor)
        msg = f"Cannot securely inspect postprocess cohort {cohort}."
        raise SourceDataBuildError(msg) from error
    except Exception:
        os.close(descriptor)
        raise


def _validate_cohort(  # noqa: PLR0913
    config: SourceDataBuildConfig,
    cohort: str,
    output_record: Mapping[str, object],
    pin: _PinnedCohort,
    *,
    grid_authority_sha256: str,
    root_authority: Mapping[str, object],
) -> dict[str, object]:
    csv_sha256 = pin.csv_file.sha256
    manifest_sha256 = pin.manifest_file.sha256
    if (
        csv_sha256 != output_record["csv_sha256"]
        or manifest_sha256 != output_record["manifest_sha256"]
    ):
        msg = f"Postprocess release receipt does not match {cohort} bytes."
        raise SourceDataBuildError(msg)
    expected_publication_binding = _sha256(
        _canonical_json(
            {
                "authority_sha256": grid_authority_sha256,
                "cohort": cohort,
                "csv_sha256": csv_sha256,
                "manifest_sha256": manifest_sha256,
                "row_count": output_record["rows"],
            },
        ),
    )
    if output_record["publication_binding_sha256"] != expected_publication_binding:
        msg = f"Postprocess publication binding is invalid for {cohort}."
        raise SourceDataBuildError(msg)
    manifest = _parse_canonical_json(
        _retained_content(
            pin.manifest_file,
            label=f"{cohort} postprocess manifest",
        ),
        label=f"{cohort} postprocess manifest",
    )
    _require_exact_keys(
        manifest,
        {
            "axis",
            "cohort",
            "complete_family_required",
            "component_failure_semantics",
            "component_order",
            "components",
            "d3_provider_hierarchy",
            "d5_contract",
            "derivation_contract",
            "direction",
            "implementation",
            "marginal_validity",
            "multiplicity",
            "output",
            "p_value_combiner",
            "pair_filtering",
            "pair_ranking",
            "production_authority",
            "production_eligible",
            "schema",
            "tested_family",
            "valid_component_statuses",
        },
        label=f"{cohort} postprocess manifest",
    )
    axis = _require_mapping(manifest.get("axis"), label=f"{cohort} axis")
    output = _require_mapping(manifest.get("output"), label=f"{cohort} output")
    implementation = _require_mapping(
        manifest.get("implementation"),
        label=f"{cohort} implementation",
    )
    production = _require_mapping(
        manifest.get("production_authority"),
        label=f"{cohort} production authority",
    )
    rows = _require_positive_integer(output.get("rows"), label=f"{cohort} rows")
    minimum_pairs = TOP_K * (TOP_K - 1) // 2 - TOP_K // 2
    maximum_pairs = TOP_K * (TOP_K - 1) // 2
    if (
        manifest.get("schema") != POSTPROCESS_COHORT_SCHEMA
        or manifest.get("derivation_contract") != DERIVATION_CONTRACT
        or manifest.get("d5_contract") != D5_CONTRACT
        or manifest.get("cohort") != cohort
        or manifest.get("production_eligible") is not True
        or manifest.get("complete_family_required") is not True
        or manifest.get("pair_filtering") is not False
        or manifest.get("pair_ranking") is not False
        or manifest.get("component_order") != list(BMRS)
        or axis.get("feature_count") != TOP_K
        or not minimum_pairs <= rows <= maximum_pairs
        or axis.get("pair_count") != rows
        or output.get("name") != POSTPROCESS_CSV_NAME
        or output.get("bytes") != pin.csv_file.size_bytes
        or output.get("rows") != output_record["rows"]
        or output.get("sha256") != csv_sha256
        or output.get("columns") != list(OUTPUT_COLUMNS)
        or production.get("grid_authority_sha256") != grid_authority_sha256
        or production.get("sealed_completion_sha256")
        != config.expected_sealed_completion_sha256
        or production.get("canonical_input_manifest_sha256")
        != config.expected_canonical_input_sha256
        or production.get("provider_input_manifest_sha256")
        != config.expected_provider_input_sha256
        or production.get("marginal_validity_evidence_sha256")
        != config.expected_marginal_validity_evidence_sha256
        or implementation.get("combined_sha256")
        != config.expected_postprocess_implementation_sha256
    ):
        msg = f"Postprocess cohort manifest is not the exact K=500 family: {cohort}."
        raise SourceDataBuildError(msg)
    for key in (
        "ordered_features_sha256",
        "ordered_pair_sha256",
        "cohort_contract_sha256",
    ):
        _require_sha256(axis.get(key), label=f"{cohort} axis {key}")
    _validate_cohort_policy(
        manifest,
        root_authority=root_authority,
        config=config,
        cohort=cohort,
    )
    _validate_implementation(implementation, cohort=cohort)
    _validate_cohort_production_authority(
        production,
        root_authority=root_authority,
        config=config,
        cohort=cohort,
        axis=axis,
        components=_require_mapping(
            manifest["components"],
            label=f"{cohort} components",
        ),
    )
    observed_pair_sha256, observed_rows = _validate_csv_descriptor(
        pin.csv_file,
        cohort=cohort,
        manifest=manifest,
    )
    if observed_rows != rows or observed_pair_sha256 != axis["ordered_pair_sha256"]:
        msg = f"Postprocess CSV does not match the frozen pair axis for {cohort}."
        raise SourceDataBuildError(msg)
    return {
        "cohort": cohort,
        "source_csv_sha256": csv_sha256,
        "source_manifest_sha256": manifest_sha256,
        "bytes": pin.csv_file.size_bytes,
        "rows": rows,
        "ordered_features_sha256": axis["ordered_features_sha256"],
        "ordered_pair_sha256": axis["ordered_pair_sha256"],
        "cohort_contract_sha256": axis["cohort_contract_sha256"],
        "publication_binding_sha256": output_record["publication_binding_sha256"],
    }


def _validate_implementation(
    implementation: Mapping[str, object],
    *,
    cohort: str,
) -> None:
    if set(implementation) != {"combined_sha256", "files"}:
        msg = f"Postprocess implementation receipt is not closed for {cohort}."
        raise SourceDataBuildError(msg)
    files = _require_mapping(
        implementation.get("files"),
        label=f"{cohort} implementation files",
    )
    expected_files = {
        "analysis/postprocess_tcga_revision_k500.py",
        "analysis/run_tcga_revision_k500.py",
        "src/dialect/data/revision_approval.py",
        "src/dialect/data/revision_fit_policy.py",
        "src/dialect/models/interaction.py",
        "src/dialect/stats/revision_inference.py",
    }
    if set(files) != expected_files:
        msg = f"Postprocess implementation source closure is incomplete for {cohort}."
        raise SourceDataBuildError(msg)
    for path, digest in files.items():
        _require_sha256(digest, label=f"{cohort} implementation {path}")
    expected_combined = _sha256(_canonical_json(dict(files)))
    if implementation.get("combined_sha256") != expected_combined:
        msg = f"Postprocess implementation digest is invalid for {cohort}."
        raise SourceDataBuildError(msg)


def _validate_cohort_policy(
    manifest: Mapping[str, object],
    *,
    root_authority: Mapping[str, object],
    config: SourceDataBuildConfig,
    cohort: str,
) -> None:
    hierarchy = _require_mapping(
        manifest.get("d3_provider_hierarchy"),
        label=f"{cohort} provider hierarchy",
    )
    fit_policy = _require_mapping(
        root_authority.get("fit_policy"),
        label="postprocess fit policy",
    )
    root_d3 = _require_mapping(
        fit_policy.get("d3"),
        label="postprocess D3 policy",
    )
    expected_hierarchy = {**root_d3, "synthetic_qa_default": False}
    if (
        hierarchy != expected_hierarchy
        or hierarchy.get("primary_provider") != "cbase"
        or hierarchy.get("sensitivity_providers") != ["dig", "mutsig"]
        or hierarchy.get("all_three_conjunction_role") != "secondary"
        or hierarchy.get("burden_dependent_switching") is not False
        or hierarchy.get("synthetic_qa_default") is not False
        or not isinstance(hierarchy.get("rationale"), str)
        or not hierarchy["rationale"]
        or not isinstance(hierarchy.get("mutsig_support"), dict)
        or not hierarchy["mutsig_support"]
        or not isinstance(hierarchy.get("implementation_binding"), dict)
        or not hierarchy["implementation_binding"]
    ):
        msg = f"Postprocess D3 production hierarchy is invalid for {cohort}."
        raise SourceDataBuildError(msg)

    expected_tested_family = {
        "top_k": TOP_K,
        "feature_ranking": "descending-total-eligible-mutation-event-count",
        "tie_break": "canonical-count-matrix-column-order",
        "provider_support": "shared-native-cbase-dig-mutsig",
        "pair_construction": "all-unordered-pairs-of-ordered-feature-axis",
        "same_base_missense_nonsense": "exclude-before-fitting-and-testing",
        "epsilon_pretest_filter": "none",
        "marginal_effect_pretest_filter": "none",
        "family": "one-complete-within-cohort-tested-pair-family",
    }
    direction = {
        "provider_rule": "rho-negative-me-positive-co-zero-neutral",
        "undefined_rho_rule": "unavailable",
        "consensus_rule": "unanimous-me-or-co-else-not-unanimous",
        "reporting_layer": "descriptive-post-rejection",
        "directional_fdr_control": False,
    }
    multiplicity = {
        "computed_methods": ["by", "bh"],
        "primary_method": "by",
        "primary_q_threshold": 0.01,
        "primary_reporting_layer": ("confirmatory-conditional-on-valid-marginals"),
        "sensitivity_method": "bh",
        "sensitivity_q_threshold": 0.01,
        "sensitivity_reporting_layer": "nominal-sensitivity",
        "descriptive_methods": ["by", "bh"],
        "descriptive_q_threshold": 0.05,
        "descriptive_reporting_layer": "descriptive",
        "threshold_comparison": "inclusive-less-than-or-equal",
    }
    root_d5 = _require_mapping(
        fit_policy.get("d5"),
        label="postprocess D5 policy",
    )
    conjunction = _require_mapping(
        root_d5.get("conjunction"),
        label="postprocess D5 conjunction policy",
    )
    if (
        manifest.get("tested_family") != expected_tested_family
        or root_d5.get("tested_family") != expected_tested_family
        or manifest.get("direction") != direction
        or root_d5.get("direction_annotation") != direction
        or manifest.get("multiplicity") != multiplicity
        or root_d5.get("multiplicity")
        != {
            key: value
            for key, value in multiplicity.items()
            if key != "computed_methods"
        }
        or manifest.get("valid_component_statuses") != list(VALID_COMPONENT_STATUSES)
        or conjunction.get("valid_component_statuses") != list(VALID_COMPONENT_STATUSES)
        or conjunction.get("component_order") != list(BMRS)
        or conjunction.get("mode") != "nondirectional-max-p-iut"
        or conjunction.get("p_value_combiner") != "max(p_cbase,p_dig,p_mutsig)"
        or conjunction.get("direction_affects_p_or_q") is not False
        or manifest.get("p_value_combiner") != "max(p_cbase,p_dig,p_mutsig)"
        or manifest.get("component_failure_semantics")
        != "task-abort-no-published-row-no-p-one-substitution"
        or root_d5.get("component_failure_semantics")
        != "task-abort-no-published-row-no-p-one-substitution"
    ):
        msg = f"Postprocess D5 production policy is invalid for {cohort}."
        raise SourceDataBuildError(msg)
    components = _require_mapping(
        manifest.get("components"),
        label=f"{cohort} components",
    )
    if set(components) != set(BMRS):
        msg = f"Postprocess component inventory is invalid for {cohort}."
        raise SourceDataBuildError(msg)
    for provider in BMRS:
        component = _require_mapping(
            components[provider],
            label=f"{cohort} {provider} component",
        )
        if set(component) != {"pairwise_sha256", "raw_schema"} or component.get(
            "raw_schema",
        ) != list(RAW_PAIRWISE_COLUMNS):
            msg = (
                f"Postprocess raw component schema is invalid for {cohort}/{provider}."
            )
            raise SourceDataBuildError(msg)
        _require_sha256(
            component.get("pairwise_sha256"),
            label=f"{cohort} {provider} source",
        )
    _validate_marginal_validity_manifest(
        manifest,
        config=config,
        cohort=cohort,
    )
    cohort_marginal = _require_mapping(
        manifest.get("marginal_validity"),
        label=f"{cohort} marginal validity",
    )
    root_marginal = _require_mapping(
        root_authority.get("marginal_validity_evidence"),
        label="postprocess root marginal validity",
    )
    if (
        cohort_marginal.get("status") != root_marginal.get("status")
        or cohort_marginal.get("evidence_id") != root_marginal.get("evidence_id")
        or cohort_marginal.get("artifact_sha256") != root_marginal.get("sha256")
    ):
        msg = f"Postprocess marginal-validity state is inconsistent for {cohort}."
        raise SourceDataBuildError(msg)


def _validate_marginal_validity_manifest(
    manifest: Mapping[str, object],
    *,
    config: SourceDataBuildConfig,
    cohort: str,
) -> None:
    marginal = _require_mapping(
        manifest.get("marginal_validity"),
        label=f"{cohort} marginal validity",
    )
    _require_exact_keys(
        marginal,
        {
            "artifact_sha256",
            "conditional_by_inferential_eligible",
            "correction_selection_affected",
            "evidence_id",
            "q_values_affected",
            "status",
        },
        label=f"{cohort} marginal validity",
    )
    evidence_sha256 = config.expected_marginal_validity_evidence_sha256
    if evidence_sha256 is None:
        valid_state = (
            marginal.get("status") in {"absent", "invalid", "inconclusive"}
            and marginal.get("conditional_by_inferential_eligible") is False
            and marginal.get("artifact_sha256") is None
            and marginal.get("evidence_id") is None
        )
    else:
        valid_state = (
            marginal.get("status") == "certified"
            and marginal.get("conditional_by_inferential_eligible") is True
            and marginal.get("artifact_sha256") == evidence_sha256
            and isinstance(marginal.get("evidence_id"), str)
            and bool(marginal["evidence_id"])
        )
    if (
        not valid_state
        or marginal.get("correction_selection_affected") is not False
        or marginal.get("q_values_affected") is not False
    ):
        msg = f"Postprocess marginal-validity gate is invalid for {cohort}."
        raise SourceDataBuildError(msg)


def _validate_cohort_production_authority(  # noqa: PLR0913
    production: Mapping[str, object],
    *,
    root_authority: Mapping[str, object],
    config: SourceDataBuildConfig,
    cohort: str,
    axis: Mapping[str, object],
    components: Mapping[str, object],
) -> None:
    _require_exact_keys(
        production,
        {
            "calibration_approval_sha256",
            "canonical_input_manifest_sha256",
            "cohort",
            "d5_policy_receipt",
            "fit_approval_sha256",
            "grid_authority_sha256",
            "input_approval_sha256",
            "inspect_approval_sha256",
            "marginal_validity_evidence_sha256",
            "provider_input_manifest_sha256",
            "sealed_completion_sha256",
            "task_bindings",
        },
        label=f"{cohort} production authority",
    )
    approvals = _validate_root_approvals(root_authority, config=config)
    d5_receipt = _validate_root_fit_policy(root_authority)
    root_contracts = root_authority.get("contracts")
    if not isinstance(root_contracts, list):
        msg = "Postprocess root contract inventory is invalid."
        raise SourceDataBuildError(msg)
    root_contract = _require_mapping(
        root_contracts[TCGA_COHORTS.index(cohort)],
        label=f"{cohort} root contract",
    )
    expected = {
        "grid_authority_sha256": config.expected_postprocess_authority_sha256,
        "sealed_completion_sha256": config.expected_sealed_completion_sha256,
        "canonical_input_manifest_sha256": config.expected_canonical_input_sha256,
        "provider_input_manifest_sha256": config.expected_provider_input_sha256,
        "input_approval_sha256": approvals["input"],
        "fit_approval_sha256": approvals["fit"],
        "inspect_approval_sha256": approvals["inspect"],
        "calibration_approval_sha256": approvals["calibration"],
        "marginal_validity_evidence_sha256": (
            config.expected_marginal_validity_evidence_sha256
        ),
        "cohort": cohort,
        "d5_policy_receipt": d5_receipt,
    }
    if root_contract.get("contract_sha256") != axis.get(
        "cohort_contract_sha256",
    ) or any(production.get(key) != value for key, value in expected.items()):
        msg = f"Postprocess cohort authority is inconsistent for {cohort}."
        raise SourceDataBuildError(msg)
    bindings = production.get("task_bindings")
    if not isinstance(bindings, list) or len(bindings) != len(BMRS):
        msg = f"Postprocess task bindings are incomplete for {cohort}."
        raise SourceDataBuildError(msg)
    for provider, raw in zip(BMRS, bindings, strict=True):
        binding = _require_mapping(
            raw,
            label=f"{cohort} {provider} task binding",
        )
        _require_exact_keys(
            binding,
            {
                "contract_sha256",
                "pairwise_sha256",
                "provider",
                "single_gene_sha256",
                "task_manifest_sha256",
            },
            label=f"{cohort} {provider} task binding",
        )
        component = _require_mapping(
            components.get(provider),
            label=f"{cohort} {provider} component",
        )
        if (
            binding.get("provider") != provider
            or binding.get("contract_sha256") != axis.get("cohort_contract_sha256")
            or binding.get("pairwise_sha256") != component.get("pairwise_sha256")
        ):
            msg = f"Postprocess task binding is inconsistent for {cohort}/{provider}."
            raise SourceDataBuildError(msg)
        for key in (
            "contract_sha256",
            "pairwise_sha256",
            "single_gene_sha256",
            "task_manifest_sha256",
        ):
            _require_sha256(
                binding.get(key),
                label=f"{cohort} {provider} task {key}",
            )


def _validate_root_approvals(
    authority: Mapping[str, object],
    *,
    config: SourceDataBuildConfig,
) -> dict[str, str | None]:
    approvals = _require_mapping(
        authority.get("approvals"),
        label="postprocess authority approvals",
    )
    _require_exact_keys(
        approvals,
        {"calibration", "fit", "input", "inspect"},
        label="postprocess authority approvals",
    )
    output: dict[str, str | None] = {}
    for name in ("input", "fit", "inspect"):
        record = _require_mapping(
            approvals.get(name),
            label=f"postprocess {name} approval",
        )
        expected_keys = {"path", "sha256"}
        if name == "inspect":
            expected_keys.add("authorized_stage")
        _require_exact_keys(
            record,
            expected_keys,
            label=f"postprocess {name} approval",
        )
        if not isinstance(record.get("path"), str) or not record["path"]:
            msg = f"Postprocess {name} approval path is invalid."
            raise SourceDataBuildError(msg)
        output[name] = _require_sha256(
            record.get("sha256"),
            label=f"postprocess {name} approval",
        )
    inspect = _require_mapping(
        approvals["inspect"],
        label="postprocess inspect approval",
    )
    if inspect.get("authorized_stage") != "inspect-tcga-k500":
        msg = "Postprocess inspect approval stage is invalid."
        raise SourceDataBuildError(msg)
    calibration = approvals.get("calibration")
    if config.expected_marginal_validity_evidence_sha256 is None:
        if calibration is not None:
            msg = "Blocking marginal validity must not include calibration approval."
            raise SourceDataBuildError(msg)
        output["calibration"] = None
    else:
        record = _require_mapping(
            calibration,
            label="postprocess calibration approval",
        )
        _require_exact_keys(
            record,
            {"authorized_stage", "decision_digests", "path", "schema", "sha256"},
            label="postprocess calibration approval",
        )
        if (
            record.get("authorized_stage") != "calibration"
            or record.get("schema") != STAGE_SCOPED_APPROVAL_SCHEMA_V6
            or not isinstance(record.get("path"), str)
            or not record["path"]
        ):
            msg = "Postprocess calibration approval metadata is invalid."
            raise SourceDataBuildError(msg)
        decision_digests = _require_mapping(
            record.get("decision_digests"),
            label="postprocess calibration decision digests",
        )
        expected_decisions = tuple(f"D{index}" for index in range(1, 7))
        if tuple(decision_digests) != expected_decisions:
            msg = "Postprocess calibration approval must bind exact D1-D6."
            raise SourceDataBuildError(msg)
        for decision_id, digest in decision_digests.items():
            _require_sha256(
                digest,
                label=f"postprocess calibration {decision_id}",
            )
        output["calibration"] = _require_sha256(
            record.get("sha256"),
            label="postprocess calibration approval",
        )
    return output


def _validate_root_fit_policy(
    authority: Mapping[str, object],
) -> Mapping[str, object]:
    policy = _require_mapping(
        authority.get("fit_policy"),
        label="postprocess fit policy",
    )
    _require_exact_keys(
        policy,
        {"d3", "d4", "d5", "d6", "receipts"},
        label="postprocess fit policy",
    )
    for decision in ("d3", "d4", "d5", "d6"):
        decision_policy = _require_mapping(
            policy.get(decision),
            label=f"postprocess policy {decision}",
        )
        if not decision_policy:
            msg = f"Postprocess policy {decision} must not be empty."
            raise SourceDataBuildError(msg)
    receipts = _require_mapping(
        policy.get("receipts"),
        label="postprocess fit-policy receipts",
    )
    decision_records = {
        str(record["decision_id"]): record
        for record in _validate_fit_decision_records(authority)
    }
    expected_decisions = tuple(f"D{index}" for index in range(3, 7))
    if tuple(receipts) != expected_decisions:
        msg = "Postprocess fit policy must contain exact D3-D6 receipts."
        raise SourceDataBuildError(msg)
    for decision_id, raw in receipts.items():
        receipt = _require_mapping(
            raw,
            label=f"postprocess {decision_id} policy receipt",
        )
        _require_exact_keys(
            receipt,
            {
                "canonical_artifact_path",
                "canonical_artifact_sha256",
                "canonical_artifact_size_bytes",
                "contract",
                "decision_digest",
                "decision_id",
                "payload_sha256",
            },
            label=f"postprocess {decision_id} policy receipt",
        )
        if (
            receipt.get("decision_id") != decision_id
            or not isinstance(receipt.get("contract"), str)
            or not receipt["contract"]
            or not isinstance(receipt.get("canonical_artifact_path"), str)
            or not receipt["canonical_artifact_path"]
            or isinstance(receipt.get("canonical_artifact_size_bytes"), bool)
            or not isinstance(receipt.get("canonical_artifact_size_bytes"), int)
            or receipt["canonical_artifact_size_bytes"] <= 0
        ):
            msg = f"Postprocess {decision_id} policy receipt is invalid."
            raise SourceDataBuildError(msg)
        for key in (
            "canonical_artifact_sha256",
            "decision_digest",
            "payload_sha256",
        ):
            _require_sha256(
                receipt.get(key),
                label=f"postprocess {decision_id} {key}",
            )
        decision_record = decision_records[decision_id]
        policy_payload = _require_mapping(
            policy.get(decision_id.lower()),
            label=f"postprocess {decision_id} policy payload",
        )
        if (
            receipt.get("contract") != decision_record["contract"]
            or receipt.get("canonical_artifact_sha256")
            != decision_record["canonical_artifact_sha256"]
            or receipt.get("canonical_artifact_size_bytes")
            != decision_record["canonical_artifact_size_bytes"]
            or receipt.get("payload_sha256") != decision_record["payload_sha256"]
            or _sha256(_canonical_json(policy_payload))
            != decision_record["payload_sha256"]
        ):
            msg = f"Postprocess {decision_id} policy linkage is inconsistent."
            raise SourceDataBuildError(msg)
    return _require_mapping(receipts["D5"], label="postprocess D5 receipt")


def _validate_csv_rows(
    raw: bytes,
    *,
    cohort: str,
    manifest: Mapping[str, object],
) -> tuple[str, int]:
    stream = io.TextIOWrapper(
        io.BytesIO(raw),
        encoding="utf-8",
        errors="strict",
        newline="",
    )
    try:
        return _validate_csv_stream(stream, cohort=cohort, manifest=manifest)
    finally:
        stream.close()


def _validate_csv_descriptor(
    pin: _PinnedFile,
    *,
    cohort: str,
    manifest: Mapping[str, object],
) -> tuple[str, int]:
    before = os.fstat(pin.descriptor)
    duplicated = os.dup(pin.descriptor)
    try:
        os.lseek(duplicated, 0, os.SEEK_SET)
        stream = io.TextIOWrapper(
            io.FileIO(duplicated, mode="rb", closefd=True),
            encoding="utf-8",
            errors="strict",
            newline="",
        )
        duplicated = -1
        try:
            result = _validate_csv_stream(
                stream,
                cohort=cohort,
                manifest=manifest,
            )
        finally:
            stream.close()
    finally:
        if duplicated >= 0:
            os.close(duplicated)
    after = os.fstat(pin.descriptor)
    if not _same_file_snapshot(before, after) or not _same_file_snapshot(
        pin.identity,
        after,
    ):
        msg = f"Source-data CSV changed during parsing for {cohort}."
        raise SourceDataBuildError(msg)
    return result


def _validate_csv_stream(
    stream: io.TextIOWrapper,
    *,
    cohort: str,
    manifest: Mapping[str, object],
) -> tuple[str, int]:
    axis = _require_mapping(manifest["axis"], label=f"{cohort} axis")
    components = _require_mapping(
        manifest["components"],
        label=f"{cohort} components",
    )
    expected_sources: dict[str, str] = {}
    for provider in BMRS:
        component = _require_mapping(
            components.get(provider),
            label=f"{cohort} {provider} component",
        )
        digest = _require_sha256(
            component.get("pairwise_sha256"),
            label=f"{cohort} {provider} source",
        )
        expected_sources[provider] = digest
    pair_digest = hashlib.sha256()
    seen: set[tuple[str, str]] = set()
    features: set[str] = set()
    conjunction_p_values: list[float] = []
    observed_by_q_values: list[str] = []
    observed_bh_q_values: list[str] = []
    try:
        reader = csv.reader(stream, strict=True)
        if tuple(next(reader, ())) != OUTPUT_COLUMNS:
            msg = f"Source-data CSV header is invalid for {cohort}."
            raise SourceDataBuildError(msg)
        for index, row in enumerate(reader, start=1):
            if len(row) != len(OUTPUT_COLUMNS):
                msg = f"Source-data CSV row width is invalid for {cohort}."
                raise SourceDataBuildError(msg)
            record = dict(zip(OUTPUT_COLUMNS, row, strict=True))
            _validate_row_constants(
                record,
                cohort=cohort,
                axis=axis,
                sources=expected_sources,
            )
            gene_a = record["gene_a"]
            gene_b = record["gene_b"]
            pair = (gene_a, gene_b)
            unordered_pair = tuple(sorted(pair))
            if (
                _FEATURE_PATTERN.fullmatch(gene_a) is None
                or _FEATURE_PATTERN.fullmatch(gene_b) is None
                or gene_a == gene_b
                or _base_gene(gene_a) == _base_gene(gene_b)
                or unordered_pair in seen
            ):
                msg = f"Source-data pair axis is invalid for {cohort} row {index}."
                raise SourceDataBuildError(msg)
            seen.add(unordered_pair)
            features.update(pair)
            encoded = f"{gene_a}\t{gene_b}".encode()
            pair_digest.update(len(encoded).to_bytes(8, "big"))
            pair_digest.update(encoded)
            conjunction, by_q, bh_q = _validate_numeric_and_boolean_fields(
                record,
                cohort=cohort,
                index=index,
                manifest=manifest,
            )
            conjunction_p_values.append(conjunction)
            observed_by_q_values.append(by_q)
            observed_bh_q_values.append(bh_q)
    except (csv.Error, UnicodeDecodeError) as error:
        msg = f"Source-data CSV is not strict UTF-8 CSV for {cohort}."
        raise SourceDataBuildError(msg) from error
    if len(features) != TOP_K:
        msg = f"Source-data pair axis does not contain exactly K=500 for {cohort}."
        raise SourceDataBuildError(msg)
    variants_by_base: dict[str, set[str]] = {}
    for feature in features:
        variants_by_base.setdefault(_base_gene(feature), set()).add(feature[-1])
    same_base_exclusions = sum(
        variants == {"M", "N"} for variants in variants_by_base.values()
    )
    expected_pairs = TOP_K * (TOP_K - 1) // 2 - same_base_exclusions
    if len(seen) != expected_pairs:
        msg = f"Source-data pair family is incomplete for {cohort}."
        raise SourceDataBuildError(msg)
    _validate_complete_family_q_values(
        conjunction_p_values,
        observed_by_q_values,
        observed_bh_q_values,
        cohort=cohort,
    )
    return pair_digest.hexdigest(), len(seen)


def _validate_row_constants(
    record: Mapping[str, str],
    *,
    cohort: str,
    axis: Mapping[str, object],
    sources: Mapping[str, str],
) -> None:
    expected = {
        "schema": POSTPROCESS_COHORT_SCHEMA,
        "derivation_contract": DERIVATION_CONTRACT,
        "d5_contract": D5_CONTRACT,
        "cohort": cohort,
        "d3_conjunction_role": "secondary",
        "cbase_source_sha256": sources["cbase"],
        "dig_source_sha256": sources["dig"],
        "mutsig_source_sha256": sources["mutsig"],
        "cohort_contract_sha256": str(axis["cohort_contract_sha256"]),
        "ordered_features_sha256": str(axis["ordered_features_sha256"]),
        "ordered_pair_sha256": str(axis["ordered_pair_sha256"]),
    }
    if any(record[key] != value for key, value in expected.items()):
        msg = f"Source-data row authority binding is inconsistent for {cohort}."
        raise SourceDataBuildError(msg)


def _validate_numeric_and_boolean_fields(
    record: Mapping[str, str],
    *,
    cohort: str,
    index: int,
    manifest: Mapping[str, object],
) -> tuple[float, str, str]:
    numeric = (
        "cbase_p_value",
        "dig_p_value",
        "mutsig_p_value",
        "conjunction_p_value",
        "by_q_value",
        "bh_q_value",
    )
    probabilities: dict[str, float] = {}
    for field_name in numeric:
        value = record[field_name]
        if not value:
            msg = (
                f"Missing production probability {field_name} in {cohort} row {index}."
            )
            raise SourceDataBuildError(msg)
        try:
            parsed = float(value)
        except ValueError as error:
            msg = f"Non-numeric {field_name} in {cohort} row {index}."
            raise SourceDataBuildError(msg) from error
        if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
            msg = f"Invalid probability {field_name} in {cohort} row {index}."
            raise SourceDataBuildError(msg)
        if format(parsed, ".17g") != value:
            msg = f"Non-canonical probability {field_name} in {cohort} row {index}."
            raise SourceDataBuildError(msg)
        probabilities[field_name] = parsed
    booleans: dict[str, bool] = {}
    for field_name in (
        "conditional_by_inferential_eligible",
        "by_q_le_0_01",
        "conditional_by_q_le_0_01_reportable",
        "bh_q_le_0_01_nominal",
        "by_q_le_0_05_descriptive",
        "bh_q_le_0_05_descriptive",
    ):
        if record[field_name] not in {"true", "false"}:
            msg = f"Invalid boolean {field_name} in {cohort} row {index}."
            raise SourceDataBuildError(msg)
        booleans[field_name] = record[field_name] == "true"

    component_p_values = []
    component_directions = []
    for provider in BMRS:
        status = record[f"{provider}_component_status"]
        direction = record[f"{provider}_direction"]
        identifiability = record[f"{provider}_effect_identifiability"]
        p_value = probabilities[f"{provider}_p_value"]
        if (
            status not in VALID_COMPONENT_STATUSES
            or direction not in COMPONENT_DIRECTIONS
            or identifiability not in EFFECT_IDENTIFIABILITY_STATUSES
            or (
                status == "valid-degenerate-null-p-one"
                and (
                    p_value != 1.0
                    or direction != "unavailable"
                    or identifiability != "full-affine-rank"
                )
            )
            or (
                identifiability != "full-affine-rank"
                and (status != "valid-profile-lrt" or direction != "unavailable")
            )
            or (
                identifiability == "full-affine-rank"
                and status == "valid-profile-lrt"
                and direction == "unavailable"
            )
        ):
            msg = f"Invalid component semantics for {cohort}/{provider} row {index}."
            raise SourceDataBuildError(msg)
        component_p_values.append(p_value)
        component_directions.append(direction)
    conjunction = probabilities["conjunction_p_value"]
    by_q = probabilities["by_q_value"]
    bh_q = probabilities["bh_q_value"]
    if (
        conjunction != max(component_p_values)
        or by_q + 1e-15 < conjunction
        or bh_q + 1e-15 < conjunction
        or by_q + 1e-15 < bh_q
    ):
        msg = f"Invalid conjunction or q-value relation for {cohort} row {index}."
        raise SourceDataBuildError(msg)
    directions = tuple(component_directions)
    if "unavailable" in directions:
        expected_consensus = "unavailable"
    elif all(direction == "me" for direction in directions):
        expected_consensus = "unanimous-me"
    elif all(direction == "co" for direction in directions):
        expected_consensus = "unanimous-co"
    else:
        expected_consensus = "discordant"
    marginal = _require_mapping(
        manifest["marginal_validity"],
        label=f"{cohort} marginal validity",
    )
    eligible = bool(marginal["conditional_by_inferential_eligible"])
    expected_flags = {
        "conditional_by_inferential_eligible": eligible,
        "by_q_le_0_01": by_q <= 0.01,
        "conditional_by_q_le_0_01_reportable": by_q <= 0.01 and eligible,
        "bh_q_le_0_01_nominal": bh_q <= 0.01,
        "by_q_le_0_05_descriptive": by_q <= 0.05,
        "bh_q_le_0_05_descriptive": bh_q <= 0.05,
    }
    if (
        record["consensus_direction"] not in CONSENSUS_DIRECTIONS
        or record["consensus_direction"] != expected_consensus
        or record["marginal_validity_status"] != marginal["status"]
        or booleans != expected_flags
    ):
        msg = f"Invalid direction or reporting gate for {cohort} row {index}."
        raise SourceDataBuildError(msg)
    return conjunction, record["by_q_value"], record["bh_q_value"]


def _validate_complete_family_q_values(
    conjunction_p_values: list[float],
    observed_by_q_values: list[str],
    observed_bh_q_values: list[str],
    *,
    cohort: str,
) -> None:
    expected_by = _adjust_q_values(conjunction_p_values, method="by")
    expected_bh = _adjust_q_values(conjunction_p_values, method="bh")
    for index, (by_q, bh_q, observed_by, observed_bh) in enumerate(
        zip(
            expected_by,
            expected_bh,
            observed_by_q_values,
            observed_bh_q_values,
            strict=True,
        ),
        start=1,
    ):
        if (
            format(float(by_q), ".17g") != observed_by
            or format(float(bh_q), ".17g") != observed_bh
        ):
            msg = (
                "Postprocess complete-family q-values are inconsistent for "
                f"{cohort} row {index}."
            )
            raise SourceDataBuildError(msg)


def _adjust_q_values(p_values: list[float], *, method: str) -> list[float]:
    """Independently replay the prespecified stable complete-family BH/BY rule."""
    if not p_values or method not in {"bh", "by"}:
        msg = "Complete-family q-value replay received an invalid specification."
        raise SourceDataBuildError(msg)
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    count = len(ordered)
    factor = (
        math.fsum(1.0 / rank for rank in range(1, count + 1)) if method == "by" else 1.0
    )
    adjusted = [
        min(1.0, p_value * count * factor / rank)
        for rank, (_index, p_value) in enumerate(ordered, start=1)
    ]
    for index in range(count - 2, -1, -1):
        adjusted[index] = min(adjusted[index], adjusted[index + 1])
    restored = [0.0] * count
    for (original_index, _p_value), q_value in zip(ordered, adjusted, strict=True):
        restored[original_index] = q_value
    return restored


def _mint_validated_source_data(  # noqa: PLR0913
    config: SourceDataBuildConfig,
    *,
    approval: object,
    release: Mapping[str, object],
    cohorts: tuple[dict[str, object], ...],
    total_rows: int,
    builder_implementation: Mapping[str, object],
) -> _ValidatedSourceData:
    value = object.__new__(_ValidatedSourceData)
    fields = {
        "approval_sha256": approval.manifest_sha256,
        "decision_digests": dict(approval.decision_digests),
        "release_manifest_sha256": config.expected_postprocess_release_sha256,
        "authority_sha256": config.expected_postprocess_authority_sha256,
        "grid_authority_sha256": str(release["grid_authority_sha256"]),
        "postprocess_implementation_sha256": (
            config.expected_postprocess_implementation_sha256
        ),
        "sealed_completion_sha256": config.expected_sealed_completion_sha256,
        "canonical_input_sha256": config.expected_canonical_input_sha256,
        "provider_input_sha256": config.expected_provider_input_sha256,
        "marginal_validity_evidence_sha256": (
            config.expected_marginal_validity_evidence_sha256
        ),
        "builder_implementation": dict(builder_implementation),
        "cohorts": cohorts,
        "total_rows": total_rows,
        "_seal": _PRODUCTION_SEAL,
    }
    for name, field_value in fields.items():
        object.__setattr__(value, name, field_value)
    return value


def _publish(
    validated: _ValidatedSourceData,
    output_root: Path,
    *,
    cohort_pins: Sequence[_PinnedCohort],
    source_root: Path,
    input_revalidator: Callable[[], None],
) -> SourceDataReleaseReceipt:
    _require_validated(validated)
    parent = output_root.parent.resolve(strict=True)
    resolved_destination = parent / output_root.name
    try:
        resolved_destination.relative_to(source_root)
    except ValueError:
        pass
    else:
        msg = "Source-data output must be outside the immutable postprocess root."
        raise SourceDataBuildError(msg)
    parent_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    parent_fd = os.open(parent, parent_flags)
    try:
        parent_identity = os.fstat(parent_fd)
    except OSError as error:
        os.close(parent_fd)
        msg = "Cannot securely inspect source-data publication parent."
        raise SourceDataBuildError(msg) from error
    staging_name = f".{output_root.name}.{uuid.uuid4().hex}.tmp"
    staging_fd: int | None = None
    cohorts_fd: int | None = None
    renamed = False
    root_output_pins: list[_PinnedOutputFile] = []
    cohort_output_pins: list[_PinnedOutputFile] = []
    try:
        _require_parent(output_root.parent, parent, parent_fd, parent_identity)
        _require_destination_absent(parent_fd, output_root.name)
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_fd)
        staging_fd = os.open(staging_name, parent_flags, dir_fd=parent_fd)
        staging_identity = os.fstat(staging_fd)
        os.mkdir(COHORT_DIRECTORY_NAME, mode=0o700, dir_fd=staging_fd)
        cohorts_fd = os.open(COHORT_DIRECTORY_NAME, parent_flags, dir_fd=staging_fd)

        cohort_outputs: list[dict[str, object]] = []
        for pin, record in zip(cohort_pins, validated.cohorts, strict=True):
            output_name = f"{pin.cohort}.csv"
            _write_file_from_pin(cohorts_fd, output_name, pin.csv_file)
            cohort_outputs.append(
                {
                    **record,
                    "path": f"{COHORT_DIRECTORY_NAME}/{output_name}",
                    "sha256": pin.csv_file.sha256,
                },
            )

        dictionary_bytes = _canonical_json(_data_dictionary()) + b"\n"
        readme_bytes = _readme_bytes()
        _write_file(staging_fd, DATA_DICTIONARY_NAME, dictionary_bytes)
        _write_file(staging_fd, README_NAME, readme_bytes)
        manifest = _source_data_manifest(
            validated,
            cohort_outputs=cohort_outputs,
            dictionary_bytes=dictionary_bytes,
            readme_bytes=readme_bytes,
        )
        manifest_bytes = _canonical_json(manifest) + b"\n"
        _write_file(staging_fd, SOURCE_DATA_MANIFEST_NAME, manifest_bytes)
        cohorts_identity = os.fstat(cohorts_fd)
        for name, content in (
            (DATA_DICTIONARY_NAME, dictionary_bytes),
            (README_NAME, readme_bytes),
            (SOURCE_DATA_MANIFEST_NAME, manifest_bytes),
        ):
            root_output_pins.append(
                _pin_output_file(
                    staging_fd,
                    name,
                    expected_sha256=_sha256(content),
                    expected_size=len(content),
                ),
            )
        cohort_output_pins.extend(
            (
                _pin_output_file(
                    cohorts_fd,
                    f"{pin.cohort}.csv",
                    expected_sha256=pin.csv_file.sha256,
                    expected_size=pin.csv_file.size_bytes,
                )
                for pin in cohort_pins
            ),
        )

        _validate_pinned_output_tree(
            staging_fd,
            cohorts_fd,
            staging_identity=staging_identity,
            cohorts_identity=cohorts_identity,
            root_pins=root_output_pins,
            cohort_pins=cohort_output_pins,
            frozen=False,
        )
        for pin in cohort_pins:
            os.chmod(f"{pin.cohort}.csv", 0o400, dir_fd=cohorts_fd)
        os.fchmod(cohorts_fd, 0o500)
        for name in (DATA_DICTIONARY_NAME, README_NAME, SOURCE_DATA_MANIFEST_NAME):
            os.chmod(name, 0o400, dir_fd=staging_fd)
        os.fchmod(staging_fd, 0o500)
        for pin in (*root_output_pins, *cohort_output_pins):
            os.fsync(pin.descriptor)
        os.fsync(cohorts_fd)
        os.fsync(staging_fd)
        _validate_pinned_output_tree(
            staging_fd,
            cohorts_fd,
            staging_identity=staging_identity,
            cohorts_identity=cohorts_identity,
            root_pins=root_output_pins,
            cohort_pins=cohort_output_pins,
            frozen=True,
        )
        _revalidate_builder_implementation(validated)
        input_revalidator()
        _validate_pinned_output_tree(
            staging_fd,
            cohorts_fd,
            staging_identity=staging_identity,
            cohorts_identity=cohorts_identity,
            root_pins=root_output_pins,
            cohort_pins=cohort_output_pins,
            frozen=True,
        )
        _require_parent(output_root.parent, parent, parent_fd, parent_identity)
        _require_directory_entry_identity(
            parent_fd,
            staging_name,
            staging_identity,
            label="staged source-data root",
        )
        _rename_no_replace(parent_fd, staging_name, output_root.name)
        renamed = True
        os.fsync(parent_fd)
        _require_directory_entry_identity(
            parent_fd,
            output_root.name,
            staging_identity,
            label="published source-data root",
        )
        published_fd = os.open(output_root.name, parent_flags, dir_fd=parent_fd)
        try:
            published_identity = os.fstat(published_fd)
            if (
                published_identity.st_dev != staging_identity.st_dev
                or published_identity.st_ino != staging_identity.st_ino
            ):
                msg = "Published source-data root identity changed after rename."
                raise SourceDataBuildError(msg)
            published_cohorts_fd = os.open(
                COHORT_DIRECTORY_NAME,
                parent_flags,
                dir_fd=published_fd,
            )
            try:
                _validate_pinned_output_tree(
                    published_fd,
                    published_cohorts_fd,
                    staging_identity=staging_identity,
                    cohorts_identity=cohorts_identity,
                    root_pins=root_output_pins,
                    cohort_pins=cohort_output_pins,
                    frozen=True,
                )
            finally:
                os.close(published_cohorts_fd)
        finally:
            os.close(published_fd)
        _require_parent(output_root.parent, parent, parent_fd, parent_identity)
        _require_directory_entry_identity(
            parent_fd,
            output_root.name,
            staging_identity,
            label="published source-data root",
        )
    except Exception as error:
        if staging_fd is not None and not renamed:
            error.add_note(
                "Builder-owned staging was intentionally retained for forensic "
                f"recovery and was not deleted: {(parent / staging_name).as_posix()}",
            )
        elif renamed:
            error.add_note(
                "The atomic rename committed the destination before a later "
                f"verification failure: {resolved_destination.as_posix()}",
            )
        raise
    finally:
        for pin in root_output_pins:
            os.close(pin.descriptor)
        for pin in cohort_output_pins:
            os.close(pin.descriptor)
        if cohorts_fd is not None:
            os.close(cohorts_fd)
        if staging_fd is not None:
            os.close(staging_fd)
        os.close(parent_fd)
    return SourceDataReleaseReceipt(
        output_root=output_root.as_posix(),
        manifest_sha256=_sha256(manifest_bytes),
        total_rows=validated.total_rows,
        cohort_count=len(validated.cohorts),
    )


def _require_destination_absent(parent_fd: int, name: str) -> None:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), name)


def _source_data_manifest(
    validated: _ValidatedSourceData,
    *,
    cohort_outputs: Sequence[Mapping[str, object]],
    dictionary_bytes: bytes,
    readme_bytes: bytes,
) -> dict[str, object]:
    ordered_family_values = [
        f"{record['cohort']}\t{record['sha256']}\t{record['rows']}"
        for record in cohort_outputs
    ]
    return {
        "schema": SOURCE_DATA_SCHEMA,
        "contract": SOURCE_DATA_CONTRACT,
        "analysis": "tcga-revision-k500",
        "production_eligible": True,
        "dataset_id": "k500-complete-conjunction-families",
        "scope": (
            "complete-within-cohort-three-bmr-conjunction-families; "
            "not-raw-fit-runtime-calibration-comparator-or-msk-data"
        ),
        "top_k": TOP_K,
        "bmrs": list(BMRS),
        "cohorts": list(TCGA_COHORTS),
        "authority": {
            "release_approval_manifest_sha256": validated.approval_sha256,
            "release_decision_digests": dict(validated.decision_digests),
            "postprocess_release_manifest_sha256": (validated.release_manifest_sha256),
            "postprocess_authority_sha256": validated.authority_sha256,
            "grid_authority_sha256": validated.grid_authority_sha256,
            "postprocess_implementation_sha256": (
                validated.postprocess_implementation_sha256
            ),
            "sealed_completion_sha256": validated.sealed_completion_sha256,
            "canonical_input_manifest_sha256": validated.canonical_input_sha256,
            "provider_input_manifest_sha256": validated.provider_input_sha256,
            "marginal_validity_evidence_sha256": (
                validated.marginal_validity_evidence_sha256
            ),
        },
        "builder_implementation": dict(validated.builder_implementation),
        "dataset": {
            "total_rows": validated.total_rows,
            "ordered_family_sha256": _sequence_sha256(ordered_family_values),
            "columns": list(OUTPUT_COLUMNS),
            "cohort_files": [dict(record) for record in cohort_outputs],
        },
        "supporting_files": {
            "data_dictionary": {
                "path": DATA_DICTIONARY_NAME,
                "bytes": len(dictionary_bytes),
                "sha256": _sha256(dictionary_bytes),
            },
            "readme": {
                "path": README_NAME,
                "bytes": len(readme_bytes),
                "sha256": _sha256(readme_bytes),
            },
        },
    }


def _validate_source_data_release_manifest(
    manifest: Mapping[str, object],
) -> _SourceDataValidationPlan:
    _require_exact_keys(
        manifest,
        {
            "analysis",
            "authority",
            "bmrs",
            "builder_implementation",
            "cohorts",
            "contract",
            "dataset",
            "dataset_id",
            "production_eligible",
            "schema",
            "scope",
            "supporting_files",
            "top_k",
        },
        label="source-data release manifest",
    )
    if (
        manifest.get("schema") != SOURCE_DATA_SCHEMA
        or manifest.get("contract") != SOURCE_DATA_CONTRACT
        or manifest.get("analysis") != "tcga-revision-k500"
        or manifest.get("production_eligible") is not True
        or manifest.get("dataset_id") != "k500-complete-conjunction-families"
        or manifest.get("scope")
        != (
            "complete-within-cohort-three-bmr-conjunction-families; "
            "not-raw-fit-runtime-calibration-comparator-or-msk-data"
        )
        or manifest.get("top_k") != TOP_K
        or manifest.get("bmrs") != list(BMRS)
        or manifest.get("cohorts") != list(TCGA_COHORTS)
    ):
        msg = "Source-data release manifest metadata is invalid."
        raise SourceDataBuildError(msg)

    authority = _require_mapping(
        manifest.get("authority"),
        label="source-data release authority",
    )
    _require_exact_keys(
        authority,
        {
            "canonical_input_manifest_sha256",
            "grid_authority_sha256",
            "marginal_validity_evidence_sha256",
            "postprocess_authority_sha256",
            "postprocess_implementation_sha256",
            "postprocess_release_manifest_sha256",
            "provider_input_manifest_sha256",
            "release_approval_manifest_sha256",
            "release_decision_digests",
            "sealed_completion_sha256",
        },
        label="source-data release authority",
    )
    for key in (
        "canonical_input_manifest_sha256",
        "grid_authority_sha256",
        "postprocess_authority_sha256",
        "postprocess_implementation_sha256",
        "postprocess_release_manifest_sha256",
        "provider_input_manifest_sha256",
        "release_approval_manifest_sha256",
        "sealed_completion_sha256",
    ):
        _require_sha256(authority.get(key), label=f"source-data authority {key}")
    marginal_digest = authority.get("marginal_validity_evidence_sha256")
    if marginal_digest is not None:
        _require_sha256(
            marginal_digest,
            label="source-data marginal-validity evidence",
        )
    if authority["postprocess_authority_sha256"] != authority["grid_authority_sha256"]:
        msg = "Source-data release authority digests are inconsistent."
        raise SourceDataBuildError(msg)
    decision_digests = _require_mapping(
        authority.get("release_decision_digests"),
        label="source-data release decision digests",
    )
    if set(decision_digests) != set(DECISION_IDS):
        msg = "Source-data release decision inventory is invalid."
        raise SourceDataBuildError(msg)
    for decision_id, digest in decision_digests.items():
        _require_sha256(digest, label=f"source-data decision {decision_id}")

    builder_implementation = _require_mapping(
        manifest.get("builder_implementation"),
        label="source-data builder implementation",
    )
    _validate_builder_implementation_record(builder_implementation)

    dataset = _require_mapping(
        manifest.get("dataset"),
        label="source-data dataset",
    )
    _require_exact_keys(
        dataset,
        {"cohort_files", "columns", "ordered_family_sha256", "total_rows"},
        label="source-data dataset",
    )
    total_rows = _require_positive_integer(
        dataset.get("total_rows"),
        label="source-data total rows",
    )
    if dataset.get("columns") != list(OUTPUT_COLUMNS):
        msg = "Source-data release columns are invalid."
        raise SourceDataBuildError(msg)
    _require_sha256(
        dataset.get("ordered_family_sha256"),
        label="source-data ordered family",
    )
    raw_cohort_files = dataset.get("cohort_files")
    if not isinstance(raw_cohort_files, list) or len(raw_cohort_files) != len(
        TCGA_COHORTS,
    ):
        msg = "Source-data release does not cover the exact cohort inventory."
        raise SourceDataBuildError(msg)
    cohort_files: list[Mapping[str, object]] = []
    minimum_pairs = TOP_K * (TOP_K - 1) // 2 - TOP_K // 2
    maximum_pairs = TOP_K * (TOP_K - 1) // 2
    for cohort, raw_record in zip(TCGA_COHORTS, raw_cohort_files, strict=True):
        record = _require_mapping(
            raw_record,
            label=f"source-data cohort receipt {cohort}",
        )
        _require_exact_keys(
            record,
            {
                "bytes",
                "cohort",
                "cohort_contract_sha256",
                "ordered_features_sha256",
                "ordered_pair_sha256",
                "path",
                "publication_binding_sha256",
                "rows",
                "sha256",
                "source_csv_sha256",
                "source_manifest_sha256",
            },
            label=f"source-data cohort receipt {cohort}",
        )
        rows = _require_positive_integer(
            record.get("rows"),
            label=f"source-data {cohort} rows",
        )
        _require_positive_integer(
            record.get("bytes"),
            label=f"source-data {cohort} bytes",
        )
        for key in (
            "cohort_contract_sha256",
            "ordered_features_sha256",
            "ordered_pair_sha256",
            "publication_binding_sha256",
            "sha256",
            "source_csv_sha256",
            "source_manifest_sha256",
        ):
            _require_sha256(record.get(key), label=f"source-data {cohort} {key}")
        expected_binding = _sha256(
            _canonical_json(
                {
                    "authority_sha256": authority["postprocess_authority_sha256"],
                    "cohort": cohort,
                    "csv_sha256": record["source_csv_sha256"],
                    "manifest_sha256": record["source_manifest_sha256"],
                    "row_count": rows,
                },
            ),
        )
        if (
            record.get("cohort") != cohort
            or record.get("path") != f"{COHORT_DIRECTORY_NAME}/{cohort}.csv"
            or record.get("sha256") != record.get("source_csv_sha256")
            or record.get("publication_binding_sha256") != expected_binding
            or not minimum_pairs <= rows <= maximum_pairs
        ):
            msg = f"Source-data cohort receipt is invalid for {cohort}."
            raise SourceDataBuildError(msg)
        cohort_files.append(record)
    if total_rows != sum(int(record["rows"]) for record in cohort_files):
        msg = "Source-data total row count is inconsistent."
        raise SourceDataBuildError(msg)
    ordered_family_values = [
        f"{record['cohort']}\t{record['sha256']}\t{record['rows']}"
        for record in cohort_files
    ]
    if dataset["ordered_family_sha256"] != _sequence_sha256(
        ordered_family_values,
    ):
        msg = "Source-data ordered family digest is invalid."
        raise SourceDataBuildError(msg)

    supporting = _require_mapping(
        manifest.get("supporting_files"),
        label="source-data supporting files",
    )
    _require_exact_keys(
        supporting,
        {"data_dictionary", "readme"},
        label="source-data supporting files",
    )
    expected_support = (
        (
            "data_dictionary",
            DATA_DICTIONARY_NAME,
            _canonical_json(_data_dictionary()) + b"\n",
        ),
        ("readme", README_NAME, _readme_bytes()),
    )
    supporting_files: list[Mapping[str, object]] = []
    for key, expected_path, expected_bytes in expected_support:
        record = _require_mapping(
            supporting.get(key),
            label=f"source-data support receipt {key}",
        )
        _require_exact_keys(
            record,
            {"bytes", "path", "sha256"},
            label=f"source-data support receipt {key}",
        )
        size_bytes = _require_positive_integer(
            record.get("bytes"),
            label=f"source-data support bytes {key}",
        )
        _require_sha256(
            record.get("sha256"),
            label=f"source-data support digest {key}",
        )
        if (
            record.get("path") != expected_path
            or size_bytes != len(expected_bytes)
            or record.get("sha256") != _sha256(expected_bytes)
        ):
            msg = f"Source-data support receipt is invalid for {key}."
            raise SourceDataBuildError(msg)
        supporting_files.append(record)

    return _SourceDataValidationPlan(
        builder_implementation=dict(builder_implementation),
        supporting_files=tuple(supporting_files),
        cohort_files=tuple(cohort_files),
        total_rows=total_rows,
    )


def _data_dictionary() -> dict[str, object]:
    definitions = {
        "schema": ("string", False, "Row schema identifier.", "provenance"),
        "derivation_contract": (
            "string",
            False,
            "Complete-family conjunction derivation contract.",
            "provenance",
        ),
        "d5_contract": ("string", False, "Multiplicity policy contract.", "provenance"),
        "cohort": ("string", False, "TCGA cohort abbreviation.", "identifier"),
        "gene_a": (
            "string",
            False,
            "First ordered mutation-event feature.",
            "identifier",
        ),
        "gene_b": (
            "string",
            False,
            "Second ordered mutation-event feature.",
            "identifier",
        ),
        "conjunction_p_value": (
            "number",
            False,
            "Maximum valid component p-value for the three-BMR conjunction.",
            "inferential",
        ),
        "consensus_direction": (
            "string",
            False,
            "Direction annotation when provider directions agree.",
            "annotation",
        ),
        "d3_conjunction_role": (
            "string",
            False,
            "Prespecified reporting role of the all-three-BMR conjunction.",
            "provenance",
        ),
        "by_q_value": ("number", False, "Within-cohort BY q-value.", "inferential"),
        "bh_q_value": (
            "number",
            False,
            "Within-cohort BH sensitivity q-value.",
            "sensitivity",
        ),
        "marginal_validity_status": (
            "string",
            False,
            "Calibration-gated marginal-validity disposition.",
            "provenance",
        ),
        "conditional_by_inferential_eligible": (
            "boolean",
            False,
            "Whether the conditional BY layer is reportable.",
            "inferential_gate",
        ),
        "by_q_le_0_01": (
            "boolean",
            False,
            "BY q-value at or below 0.01.",
            "inferential",
        ),
        "conditional_by_q_le_0_01_reportable": (
            "boolean",
            False,
            "BY 0.01 flag after marginal-validity gating.",
            "inferential_gate",
        ),
        "bh_q_le_0_01_nominal": (
            "boolean",
            False,
            "Nominal BH sensitivity flag at 0.01.",
            "sensitivity",
        ),
        "by_q_le_0_05_descriptive": (
            "boolean",
            False,
            "Descriptive BY flag at 0.05.",
            "descriptive",
        ),
        "bh_q_le_0_05_descriptive": (
            "boolean",
            False,
            "Descriptive BH flag at 0.05.",
            "descriptive",
        ),
        "cohort_contract_sha256": (
            "sha256",
            False,
            "Digest of the sealed cohort contract.",
            "provenance",
        ),
        "ordered_features_sha256": (
            "sha256",
            False,
            "Digest of the ordered K=500 feature axis.",
            "provenance",
        ),
        "ordered_pair_sha256": (
            "sha256",
            False,
            "Digest of the ordered complete tested-pair axis.",
            "provenance",
        ),
    }
    for provider in BMRS:
        definitions[f"{provider}_component_status"] = (
            "string",
            False,
            f"{provider} component inference status.",
            "status",
        )
        definitions[f"{provider}_p_value"] = (
            "number",
            False,
            f"{provider} component p-value.",
            "inferential",
        )
        definitions[f"{provider}_direction"] = (
            "string",
            False,
            f"{provider} direction annotation.",
            "annotation",
        )
        definitions[f"{provider}_effect_identifiability"] = (
            "string",
            False,
            f"{provider} effect-identifiability status.",
            "status",
        )
        definitions[f"{provider}_source_sha256"] = (
            "sha256",
            False,
            f"Digest of the sealed {provider} pairwise source.",
            "provenance",
        )
    return {
        "schema": "dialect-tcga-k500-source-data-dictionary-v1",
        "dataset_id": "k500-complete-conjunction-families",
        "null_encoding": "none; empty CSV fields are forbidden",
        "boolean_encoding": ["false", "true"],
        "columns": [
            {
                "name": name,
                "data_type": definitions[name][0],
                "nullable": definitions[name][1],
                "description": definitions[name][2],
                "reporting_role": definitions[name][3],
            }
            for name in OUTPUT_COLUMNS
        ],
    }


def _readme_bytes() -> bytes:
    return (
        "# DIALECT TCGA K=500 conjunction source data\n\n"
        "This immutable package contains exact copies of the 32 authenticated "
        "within-cohort three-BMR conjunction families. Each cohort uses the same "
        "sealed K=500 event axis and includes all tested unordered pairs after "
        "the prespecified same-gene missense/nonsense exclusion.\n\n"
        "The package does not contain raw fit tables, runtime logs, calibration "
        "replicates, comparator outputs, or MSK replication data. Those are "
        "separate gated datasets. See `data_dictionary.json` for column semantics "
        "and `source_data_manifest.json` for exact hashes and authority bindings.\n"
    ).encode("ascii")


def _builder_implementation() -> dict[str, object]:
    files = {
        "analysis/build_tcga_revision_source_data.py": Path(__file__).resolve(),
        "src/dialect/data/revision_approval.py": Path(
            revision_approval_module.__file__,
        ).resolve(),
    }
    hashes = {
        label: _hash_live_implementation(path, label=label)
        for label, path in files.items()
    }
    record = {
        "files": hashes,
        "combined_sha256": _sha256(_canonical_json(hashes)),
    }
    _validate_builder_implementation_record(record)
    return record


def _hash_live_implementation(path: Path, *, label: str) -> str:
    resolved = path.resolve(strict=True)
    if resolved != path:
        msg = f"Builder implementation path is not canonical: {label}."
        raise SourceDataBuildError(msg)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(resolved, flags)
    try:
        before = os.fstat(descriptor)
        entry_before = resolved.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or entry_before.st_dev != before.st_dev
            or entry_before.st_ino != before.st_ino
        ):
            msg = f"Builder implementation is not a stable regular file: {label}."
            raise SourceDataBuildError(msg)
        digest = _hash_stable_descriptor(descriptor, before, label=label)
        entry_after = resolved.stat(follow_symlinks=False)
        if (
            entry_after.st_dev != before.st_dev
            or entry_after.st_ino != before.st_ino
            or entry_after.st_size != before.st_size
            or entry_after.st_mtime_ns != before.st_mtime_ns
            or entry_after.st_ctime_ns != before.st_ctime_ns
        ):
            msg = f"Builder implementation changed during hashing: {label}."
            raise SourceDataBuildError(msg)
        return digest
    finally:
        os.close(descriptor)


def _validate_builder_implementation_record(
    record: Mapping[str, object],
) -> None:
    if set(record) != {"combined_sha256", "files"}:
        msg = "Builder implementation provenance has an invalid closed schema."
        raise SourceDataBuildError(msg)
    files = _require_mapping(record.get("files"), label="builder implementation files")
    expected = {
        "analysis/build_tcga_revision_source_data.py",
        "src/dialect/data/revision_approval.py",
    }
    if set(files) != expected:
        msg = "Builder implementation provenance has an invalid file inventory."
        raise SourceDataBuildError(msg)
    for label, digest in files.items():
        _require_sha256(digest, label=f"builder implementation {label}")
    if record.get("combined_sha256") != _sha256(_canonical_json(dict(files))):
        msg = "Builder implementation combined digest is invalid."
        raise SourceDataBuildError(msg)


def _revalidate_builder_implementation(validated: _ValidatedSourceData) -> None:
    _validate_builder_implementation_record(validated.builder_implementation)
    if _builder_implementation() != validated.builder_implementation:
        msg = "Builder implementation changed before atomic publication."
        raise SourceDataBuildError(msg)


def _require_live_builder_implementation(
    expected: Mapping[str, object],
) -> None:
    _validate_builder_implementation_record(expected)
    if _builder_implementation() != expected:
        msg = "Source-data release builder implementation has drifted."
        raise SourceDataBuildError(msg)


def _require_validated(validated: _ValidatedSourceData) -> None:
    if (
        not isinstance(validated, _ValidatedSourceData)
        or getattr(validated, "_seal", None) is not _PRODUCTION_SEAL
        or tuple(record.get("cohort") for record in validated.cohorts) != TCGA_COHORTS
        or len(validated.decision_digests) != len(DECISION_IDS)
        or tuple(validated.decision_digests) != DECISION_IDS
        or validated.total_rows
        != sum(int(record["rows"]) for record in validated.cohorts)
    ):
        msg = "Source-data publication requires a sealed production authority."
        raise SourceDataBuildError(msg)
    _validate_builder_implementation_record(validated.builder_implementation)


def _open_frozen_directory(
    path: Path,
    *,
    label: str,
) -> tuple[Path, int, os.stat_result]:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        msg = f"Cannot securely resolve {label}."
        raise SourceDataBuildError(msg) from error
    if resolved != path:
        msg = f"{label} must not use symlinked path components."
        raise SourceDataBuildError(msg)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(resolved, flags)
    except OSError as error:
        msg = f"Cannot securely open {label}."
        raise SourceDataBuildError(msg) from error
    try:
        identity = os.fstat(descriptor)
        _require_frozen_directory_identity(identity, label=label)
        path_identity = resolved.stat(follow_symlinks=False)
        _require_matching_directory_identity(path_identity, identity, label=label)
    except OSError as error:
        os.close(descriptor)
        msg = f"Cannot securely inspect {label}."
        raise SourceDataBuildError(msg) from error
    except Exception:
        os.close(descriptor)
        raise
    return resolved, descriptor, identity


def _open_frozen_child_directory(
    parent_fd: int,
    name: str,
    *,
    label: str,
) -> tuple[int, os.stat_result]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        msg = f"Cannot securely open {label}."
        raise SourceDataBuildError(msg) from error
    try:
        identity = os.fstat(descriptor)
        _require_frozen_directory_identity(identity, label=label)
        _require_directory_entry_identity(
            parent_fd,
            name,
            identity,
            label=label,
        )
    except OSError as error:
        os.close(descriptor)
        msg = f"Cannot securely open {label}."
        raise SourceDataBuildError(msg) from error
    except Exception:
        os.close(descriptor)
        raise
    return descriptor, identity


def _open_frozen_file(  # noqa: PLR0913
    directory_fd: int,
    name: str,
    *,
    label: str,
    retain_content: bool = True,
    expected_size_bytes: int | None = None,
    max_bytes: int | None = None,
) -> _PinnedFile:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        msg = f"Cannot securely open {label}."
        raise SourceDataBuildError(msg) from error
    try:
        identity = os.fstat(descriptor)
        _require_frozen_file_identity(identity, label=label)
        _require_bounded_file_size(
            identity,
            expected_size_bytes=expected_size_bytes,
            max_bytes=max_bytes,
            label=label,
        )
        _require_file_entry_identity(
            directory_fd,
            name,
            identity,
            label=label,
        )
        content = (
            _read_stable_descriptor(descriptor, identity, label=label)
            if retain_content
            else None
        )
        digest = (
            _sha256(content)
            if content is not None
            else _hash_stable_descriptor(descriptor, identity, label=label)
        )
        _require_file_entry_identity(
            directory_fd,
            name,
            identity,
            label=label,
        )
        pin = _PinnedFile(
            name=name,
            descriptor=descriptor,
            identity=identity,
            sha256=digest,
            size_bytes=identity.st_size,
            content=content,
        )
    except OSError as error:
        os.close(descriptor)
        msg = f"Cannot securely open {label}."
        raise SourceDataBuildError(msg) from error
    except Exception:
        os.close(descriptor)
        raise
    return pin


def _read_stable_descriptor(
    descriptor: int,
    identity: os.stat_result,
    *,
    label: str,
) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    after = os.fstat(descriptor)
    if not _same_file_snapshot(identity, after):
        msg = f"{label} changed during descriptor-pinned read."
        raise SourceDataBuildError(msg)
    return b"".join(chunks)


def _hash_stable_descriptor(
    descriptor: int,
    identity: os.stat_result,
    *,
    label: str,
) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
    after = os.fstat(descriptor)
    if not _same_file_snapshot(identity, after):
        msg = f"{label} changed during descriptor-pinned hashing."
        raise SourceDataBuildError(msg)
    return digest.hexdigest()


def _retained_content(pin: _PinnedFile, *, label: str) -> bytes:
    if pin.content is None:
        msg = f"{label} content was not retained by the metadata validator."
        raise SourceDataBuildError(msg)
    return pin.content


def _require_pinned_release_member(
    pin: _PinnedFile,
    record: Mapping[str, object],
    *,
    label: str,
) -> None:
    if pin.size_bytes != record.get("bytes") or pin.sha256 != record.get("sha256"):
        msg = f"Source-data release member does not match its receipt: {label}."
        raise SourceDataBuildError(msg)


def _revalidate_source_data_release_tree(  # noqa: PLR0913
    root_path: Path,
    root_fd: int,
    root_identity: os.stat_result,
    *,
    root_pins: Sequence[_PinnedFile],
    cohorts_fd: int,
    cohorts_identity: os.stat_result,
    cohort_pins: Sequence[_PinnedFile],
) -> None:
    _require_frozen_directory_snapshot(
        root_path,
        root_fd,
        root_identity,
        label="source-data release root",
    )
    _require_frozen_child_directory_snapshot(
        root_fd,
        COHORT_DIRECTORY_NAME,
        cohorts_fd,
        cohorts_identity,
        label="source-data cohorts directory",
    )
    _require_inventory(
        root_fd,
        {
            COHORT_DIRECTORY_NAME,
            DATA_DICTIONARY_NAME,
            README_NAME,
            SOURCE_DATA_MANIFEST_NAME,
        },
        label="source-data release root",
    )
    _require_inventory(
        cohorts_fd,
        {f"{cohort}.csv" for cohort in TCGA_COHORTS},
        label="source-data cohort files",
    )
    if [pin.name for pin in root_pins] != [
        SOURCE_DATA_MANIFEST_NAME,
        DATA_DICTIONARY_NAME,
        README_NAME,
    ]:
        msg = "Pinned source-data root inventory is invalid."
        raise SourceDataBuildError(msg)
    if [pin.name for pin in cohort_pins] != [
        f"{cohort}.csv" for cohort in TCGA_COHORTS
    ]:
        msg = "Pinned source-data cohort inventory is invalid."
        raise SourceDataBuildError(msg)
    for pin in root_pins:
        _revalidate_frozen_release_member(root_fd, pin)
    for pin in cohort_pins:
        _revalidate_frozen_release_member(cohorts_fd, pin)
    _require_inventory(
        cohorts_fd,
        {f"{cohort}.csv" for cohort in TCGA_COHORTS},
        label="source-data cohort files",
    )
    _require_inventory(
        root_fd,
        {
            COHORT_DIRECTORY_NAME,
            DATA_DICTIONARY_NAME,
            README_NAME,
            SOURCE_DATA_MANIFEST_NAME,
        },
        label="source-data release root",
    )
    _require_frozen_child_directory_snapshot(
        root_fd,
        COHORT_DIRECTORY_NAME,
        cohorts_fd,
        cohorts_identity,
        label="source-data cohorts directory",
    )
    _require_frozen_directory_snapshot(
        root_path,
        root_fd,
        root_identity,
        label="source-data release root",
    )


def _revalidate_frozen_release_member(
    directory_fd: int,
    pin: _PinnedFile,
) -> None:
    identity = os.fstat(pin.descriptor)
    _require_frozen_file_identity(identity, label=pin.name)
    if not _same_file_snapshot(pin.identity, identity):
        msg = f"Pinned source-data member identity changed: {pin.name}."
        raise SourceDataBuildError(msg)
    _require_file_entry_identity(
        directory_fd,
        pin.name,
        pin.identity,
        label=f"source-data member {pin.name}",
    )
    digest = _hash_stable_descriptor(
        pin.descriptor,
        pin.identity,
        label=f"source-data member {pin.name}",
    )
    _require_file_entry_identity(
        directory_fd,
        pin.name,
        pin.identity,
        label=f"source-data member {pin.name}",
    )
    if digest != pin.sha256:
        msg = f"Pinned source-data member bytes changed: {pin.name}."
        raise SourceDataBuildError(msg)


def _require_frozen_directory_snapshot(
    path: Path,
    descriptor: int,
    expected: os.stat_result,
    *,
    label: str,
) -> None:
    try:
        resolved = path.resolve(strict=True)
        path_identity = path.stat(follow_symlinks=False)
        current = os.fstat(descriptor)
    except (OSError, RuntimeError) as error:
        msg = f"{label} cannot be revalidated."
        raise SourceDataBuildError(msg) from error
    if (
        resolved != path
        or not _same_frozen_directory_snapshot(expected, current)
        or not _same_frozen_directory_snapshot(expected, path_identity)
    ):
        msg = f"{label} changed during validation."
        raise SourceDataBuildError(msg)


def _require_frozen_child_directory_snapshot(
    parent_fd: int,
    name: str,
    descriptor: int,
    expected: os.stat_result,
    *,
    label: str,
) -> None:
    try:
        current = os.fstat(descriptor)
        entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        msg = f"{label} cannot be revalidated."
        raise SourceDataBuildError(msg) from error
    if not _same_frozen_directory_snapshot(
        expected,
        current,
    ) or not _same_frozen_directory_snapshot(expected, entry):
        msg = f"{label} changed during validation."
        raise SourceDataBuildError(msg)


def _same_frozen_directory_snapshot(
    first: os.stat_result,
    second: os.stat_result,
) -> bool:
    return (
        stat.S_ISDIR(second.st_mode)
        and stat.S_IMODE(second.st_mode) == 0o500
        and second.st_dev == first.st_dev
        and second.st_ino == first.st_ino
        and second.st_nlink == first.st_nlink
        and second.st_size == first.st_size
        and second.st_mtime_ns == first.st_mtime_ns
        and second.st_ctime_ns == first.st_ctime_ns
    )


def _revalidate_input_tree(
    root_path: Path,
    root_fd: int,
    root_identity: os.stat_result,
    root_files: Sequence[_PinnedFile],
    cohort_pins: Sequence[_PinnedCohort],
) -> None:
    expected_root_inventory = {
        *TCGA_COHORTS,
        POSTPROCESS_AUTHORITY_NAME,
        POSTPROCESS_RELEASE_MANIFEST_NAME,
    }
    _require_frozen_directory_snapshot(
        root_path,
        root_fd,
        root_identity,
        label="postprocess release root",
    )
    _require_inventory(
        root_fd,
        expected_root_inventory,
        label="postprocess release root",
    )
    for pin in root_files:
        _require_file_entry_identity(
            root_fd,
            pin.name,
            pin.identity,
            label=pin.name,
        )
        digest = _hash_stable_descriptor(
            pin.descriptor,
            pin.identity,
            label=pin.name,
        )
        _require_file_entry_identity(
            root_fd,
            pin.name,
            pin.identity,
            label=pin.name,
        )
        if digest != pin.sha256:
            msg = f"Pinned postprocess file changed: {pin.name}."
            raise SourceDataBuildError(msg)
    for pin in cohort_pins:
        _require_frozen_child_directory_snapshot(
            root_fd,
            pin.cohort,
            pin.descriptor,
            pin.identity,
            label=f"postprocess cohort {pin.cohort}",
        )
        _require_inventory(
            pin.descriptor,
            {POSTPROCESS_CSV_NAME, POSTPROCESS_COHORT_MANIFEST_NAME},
            label=f"postprocess cohort {pin.cohort}",
        )
        for file_pin in (pin.csv_file, pin.manifest_file):
            _require_file_entry_identity(
                pin.descriptor,
                file_pin.name,
                file_pin.identity,
                label=f"{pin.cohort}/{file_pin.name}",
            )
            digest = _hash_stable_descriptor(
                file_pin.descriptor,
                file_pin.identity,
                label=f"{pin.cohort}/{file_pin.name}",
            )
            _require_file_entry_identity(
                pin.descriptor,
                file_pin.name,
                file_pin.identity,
                label=f"{pin.cohort}/{file_pin.name}",
            )
            if digest != file_pin.sha256:
                msg = f"Pinned postprocess bytes changed: {pin.cohort}/{file_pin.name}."
                raise SourceDataBuildError(msg)
        _require_inventory(
            pin.descriptor,
            {POSTPROCESS_CSV_NAME, POSTPROCESS_COHORT_MANIFEST_NAME},
            label=f"postprocess cohort {pin.cohort}",
        )
        _require_frozen_child_directory_snapshot(
            root_fd,
            pin.cohort,
            pin.descriptor,
            pin.identity,
            label=f"postprocess cohort {pin.cohort}",
        )
    _require_inventory(
        root_fd,
        expected_root_inventory,
        label="postprocess release root",
    )
    for pin in cohort_pins:
        _require_frozen_child_directory_snapshot(
            root_fd,
            pin.cohort,
            pin.descriptor,
            pin.identity,
            label=f"postprocess cohort {pin.cohort}",
        )
    _require_frozen_directory_snapshot(
        root_path,
        root_fd,
        root_identity,
        label="postprocess release root",
    )


def _pin_output_file(
    directory_fd: int,
    name: str,
    *,
    expected_sha256: str,
    expected_size: int,
) -> _PinnedOutputFile:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        identity = os.fstat(descriptor)
        _require_staged_output_identity(
            descriptor,
            identity,
            name=name,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        )
        _require_file_entry_identity(
            directory_fd,
            name,
            identity,
            label=f"staged output {name}",
        )
        return _PinnedOutputFile(
            name=name,
            descriptor=descriptor,
            device=identity.st_dev,
            inode=identity.st_ino,
            sha256=expected_sha256,
            size_bytes=expected_size,
        )
    except Exception:
        os.close(descriptor)
        raise


def _require_staged_output_identity(
    descriptor: int,
    identity: os.stat_result,
    *,
    name: str,
    expected_sha256: str,
    expected_size: int,
) -> None:
    if (
        not stat.S_ISREG(identity.st_mode)
        or identity.st_nlink != 1
        or stat.S_IMODE(identity.st_mode) != 0o600
        or identity.st_size != expected_size
        or _hash_stable_descriptor(descriptor, identity, label=name) != expected_sha256
    ):
        msg = f"Cannot pin staged source-data output: {name}."
        raise SourceDataBuildError(msg)


def _validate_pinned_output_tree(  # noqa: PLR0913
    root_fd: int,
    cohorts_fd: int,
    *,
    staging_identity: os.stat_result,
    cohorts_identity: os.stat_result,
    root_pins: Sequence[_PinnedOutputFile],
    cohort_pins: Sequence[_PinnedOutputFile],
    frozen: bool,
) -> None:
    expected_mode = 0o500 if frozen else 0o700
    root_identity = os.fstat(root_fd)
    current_cohorts_identity = os.fstat(cohorts_fd)
    if (
        root_identity.st_dev != staging_identity.st_dev
        or root_identity.st_ino != staging_identity.st_ino
        or stat.S_IMODE(root_identity.st_mode) != expected_mode
    ):
        msg = "Source-data staging root mode is invalid."
        raise SourceDataBuildError(msg)
    if (
        current_cohorts_identity.st_dev != cohorts_identity.st_dev
        or current_cohorts_identity.st_ino != cohorts_identity.st_ino
        or stat.S_IMODE(current_cohorts_identity.st_mode) != expected_mode
    ):
        msg = "Source-data cohort directory mode is invalid."
        raise SourceDataBuildError(msg)
    _require_directory_entry_identity(
        root_fd,
        COHORT_DIRECTORY_NAME,
        cohorts_identity,
        label="source-data cohorts directory",
    )
    _require_inventory(
        root_fd,
        {
            COHORT_DIRECTORY_NAME,
            DATA_DICTIONARY_NAME,
            README_NAME,
            SOURCE_DATA_MANIFEST_NAME,
        },
        label="source-data output root",
    )
    _require_inventory(
        cohorts_fd,
        {f"{cohort}.csv" for cohort in TCGA_COHORTS},
        label="source-data cohort output",
    )
    if [pin.name for pin in root_pins] != [
        DATA_DICTIONARY_NAME,
        README_NAME,
        SOURCE_DATA_MANIFEST_NAME,
    ]:
        msg = "Pinned source-data root-file inventory is invalid."
        raise SourceDataBuildError(msg)
    if [pin.name for pin in cohort_pins] != [
        f"{cohort}.csv" for cohort in TCGA_COHORTS
    ]:
        msg = "Pinned source-data cohort-file inventory is invalid."
        raise SourceDataBuildError(msg)
    for pin in root_pins:
        _validate_output_pin(root_fd, pin, frozen=frozen)
    for pin in cohort_pins:
        _validate_output_pin(cohorts_fd, pin, frozen=frozen)
    _require_inventory(
        cohorts_fd,
        {f"{cohort}.csv" for cohort in TCGA_COHORTS},
        label="source-data cohort output",
    )
    _require_inventory(
        root_fd,
        {
            COHORT_DIRECTORY_NAME,
            DATA_DICTIONARY_NAME,
            README_NAME,
            SOURCE_DATA_MANIFEST_NAME,
        },
        label="source-data output root",
    )
    _require_directory_entry_identity(
        root_fd,
        COHORT_DIRECTORY_NAME,
        cohorts_identity,
        label="source-data cohorts directory",
    )
    final_cohorts_identity = os.fstat(cohorts_fd)
    final_root_identity = os.fstat(root_fd)
    if not _same_output_directory_snapshot(
        current_cohorts_identity,
        final_cohorts_identity,
        expected_mode=expected_mode,
    ):
        msg = "Source-data cohort directory changed during validation."
        raise SourceDataBuildError(msg)
    if not _same_output_directory_snapshot(
        root_identity,
        final_root_identity,
        expected_mode=expected_mode,
    ):
        msg = "Source-data staging root changed during validation."
        raise SourceDataBuildError(msg)


def _same_output_directory_snapshot(
    first: os.stat_result,
    second: os.stat_result,
    *,
    expected_mode: int,
) -> bool:
    return (
        stat.S_ISDIR(second.st_mode)
        and stat.S_IMODE(second.st_mode) == expected_mode
        and second.st_dev == first.st_dev
        and second.st_ino == first.st_ino
        and second.st_nlink == first.st_nlink
        and second.st_size == first.st_size
        and second.st_mtime_ns == first.st_mtime_ns
        and second.st_ctime_ns == first.st_ctime_ns
    )


def _validate_output_pin(
    directory_fd: int,
    pin: _PinnedOutputFile,
    *,
    frozen: bool,
) -> None:
    identity = os.fstat(pin.descriptor)
    expected_mode = 0o400 if frozen else 0o600
    _require_file_entry_identity(
        directory_fd,
        pin.name,
        identity,
        label=f"source-data output {pin.name}",
    )
    digest = _hash_stable_descriptor(pin.descriptor, identity, label=pin.name)
    _require_file_entry_identity(
        directory_fd,
        pin.name,
        identity,
        label=f"source-data output {pin.name}",
    )
    if (
        identity.st_dev != pin.device
        or identity.st_ino != pin.inode
        or stat.S_IMODE(identity.st_mode) != expected_mode
        or identity.st_size != pin.size_bytes
        or digest != pin.sha256
    ):
        msg = f"Pinned source-data output verification failed: {pin.name}."
        raise SourceDataBuildError(msg)


def _write_file(directory_fd: int, name: str, content: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            msg = f"Staged output is not a single-link regular file: {name}."
            raise SourceDataBuildError(msg)
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                msg = f"Staged output write made no progress: {name}."
                raise OSError(msg)
            view = view[written:]
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_nlink != 1
            or after.st_size != len(content)
        ):
            msg = f"Staged output changed while writing: {name}."
            raise SourceDataBuildError(msg)
    finally:
        os.close(descriptor)


def _write_file_from_pin(
    directory_fd: int,
    name: str,
    source: _PinnedFile,
) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        output_before = os.fstat(descriptor)
        source_before = os.fstat(source.descriptor)
        if (
            not stat.S_ISREG(output_before.st_mode)
            or output_before.st_nlink != 1
            or not _same_file_snapshot(source.identity, source_before)
        ):
            msg = f"Source-data streaming precondition failed: {name}."
            raise SourceDataBuildError(msg)
        os.lseek(source.descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        copied = 0
        while chunk := os.read(source.descriptor, 1024 * 1024):
            digest.update(chunk)
            copied += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    msg = f"Source-data streaming write made no progress: {name}."
                    raise OSError(msg)
                view = view[written:]
        os.fsync(descriptor)
        output_after = os.fstat(descriptor)
        source_after = os.fstat(source.descriptor)
        if (
            not _same_file_snapshot(source_before, source_after)
            or output_after.st_dev != output_before.st_dev
            or output_after.st_ino != output_before.st_ino
            or output_after.st_nlink != 1
            or output_after.st_size != source.size_bytes
            or copied != source.size_bytes
            or digest.hexdigest() != source.sha256
        ):
            msg = f"Source-data streaming copy failed verification: {name}."
            raise SourceDataBuildError(msg)
    finally:
        os.close(descriptor)


def _rename_no_replace(parent_fd: int, source: str, destination: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if hasattr(libc, "renameatx_np"):
        rename = libc.renameatx_np
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(parent_fd, source_bytes, parent_fd, destination_bytes, 4)
    elif hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(parent_fd, source_bytes, parent_fd, destination_bytes, 1)
    else:
        msg = "Platform lacks an atomic no-replace rename primitive."
        raise OSError(errno.ENOTSUP, msg, destination)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def _require_parent(
    lexical: Path,
    resolved: Path,
    descriptor: int,
    expected: os.stat_result,
) -> None:
    current = os.fstat(descriptor)
    path_identity = resolved.stat(follow_symlinks=False)
    if (
        lexical.resolve(strict=True) != resolved
        or current.st_dev != expected.st_dev
        or current.st_ino != expected.st_ino
        or path_identity.st_dev != expected.st_dev
        or path_identity.st_ino != expected.st_ino
        or not stat.S_ISDIR(current.st_mode)
    ):
        msg = "Source-data publication parent changed during publication."
        raise SourceDataBuildError(msg)


def _close_cohort(pin: _PinnedCohort) -> None:
    os.close(pin.csv_file.descriptor)
    os.close(pin.manifest_file.descriptor)
    os.close(pin.descriptor)


def _require_inventory(directory_fd: int, expected: set[str], *, label: str) -> None:
    if set(os.listdir(directory_fd)) != expected:
        msg = f"{label} inventory is not closed."
        raise SourceDataBuildError(msg)


def _parse_canonical_json(raw: bytes, *, label: str) -> Mapping[str, object]:
    try:
        parsed = json.loads(raw)
        canonical = _canonical_json(parsed)
    except (
        TypeError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        msg = f"{label} is not valid JSON."
        raise SourceDataBuildError(msg) from error
    if not isinstance(parsed, dict) or canonical + b"\n" != raw:
        msg = f"{label} is not canonical JSON."
        raise SourceDataBuildError(msg)
    return parsed


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        msg = f"{label} must be a JSON object."
        raise SourceDataBuildError(msg)
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(value) != expected:
        msg = f"{label} has an invalid closed schema."
        raise SourceDataBuildError(msg)


def _require_positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"{label} must be a positive integer."
        raise SourceDataBuildError(msg)
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        msg = f"{label} must be a lowercase SHA-256."
        raise SourceDataBuildError(msg)
    return value


def _require_frozen_directory_identity(
    identity: os.stat_result,
    *,
    label: str,
) -> None:
    if not stat.S_ISDIR(identity.st_mode) or stat.S_IMODE(identity.st_mode) != 0o500:
        msg = f"{label} is not a frozen directory."
        raise SourceDataBuildError(msg)


def _require_matching_directory_identity(
    observed: os.stat_result,
    expected: os.stat_result,
    *,
    label: str,
) -> None:
    if (
        not stat.S_ISDIR(observed.st_mode)
        or observed.st_dev != expected.st_dev
        or observed.st_ino != expected.st_ino
    ):
        msg = f"{label} path identity changed while opening."
        raise SourceDataBuildError(msg)


def _require_bounded_file_size(
    identity: os.stat_result,
    *,
    expected_size_bytes: int | None,
    max_bytes: int | None,
    label: str,
) -> None:
    if expected_size_bytes is not None and identity.st_size != expected_size_bytes:
        msg = f"{label} does not match its manifest-declared receipt size."
        raise SourceDataBuildError(msg)
    if max_bytes is not None and identity.st_size > max_bytes:
        msg = f"{label} exceeds its bounded file size."
        raise SourceDataBuildError(msg)


def _require_frozen_file_identity(
    identity: os.stat_result,
    *,
    label: str,
) -> None:
    if (
        not stat.S_ISREG(identity.st_mode)
        or identity.st_nlink != 1
        or stat.S_IMODE(identity.st_mode) != 0o400
    ):
        msg = f"{label} must be a frozen single-link regular file."
        raise SourceDataBuildError(msg)


def _same_file_snapshot(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        stat.S_ISREG(second.st_mode)
        and second.st_dev == first.st_dev
        and second.st_ino == first.st_ino
        and second.st_nlink == 1
        and second.st_size == first.st_size
        and second.st_mtime_ns == first.st_mtime_ns
        and second.st_ctime_ns == first.st_ctime_ns
    )


def _require_directory_entry_identity(
    directory_fd: int,
    name: str,
    expected: os.stat_result,
    *,
    label: str,
) -> None:
    try:
        observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        msg = f"{label} directory entry cannot be revalidated."
        raise SourceDataBuildError(msg) from error
    if (
        not stat.S_ISDIR(observed.st_mode)
        or observed.st_dev != expected.st_dev
        or observed.st_ino != expected.st_ino
    ):
        msg = f"{label} directory entry identity changed."
        raise SourceDataBuildError(msg)


def _require_file_entry_identity(
    directory_fd: int,
    name: str,
    expected: os.stat_result,
    *,
    label: str,
) -> None:
    try:
        observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        msg = f"{label} file entry cannot be revalidated."
        raise SourceDataBuildError(msg) from error
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_dev != expected.st_dev
        or observed.st_ino != expected.st_ino
        or observed.st_nlink != 1
    ):
        msg = f"{label} file entry identity changed."
        raise SourceDataBuildError(msg)


def _base_gene(feature: str) -> str:
    return feature[:-2]


def _sequence_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _add_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--postprocess-root", type=Path, required=True)
    parser.add_argument("--release-approval-manifest", type=Path, required=True)
    parser.add_argument("--expected-postprocess-release-sha256", required=True)
    parser.add_argument("--expected-postprocess-authority-sha256", required=True)
    parser.add_argument("--expected-postprocess-implementation-sha256", required=True)
    parser.add_argument("--expected-sealed-completion-sha256", required=True)
    parser.add_argument("--expected-canonical-input-sha256", required=True)
    parser.add_argument("--expected-provider-input-sha256", required=True)
    parser.add_argument("--expected-release-approval-sha256", required=True)
    parser.add_argument("--expected-marginal-validity-evidence-sha256")
    parser.add_argument("--output-root", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser(
        "build",
        help="Build and atomically publish a source-data release.",
    )
    _add_build_arguments(build_parser)
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a frozen source-data release as opaque bytes.",
    )
    validate_parser.add_argument("--source-data-root", type=Path, required=True)
    validate_parser.add_argument("--expected-manifest-sha256", required=True)
    return parser


def _parse_cli_arguments(argv: Sequence[str]) -> argparse.Namespace:
    arguments = list(argv)
    if not arguments or arguments[0] not in {"build", "validate"}:
        arguments.insert(0, "build")
    return _parser().parse_args(arguments)


def main() -> None:
    """Run the closed source-data build or opaque-byte validation entrypoint."""
    args = _parse_cli_arguments(tuple(sys.argv[1:]))

    def absolute(path: Path) -> Path:
        return Path(os.path.abspath(path))  # noqa: PTH100

    if args.command == "validate":
        validation_receipt = validate_source_data_release(
            args.source_data_root,
            args.expected_manifest_sha256,
        )
        print(json.dumps(asdict(validation_receipt), sort_keys=True))
        return

    config = SourceDataBuildConfig(
        postprocess_root=absolute(args.postprocess_root),
        release_approval_manifest=absolute(args.release_approval_manifest),
        expected_postprocess_release_sha256=args.expected_postprocess_release_sha256,
        expected_postprocess_authority_sha256=(
            args.expected_postprocess_authority_sha256
        ),
        expected_postprocess_implementation_sha256=(
            args.expected_postprocess_implementation_sha256
        ),
        expected_sealed_completion_sha256=args.expected_sealed_completion_sha256,
        expected_canonical_input_sha256=args.expected_canonical_input_sha256,
        expected_provider_input_sha256=args.expected_provider_input_sha256,
        expected_release_approval_sha256=args.expected_release_approval_sha256,
        expected_marginal_validity_evidence_sha256=(
            args.expected_marginal_validity_evidence_sha256
        ),
    )
    receipt = build_source_data_release(config, absolute(args.output_root))
    print(json.dumps(asdict(receipt), sort_keys=True))


if __name__ == "__main__":
    main()
