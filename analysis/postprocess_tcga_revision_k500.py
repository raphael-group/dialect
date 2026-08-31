"""Derive the frozen D5 conjunction family from sealed K=500 pair tables.

This module never filters, ranks, or selects pairs.  Production callers provide only
independently pinned immutable roots and approval artifacts: the feature axes,
pairwise bytes, D3--D6 policies, and validity status are derived from those validated
authorities rather than accepted as caller data.  A separate inspect-stage approval
is mandatory before any result row is parsed.

The conjunction is explicitly D3-secondary.  Its conditional BY layer is reportable
only when a separate calibration-stage approval and marginal-validity artifact are
certified.  Calibration status never chooses a correction and never changes either
the BY or BH values.
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
import stat as stat_module
import sys
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from analysis import run_tcga_revision_k500 as runner
from analysis.run_tcga_revision_k500 import (
    BMRS,
    PAIRWISE_COLUMNS,
    REQUIRED_LRT_CONTRACT,
    REQUIRED_LRT_NESTEDNESS_TOL,
    REQUIRED_NONIDENTIFIED_EFFECT_BLANK_FIELDS,
    REQUIRED_OUTPUT_RECOMPUTATION_ATOL,
    REQUIRED_PAIR_EFFECT_IDENTIFIED_STATUS,
    REQUIRED_PAIR_EFFECT_RANK_DEFICIENT_STATUS,
    REQUIRED_PAIR_EFFECT_UNDERFLOW_STATUS,
    REQUIRED_PAIR_FIT_CONTRACT,
    REQUIRED_PAIR_FIT_KKT_TOL,
    REQUIRED_PAIR_FIT_MAX_ITER,
    REQUIRED_PAIR_SIMPLEX_TOL,
    REQUIRED_UNDEFINED_RHO_LRT_TOL,
    SEALED_COMPLETION_CONTRACT,
    SEALED_COMPLETION_NAME,
    SEALED_COMPLETION_SCHEMA,
    TCGA_COHORTS,
    TOP_K,
    RunPaths,
    Task,
    iter_tested_pairs,
)
from dialect.data import revision_fit_policy as revision_fit_policy_module
from dialect.data.revision_approval import (
    CALIBRATION_STAGE,
    FIT_SEALED_TCGA_K500_STAGE,
    INSPECT_TCGA_K500_STAGE,
    MATERIALIZE_FINAL_INPUTS_STAGE,
    STAGE_MINIMUM_DECISIONS_V6,
    STAGE_SCOPED_APPROVAL_SCHEMA_V6,
    RevisionApproval,
    validate_revision_approval,
)
from dialect.data.revision_fit_policy import (
    COMPONENT_FAILURE_SEMANTICS,
    CONJUNCTION_P_VALUE_COMBINER,
    CONJUNCTION_SECONDARY,
    D5_CONTRACT,
    DESCRIPTIVE_REPORTING_LAYER,
    DIRECTION_CONSENSUS_RULE,
    DIRECTION_PROVIDER_RULE,
    DIRECTION_REPORTING_LAYER,
    INCLUSIVE_THRESHOLD,
    PRIMARY_REPORTING_LAYER,
    SENSITIVITY_REPORTING_LAYER,
    UNDEFINED_RHO_DIRECTION_RULE,
    VALID_CONJUNCTION_COMPONENT_STATUSES,
    ConjunctionMultiplicityPolicy,
    ProviderHierarchyPolicy,
    RevisionFitPolicy,
    validate_revision_fit_policy,
)
from dialect.models.interaction import compute_marshall_olkin_rho
from dialect.stats import revision_inference as revision_inference_module
from dialect.stats.revision_inference import (
    DESCRIPTIVE_Q_THRESHOLD,
    PRIMARY_Q_THRESHOLD,
    SENSITIVITY_Q_THRESHOLD,
    ComponentInference,
    RevisionInferenceError,
    classify_component,
    derive_complete_family,
    validate_policy,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

POSTPROCESS_SCHEMA: Final = "dialect-tcga-k500-d5-postprocess-v1"
DERIVATION_CONTRACT: Final = "complete-family-max-p-by-bh-derivation-v1"
OUTPUT_CSV_NAME: Final = "conjunction_interaction_results.csv"
OUTPUT_MANIFEST_NAME: Final = "postprocess_manifest.json"
RELEASE_MANIFEST_NAME: Final = "postprocess_release_manifest.json"
AUTHORITY_RECEIPT_NAME: Final = "postprocess_authority.json"
RELEASE_SCHEMA: Final = "dialect-tcga-k500-d5-postprocess-release-v1"
RELEASE_CONTRACT: Final = "sealed-whole-grid-no-replace-publication-v1"

MARGINAL_VALIDITY_EVIDENCE_SCHEMA: Final = (
    "dialect-tcga-k500-marginal-validity-evidence-v2"
)
MARGINAL_VALIDITY_EVIDENCE_CONTRACT: Final = (
    "signed-d1-d6-complete-family-validity-gate-v2"
)
POSTPROCESS_AUTHORITY_SCHEMA: Final = "dialect-tcga-k500-postprocess-authority-v2"
POSTPROCESS_AUTHORITY_CONTRACT: Final = (
    "pinned-roots-sealed-grid-signed-d1-d6-evidence-v2"
)
CALIBRATION_DECISION_IDS: Final = STAGE_MINIMUM_DECISIONS_V6[CALIBRATION_STAGE]

MARGINAL_VALIDITY_CERTIFIED: Final = "certified"
MARGINAL_VALIDITY_BLOCKING: Final = frozenset(
    {"absent", "invalid", "inconclusive"},
)
MARGINAL_VALIDITY_STATUSES: Final = frozenset(
    {MARGINAL_VALIDITY_CERTIFIED, *MARGINAL_VALIDITY_BLOCKING},
)

_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_COHORT_PATTERN: Final = re.compile(r"[A-Z0-9]+")
_FEATURE_PATTERN: Final = re.compile(r".+_[MN]")
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

_AUTHORITY_SEAL: Final = object()
_EVIDENCE_SEAL: Final = object()
_DERIVATION_SEAL: Final = object()


class D5DerivationError(ValueError):
    """Raised when a D5 family cannot be published without substitution."""


@dataclass(frozen=True, slots=True)
class FrozenFamilyAxis:
    """Digest-bound cohort feature axis used to prove family completeness."""

    cohort: str
    ordered_features: tuple[str, ...]
    ordered_features_sha256: str
    ordered_pair_sha256: str
    cohort_contract_sha256: str


@dataclass(frozen=True, slots=True)
class MarginalValidityEvidence:
    """Finite-sample validity state; certification requires authenticated bytes.

    Callers may construct blocking states for synthetic QA.  The ``certified``
    state is minted only by :func:`validate_postprocess_authority` after the
    independently pinned evidence artifact and calibration-stage approval have
    both passed their closed-schema and upstream-binding checks.
    """

    status: str
    evidence_id: str | None = None
    artifact_sha256: str | None = None
    artifact_bytes: bytes | None = field(default=None, repr=False, compare=False)
    _seal: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Reject caller-forged certification at object construction."""
        if self.status == MARGINAL_VALIDITY_CERTIFIED and self._seal is not (
            _EVIDENCE_SEAL
        ):
            msg = (
                "Certified marginal-validity evidence must come from the "
                "authenticated calibration-evidence validator."
            )
            raise D5DerivationError(msg)

    @property
    def inferentially_eligible(self) -> bool:
        """Return whether the conditional primary layer may be reported."""
        return self.status == MARGINAL_VALIDITY_CERTIFIED


@dataclass(frozen=True, slots=True)
class DerivedCohortFamily:
    """Deterministic bytes for one family; never itself authorizes publication."""

    csv_bytes: bytes
    manifest_bytes: bytes
    row_count: int
    csv_sha256: str
    manifest_sha256: str
    production_eligible: bool


@dataclass(frozen=True, slots=True)
class ProductionPostprocessConfig:
    """Independently pinned immutable roots required for production derivation."""

    run_output_root: Path
    canonical_input_root: Path
    provider_input_root: Path
    input_approval_manifest: Path
    fit_approval_manifest: Path
    inspect_approval_manifest: Path
    calibration_approval_manifest: Path | None
    calibration_evidence_artifact: Path | None
    calibration_design_manifest: Path | None
    marginal_validity_status: str
    expected_sealed_completion_sha256: str
    expected_canonical_input_sha256: str
    expected_provider_input_manifest_sha256: str
    expected_input_approval_sha256: str
    expected_fit_approval_sha256: str
    expected_inspect_approval_sha256: str
    expected_calibration_approval_sha256: str | None
    expected_calibration_evidence_sha256: str | None


@dataclass(frozen=True, slots=True, init=False)
class ValidatedPostprocessAuthority:
    """Opaque receipt minted after the full production trust chain validates."""

    config: ProductionPostprocessConfig
    fit_policy: RevisionFitPolicy
    marginal_validity: MarginalValidityEvidence
    completion_bytes: bytes = field(repr=False)
    run_manifest_bytes: bytes = field(repr=False)
    contract_bytes: tuple[bytes, ...] = field(repr=False)
    authority_record_bytes: bytes = field(repr=False)
    authority_sha256: str
    _seal: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True, init=False)
class SealedDerivedCohortFamily:
    """Opaque publication receipt minted from validated production authority."""

    cohort: str
    derived: DerivedCohortFamily = field(repr=False)
    authority: ValidatedPostprocessAuthority = field(repr=False, compare=False)
    publication_binding_sha256: str
    _seal: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _ComponentRow:
    coordinate: tuple[str, str]
    status: str
    p_value: float
    direction: str
    effect_identifiability: str
    contingency: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class _PinnedPublicationFile:
    """Open descriptor and identity for one immutable staged release file."""

    name: str
    descriptor: int = field(repr=False, compare=False)
    identity: os.stat_result = field(repr=False)
    expected_content: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class _PinnedCohortDirectory:
    """Pinned child directory and exact two-file publication inventory."""

    cohort: str
    descriptor: int = field(repr=False, compare=False)
    identity: os.stat_result = field(repr=False)
    csv_file: _PinnedPublicationFile = field(repr=False)
    manifest_file: _PinnedPublicationFile = field(repr=False)


def sequence_sha256(values: Sequence[str]) -> str:
    """Hash an ordered string sequence using the K=500 runner encoding."""
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def ordered_pair_sha256(pairs: Sequence[tuple[str, str]]) -> str:
    """Hash an ordered pair sequence using the K=500 runner encoding."""
    digest = hashlib.sha256()
    for gene_a, gene_b in pairs:
        encoded = f"{gene_a}\t{gene_b}".encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def build_frozen_family_axis(
    cohort: str,
    ordered_features: Sequence[str],
    *,
    cohort_contract_sha256: str,
) -> FrozenFamilyAxis:
    """Build the exact pair-axis binding from a validated runner feature axis."""
    features = tuple(ordered_features)
    pairs = tuple(iter_tested_pairs(features))
    axis = FrozenFamilyAxis(
        cohort=cohort,
        ordered_features=features,
        ordered_features_sha256=sequence_sha256(features),
        ordered_pair_sha256=ordered_pair_sha256(pairs),
        cohort_contract_sha256=cohort_contract_sha256,
    )
    _validate_axis(axis)
    return axis


def validate_postprocess_authority(
    config: ProductionPostprocessConfig,
) -> ValidatedPostprocessAuthority:
    """Validate and seal every upstream authority before scientific row access.

    The independently supplied SHA-256 arguments are trust anchors.  The function
    first validates the signed fit and calibration approvals, canonical/provider
    roots, run manifest, sealed completion, all cohort contracts, and all 96 task
    metadata/raw-file receipts.  Pairwise rows are not parsed here.
    """
    _validate_production_config(config)
    paths = _runner_paths(config)
    fit_approval = runner._validated_fit_approval(  # noqa: SLF001
        config.fit_approval_manifest,
        config.expected_fit_approval_sha256,
    )
    runner._require_fit_stage_binding(fit_approval, paths)  # noqa: SLF001
    fit_policy = validate_revision_fit_policy(
        fit_approval,
        expected_d4_implementation=runner.REQUIRED_D4_IMPLEMENTATION,
        expected_tested_family=runner.REQUIRED_TESTED_FAMILY,
    )
    _validate_signed_hierarchy(fit_policy.d3)
    _validate_policy(fit_policy.d5)

    # Reuse the runner's full, result-blind root replays.  This validates the
    # canonical/provider inventories and exact fit-policy receipts before the
    # postprocessor opens a result-bearing CSV.
    runner._prime_parent_revision_authority(paths, TCGA_COHORTS[0])  # noqa: SLF001
    run_manifest_path = config.run_output_root / "run_manifest.json"
    run_manifest_bytes = runner._read_secure_regular_bytes(  # noqa: SLF001
        run_manifest_path,
        label="postprocess run manifest",
    )
    run_manifest = runner._parse_json_bytes(  # noqa: SLF001
        run_manifest_bytes,
        path=run_manifest_path,
    )
    run_authority = runner._require_completion_run_manifest(  # noqa: SLF001
        paths,
        run_manifest,
    )

    completion_path = config.run_output_root / SEALED_COMPLETION_NAME
    completion_bytes = runner._read_secure_regular_bytes(  # noqa: SLF001
        completion_path,
        label="sealed K=500 completion",
    )
    observed_completion_sha256 = hashlib.sha256(completion_bytes).hexdigest()
    if observed_completion_sha256 != config.expected_sealed_completion_sha256:
        msg = "Sealed completion does not match its independent SHA-256 anchor."
        raise D5DerivationError(msg)
    completion = runner._parse_json_bytes(  # noqa: SLF001
        completion_bytes,
        path=completion_path,
    )
    inspect_approval = _validate_inspect_approval(
        config,
        fit_approval=fit_approval,
    )
    contract_bytes = _validate_sealed_grid(
        paths,
        completion=completion,
        run_manifest_bytes=run_manifest_bytes,
        run_authority=run_authority,
    )
    calibration_approval: RevisionApproval | None = None
    if config.marginal_validity_status == MARGINAL_VALIDITY_CERTIFIED:
        calibration_approval = _validate_calibration_approval(
            config,
            fit_approval=fit_approval,
        )
        marginal_validity = _validate_calibration_evidence(
            config,
            fit_policy=fit_policy,
            fit_approval=fit_approval,
            calibration_approval=calibration_approval,
        )
    else:
        marginal_validity = MarginalValidityEvidence(
            config.marginal_validity_status,
        )
    authority_record = _build_authority_record(
        config,
        completion=completion,
        fit_policy=fit_policy,
        fit_approval=fit_approval,
        inspect_approval=inspect_approval,
        calibration_approval=calibration_approval,
        marginal_validity=marginal_validity,
        contract_bytes=contract_bytes,
    )
    authority_record_bytes = _canonical_json(authority_record) + b"\n"
    authority = object.__new__(ValidatedPostprocessAuthority)
    values = {
        "config": config,
        "fit_policy": fit_policy,
        "marginal_validity": marginal_validity,
        "completion_bytes": completion_bytes,
        "run_manifest_bytes": run_manifest_bytes,
        "contract_bytes": contract_bytes,
        "authority_record_bytes": authority_record_bytes,
        "authority_sha256": hashlib.sha256(authority_record_bytes).hexdigest(),
        "_seal": _AUTHORITY_SEAL,
    }
    for name, value in values.items():
        object.__setattr__(authority, name, value)
    _require_validated_authority(authority)
    return authority


def _validate_production_config(config: ProductionPostprocessConfig) -> None:
    if not isinstance(config, ProductionPostprocessConfig):
        msg = "Production authority requires ProductionPostprocessConfig."
        raise TypeError(msg)
    required_paths = (
        config.run_output_root,
        config.canonical_input_root,
        config.provider_input_root,
        config.input_approval_manifest,
        config.fit_approval_manifest,
        config.inspect_approval_manifest,
    )
    if any(
        not isinstance(path, Path) or not path.is_absolute() for path in required_paths
    ):
        msg = "Every production authority path must be an absolute pathlib.Path."
        raise D5DerivationError(msg)
    optional_paths = (
        config.calibration_approval_manifest,
        config.calibration_evidence_artifact,
        config.calibration_design_manifest,
    )
    if any(
        path is not None and (not isinstance(path, Path) or not path.is_absolute())
        for path in optional_paths
    ):
        msg = "Optional calibration paths must be absolute when supplied."
        raise D5DerivationError(msg)
    if (
        not isinstance(config.marginal_validity_status, str)
        or config.marginal_validity_status not in MARGINAL_VALIDITY_STATUSES
    ):
        msg = "Production marginal-validity status is unsupported."
        raise D5DerivationError(msg)
    calibration_fields = (
        config.calibration_approval_manifest,
        config.calibration_evidence_artifact,
        config.expected_calibration_approval_sha256,
        config.expected_calibration_evidence_sha256,
    )
    if config.marginal_validity_status == MARGINAL_VALIDITY_CERTIFIED:
        if any(value is None for value in calibration_fields):
            msg = "Certified validity requires calibration approval and evidence."
            raise D5DerivationError(msg)
    elif any(
        value is not None
        for value in (*calibration_fields, config.calibration_design_manifest)
    ):
        msg = "Blocking validity states must not smuggle calibration certification."
        raise D5DerivationError(msg)
    if config.calibration_design_manifest is not None and (
        not isinstance(config.calibration_design_manifest, Path)
        or not config.calibration_design_manifest.is_absolute()
    ):
        msg = "The calibration design path must be absolute when supplied."
        raise D5DerivationError(msg)
    for label, digest in (
        ("sealed completion", config.expected_sealed_completion_sha256),
        ("canonical input", config.expected_canonical_input_sha256),
        ("provider input", config.expected_provider_input_manifest_sha256),
        ("input approval", config.expected_input_approval_sha256),
        ("fit approval", config.expected_fit_approval_sha256),
        ("inspect approval", config.expected_inspect_approval_sha256),
    ):
        _require_sha256(digest, label=label)
    if config.marginal_validity_status == MARGINAL_VALIDITY_CERTIFIED:
        _require_sha256(
            config.expected_calibration_approval_sha256,
            label="calibration approval",
        )
        _require_sha256(
            config.expected_calibration_evidence_sha256,
            label="calibration evidence",
        )


def _runner_paths(config: ProductionPostprocessConfig) -> RunPaths:
    return RunPaths(
        source_root=config.provider_input_root / "cohorts",
        mutsig_root=config.provider_input_root / "mutsig",
        output_root=config.run_output_root,
        canonical_input_root=config.canonical_input_root,
        input_approval_manifest=config.input_approval_manifest,
        expected_input_approval_sha256=config.expected_input_approval_sha256,
        fit_approval_manifest=config.fit_approval_manifest,
        expected_fit_approval_sha256=config.expected_fit_approval_sha256,
        expected_canonical_input_sha256=config.expected_canonical_input_sha256,
        provider_input_root=config.provider_input_root,
        expected_provider_input_manifest_sha256=(
            config.expected_provider_input_manifest_sha256
        ),
    )


def _validate_signed_hierarchy(policy: ProviderHierarchyPolicy) -> None:
    if (
        policy.primary_provider != "cbase"
        or policy.sensitivity_providers != ("dig", "mutsig")
        or policy.all_three_conjunction_role != CONJUNCTION_SECONDARY
        or policy.burden_dependent_switching is not False
    ):
        msg = "Signed D3 provider hierarchy does not authorize this conjunction."
        raise D5DerivationError(msg)


def _validate_sealed_grid(
    paths: RunPaths,
    *,
    completion: Mapping[str, object],
    run_manifest_bytes: bytes,
    run_authority: Mapping[str, object],
) -> tuple[bytes, ...]:
    expected_keys = {
        "analysis",
        "authority",
        "bmrs",
        "cohorts",
        "contract",
        "contracts",
        "downstream_binding",
        "grid",
        "result_rows_opened",
        "run_manifest",
        "schema",
        "tasks",
        "top_k",
    }
    coordinates = [
        f"{cohort}/{provider}" for cohort in TCGA_COHORTS for provider in BMRS
    ]
    grid = completion.get("grid")
    run_receipt = completion.get("run_manifest")
    if (
        set(completion) != expected_keys
        or completion.get("schema") != SEALED_COMPLETION_SCHEMA
        or completion.get("contract") != SEALED_COMPLETION_CONTRACT
        or completion.get("analysis") != "tcga-revision-k500"
        or completion.get("top_k") != TOP_K
        or completion.get("cohorts") != list(TCGA_COHORTS)
        or completion.get("bmrs") != list(BMRS)
        or completion.get("authority") != run_authority
        or completion.get("result_rows_opened") is not False
        or completion.get("downstream_binding")
        != {
            "field": "upstream_result_manifest_sha256",
            "stage": "inspect-tcga-k500",
        }
        or not isinstance(grid, Mapping)
        or dict(grid)
        != {
            "ordered_coordinates_sha256": sequence_sha256(coordinates),
            "task_count": len(coordinates),
        }
        or not isinstance(run_receipt, Mapping)
        or dict(run_receipt)
        != {
            "bytes": len(run_manifest_bytes),
            "sha256": hashlib.sha256(run_manifest_bytes).hexdigest(),
        }
    ):
        msg = "Sealed completion has an invalid or substituted whole-grid authority."
        raise D5DerivationError(msg)
    contracts = completion.get("contracts")
    tasks = completion.get("tasks")
    if (
        not isinstance(contracts, list)
        or len(contracts) != len(TCGA_COHORTS)
        or not isinstance(tasks, list)
        or len(tasks) != len(coordinates)
    ):
        msg = "Sealed completion does not contain the exact 32 by 3 grid."
        raise D5DerivationError(msg)

    runner._require_closed_completion_layout(paths)  # noqa: SLF001
    contract_snapshots: list[bytes] = []
    task_index = 0
    for cohort_index, cohort in enumerate(TCGA_COHORTS):
        contract_path = paths.output_root / "contracts" / f"{cohort}.json"
        raw = runner._read_secure_regular_bytes(  # noqa: SLF001
            contract_path,
            label=f"postprocess contract {cohort}",
        )
        contract = runner._parse_json_bytes(raw, path=contract_path)  # noqa: SLF001
        verified = runner._load_verified_contract(  # noqa: SLF001
            paths,
            cohort,
            top_k=TOP_K,
        )
        receipt = contracts[cohort_index]
        contract_sha256 = runner._json_sha256(contract)  # noqa: SLF001
        if (
            verified != contract
            or not isinstance(receipt, Mapping)
            or dict(receipt)
            != {
                "bytes": len(raw),
                "cohort": cohort,
                "contract_sha256": contract_sha256,
                "file_sha256": hashlib.sha256(raw).hexdigest(),
            }
        ):
            msg = f"Sealed completion contract binding changed for {cohort}."
            raise D5DerivationError(msg)
        contract_snapshots.append(raw)
        for provider in BMRS:
            fresh = runner._metadata_task_receipt(  # noqa: SLF001
                paths.output_root / "tasks" / cohort / provider,
                contract,
                Task(cohort, provider),
            )
            if (
                fresh.pop("implementation_sha256")
                != runner._parse_json_bytes(  # noqa: SLF001
                    run_manifest_bytes,
                    path=paths.output_root / "run_manifest.json",
                )["implementation_sha256"]
                or fresh.pop("provider_input_root_receipt")
                != run_authority["provider_input"]
                or fresh != tasks[task_index]
            ):
                msg = f"Sealed task/raw-output binding changed for {cohort}/{provider}."
                raise D5DerivationError(msg)
            task_index += 1
    return tuple(contract_snapshots)


def _validate_calibration_approval(
    config: ProductionPostprocessConfig,
    *,
    fit_approval: RevisionApproval,
) -> RevisionApproval:
    if (
        config.calibration_approval_manifest is None
        or config.expected_calibration_approval_sha256 is None
    ):
        msg = "Certified validity lacks its calibration approval authority."
        raise D5DerivationError(msg)
    approval = validate_revision_approval(
        config.calibration_approval_manifest,
        config.expected_calibration_approval_sha256,
        CALIBRATION_STAGE,
    )
    expected_binding = {
        "canonical_input_manifest_sha256": config.expected_canonical_input_sha256,
        "provider_input_manifest_sha256": (
            config.expected_provider_input_manifest_sha256
        ),
        "upstream_result_manifest_sha256": (config.expected_sealed_completion_sha256),
    }
    if (
        approval.schema != STAGE_SCOPED_APPROVAL_SCHEMA_V6
        or approval.allowed_stages != (CALIBRATION_STAGE,)
        or set(approval.stage_bindings) != {CALIBRATION_STAGE}
        or dict(approval.stage_bindings[CALIBRATION_STAGE]) != expected_binding
        or tuple(approval.decisions) != CALIBRATION_DECISION_IDS
        or tuple(approval.decision_digests) != CALIBRATION_DECISION_IDS
    ):
        msg = (
            "Calibration approval must be an exact singleton stage-scoped v6 "
            "D1-D6 authority bound to the sealed K=500 run."
        )
        raise D5DerivationError(msg)
    for decision_id in CALIBRATION_DECISION_IDS:
        if runner._decision_reauthorization_record(  # noqa: SLF001
            fit_approval.decisions[decision_id],
        ) != runner._decision_reauthorization_record(  # noqa: SLF001
            approval.decisions[decision_id],
        ):
            msg = f"Calibration approval does not reauthorize signed {decision_id}."
            raise D5DerivationError(msg)
    return approval


def _validate_inspect_approval(
    config: ProductionPostprocessConfig,
    *,
    fit_approval: RevisionApproval,
) -> RevisionApproval:
    """Require an inspect-only authority before any pairwise row can be parsed."""
    approval = validate_revision_approval(
        config.inspect_approval_manifest,
        config.expected_inspect_approval_sha256,
        INSPECT_TCGA_K500_STAGE,
    )
    expected_binding = {
        "canonical_input_manifest_sha256": config.expected_canonical_input_sha256,
        "provider_input_manifest_sha256": (
            config.expected_provider_input_manifest_sha256
        ),
        "upstream_result_manifest_sha256": (config.expected_sealed_completion_sha256),
    }
    if (
        approval.allowed_stages != (INSPECT_TCGA_K500_STAGE,)
        or set(approval.stage_bindings) != {INSPECT_TCGA_K500_STAGE}
        or dict(approval.stage_bindings[INSPECT_TCGA_K500_STAGE]) != expected_binding
    ):
        msg = "Inspection approval does not exclusively bind the sealed K=500 run."
        raise D5DerivationError(msg)
    for decision_id in ("D1", "D2", "D3", "D4", "D5", "D6"):
        if runner._decision_reauthorization_record(  # noqa: SLF001
            fit_approval.decisions[decision_id],
        ) != runner._decision_reauthorization_record(  # noqa: SLF001
            approval.decisions[decision_id],
        ):
            msg = f"Inspection approval does not reauthorize signed {decision_id}."
            raise D5DerivationError(msg)
    return approval


def _validate_calibration_evidence(
    config: ProductionPostprocessConfig,
    *,
    fit_policy: RevisionFitPolicy,
    fit_approval: RevisionApproval,
    calibration_approval: RevisionApproval,
) -> MarginalValidityEvidence:
    if (
        config.calibration_evidence_artifact is None
        or config.expected_calibration_evidence_sha256 is None
    ):
        msg = "Certified validity lacks its pinned calibration-evidence artifact."
        raise D5DerivationError(msg)
    path = config.calibration_evidence_artifact
    raw = runner._read_secure_regular_bytes(  # noqa: SLF001
        path,
        label="marginal-validity evidence",
    )
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_sha256 != config.expected_calibration_evidence_sha256:
        msg = "Marginal-validity evidence does not match its independent digest."
        raise D5DerivationError(msg)
    artifact = runner._parse_json_bytes(raw, path=path)  # noqa: SLF001
    d6 = fit_policy.d6
    design = None if d6.design_authority is None else asdict(d6.design_authority)
    compute = None if d6.compute_authority is None else asdict(d6.compute_authority)
    if d6.design_authority is None:
        if config.calibration_design_manifest is not None:
            msg = "D6 no-extension cannot accept a calibration design artifact."
            raise D5DerivationError(msg)
    else:
        if config.calibration_design_manifest is None:
            msg = "Signed D6 calibration requires its exact design manifest."
            raise D5DerivationError(msg)
        design_raw = runner._read_secure_regular_bytes(  # noqa: SLF001
            config.calibration_design_manifest,
            label="signed calibration design",
        )
        if (
            hashlib.sha256(design_raw).hexdigest()
            != d6.design_authority.manifest_sha256
        ):
            msg = "Calibration design does not match the signed D6 digest."
            raise D5DerivationError(msg)
    expected_keys = {
        "analysis",
        "bmrs",
        "calibration_authority",
        "cohorts",
        "contract",
        "correction_policy",
        "marginal_validity",
        "schema",
        "tested_family",
        "top_k",
        "upstream",
    }
    calibration_authority = artifact.get("calibration_authority")
    correction_policy = artifact.get("correction_policy")
    marginal = artifact.get("marginal_validity")
    upstream = artifact.get("upstream")
    expected_calibration_authority = {
        "approval_manifest_sha256": calibration_approval.manifest_sha256,
        "approval_schema": calibration_approval.schema,
        "authorized_stage": CALIBRATION_STAGE,
        "calibration_decision_digests": {
            decision_id: calibration_approval.decision_digests[decision_id]
            for decision_id in CALIBRATION_DECISION_IDS
        },
        "compute_authority": compute,
        "design_authority": design,
        "fit_decision_digests": {
            decision_id: fit_approval.decision_digests[decision_id]
            for decision_id in CALIBRATION_DECISION_IDS
        },
        "fit_d6_artifact_sha256": fit_policy.receipts["D6"].canonical_artifact_sha256,
        "fit_d6_decision_digest": fit_policy.receipts["D6"].decision_digest,
        "fit_d6_payload_sha256": fit_policy.receipts["D6"].payload_sha256,
        "path": d6.path,
        "claim_scope": d6.claim_scope,
    }
    expected_upstream = {
        "canonical_input_manifest_sha256": config.expected_canonical_input_sha256,
        "fit_approval_manifest_sha256": fit_approval.manifest_sha256,
        "provider_input_manifest_sha256": (
            config.expected_provider_input_manifest_sha256
        ),
        "sealed_completion_manifest_sha256": (config.expected_sealed_completion_sha256),
    }
    if (
        set(artifact) != expected_keys
        or artifact.get("schema") != MARGINAL_VALIDITY_EVIDENCE_SCHEMA
        or artifact.get("contract") != MARGINAL_VALIDITY_EVIDENCE_CONTRACT
        or artifact.get("analysis") != "tcga-revision-k500"
        or artifact.get("top_k") != TOP_K
        or artifact.get("cohorts") != list(TCGA_COHORTS)
        or artifact.get("bmrs") != list(BMRS)
        or artifact.get("tested_family") != asdict(fit_policy.d5.tested_family)
        or calibration_authority != expected_calibration_authority
        or upstream != expected_upstream
        or correction_policy
        != {
            "computed_methods": ["by", "bh"],
            "correction_selection_affected": False,
            "q_values_affected": False,
        }
        or not isinstance(marginal, Mapping)
        or set(marginal)
        != {
            "evidence_id",
            "finite_sample_super_uniformity_certified",
            "scope",
            "status",
        }
    ):
        msg = "Marginal-validity evidence has an invalid authenticated schema."
        raise D5DerivationError(msg)
    status = marginal["status"]
    evidence_id = marginal["evidence_id"]
    if (
        not isinstance(status, str)
        or status not in MARGINAL_VALIDITY_STATUSES
        or not isinstance(evidence_id, str)
        or not evidence_id
        or evidence_id != evidence_id.strip()
        or any(ord(character) < 0x20 for character in evidence_id)
        or marginal["scope"] != "complete-within-cohort-conjunction-family"
        or marginal["finite_sample_super_uniformity_certified"]
        is not (status == MARGINAL_VALIDITY_CERTIFIED)
        or (status == MARGINAL_VALIDITY_CERTIFIED and d6.design_authority is None)
    ):
        msg = "Marginal-validity evidence status is not authorized by signed D6."
        raise D5DerivationError(msg)
    if status == MARGINAL_VALIDITY_CERTIFIED:
        if d6.path != "full-external":
            msg = (
                "Narrow/local calibration cannot certify finite-sample marginal "
                "validity."
            )
            raise D5DerivationError(msg)
        msg = (
            "Certified marginal validity requires a production validator that "
            "reconstructs the gate from authenticated calibration outputs; a "
            "pinned self-asserted evidence document is insufficient."
        )
        raise D5DerivationError(msg)
    evidence = MarginalValidityEvidence(
        status=status,
        evidence_id=evidence_id,
        artifact_sha256=observed_sha256,
        artifact_bytes=raw,
        _seal=_EVIDENCE_SEAL,
    )
    _validate_marginal_validity(evidence)
    return evidence


def _build_authority_record(  # noqa: PLR0913
    config: ProductionPostprocessConfig,
    *,
    completion: Mapping[str, object],
    fit_policy: RevisionFitPolicy,
    fit_approval: RevisionApproval,
    inspect_approval: RevisionApproval,
    calibration_approval: RevisionApproval | None,
    marginal_validity: MarginalValidityEvidence,
    contract_bytes: tuple[bytes, ...],
) -> dict[str, object]:
    return {
        "schema": POSTPROCESS_AUTHORITY_SCHEMA,
        "contract": POSTPROCESS_AUTHORITY_CONTRACT,
        "analysis": "tcga-revision-k500",
        "top_k": TOP_K,
        "cohorts": list(TCGA_COHORTS),
        "bmrs": list(BMRS),
        "roots": {
            "run_output_root": config.run_output_root.as_posix(),
            "canonical_input_root": config.canonical_input_root.as_posix(),
            "canonical_input_manifest_sha256": (config.expected_canonical_input_sha256),
            "provider_input_root": config.provider_input_root.as_posix(),
            "provider_input_manifest_sha256": (
                config.expected_provider_input_manifest_sha256
            ),
        },
        "approvals": {
            "input": {
                "path": config.input_approval_manifest.as_posix(),
                "sha256": config.expected_input_approval_sha256,
            },
            "fit": {
                "path": config.fit_approval_manifest.as_posix(),
                "sha256": fit_approval.manifest_sha256,
            },
            "inspect": {
                "path": config.inspect_approval_manifest.as_posix(),
                "sha256": inspect_approval.manifest_sha256,
                "authorized_stage": INSPECT_TCGA_K500_STAGE,
            },
            "calibration": (
                None
                if calibration_approval is None
                else {
                    "path": config.calibration_approval_manifest.as_posix(),
                    "sha256": calibration_approval.manifest_sha256,
                    "schema": calibration_approval.schema,
                    "authorized_stage": CALIBRATION_STAGE,
                    "decision_digests": dict(calibration_approval.decision_digests),
                }
            ),
        },
        "sealed_completion": {
            "name": SEALED_COMPLETION_NAME,
            "sha256": config.expected_sealed_completion_sha256,
            "task_count": completion["grid"]["task_count"],
        },
        "fit_policy": runner._fit_policy_record(fit_policy),  # noqa: SLF001
        "d3_conjunction_role": fit_policy.d3.all_three_conjunction_role,
        "contracts": [
            {
                "cohort": cohort,
                "file_sha256": hashlib.sha256(raw).hexdigest(),
            }
            for cohort, raw in zip(TCGA_COHORTS, contract_bytes, strict=True)
        ],
        "marginal_validity_evidence": {
            "path": (
                None
                if config.calibration_evidence_artifact is None
                else config.calibration_evidence_artifact.as_posix()
            ),
            "sha256": marginal_validity.artifact_sha256,
            "evidence_id": marginal_validity.evidence_id,
            "status": marginal_validity.status,
        },
    }


def _require_validated_authority(authority: ValidatedPostprocessAuthority) -> None:
    """Revalidate the opaque receipt and its live descriptor-bound anchors."""
    if not isinstance(authority, ValidatedPostprocessAuthority):
        msg = "Production derivation requires a validated authority receipt."
        raise TypeError(msg)
    if getattr(authority, "_seal", None) is not _AUTHORITY_SEAL:
        msg = "Production authority receipt was not minted by its validator."
        raise D5DerivationError(msg)
    _validate_production_config(authority.config)
    if (
        not isinstance(authority.completion_bytes, bytes)
        or not isinstance(authority.run_manifest_bytes, bytes)
        or not isinstance(authority.contract_bytes, tuple)
        or len(authority.contract_bytes) != len(TCGA_COHORTS)
        or not all(isinstance(raw, bytes) for raw in authority.contract_bytes)
        or not isinstance(authority.authority_record_bytes, bytes)
        or hashlib.sha256(authority.authority_record_bytes).hexdigest()
        != authority.authority_sha256
    ):
        msg = "Production authority receipt has invalid immutable fields."
        raise D5DerivationError(msg)
    record = runner._parse_json_bytes(  # noqa: SLF001
        authority.authority_record_bytes,
        path=Path("validated-postprocess-authority.json"),
    )
    expected_record_keys = {
        "analysis",
        "approvals",
        "bmrs",
        "cohorts",
        "contract",
        "contracts",
        "d3_conjunction_role",
        "fit_policy",
        "marginal_validity_evidence",
        "roots",
        "schema",
        "sealed_completion",
        "top_k",
    }
    if (
        set(record) != expected_record_keys
        or record.get("schema") != POSTPROCESS_AUTHORITY_SCHEMA
        or record.get("contract") != POSTPROCESS_AUTHORITY_CONTRACT
        or record.get("analysis") != "tcga-revision-k500"
        or record.get("top_k") != TOP_K
        or record.get("cohorts") != list(TCGA_COHORTS)
        or record.get("bmrs") != list(BMRS)
        or _canonical_json(record.get("fit_policy"))
        != _canonical_json(
            runner._fit_policy_record(authority.fit_policy),  # noqa: SLF001
        )
        or record.get("d3_conjunction_role") != CONJUNCTION_SECONDARY
    ):
        msg = "Production authority record is not the validated D1-D6 K=500 record."
        raise D5DerivationError(msg)
    _validate_signed_hierarchy(authority.fit_policy.d3)
    _validate_policy(authority.fit_policy.d5)
    _validate_marginal_validity(authority.marginal_validity)
    if record.get("roots") != {
        "run_output_root": authority.config.run_output_root.as_posix(),
        "canonical_input_root": authority.config.canonical_input_root.as_posix(),
        "canonical_input_manifest_sha256": (
            authority.config.expected_canonical_input_sha256
        ),
        "provider_input_root": authority.config.provider_input_root.as_posix(),
        "provider_input_manifest_sha256": (
            authority.config.expected_provider_input_manifest_sha256
        ),
    }:
        msg = "Production authority root record changed after validation."
        raise D5DerivationError(msg)
    _require_current_authority_files(authority)
    paths = _runner_paths(authority.config)
    input_approval = validate_revision_approval(
        authority.config.input_approval_manifest,
        authority.config.expected_input_approval_sha256,
        MATERIALIZE_FINAL_INPUTS_STAGE,
    )
    runner._require_materialize_stage_binding(input_approval)  # noqa: SLF001
    fit_approval = validate_revision_approval(
        authority.config.fit_approval_manifest,
        authority.config.expected_fit_approval_sha256,
        FIT_SEALED_TCGA_K500_STAGE,
    )
    runner._require_fit_stage_binding(fit_approval, paths)  # noqa: SLF001
    current_policy = validate_revision_fit_policy(
        fit_approval,
        expected_d4_implementation=runner.REQUIRED_D4_IMPLEMENTATION,
        expected_tested_family=runner.REQUIRED_TESTED_FAMILY,
    )
    if _canonical_json(runner._fit_policy_record(current_policy)) != _canonical_json(  # noqa: SLF001
        runner._fit_policy_record(authority.fit_policy),  # noqa: SLF001
    ):
        msg = "Live signed fit policy changed after authority validation."
        raise D5DerivationError(msg)
    _validate_inspect_approval(authority.config, fit_approval=fit_approval)
    if authority.marginal_validity.inferentially_eligible:
        calibration = _validate_calibration_approval(
            authority.config,
            fit_approval=fit_approval,
        )
        current_evidence = _validate_calibration_evidence(
            authority.config,
            fit_policy=current_policy,
            fit_approval=fit_approval,
            calibration_approval=calibration,
        )
        if (
            current_evidence.artifact_sha256
            != authority.marginal_validity.artifact_sha256
            or current_evidence.status != authority.marginal_validity.status
        ):
            msg = "Live marginal-validity authority changed after validation."
            raise D5DerivationError(msg)


def _require_current_authority_files(authority: ValidatedPostprocessAuthority) -> None:
    config = authority.config
    exact_files: list[tuple[Path, str, bytes | None, str]] = [
        (
            config.run_output_root / SEALED_COMPLETION_NAME,
            config.expected_sealed_completion_sha256,
            authority.completion_bytes,
            "sealed completion",
        ),
        (
            config.run_output_root / "run_manifest.json",
            hashlib.sha256(authority.run_manifest_bytes).hexdigest(),
            authority.run_manifest_bytes,
            "run manifest",
        ),
        (
            config.canonical_input_root / "input_manifest.json",
            config.expected_canonical_input_sha256,
            None,
            "canonical input manifest",
        ),
        (
            config.provider_input_root / "provider_input_manifest.json",
            config.expected_provider_input_manifest_sha256,
            None,
            "provider input manifest",
        ),
        (
            config.input_approval_manifest,
            config.expected_input_approval_sha256,
            None,
            "input approval",
        ),
        (
            config.fit_approval_manifest,
            config.expected_fit_approval_sha256,
            None,
            "fit approval",
        ),
        (
            config.inspect_approval_manifest,
            config.expected_inspect_approval_sha256,
            None,
            "inspection approval",
        ),
    ]
    if authority.marginal_validity.inferentially_eligible:
        if (
            config.calibration_approval_manifest is None
            or config.expected_calibration_approval_sha256 is None
            or config.calibration_evidence_artifact is None
            or config.expected_calibration_evidence_sha256 is None
        ):
            msg = "Certified authority lost its calibration files."
            raise D5DerivationError(msg)
        exact_files.extend(
            [
                (
                    config.calibration_approval_manifest,
                    config.expected_calibration_approval_sha256,
                    None,
                    "calibration approval",
                ),
                (
                    config.calibration_evidence_artifact,
                    config.expected_calibration_evidence_sha256,
                    authority.marginal_validity.artifact_bytes,
                    "marginal-validity evidence",
                ),
            ],
        )
    for path, digest, frozen, label in exact_files:
        current = runner._read_secure_regular_bytes(path, label=label)  # noqa: SLF001
        if hashlib.sha256(current).hexdigest() != digest or (
            frozen is not None and current != frozen
        ):
            msg = f"Live {label} changed after authority validation."
            raise D5DerivationError(msg)
    design = (
        authority.fit_policy.d6.design_authority
        if authority.marginal_validity.inferentially_eligible
        else None
    )
    if design is not None:
        path = config.calibration_design_manifest
        if path is None:
            msg = "Signed D6 design path disappeared after validation."
            raise D5DerivationError(msg)
        raw = runner._read_secure_regular_bytes(  # noqa: SLF001
            path,
            label="signed calibration design",
        )
        if hashlib.sha256(raw).hexdigest() != design.manifest_sha256:
            msg = "Live calibration design changed after authority validation."
            raise D5DerivationError(msg)
    contracts = runner._parse_json_bytes(  # noqa: SLF001
        authority.authority_record_bytes,
        path=Path("validated-postprocess-authority.json"),
    )["contracts"]
    for index, cohort in enumerate(TCGA_COHORTS):
        current = runner._read_secure_regular_bytes(  # noqa: SLF001
            config.run_output_root / "contracts" / f"{cohort}.json",
            label=f"postprocess contract {cohort}",
        )
        if (
            current != authority.contract_bytes[index]
            or hashlib.sha256(current).hexdigest() != contracts[index]["file_sha256"]
        ):
            msg = f"Live cohort contract changed after validation: {cohort}."
            raise D5DerivationError(msg)


def _completion_task_receipt(
    authority: ValidatedPostprocessAuthority,
    *,
    cohort: str,
    provider: str,
) -> Mapping[str, object]:
    completion = runner._parse_json_bytes(  # noqa: SLF001
        authority.completion_bytes,
        path=authority.config.run_output_root / SEALED_COMPLETION_NAME,
    )
    index = TCGA_COHORTS.index(cohort) * len(BMRS) + BMRS.index(provider)
    receipt = completion["tasks"][index]
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("cohort") != cohort
        or receipt.get("bmr") != provider
    ):
        msg = f"Sealed task coordinate is invalid for {cohort}/{provider}."
        raise D5DerivationError(msg)
    return receipt


def _read_pinned_validated_task_snapshot(
    task_dir: Path,
    contract: dict[str, object],
    *,
    provider: str,
) -> tuple[bytes, bytes, bytes]:
    """Validate and return one immutable snapshot through one pinned directory."""
    directory_fd = runner._open_secure_directory(  # noqa: SLF001
        task_dir,
        label=f"D5 task {provider}",
    )
    try:
        runner._require_directory_path_identity(  # noqa: SLF001
            task_dir,
            directory_fd,
            label=f"D5 task {provider}",
        )
        before = runner._read_task_output_snapshot(  # noqa: SLF001
            task_dir,
            require_manifest=True,
            directory_fd=directory_fd,
        )
        validation = runner.validate_task_output(
            task_dir,
            contract,
            bmr=provider,
            directory_fd=directory_fd,
        )
        after = runner._read_task_output_snapshot(  # noqa: SLF001
            task_dir,
            require_manifest=True,
            directory_fd=directory_fd,
        )
        runner._require_directory_path_identity(  # noqa: SLF001
            task_dir,
            directory_fd,
            label=f"D5 task {provider}",
        )
        single_raw, pairwise_raw, manifest_raw = before
        if (
            before != after
            or manifest_raw is None
            or validation.get("single_gene_sha256")
            != hashlib.sha256(single_raw).hexdigest()
            or validation.get("pairwise_sha256")
            != hashlib.sha256(pairwise_raw).hexdigest()
        ):
            msg = f"Task changed across pinned validation: {task_dir}."
            raise D5DerivationError(msg)  # noqa: TRY301
        return single_raw, pairwise_raw, manifest_raw  # noqa: TRY300
    except D5DerivationError:
        raise
    except (OSError, ValueError, runner.SealedFitError) as error:
        msg = f"Pinned task validation failed closed: {task_dir}."
        raise D5DerivationError(msg) from error
    finally:
        os.close(directory_fd)


def _read_validated_component_bytes(
    authority: ValidatedPostprocessAuthority,
    *,
    cohort: str,
    contract: dict[str, object],
) -> tuple[dict[str, bytes], tuple[dict[str, object], ...]]:
    paths = _runner_paths(authority.config)
    current_contract = runner._load_verified_contract(  # noqa: SLF001
        paths,
        cohort,
        top_k=TOP_K,
    )
    if current_contract != contract:
        msg = f"Cohort contract changed before row validation: {cohort}."
        raise D5DerivationError(msg)
    component_bytes: dict[str, bytes] = {}
    bindings: list[dict[str, object]] = []
    for provider in BMRS:
        task_dir = authority.config.run_output_root / "tasks" / cohort / provider
        single_raw, pairwise_raw, manifest_raw = _read_pinned_validated_task_snapshot(
            task_dir,
            contract,
            provider=provider,
        )
        expected = _completion_task_receipt(
            authority,
            cohort=cohort,
            provider=provider,
        )
        manifest = runner._parse_json_bytes(  # noqa: SLF001
            manifest_raw,
            path=task_dir / "task_manifest.json",
        )
        pair_receipt = expected.get("pairwise_interaction_results")
        single_receipt = expected.get("single_gene_results")
        task_receipt = expected.get("task_manifest")
        if (
            not isinstance(pair_receipt, Mapping)
            or dict(pair_receipt)
            != {
                "bytes": len(pairwise_raw),
                "sha256": hashlib.sha256(pairwise_raw).hexdigest(),
            }
            or not isinstance(single_receipt, Mapping)
            or dict(single_receipt)
            != {
                "bytes": len(single_raw),
                "sha256": hashlib.sha256(single_raw).hexdigest(),
            }
            or not isinstance(task_receipt, Mapping)
            or dict(task_receipt)
            != {
                "bytes": len(manifest_raw),
                "sha256": hashlib.sha256(manifest_raw).hexdigest(),
            }
            or manifest.get("cohort") != cohort
            or manifest.get("bmr") != provider
            or manifest.get("contract_sha256") != runner._json_sha256(contract)  # noqa: SLF001
            or manifest.get("validation", {}).get("pairwise_sha256")
            != pair_receipt["sha256"]
        ):
            msg = f"Task/output snapshot differs from seal: {cohort}/{provider}."
            raise D5DerivationError(msg)
        component_bytes[provider] = pairwise_raw
        bindings.append(
            {
                "provider": provider,
                "contract_sha256": expected["contract_sha256"],
                "task_manifest_sha256": task_receipt["sha256"],
                "pairwise_sha256": pair_receipt["sha256"],
                "single_gene_sha256": single_receipt["sha256"],
            },
        )
    return component_bytes, tuple(bindings)


def _cohort_authority_record(
    authority: ValidatedPostprocessAuthority,
    *,
    cohort: str,
    task_bindings: tuple[dict[str, object], ...],
) -> Mapping[str, object]:
    return MappingProxyType(
        {
            "grid_authority_sha256": authority.authority_sha256,
            "sealed_completion_sha256": (
                authority.config.expected_sealed_completion_sha256
            ),
            "canonical_input_manifest_sha256": (
                authority.config.expected_canonical_input_sha256
            ),
            "provider_input_manifest_sha256": (
                authority.config.expected_provider_input_manifest_sha256
            ),
            "input_approval_sha256": (authority.config.expected_input_approval_sha256),
            "cohort": cohort,
            "task_bindings": [dict(binding) for binding in task_bindings],
            "fit_approval_sha256": authority.config.expected_fit_approval_sha256,
            "inspect_approval_sha256": (
                authority.config.expected_inspect_approval_sha256
            ),
            "d5_policy_receipt": runner._fit_policy_record(  # noqa: SLF001
                authority.fit_policy,
            )["receipts"]["D5"],
            "calibration_approval_sha256": (
                authority.config.expected_calibration_approval_sha256
            ),
            "marginal_validity_evidence_sha256": (
                authority.config.expected_calibration_evidence_sha256
            ),
        },
    )


def _seal_derived_family(
    authority: ValidatedPostprocessAuthority,
    *,
    cohort: str,
    derived: DerivedCohortFamily,
) -> SealedDerivedCohortFamily:
    _validate_derived_for_publication(derived)
    binding = _publication_binding(
        authority_sha256=authority.authority_sha256,
        cohort=cohort,
        derived=derived,
    )
    sealed = object.__new__(SealedDerivedCohortFamily)
    for name, value in {
        "cohort": cohort,
        "derived": derived,
        "authority": authority,
        "publication_binding_sha256": binding,
        "_seal": _DERIVATION_SEAL,
    }.items():
        object.__setattr__(sealed, name, value)
    return sealed


def _publication_binding(
    *,
    authority_sha256: str,
    cohort: str,
    derived: DerivedCohortFamily,
) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "authority_sha256": authority_sha256,
                "cohort": cohort,
                "csv_sha256": derived.csv_sha256,
                "manifest_sha256": derived.manifest_sha256,
                "row_count": derived.row_count,
            },
        ),
    ).hexdigest()


def _require_sealed_derived_family(sealed: SealedDerivedCohortFamily) -> None:
    if not isinstance(sealed, SealedDerivedCohortFamily):
        msg = "Production publication requires a sealed derivation receipt."
        raise TypeError(msg)
    if getattr(sealed, "_seal", None) is not _DERIVATION_SEAL:
        msg = "Derivation receipt was not minted by the production derivation."
        raise D5DerivationError(msg)
    _require_validated_authority(sealed.authority)
    _validate_derived_for_publication(sealed.derived)
    if (
        sealed.cohort not in TCGA_COHORTS
        or not sealed.derived.production_eligible
        or sealed.publication_binding_sha256
        != _publication_binding(
            authority_sha256=sealed.authority.authority_sha256,
            cohort=sealed.cohort,
            derived=sealed.derived,
        )
    ):
        msg = "Sealed derivation publication binding is invalid."
        raise D5DerivationError(msg)
    manifest = runner._parse_json_bytes(  # noqa: SLF001
        sealed.derived.manifest_bytes,
        path=Path("sealed-derived-family.json"),
    )
    production = manifest.get("production_authority")
    if (
        manifest.get("cohort") != sealed.cohort
        or not isinstance(production, Mapping)
        or production.get("grid_authority_sha256") != sealed.authority.authority_sha256
        or production.get("sealed_completion_sha256")
        != sealed.authority.config.expected_sealed_completion_sha256
        or production.get("marginal_validity_evidence_sha256")
        != sealed.authority.config.expected_calibration_evidence_sha256
    ):
        msg = "Derived manifest does not bind the live production authority."
        raise D5DerivationError(msg)
    _revalidate_sealed_task_bindings(sealed)


def _revalidate_sealed_task_bindings(sealed: SealedDerivedCohortFamily) -> None:
    manifest = runner._parse_json_bytes(  # noqa: SLF001
        sealed.derived.manifest_bytes,
        path=Path("sealed-derived-family.json"),
    )
    bindings = manifest["production_authority"].get("task_bindings")
    if not isinstance(bindings, list) or len(bindings) != len(BMRS):
        msg = "Derived task bindings do not cover all three providers."
        raise D5DerivationError(msg)
    contract = runner._parse_json_bytes(  # noqa: SLF001
        sealed.authority.contract_bytes[TCGA_COHORTS.index(sealed.cohort)],
        path=(
            sealed.authority.config.run_output_root
            / "contracts"
            / f"{sealed.cohort}.json"
        ),
    )
    for provider, binding in zip(BMRS, bindings, strict=True):
        task_dir = (
            sealed.authority.config.run_output_root / "tasks" / sealed.cohort / provider
        )
        single_raw, pairwise_raw, task_raw = _read_pinned_validated_task_snapshot(
            task_dir,
            contract,
            provider=provider,
        )
        if binding != {
            "provider": provider,
            "contract_sha256": manifest["axis"]["cohort_contract_sha256"],
            "task_manifest_sha256": hashlib.sha256(task_raw).hexdigest(),
            "pairwise_sha256": hashlib.sha256(pairwise_raw).hexdigest(),
            "single_gene_sha256": hashlib.sha256(single_raw).hexdigest(),
        }:
            msg = (
                "Live task bytes changed before publication: "
                f"{sealed.cohort}/{provider}."
            )
            raise D5DerivationError(msg)


def derive_k500_cohort_conjunction(
    authority: ValidatedPostprocessAuthority,
    cohort: str,
) -> SealedDerivedCohortFamily:
    """Derive one K=500 family only from an opaque validated grid authority.

    No caller-supplied scientific bytes, axis, policy, or validity label crosses
    this seam.  The three component files are reopened securely, revalidated
    against their task manifests and the sealed whole-grid receipt, and only then
    parsed into the complete within-cohort D5 family.
    """
    _require_validated_authority(authority)
    if cohort not in TCGA_COHORTS:
        msg = f"Unknown production cohort: {cohort!r}."
        raise D5DerivationError(msg)
    contract = runner._parse_json_bytes(  # noqa: SLF001
        authority.contract_bytes[TCGA_COHORTS.index(cohort)],
        path=authority.config.run_output_root / "contracts" / f"{cohort}.json",
    )
    axis = build_frozen_family_axis(
        cohort,
        contract["features"],
        cohort_contract_sha256=runner._json_sha256(contract),  # noqa: SLF001
    )
    if len(axis.ordered_features) != TOP_K:
        msg = f"Production D5 derivation requires exactly {TOP_K} features."
        raise D5DerivationError(msg)
    components, task_bindings = _read_validated_component_bytes(
        authority,
        cohort=cohort,
        contract=contract,
    )
    derived = _derive_conjunction_family(
        components,
        axis=axis,
        policy=authority.fit_policy.d5,
        marginal_validity=authority.marginal_validity,
        provider_hierarchy=authority.fit_policy.d3,
        production_authority=_cohort_authority_record(
            authority,
            cohort=cohort,
            task_bindings=task_bindings,
        ),
    )
    if not derived.production_eligible:
        msg = "K=500 derivation did not retain its production eligibility."
        raise D5DerivationError(msg)
    return _seal_derived_family(authority, cohort=cohort, derived=derived)


def derive_conjunction_family(
    component_pairwise_csv: Mapping[str, bytes],
    *,
    axis: FrozenFamilyAxis,
    policy: ConjunctionMultiplicityPolicy,
    marginal_validity: MarginalValidityEvidence,
) -> DerivedCohortFamily:
    """Derive a complete synthetic QA family under the production D5 logic.

    Use :func:`derive_k500_cohort_conjunction` for a publishable production family.
    Outputs whose axis is not exactly K=500 are marked ineligible and the public
    publication seam rejects them.
    """
    if marginal_validity.inferentially_eligible:
        msg = "Synthetic derivation cannot claim certified marginal validity."
        raise D5DerivationError(msg)
    return _derive_conjunction_family(
        component_pairwise_csv,
        axis=axis,
        policy=policy,
        marginal_validity=marginal_validity,
        provider_hierarchy=None,
        production_authority=None,
    )


def _derive_conjunction_family(  # noqa: PLR0913
    component_pairwise_csv: Mapping[str, bytes],
    *,
    axis: FrozenFamilyAxis,
    policy: ConjunctionMultiplicityPolicy,
    marginal_validity: MarginalValidityEvidence,
    provider_hierarchy: ProviderHierarchyPolicy | None,
    production_authority: Mapping[str, object] | None,
) -> DerivedCohortFamily:
    """Apply the pure complete-family derivation after the trust-boundary gate."""
    _validate_axis(axis)
    _validate_policy(policy)
    _validate_marginal_validity(marginal_validity)
    _validate_provider_mapping(component_pairwise_csv)

    expected_pairs = tuple(iter_tested_pairs(axis.ordered_features))
    if not expected_pairs:
        msg = "D5 derivation requires a non-empty complete pair family."
        raise D5DerivationError(msg)

    source_sha256: dict[str, str] = {}
    component_rows: dict[str, tuple[_ComponentRow, ...]] = {}
    for provider in BMRS:
        raw = component_pairwise_csv[provider]
        if not isinstance(raw, bytes):
            msg = f"{provider} pairwise input must be immutable bytes."
            raise D5DerivationError(msg)
        source_sha256[provider] = hashlib.sha256(raw).hexdigest()
        component_rows[provider] = _parse_component_csv(
            raw,
            provider=provider,
            expected_pairs=expected_pairs,
        )

    _validate_cross_component_coordinates(component_rows, expected_pairs)
    rows = _derive_output_rows(
        component_rows,
        axis=axis,
        policy=policy,
        source_sha256=source_sha256,
        marginal_validity=marginal_validity,
        d3_conjunction_role=(
            CONJUNCTION_SECONDARY
            if provider_hierarchy is None
            else provider_hierarchy.all_three_conjunction_role
        ),
    )
    csv_bytes = _serialize_rows(rows)
    csv_sha256 = hashlib.sha256(csv_bytes).hexdigest()
    production_eligible = (
        len(axis.ordered_features) == TOP_K
        and policy.tested_family.top_k == TOP_K
        and production_authority is not None
        and provider_hierarchy is not None
        and (
            not marginal_validity.inferentially_eligible
            or marginal_validity._seal is _EVIDENCE_SEAL  # noqa: SLF001
        )
    )
    manifest = _build_manifest(
        axis=axis,
        policy=policy,
        source_sha256=source_sha256,
        row_count=len(rows),
        csv_bytes=csv_bytes,
        csv_sha256=csv_sha256,
        marginal_validity=marginal_validity,
        production_eligible=production_eligible,
        provider_hierarchy=provider_hierarchy,
        production_authority=production_authority,
    )
    manifest_bytes = _canonical_json(manifest) + b"\n"
    return DerivedCohortFamily(
        csv_bytes=csv_bytes,
        manifest_bytes=manifest_bytes,
        row_count=len(rows),
        csv_sha256=csv_sha256,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        production_eligible=production_eligible,
    )


def write_derived_cohort_family(
    output_directory: Path,
    sealed: SealedDerivedCohortFamily,
) -> None:
    """Reject standalone production writes outside the whole-grid release seam."""
    if not isinstance(output_directory, Path):
        msg = "Output directory must be a pathlib.Path."
        raise TypeError(msg)
    _require_sealed_derived_family(sealed)
    msg = (
        "Standalone production cohort publication is forbidden; use "
        "run_production_postprocess for one atomic complete-grid release."
    )
    raise D5DerivationError(msg)


def _write_production_cohort_for_release(
    output_directory: Path,
    sealed: SealedDerivedCohortFamily,
) -> None:
    """Stage one sealed cohort only inside the validated whole-grid workflow."""
    _require_sealed_derived_family(sealed)
    _write_derived_cohort_family(
        output_directory,
        sealed.derived,
        allow_nonproduction=False,
    )


def _write_derived_cohort_family_for_qa(
    output_directory: Path,
    derived: DerivedCohortFamily,
) -> None:
    """Exercise the production publication boundary with a synthetic QA family."""
    _write_derived_cohort_family(
        output_directory,
        derived,
        allow_nonproduction=True,
    )


def _write_derived_cohort_family(
    output_directory: Path,
    derived: DerivedCohortFamily,
    *,
    allow_nonproduction: bool,
) -> None:
    """Implement the shared atomic writer after an explicit eligibility gate."""
    _validate_derived_for_publication(derived)
    if not derived.production_eligible and not allow_nonproduction:
        msg = "Synthetic QA derivations cannot use the production publication seam."
        raise D5DerivationError(msg)
    if not isinstance(output_directory, Path):
        msg = "Output directory must be a pathlib.Path."
        raise TypeError(msg)
    if output_directory.name in {"", ".", ".."}:
        msg = "Output directory must have a safe final path component."
        raise D5DerivationError(msg)
    lexical_parent = output_directory.parent
    parent = lexical_parent.resolve(strict=True)
    parent_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    parent_identity = parent.stat(follow_symlinks=False)
    parent_fd = os.open(parent, parent_flags)
    staging_name = f".{output_directory.name}.{uuid.uuid4().hex}.tmp"
    staging_fd: int | None = None
    try:
        _require_stable_publication_parent(
            lexical_parent,
            parent,
            parent_fd,
            parent_identity,
        )
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_fd)
        staging_fd = os.open(staging_name, parent_flags, dir_fd=parent_fd)
        staging_identity = os.fstat(staging_fd)
        if not stat_module.S_ISDIR(staging_identity.st_mode):
            msg = "Staging publication entry is not a directory."
            raise D5DerivationError(msg)
        # If either staged write fails, the diagnostic staging tree remains while
        # the user-visible destination remains absent.  The receipt is written last.
        _write_exclusive_synced_at(
            staging_fd,
            OUTPUT_CSV_NAME,
            derived.csv_bytes,
        )
        _write_exclusive_synced_at(
            staging_fd,
            OUTPUT_MANIFEST_NAME,
            derived.manifest_bytes,
        )
        _verify_staged_file_at(staging_fd, OUTPUT_CSV_NAME, derived.csv_bytes)
        _verify_staged_file_at(
            staging_fd,
            OUTPUT_MANIFEST_NAME,
            derived.manifest_bytes,
        )
        os.fsync(staging_fd)
        _require_same_directory_entry(
            parent_fd,
            staging_name,
            staging_identity,
        )
        _require_stable_publication_parent(
            lexical_parent,
            parent,
            parent_fd,
            parent_identity,
        )
        _rename_no_replace_at(
            parent_fd,
            staging_name,
            output_directory.name,
        )
        published = os.stat(
            output_directory.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat_module.S_ISDIR(published.st_mode)
            or published.st_dev != staging_identity.st_dev
            or published.st_ino != staging_identity.st_ino
        ):
            msg = "Published directory identity changed during atomic publication."
            raise D5DerivationError(msg)
        os.fsync(parent_fd)
        _require_stable_publication_parent(
            lexical_parent,
            parent,
            parent_fd,
            parent_identity,
        )
    finally:
        if staging_fd is not None:
            os.close(staging_fd)
        os.close(parent_fd)


def run_production_postprocess(
    config: ProductionPostprocessConfig,
    output_root: Path,
) -> dict[str, object]:
    """Validate, derive, and atomically publish the complete 32-cohort release."""
    authority = validate_postprocess_authority(config)
    if not isinstance(output_root, Path) or not output_root.is_absolute():
        msg = "Production output root must be an absolute pathlib.Path."
        raise D5DerivationError(msg)
    if output_root.name in {"", ".", ".."}:
        msg = "Production output root must have a safe final path component."
        raise D5DerivationError(msg)
    source_roots = (
        config.run_output_root,
        config.canonical_input_root,
        config.provider_input_root,
    )
    for source_root in source_roots:
        try:
            output_root.relative_to(source_root)
        except ValueError:
            continue
        msg = "Production output root must be outside every immutable input root."
        raise D5DerivationError(msg)

    lexical_parent = output_root.parent
    parent = lexical_parent.resolve(strict=True)
    resolved_destination = parent / output_root.name
    resolved_source_roots = tuple(
        source_root.resolve(strict=True) for source_root in source_roots
    )
    resolved_source_identities = tuple(
        source_root.stat(follow_symlinks=False) for source_root in resolved_source_roots
    )
    _require_destination_outside_source_roots(
        resolved_destination,
        resolved_source_roots,
    )
    parent_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    parent_identity = parent.stat(follow_symlinks=False)
    parent_fd = os.open(parent, parent_flags)
    source_root_pins: list[tuple[Path, Path, int, os.stat_result]] = []
    staging_name = f".{output_root.name}.{uuid.uuid4().hex}.tmp"
    staging_path = parent / staging_name
    staging_fd: int | None = None
    published_fd: int | None = None
    cohort_pins: list[_PinnedCohortDirectory] = []
    root_file_pins: list[_PinnedPublicationFile] = []
    release_records: list[dict[str, object]] = []
    try:
        for source_root, resolved_source_root, source_identity in zip(
            source_roots,
            resolved_source_roots,
            resolved_source_identities,
            strict=True,
        ):
            source_fd = os.open(resolved_source_root, parent_flags)
            source_root_pins.append(
                (
                    source_root,
                    resolved_source_root,
                    source_fd,
                    source_identity,
                ),
            )
        _require_stable_publication_parent(
            lexical_parent,
            parent,
            parent_fd,
            parent_identity,
        )
        for (
            source_root,
            resolved_source_root,
            source_fd,
            source_identity,
        ) in source_root_pins:
            _require_stable_publication_parent(
                source_root,
                resolved_source_root,
                source_fd,
                source_identity,
            )
        _require_destination_outside_source_roots(
            parent / output_root.name,
            tuple(pin[1] for pin in source_root_pins),
        )
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_fd)
        staging_fd = os.open(staging_name, parent_flags, dir_fd=parent_fd)
        staging_identity = os.fstat(staging_fd)
        for cohort in TCGA_COHORTS:
            sealed = derive_k500_cohort_conjunction(authority, cohort)
            _write_production_cohort_for_release(staging_path / cohort, sealed)
            cohort_pins.append(
                _pin_staged_cohort_directory(staging_fd, cohort, sealed),
            )
            release_records.append(
                {
                    "cohort": cohort,
                    "directory": cohort,
                    "csv_sha256": sealed.derived.csv_sha256,
                    "manifest_sha256": sealed.derived.manifest_sha256,
                    "rows": sealed.derived.row_count,
                    "publication_binding_sha256": (sealed.publication_binding_sha256),
                },
            )
        _write_exclusive_synced_at(
            staging_fd,
            AUTHORITY_RECEIPT_NAME,
            authority.authority_record_bytes,
        )
        _verify_staged_file_at(
            staging_fd,
            AUTHORITY_RECEIPT_NAME,
            authority.authority_record_bytes,
        )
        root_file_pins.append(
            _open_pinned_publication_file(
                staging_fd,
                AUTHORITY_RECEIPT_NAME,
                authority.authority_record_bytes,
            ),
        )
        release_manifest = {
            "schema": RELEASE_SCHEMA,
            "contract": RELEASE_CONTRACT,
            "analysis": "tcga-revision-k500",
            "top_k": TOP_K,
            "cohorts": list(TCGA_COHORTS),
            "bmrs": list(BMRS),
            "grid_authority_sha256": authority.authority_sha256,
            "authority_receipt": {
                "name": AUTHORITY_RECEIPT_NAME,
                "sha256": authority.authority_sha256,
            },
            "sealed_completion_sha256": (config.expected_sealed_completion_sha256),
            "marginal_validity_evidence_sha256": (
                config.expected_calibration_evidence_sha256
            ),
            "outputs": release_records,
        }
        release_bytes = _canonical_json(release_manifest) + b"\n"
        # The release receipt is the final staged write; the public root does not
        # exist until every cohort has validated and this manifest is durable.
        _write_exclusive_synced_at(
            staging_fd,
            RELEASE_MANIFEST_NAME,
            release_bytes,
        )
        _verify_staged_file_at(staging_fd, RELEASE_MANIFEST_NAME, release_bytes)
        root_file_pins.append(
            _open_pinned_publication_file(
                staging_fd,
                RELEASE_MANIFEST_NAME,
                release_bytes,
            ),
        )
        expected_inventory = {
            *TCGA_COHORTS,
            AUTHORITY_RECEIPT_NAME,
            RELEASE_MANIFEST_NAME,
        }
        if set(os.listdir(staging_fd)) != expected_inventory:  # noqa: PTH208
            msg = "Staged postprocess release inventory is not closed."
            raise D5DerivationError(msg)
        _validate_complete_release_tree(
            staging_fd,
            staging_identity,
            cohort_pins,
            root_file_pins,
            expected_inventory,
            frozen=False,
        )
        for pin in cohort_pins:
            _freeze_pinned_cohort(pin)
        for pin in root_file_pins:
            _freeze_pinned_file(pin)
        os.fchmod(staging_fd, 0o500)
        os.fsync(staging_fd)
        _validate_complete_release_tree(
            staging_fd,
            staging_identity,
            cohort_pins,
            root_file_pins,
            expected_inventory,
            frozen=True,
        )
        _require_same_directory_entry(parent_fd, staging_name, staging_identity)
        _require_stable_publication_parent(
            lexical_parent,
            parent,
            parent_fd,
            parent_identity,
        )
        _rename_no_replace_at(parent_fd, staging_name, output_root.name)
        published_fd = os.open(
            output_root.name,
            parent_flags,
            dir_fd=parent_fd,
        )
        _require_same_directory_entry(parent_fd, output_root.name, staging_identity)
        _validate_complete_release_tree(
            published_fd,
            staging_identity,
            cohort_pins,
            root_file_pins,
            expected_inventory,
            frozen=True,
        )
        for pin in cohort_pins:
            os.fsync(pin.csv_file.descriptor)
            os.fsync(pin.manifest_file.descriptor)
            os.fsync(pin.descriptor)
        for pin in root_file_pins:
            os.fsync(pin.descriptor)
        os.fsync(published_fd)
        os.fsync(parent_fd)
        _require_stable_publication_parent(
            lexical_parent,
            parent,
            parent_fd,
            parent_identity,
        )
        _require_same_directory_entry(parent_fd, output_root.name, staging_identity)
        _validate_complete_release_tree(
            published_fd,
            staging_identity,
            cohort_pins,
            root_file_pins,
            expected_inventory,
            frozen=True,
        )
    finally:
        if published_fd is not None:
            os.close(published_fd)
        for pin in root_file_pins:
            os.close(pin.descriptor)
        for pin in cohort_pins:
            _close_pinned_cohort(pin)
        if staging_fd is not None:
            os.close(staging_fd)
        for _source_root, _resolved_root, source_fd, _identity in source_root_pins:
            os.close(source_fd)
        os.close(parent_fd)
    return {
        "output_root": output_root.as_posix(),
        "authority_sha256": authority.authority_sha256,
        "release_manifest_sha256": hashlib.sha256(release_bytes).hexdigest(),
        "cohorts": len(release_records),
    }


def _validate_derived_for_publication(derived: DerivedCohortFamily) -> None:
    if not isinstance(derived, DerivedCohortFamily):
        msg = "Publication requires a typed derived cohort family."
        raise TypeError(msg)
    if (
        not isinstance(derived.csv_bytes, bytes)
        or not isinstance(derived.manifest_bytes, bytes)
        or isinstance(derived.row_count, bool)
        or not isinstance(derived.row_count, int)
        or derived.row_count <= 0
        or not isinstance(derived.production_eligible, bool)
    ):
        msg = "Derived publication receipt has invalid field types."
        raise D5DerivationError(msg)
    _require_sha256(derived.csv_sha256, label="derived CSV")
    _require_sha256(derived.manifest_sha256, label="derived manifest")
    if (
        hashlib.sha256(derived.csv_bytes).hexdigest() != derived.csv_sha256
        or hashlib.sha256(derived.manifest_bytes).hexdigest() != derived.manifest_sha256
    ):
        msg = "Derived publication bytes do not match their immutable digests."
        raise D5DerivationError(msg)
    try:
        manifest = json.loads(derived.manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        msg = "Derived publication manifest is invalid canonical JSON."
        raise D5DerivationError(msg) from error
    if _canonical_json(manifest) + b"\n" != derived.manifest_bytes:
        msg = "Derived publication manifest is not canonical JSON."
        raise D5DerivationError(msg)
    _validate_derived_csv_shape(derived)
    output = manifest.get("output") if isinstance(manifest, dict) else None
    axis = manifest.get("axis") if isinstance(manifest, dict) else None
    tested_family = (
        manifest.get("tested_family") if isinstance(manifest, dict) else None
    )
    production_authority = (
        manifest.get("production_authority") if isinstance(manifest, dict) else None
    )
    provider_hierarchy = (
        manifest.get("d3_provider_hierarchy") if isinstance(manifest, dict) else None
    )
    if (
        not isinstance(manifest, dict)
        or manifest.get("production_eligible") is not derived.production_eligible
        or manifest.get("implementation") != _implementation_provenance()
        or not isinstance(output, dict)
        or (
            output.get("name") != OUTPUT_CSV_NAME
            or output.get("bytes") != len(derived.csv_bytes)
            or output.get("rows") != derived.row_count
            or output.get("sha256") != derived.csv_sha256
            or output.get("columns") != list(OUTPUT_COLUMNS)
        )
    ):
        msg = "Derived publication manifest does not bind its CSV output."
        raise D5DerivationError(msg)
    if derived.production_eligible and (
        not isinstance(axis, dict)
        or not isinstance(tested_family, dict)
        or not isinstance(production_authority, dict)
        or not isinstance(provider_hierarchy, dict)
        or axis.get("feature_count") != TOP_K
        or tested_family.get("top_k") != TOP_K
        or provider_hierarchy.get("primary_provider") != "cbase"
        or provider_hierarchy.get("sensitivity_providers") != ["dig", "mutsig"]
        or provider_hierarchy.get("all_three_conjunction_role") != CONJUNCTION_SECONDARY
        or provider_hierarchy.get("burden_dependent_switching") is not False
    ):
        msg = "Production publication is not bound to the exact K=500 family."
        raise D5DerivationError(msg)


def _validate_derived_csv_shape(derived: DerivedCohortFamily) -> None:
    try:
        stream = io.TextIOWrapper(
            io.BytesIO(derived.csv_bytes),
            encoding="utf-8",
            errors="strict",
            newline="",
        )
        reader = csv.reader(stream, strict=True)
        if tuple(next(reader, ())) != OUTPUT_COLUMNS:
            msg = "Derived publication CSV has an invalid output schema."
            raise D5DerivationError(msg)
        row_count = 0
        for row in reader:
            if len(row) != len(OUTPUT_COLUMNS):
                msg = "Derived publication CSV contains an invalid-width row."
                raise D5DerivationError(msg)
            row_count += 1
    except (csv.Error, UnicodeDecodeError) as error:
        msg = "Derived publication CSV is not canonical UTF-8 CSV."
        raise D5DerivationError(msg) from error
    if row_count != derived.row_count:
        msg = "Derived publication CSV row count does not match its receipt."
        raise D5DerivationError(msg)


def _write_exclusive_synced_at(
    directory_fd: int,
    name: str,
    content: bytes,
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
        before = os.fstat(descriptor)
        if not stat_module.S_ISREG(before.st_mode) or before.st_nlink != 1:
            msg = "Staged publication file must be a single-link regular file."
            raise D5DerivationError(msg)
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                msg = "Staged publication write made no progress."
                raise OSError(msg)
            view = view[written:]
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        if (
            not stat_module.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_size != len(content)
        ):
            msg = "Staged publication file identity changed during write."
            raise D5DerivationError(msg)
    finally:
        os.close(descriptor)


def _verify_staged_file_at(
    directory_fd: int,
    name: str,
    expected_content: bytes,
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        if (
            not stat_module.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != len(expected_content)
        ):
            msg = "Staged publication file is not a stable single-link file."
            raise D5DerivationError(msg)
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            not stat_module.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_size != before.st_size
            or digest.hexdigest() != hashlib.sha256(expected_content).hexdigest()
        ):
            msg = "Staged publication file changed before atomic publication."
            raise D5DerivationError(msg)
    finally:
        os.close(descriptor)


def _open_pinned_publication_file(
    directory_fd: int,
    name: str,
    expected_content: bytes,
) -> _PinnedPublicationFile:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        identity = os.fstat(descriptor)
        if not stat_module.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
            msg = f"Pinned release file must be single-link regular: {name}."
            raise D5DerivationError(msg)  # noqa: TRY301
        pin = _PinnedPublicationFile(
            name=name,
            descriptor=descriptor,
            identity=identity,
            expected_content=expected_content,
        )
        _validate_pinned_file_descriptor(pin, frozen=False)
    except Exception:
        os.close(descriptor)
        raise
    return pin


def _validate_pinned_file_descriptor(
    pin: _PinnedPublicationFile,
    *,
    frozen: bool,
) -> None:
    before = os.fstat(pin.descriptor)
    if (
        not stat_module.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_dev != pin.identity.st_dev
        or before.st_ino != pin.identity.st_ino
        or before.st_size != len(pin.expected_content)
        or stat_module.S_IMODE(before.st_mode) != (0o400 if frozen else 0o600)
    ):
        msg = f"Pinned release file identity or mode changed: {pin.name}."
        raise D5DerivationError(msg)
    os.lseek(pin.descriptor, 0, os.SEEK_SET)
    chunks = []
    while chunk := os.read(pin.descriptor, 1024 * 1024):
        chunks.append(chunk)
    content = b"".join(chunks)
    after = os.fstat(pin.descriptor)
    if (
        content != pin.expected_content
        or after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_size != before.st_size
        or after.st_nlink != 1
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_ctime_ns != before.st_ctime_ns
    ):
        msg = f"Pinned release file changed during readback: {pin.name}."
        raise D5DerivationError(msg)


def _pin_staged_cohort_directory(
    staging_fd: int,
    cohort: str,
    sealed: SealedDerivedCohortFamily,
) -> _PinnedCohortDirectory:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(cohort, flags, dir_fd=staging_fd)
    csv_pin: _PinnedPublicationFile | None = None
    manifest_pin: _PinnedPublicationFile | None = None
    try:
        identity = os.fstat(descriptor)
        if (
            not stat_module.S_ISDIR(identity.st_mode)
            or stat_module.S_IMODE(identity.st_mode) != 0o700
            or set(os.listdir(descriptor))  # noqa: PTH208
            != {OUTPUT_CSV_NAME, OUTPUT_MANIFEST_NAME}
        ):
            msg = f"Staged cohort inventory is not closed: {cohort}."
            raise D5DerivationError(msg)  # noqa: TRY301
        csv_pin = _open_pinned_publication_file(
            descriptor,
            OUTPUT_CSV_NAME,
            sealed.derived.csv_bytes,
        )
        manifest_pin = _open_pinned_publication_file(
            descriptor,
            OUTPUT_MANIFEST_NAME,
            sealed.derived.manifest_bytes,
        )
    except Exception:
        if csv_pin is not None:
            os.close(csv_pin.descriptor)
        if manifest_pin is not None:
            os.close(manifest_pin.descriptor)
        os.close(descriptor)
        raise
    return _PinnedCohortDirectory(
        cohort=cohort,
        descriptor=descriptor,
        identity=identity,
        csv_file=csv_pin,
        manifest_file=manifest_pin,
    )


def _validate_pinned_cohort_entry(
    release_root_fd: int,
    pin: _PinnedCohortDirectory,
    *,
    frozen: bool,
) -> None:
    expected_mode = 0o500 if frozen else 0o700
    pinned_identity = os.fstat(pin.descriptor)
    named_identity = os.stat(
        pin.cohort,
        dir_fd=release_root_fd,
        follow_symlinks=False,
    )
    if (
        not stat_module.S_ISDIR(pinned_identity.st_mode)
        or not stat_module.S_ISDIR(named_identity.st_mode)
        or pinned_identity.st_dev != pin.identity.st_dev
        or pinned_identity.st_ino != pin.identity.st_ino
        or named_identity.st_dev != pin.identity.st_dev
        or named_identity.st_ino != pin.identity.st_ino
        or stat_module.S_IMODE(pinned_identity.st_mode) != expected_mode
        or stat_module.S_IMODE(named_identity.st_mode) != expected_mode
    ):
        msg = f"Staged cohort directory identity changed: {pin.cohort}."
        raise D5DerivationError(msg)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    current_fd = os.open(pin.cohort, flags, dir_fd=release_root_fd)
    try:
        current_identity = os.fstat(current_fd)
        if (
            current_identity.st_dev != pin.identity.st_dev
            or current_identity.st_ino != pin.identity.st_ino
            or set(os.listdir(current_fd))  # noqa: PTH208
            != {OUTPUT_CSV_NAME, OUTPUT_MANIFEST_NAME}
        ):
            msg = f"Published cohort entry was substituted: {pin.cohort}."
            raise D5DerivationError(msg)
        for file_pin in (pin.csv_file, pin.manifest_file):
            named = os.stat(
                file_pin.name,
                dir_fd=current_fd,
                follow_symlinks=False,
            )
            if (
                not stat_module.S_ISREG(named.st_mode)
                or named.st_nlink != 1
                or named.st_dev != file_pin.identity.st_dev
                or named.st_ino != file_pin.identity.st_ino
                or stat_module.S_IMODE(named.st_mode) != (0o400 if frozen else 0o600)
            ):
                msg = (
                    "Published cohort file was substituted: "
                    f"{pin.cohort}/{file_pin.name}."
                )
                raise D5DerivationError(msg)
            _validate_pinned_file_descriptor(file_pin, frozen=frozen)
    finally:
        os.close(current_fd)


def _freeze_pinned_cohort(pin: _PinnedCohortDirectory) -> None:
    for file_pin in (pin.csv_file, pin.manifest_file):
        os.fchmod(file_pin.descriptor, 0o400)
        os.fsync(file_pin.descriptor)
    os.fchmod(pin.descriptor, 0o500)
    os.fsync(pin.descriptor)


def _freeze_pinned_file(pin: _PinnedPublicationFile) -> None:
    os.fchmod(pin.descriptor, 0o400)
    os.fsync(pin.descriptor)


def _validate_pinned_root_file_entry(
    release_root_fd: int,
    pin: _PinnedPublicationFile,
    *,
    frozen: bool,
) -> None:
    named = os.stat(
        pin.name,
        dir_fd=release_root_fd,
        follow_symlinks=False,
    )
    if (
        not stat_module.S_ISREG(named.st_mode)
        or named.st_nlink != 1
        or named.st_dev != pin.identity.st_dev
        or named.st_ino != pin.identity.st_ino
        or stat_module.S_IMODE(named.st_mode) != (0o400 if frozen else 0o600)
    ):
        msg = f"Published release file was substituted: {pin.name}."
        raise D5DerivationError(msg)
    _validate_pinned_file_descriptor(pin, frozen=frozen)


def _validate_complete_release_tree(  # noqa: PLR0913
    release_root_fd: int,
    expected_root_identity: os.stat_result,
    cohort_pins: Sequence[_PinnedCohortDirectory],
    root_file_pins: Sequence[_PinnedPublicationFile],
    expected_inventory: set[str],
    *,
    frozen: bool,
) -> None:
    root_identity = os.fstat(release_root_fd)
    if (
        not stat_module.S_ISDIR(root_identity.st_mode)
        or root_identity.st_dev != expected_root_identity.st_dev
        or root_identity.st_ino != expected_root_identity.st_ino
        or stat_module.S_IMODE(root_identity.st_mode) != (0o500 if frozen else 0o700)
        or set(os.listdir(release_root_fd)) != expected_inventory
    ):
        msg = "Published release root identity, mode, or inventory changed."
        raise D5DerivationError(msg)
    if [pin.cohort for pin in cohort_pins] != list(TCGA_COHORTS):
        msg = "Pinned cohort directories do not cover the canonical release grid."
        raise D5DerivationError(msg)
    if [pin.name for pin in root_file_pins] != [
        AUTHORITY_RECEIPT_NAME,
        RELEASE_MANIFEST_NAME,
    ]:
        msg = "Pinned release receipts do not cover the closed root inventory."
        raise D5DerivationError(msg)
    for pin in cohort_pins:
        _validate_pinned_cohort_entry(release_root_fd, pin, frozen=frozen)
    for pin in root_file_pins:
        _validate_pinned_root_file_entry(release_root_fd, pin, frozen=frozen)


def _close_pinned_cohort(pin: _PinnedCohortDirectory) -> None:
    os.close(pin.csv_file.descriptor)
    os.close(pin.manifest_file.descriptor)
    os.close(pin.descriptor)


def _require_same_directory_entry(
    parent_fd: int,
    name: str,
    expected: os.stat_result,
) -> None:
    observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat_module.S_ISDIR(observed.st_mode)
        or observed.st_dev != expected.st_dev
        or observed.st_ino != expected.st_ino
    ):
        msg = "Staging directory identity changed before publication."
        raise D5DerivationError(msg)


def _require_stable_publication_parent(
    lexical_parent: Path,
    canonical_parent: Path,
    parent_fd: int,
    expected: os.stat_result,
) -> None:
    try:
        resolved = lexical_parent.resolve(strict=True)
        canonical_identity = canonical_parent.stat(follow_symlinks=False)
        descriptor_identity = os.fstat(parent_fd)
    except OSError as error:
        msg = "Publication parent or ancestor changed during publication."
        raise D5DerivationError(msg) from error
    if (
        resolved != canonical_parent
        or not stat_module.S_ISDIR(canonical_identity.st_mode)
        or not stat_module.S_ISDIR(descriptor_identity.st_mode)
        or canonical_identity.st_dev != expected.st_dev
        or canonical_identity.st_ino != expected.st_ino
        or descriptor_identity.st_dev != expected.st_dev
        or descriptor_identity.st_ino != expected.st_ino
    ):
        msg = "Publication parent or ancestor identity changed."
        raise D5DerivationError(msg)


def _require_destination_outside_source_roots(
    destination: Path,
    source_roots: Sequence[Path],
) -> None:
    for source_root in source_roots:
        try:
            destination.relative_to(source_root)
        except ValueError:
            continue
        msg = (
            "Production output root resolves inside an immutable input root; "
            "path aliases are forbidden."
        )
        raise D5DerivationError(msg)


def _rename_no_replace_at(
    parent_fd: int,
    source_name: str,
    destination_name: str,
) -> None:
    """Atomically rename one sibling directory without replacing any entry."""
    libc = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
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
        result = rename(parent_fd, source, parent_fd, destination, 0x00000004)
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
        result = rename(parent_fd, source, parent_fd, destination, 0x00000001)
    else:
        msg = "Platform lacks an atomic no-replace rename primitive."
        raise OSError(errno.ENOTSUP, msg, destination_name)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            destination_name,
        )
    raise OSError(error_number, os.strerror(error_number), destination_name)


def _validate_axis(axis: FrozenFamilyAxis) -> None:
    if not isinstance(axis, FrozenFamilyAxis):
        msg = "axis must be a FrozenFamilyAxis."
        raise D5DerivationError(msg)
    if (
        not isinstance(axis.cohort, str)
        or _COHORT_PATTERN.fullmatch(axis.cohort) is None
    ):
        msg = "Cohort must use the canonical uppercase identifier vocabulary."
        raise D5DerivationError(msg)
    features = axis.ordered_features
    if not isinstance(features, tuple) or len(features) < 2:
        msg = "Frozen feature axis must contain at least two features."
        raise D5DerivationError(msg)
    for feature in features:
        if (
            not isinstance(feature, str)
            or feature != feature.strip()
            or _FEATURE_PATTERN.fullmatch(feature) is None
        ):
            msg = "Frozen feature axis contains an invalid mutation-event feature."
            raise D5DerivationError(msg)
    if len(set(features)) != len(features):
        msg = "Frozen feature axis must contain unique features."
        raise D5DerivationError(msg)
    _require_sha256(axis.cohort_contract_sha256, label="cohort contract")
    _require_sha256(axis.ordered_features_sha256, label="ordered feature axis")
    _require_sha256(axis.ordered_pair_sha256, label="ordered pair axis")
    if sequence_sha256(features) != axis.ordered_features_sha256:
        msg = "Frozen ordered-feature digest does not match its feature axis."
        raise D5DerivationError(msg)
    expected_pairs = tuple(iter_tested_pairs(features))
    if ordered_pair_sha256(expected_pairs) != axis.ordered_pair_sha256:
        msg = "Frozen ordered-pair digest does not match its feature axis."
        raise D5DerivationError(msg)


def _validate_policy(policy: ConjunctionMultiplicityPolicy) -> None:
    try:
        validate_policy(policy)
    except RevisionInferenceError as error:
        raise D5DerivationError(str(error)) from error


def _validate_marginal_validity(evidence: MarginalValidityEvidence) -> None:
    if not isinstance(evidence, MarginalValidityEvidence):
        msg = "Marginal-validity evidence must use the typed evidence contract."
        raise D5DerivationError(msg)
    if (
        not isinstance(evidence.status, str)
        or evidence.status not in MARGINAL_VALIDITY_STATUSES
    ):
        msg = "Marginal-validity evidence has an unsupported status."
        raise D5DerivationError(msg)
    if (evidence.evidence_id is None) != (evidence.artifact_sha256 is None):
        msg = "Marginal-validity evidence identifier and digest must travel together."
        raise D5DerivationError(msg)
    if evidence.status == MARGINAL_VALIDITY_CERTIFIED and evidence.evidence_id is None:
        msg = "Certified marginal validity requires immutable evidence provenance."
        raise D5DerivationError(msg)
    if evidence.status == MARGINAL_VALIDITY_CERTIFIED and (
        evidence._seal is not _EVIDENCE_SEAL  # noqa: SLF001
        or evidence.artifact_bytes is None
    ):
        msg = "Certified marginal validity lacks an authenticated artifact receipt."
        raise D5DerivationError(msg)
    if evidence.evidence_id is not None:
        if (
            not isinstance(evidence.evidence_id, str)
            or not evidence.evidence_id
            or evidence.evidence_id != evidence.evidence_id.strip()
            or any(ord(character) < 0x20 for character in evidence.evidence_id)
        ):
            msg = "Marginal-validity evidence identifier is invalid."
            raise D5DerivationError(msg)
        _require_sha256(evidence.artifact_sha256, label="marginal-validity artifact")
        if (
            evidence.artifact_bytes is not None
            and hashlib.sha256(
                evidence.artifact_bytes,
            ).hexdigest()
            != evidence.artifact_sha256
        ):
            msg = "Marginal-validity evidence bytes do not match their digest."
            raise D5DerivationError(msg)


def _validate_provider_mapping(component_pairwise_csv: Mapping[str, bytes]) -> None:
    if not isinstance(component_pairwise_csv, Mapping):
        msg = "Component pairwise inputs must be a provider mapping."
        raise D5DerivationError(msg)
    if set(component_pairwise_csv) != set(BMRS) or len(component_pairwise_csv) != len(
        BMRS,
    ):
        msg = "D5 requires exactly one cbase, dig, and mutsig component table."
        raise D5DerivationError(msg)


def _parse_component_csv(
    raw: bytes,
    *,
    provider: str,
    expected_pairs: Sequence[tuple[str, str]],
) -> tuple[_ComponentRow, ...]:
    if not raw:
        msg = f"{provider} pairwise input is empty."
        raise D5DerivationError(msg)
    rows: list[_ComponentRow] = []
    seen: set[tuple[str, str]] = set()
    sample_count: int | None = None
    try:
        stream = io.TextIOWrapper(
            io.BytesIO(raw),
            encoding="utf-8",
            errors="strict",
            newline="",
        )
        reader = csv.reader(stream, strict=True)
        header = next(reader, None)
        if tuple(header or ()) != PAIRWISE_COLUMNS:
            msg = f"{provider} pairwise input has an unexpected schema."
            raise D5DerivationError(msg)
        for row_index, expected_pair in enumerate(expected_pairs):
            values = next(reader, None)
            if values is None:
                msg = f"{provider} pairwise family ended before row {row_index}."
                raise D5DerivationError(msg)
            if len(values) != len(PAIRWISE_COLUMNS):
                msg = f"{provider} pairwise row {row_index} has invalid width."
                raise D5DerivationError(msg)
            row = dict(zip(PAIRWISE_COLUMNS, values, strict=True))
            coordinate = (row["Gene A"], row["Gene B"])
            if coordinate in seen:
                msg = f"{provider} pairwise family contains a duplicate coordinate."
                raise D5DerivationError(msg)
            seen.add(coordinate)
            if coordinate != expected_pair:
                msg = f"{provider} pairwise coordinate mismatch at row {row_index}."
                raise D5DerivationError(msg)
            component, observed_sample_count = _parse_component_row(
                row,
                provider=provider,
                row_index=row_index,
            )
            if sample_count is None:
                sample_count = observed_sample_count
            elif observed_sample_count != sample_count:
                msg = f"{provider} pairwise rows do not share one sample universe."
                raise D5DerivationError(msg)
            rows.append(component)
        extra = next(reader, None)
        if extra is not None:
            msg = f"{provider} pairwise family contains rows outside the frozen axis."
            raise D5DerivationError(msg)
    except (csv.Error, UnicodeDecodeError) as error:
        msg = f"{provider} pairwise input is not canonical UTF-8 CSV."
        raise D5DerivationError(msg) from error
    if sample_count is None or sample_count <= 0:
        msg = f"{provider} pairwise family has no supported observations."
        raise D5DerivationError(msg)
    return tuple(rows)


def _parse_component_row(
    row: Mapping[str, str],
    *,
    provider: str,
    row_index: int,
) -> tuple[_ComponentRow, int]:
    label = f"{provider} row {row_index}"
    if row["Fit Converged"] != "True":
        msg = f"{label} is not a converged publishable fit."
        raise D5DerivationError(msg)
    if (
        row["Fit Algorithm"] != REQUIRED_PAIR_FIT_CONTRACT
        or row["Pair Fit Contract"] != REQUIRED_PAIR_FIT_CONTRACT
        or row["LRT Contract"] != REQUIRED_LRT_CONTRACT
    ):
        msg = f"{label} has invalid fit/LRT provenance."
        raise D5DerivationError(msg)

    fit_iterations = _parse_integer(row["Fit Iterations"], label=f"{label} iterations")
    last_gain = _parse_finite_float(row["Fit Last LL Gain"], label=f"{label} gain")
    fixed_point = _parse_finite_float(
        row["Fit Fixed-Point Residual"],
        label=f"{label} fixed-point residual",
    )
    kkt = _parse_finite_float(row["Fit KKT Residual"], label=f"{label} KKT residual")
    if (
        fit_iterations < 0
        or fit_iterations > REQUIRED_PAIR_FIT_MAX_ITER
        or (fit_iterations == 0 and last_gain != 0)
        or last_gain < 0
        or fixed_point < 0
        or fixed_point > REQUIRED_PAIR_FIT_KKT_TOL
        or kkt < 0
        or kkt > REQUIRED_PAIR_FIT_KKT_TOL
    ):
        msg = f"{label} violates the certified convergence contract."
        raise D5DerivationError(msg)

    taus = tuple(
        _parse_finite_float(row[column], label=f"{label} {column}")
        for column in ("Tau_00", "Tau_10", "Tau_01", "Tau_11")
    )
    if (
        any(value < -REQUIRED_PAIR_SIMPLEX_TOL for value in taus)
        or any(value > 1 + REQUIRED_PAIR_SIMPLEX_TOL for value in taus)
        or not math.isclose(
            math.fsum(taus),
            1.0,
            rel_tol=0,
            abs_tol=REQUIRED_PAIR_SIMPLEX_TOL,
        )
    ):
        msg = f"{label} violates the fitted tau simplex."
        raise D5DerivationError(msg)

    null_ll = _parse_finite_float(
        row["Null Log Likelihood"],
        label=f"{label} null likelihood",
    )
    alternative_ll = _parse_finite_float(
        row["Alternative Log Likelihood"],
        label=f"{label} alternative likelihood",
    )
    likelihood_ratio = _parse_finite_float(
        row["Likelihood Ratio"],
        label=f"{label} likelihood ratio",
    )
    expected_likelihood_ratio = max(0.0, 2.0 * (alternative_ll - null_ll))
    if (
        likelihood_ratio < 0
        or alternative_ll + REQUIRED_LRT_NESTEDNESS_TOL < null_ll
        or not math.isclose(
            likelihood_ratio,
            expected_likelihood_ratio,
            rel_tol=0,
            abs_tol=REQUIRED_OUTPUT_RECOMPUTATION_ATOL,
        )
    ):
        msg = f"{label} violates the profile-LRT contract."
        raise D5DerivationError(msg)

    effect_identifiability = row["Effect Identifiability"]
    valid_effect_statuses = {
        REQUIRED_PAIR_EFFECT_IDENTIFIED_STATUS,
        REQUIRED_PAIR_EFFECT_RANK_DEFICIENT_STATUS,
        REQUIRED_PAIR_EFFECT_UNDERFLOW_STATUS,
    }
    if effect_identifiability not in valid_effect_statuses:
        msg = f"{label} has an invalid effect-identifiability status."
        raise D5DerivationError(msg)

    inference = _component_inference(
        row,
        taus=taus,
        likelihood_ratio=likelihood_ratio,
        effect_identifiability=effect_identifiability,
        label=label,
    )
    contingency = tuple(
        _parse_integer(row[column], label=f"{label} {column}")
        for column in ("_00_", "_10_", "_01_", "_11_")
    )
    if any(value < 0 for value in contingency) or sum(contingency) <= 0:
        msg = f"{label} has an invalid observation-support table."
        raise D5DerivationError(msg)

    return (
        _ComponentRow(
            coordinate=(row["Gene A"], row["Gene B"]),
            status=inference.status,
            p_value=inference.p_value,
            direction=inference.direction,
            effect_identifiability=effect_identifiability,
            contingency=contingency,
        ),
        sum(contingency),
    )


def _component_inference(
    row: Mapping[str, str],
    *,
    taus: tuple[float, float, float, float],
    likelihood_ratio: float,
    effect_identifiability: str,
    label: str,
) -> ComponentInference:
    raw_rho = row["Rho"]
    if effect_identifiability != REQUIRED_PAIR_EFFECT_IDENTIFIED_STATUS:
        if any(
            row[field] != "" for field in REQUIRED_NONIDENTIFIED_EFFECT_BLANK_FIELDS
        ):
            msg = f"{label} reports an effect field for an unidentified effect."
            raise D5DerivationError(msg)
        return _classify_component(
            likelihood_ratio,
            effect_identifiability,
            None,
            label=label,
        )

    tau_00, tau_10, tau_01, tau_11 = taus
    tau_1x = _parse_finite_float(row["Tau_1X"], label=f"{label} Tau_1X")
    tau_x1 = _parse_finite_float(row["Tau_X1"], label=f"{label} Tau_X1")
    if not math.isclose(
        tau_1x,
        tau_10 + tau_11,
        rel_tol=0,
        abs_tol=REQUIRED_OUTPUT_RECOMPUTATION_ATOL,
    ) or not math.isclose(
        tau_x1,
        tau_01 + tau_11,
        rel_tol=0,
        abs_tol=REQUIRED_OUTPUT_RECOMPUTATION_ATOL,
    ):
        msg = f"{label} has invalid identified-effect marginals."
        raise D5DerivationError(msg)

    expected_rho = compute_marshall_olkin_rho(
        [tau_00, tau_01, tau_10, tau_11],
    )
    if expected_rho is None:
        if raw_rho != "" or likelihood_ratio > REQUIRED_UNDEFINED_RHO_LRT_TOL:
            msg = f"{label} has an invalid undefined-rho boundary."
            raise D5DerivationError(msg)
        return _classify_component(
            likelihood_ratio,
            effect_identifiability,
            None,
            label=label,
        )
    rho = _parse_finite_float(raw_rho, label=f"{label} rho")
    if abs(rho) > 1 + REQUIRED_PAIR_SIMPLEX_TOL or not math.isclose(
        rho,
        expected_rho,
        rel_tol=0,
        abs_tol=REQUIRED_OUTPUT_RECOMPUTATION_ATOL,
    ):
        msg = f"{label} reports rho inconsistent with its fitted taus."
        raise D5DerivationError(msg)
    return _classify_component(
        likelihood_ratio,
        effect_identifiability,
        rho,
        label=label,
    )


def _classify_component(
    likelihood_ratio: float,
    effect_identifiability: str,
    rho: float | None,
    *,
    label: str,
) -> ComponentInference:
    try:
        return classify_component(likelihood_ratio, effect_identifiability, rho)
    except RevisionInferenceError as error:
        msg = f"{label} cannot form a valid D5 component: {error}"
        raise D5DerivationError(msg) from error


def _validate_cross_component_coordinates(
    component_rows: Mapping[str, tuple[_ComponentRow, ...]],
    expected_pairs: Sequence[tuple[str, str]],
) -> None:
    for row_index, coordinate in enumerate(expected_pairs):
        records = [component_rows[provider][row_index] for provider in BMRS]
        if any(record.coordinate != coordinate for record in records):
            msg = "Provider components do not share the frozen pair coordinates."
            raise D5DerivationError(msg)
        if any(record.contingency != records[0].contingency for record in records[1:]):
            msg = "Provider components do not share one observed sample universe."
            raise D5DerivationError(msg)


def _derive_output_rows(  # noqa: PLR0913
    component_rows: Mapping[str, tuple[_ComponentRow, ...]],
    *,
    axis: FrozenFamilyAxis,
    policy: ConjunctionMultiplicityPolicy,
    source_sha256: Mapping[str, str],
    marginal_validity: MarginalValidityEvidence,
    d3_conjunction_role: str,
) -> list[dict[str, object]]:
    components = {
        provider: tuple(
            ComponentInference(
                status=row.status,
                p_value=row.p_value,
                direction=row.direction,
                effect_identifiability=row.effect_identifiability,
            )
            for row in component_rows[provider]
        )
        for provider in BMRS
    }
    try:
        family = derive_complete_family(
            components,
            policy=policy,
            marginal_validity_certified=(marginal_validity.inferentially_eligible),
        )
    except RevisionInferenceError as error:
        raise D5DerivationError(str(error)) from error
    rows: list[dict[str, object]] = []
    for index, (gene_a, gene_b) in enumerate(
        iter_tested_pairs(axis.ordered_features),
    ):
        records = {provider: component_rows[provider][index] for provider in BMRS}
        by_q = family.by_q_values[index]
        bh_q = family.bh_q_values[index]
        rows.append(
            {
                "schema": POSTPROCESS_SCHEMA,
                "derivation_contract": DERIVATION_CONTRACT,
                "d5_contract": D5_CONTRACT,
                "cohort": axis.cohort,
                "gene_a": gene_a,
                "gene_b": gene_b,
                **{
                    f"{provider}_component_status": records[provider].status
                    for provider in BMRS
                },
                **{
                    f"{provider}_p_value": records[provider].p_value
                    for provider in BMRS
                },
                **{
                    f"{provider}_direction": records[provider].direction
                    for provider in BMRS
                },
                **{
                    f"{provider}_effect_identifiability": (
                        records[provider].effect_identifiability
                    )
                    for provider in BMRS
                },
                "conjunction_p_value": family.conjunction_p_values[index],
                "consensus_direction": family.consensus_directions[index],
                "d3_conjunction_role": d3_conjunction_role,
                "by_q_value": by_q,
                "bh_q_value": bh_q,
                "marginal_validity_status": marginal_validity.status,
                "conditional_by_inferential_eligible": (
                    family.primary_inferential_eligible
                ),
                "by_q_le_0_01": family.by_primary_threshold_crossings[index],
                "conditional_by_q_le_0_01_reportable": (
                    family.by_primary_reportable[index]
                ),
                "bh_q_le_0_01_nominal": (family.bh_nominal_threshold_crossings[index]),
                "by_q_le_0_05_descriptive": (
                    family.by_descriptive_threshold_crossings[index]
                ),
                "bh_q_le_0_05_descriptive": (
                    family.bh_descriptive_threshold_crossings[index]
                ),
                **{
                    f"{provider}_source_sha256": source_sha256[provider]
                    for provider in BMRS
                },
                "cohort_contract_sha256": axis.cohort_contract_sha256,
                "ordered_features_sha256": axis.ordered_features_sha256,
                "ordered_pair_sha256": axis.ordered_pair_sha256,
            },
        )
    return rows


def _serialize_rows(rows: Sequence[Mapping[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=OUTPUT_COLUMNS,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        serialized = {key: _serialize_cell(row[key]) for key in OUTPUT_COLUMNS}
        writer.writerow(serialized)
    return stream.getvalue().encode("utf-8")


def _serialize_cell(value: object) -> object:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            msg = "Derived output contains a non-finite value."
            raise D5DerivationError(msg)
        return format(value, ".17g")
    return value


def _build_manifest(  # noqa: PLR0913
    *,
    axis: FrozenFamilyAxis,
    policy: ConjunctionMultiplicityPolicy,
    source_sha256: Mapping[str, str],
    row_count: int,
    csv_bytes: bytes,
    csv_sha256: str,
    marginal_validity: MarginalValidityEvidence,
    production_eligible: bool,
    provider_hierarchy: ProviderHierarchyPolicy | None,
    production_authority: Mapping[str, object] | None,
) -> dict[str, object]:
    return {
        "schema": POSTPROCESS_SCHEMA,
        "derivation_contract": DERIVATION_CONTRACT,
        "d5_contract": D5_CONTRACT,
        "cohort": axis.cohort,
        "production_eligible": production_eligible,
        "implementation": _implementation_provenance(),
        "production_authority": (
            None if production_authority is None else dict(production_authority)
        ),
        "d3_provider_hierarchy": (
            {
                "primary_provider": "cbase",
                "sensitivity_providers": ["dig", "mutsig"],
                "all_three_conjunction_role": CONJUNCTION_SECONDARY,
                "burden_dependent_switching": False,
                "synthetic_qa_default": True,
            }
            if provider_hierarchy is None
            else {
                **asdict(provider_hierarchy),
                "sensitivity_providers": list(
                    provider_hierarchy.sensitivity_providers,
                ),
                "synthetic_qa_default": False,
            }
        ),
        "tested_family": {
            "top_k": policy.tested_family.top_k,
            "feature_ranking": policy.tested_family.feature_ranking,
            "tie_break": policy.tested_family.tie_break,
            "provider_support": policy.tested_family.provider_support,
            "pair_construction": policy.tested_family.pair_construction,
            "same_base_missense_nonsense": (
                policy.tested_family.same_base_missense_nonsense
            ),
            "epsilon_pretest_filter": (policy.tested_family.epsilon_pretest_filter),
            "marginal_effect_pretest_filter": (
                policy.tested_family.marginal_effect_pretest_filter
            ),
            "family": policy.tested_family.family,
        },
        "complete_family_required": True,
        "pair_filtering": False,
        "pair_ranking": False,
        "component_order": list(BMRS),
        "valid_component_statuses": list(VALID_CONJUNCTION_COMPONENT_STATUSES),
        "component_failure_semantics": COMPONENT_FAILURE_SEMANTICS,
        "p_value_combiner": CONJUNCTION_P_VALUE_COMBINER,
        "direction": {
            "provider_rule": DIRECTION_PROVIDER_RULE,
            "undefined_rho_rule": UNDEFINED_RHO_DIRECTION_RULE,
            "consensus_rule": DIRECTION_CONSENSUS_RULE,
            "reporting_layer": DIRECTION_REPORTING_LAYER,
            "directional_fdr_control": False,
        },
        "multiplicity": {
            "computed_methods": ["by", "bh"],
            "primary_method": "by",
            "primary_q_threshold": PRIMARY_Q_THRESHOLD,
            "primary_reporting_layer": PRIMARY_REPORTING_LAYER,
            "sensitivity_method": "bh",
            "sensitivity_q_threshold": SENSITIVITY_Q_THRESHOLD,
            "sensitivity_reporting_layer": SENSITIVITY_REPORTING_LAYER,
            "descriptive_methods": list(
                policy.multiplicity.descriptive_methods,
            ),
            "descriptive_q_threshold": DESCRIPTIVE_Q_THRESHOLD,
            "descriptive_reporting_layer": DESCRIPTIVE_REPORTING_LAYER,
            "threshold_comparison": INCLUSIVE_THRESHOLD,
        },
        "marginal_validity": {
            "status": marginal_validity.status,
            "conditional_by_inferential_eligible": (
                marginal_validity.inferentially_eligible
            ),
            "evidence_id": marginal_validity.evidence_id,
            "artifact_sha256": marginal_validity.artifact_sha256,
            "correction_selection_affected": False,
            "q_values_affected": False,
        },
        "axis": {
            "feature_count": len(axis.ordered_features),
            "ordered_features_sha256": axis.ordered_features_sha256,
            "pair_count": row_count,
            "ordered_pair_sha256": axis.ordered_pair_sha256,
            "cohort_contract_sha256": axis.cohort_contract_sha256,
        },
        "components": {
            provider: {
                "pairwise_sha256": source_sha256[provider],
                "raw_schema": list(PAIRWISE_COLUMNS),
            }
            for provider in BMRS
        },
        "output": {
            "name": OUTPUT_CSV_NAME,
            "bytes": len(csv_bytes),
            "rows": row_count,
            "sha256": csv_sha256,
            "columns": list(OUTPUT_COLUMNS),
        },
    }


def _implementation_provenance() -> dict[str, object]:
    files = {
        "analysis/postprocess_tcga_revision_k500.py": Path(__file__).resolve(),
        "src/dialect/data/revision_fit_policy.py": Path(
            revision_fit_policy_module.__file__,
        ).resolve(),
        "src/dialect/stats/revision_inference.py": Path(
            revision_inference_module.__file__,
        ).resolve(),
    }
    hashes = {
        label: hashlib.sha256(path.read_bytes()).hexdigest()
        for label, path in files.items()
    }
    return {
        "files": hashes,
        "combined_sha256": hashlib.sha256(_canonical_json(hashes)).hexdigest(),
    }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _parse_finite_float(value: str, *, label: str) -> float:
    if not value or value != value.strip():
        msg = f"{label} must be a canonical finite number."
        raise D5DerivationError(msg)
    try:
        parsed = float(value)
    except ValueError as error:
        msg = f"{label} must be a canonical finite number."
        raise D5DerivationError(msg) from error
    if not math.isfinite(parsed):
        msg = f"{label} must be a canonical finite number."
        raise D5DerivationError(msg)
    return parsed


def _parse_integer(value: str, *, label: str) -> int:
    if (
        not value
        or value != value.strip()
        or re.fullmatch(r"0|[1-9][0-9]*", value) is None
    ):
        msg = f"{label} must be a canonical non-negative integer."
        raise D5DerivationError(msg)
    return int(value)


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        msg = f"{label} digest must be a lowercase SHA-256."
        raise D5DerivationError(msg)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output-root", type=Path, required=True)
    parser.add_argument("--canonical-input-root", type=Path, required=True)
    parser.add_argument("--provider-input-root", type=Path, required=True)
    parser.add_argument("--input-approval-manifest", type=Path, required=True)
    parser.add_argument("--fit-approval-manifest", type=Path, required=True)
    parser.add_argument("--inspect-approval-manifest", type=Path, required=True)
    parser.add_argument("--calibration-approval-manifest", type=Path)
    parser.add_argument("--calibration-evidence-artifact", type=Path)
    parser.add_argument("--calibration-design-manifest", type=Path)
    parser.add_argument(
        "--marginal-validity-status",
        choices=tuple(sorted(MARGINAL_VALIDITY_STATUSES)),
        required=True,
    )
    parser.add_argument("--expected-sealed-completion-sha256", required=True)
    parser.add_argument("--expected-canonical-input-sha256", required=True)
    parser.add_argument(
        "--expected-provider-input-manifest-sha256",
        required=True,
    )
    parser.add_argument("--expected-input-approval-sha256", required=True)
    parser.add_argument("--expected-fit-approval-sha256", required=True)
    parser.add_argument("--expected-inspect-approval-sha256", required=True)
    parser.add_argument("--expected-calibration-approval-sha256")
    parser.add_argument("--expected-calibration-evidence-sha256")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    """Run the sealed result-bound production postprocessor."""
    args = _parser().parse_args()

    def absolute(path: Path) -> Path:
        return Path(os.path.abspath(path))  # noqa: PTH100

    def optional_absolute(path: Path | None) -> Path | None:
        return None if path is None else absolute(path)

    config = ProductionPostprocessConfig(
        run_output_root=absolute(args.run_output_root),
        canonical_input_root=absolute(args.canonical_input_root),
        provider_input_root=absolute(args.provider_input_root),
        input_approval_manifest=absolute(args.input_approval_manifest),
        fit_approval_manifest=absolute(args.fit_approval_manifest),
        inspect_approval_manifest=absolute(args.inspect_approval_manifest),
        calibration_approval_manifest=optional_absolute(
            args.calibration_approval_manifest,
        ),
        calibration_evidence_artifact=optional_absolute(
            args.calibration_evidence_artifact,
        ),
        calibration_design_manifest=optional_absolute(
            args.calibration_design_manifest,
        ),
        marginal_validity_status=args.marginal_validity_status,
        expected_sealed_completion_sha256=(args.expected_sealed_completion_sha256),
        expected_canonical_input_sha256=args.expected_canonical_input_sha256,
        expected_provider_input_manifest_sha256=(
            args.expected_provider_input_manifest_sha256
        ),
        expected_input_approval_sha256=args.expected_input_approval_sha256,
        expected_fit_approval_sha256=args.expected_fit_approval_sha256,
        expected_inspect_approval_sha256=args.expected_inspect_approval_sha256,
        expected_calibration_approval_sha256=(
            args.expected_calibration_approval_sha256
        ),
        expected_calibration_evidence_sha256=(
            args.expected_calibration_evidence_sha256
        ),
    )
    result = run_production_postprocess(
        config,
        absolute(args.output_root),
    )
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
