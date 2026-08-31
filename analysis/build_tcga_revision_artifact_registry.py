"""Build a result-blind registry for conditional revision artifacts.

The registry is a provenance and release-control layer, not a scientific-data
transform.  It accepts one small, canonical JSON reconciliation document and
never accepts or opens source-data tables.  Source data are referenced only by
release-relative member names, byte sizes, and SHA-256 digests.  Renderer
scripts and rendered outputs are opened as opaque byte streams solely to verify
their declared digests.

Every planned revision artifact has a stable semantic identifier.  An artifact
must take exactly one of two branches:

* ``ready``: all required gate receipts, source-data bindings, a verified
  renderer, rendered outputs, and a fixed result-blind claim vocabulary; or
* ``omitted``: an exact missing-gate accounting or a digest-bound release-scope
  decision to omit it.

The publication is canonical, deterministic, atomic, and no-replace.  No
figure or table number is assigned here; numbering remains a later manuscript
layout decision.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Final

# Validation failures intentionally carry precise field paths.  Keeping those
# messages at the validation site is substantially clearer than a large family
# of one-use exception subclasses.
# ruff: noqa: EM101, EM102, TRY003, TRY301

RECONCILIATION_INPUT_SCHEMA: Final = (
    "dialect-revision-artifact-reconciliation-input-v1"
)
ARTIFACT_REGISTRY_SCHEMA: Final = "dialect-revision-artifact-registry-v1"
ARTIFACT_REGISTRY_CONTRACT: Final = (
    "semantic-artifact-gates-source-to-render-reconciliation-v1"
)
ARTIFACT_CATALOG_SCHEMA: Final = "dialect-revision-semantic-artifact-catalog-v1"

GATE_ORDER: Final = ("K500", "CAL", "COAUTH", "COMP", "MSK", "SIM")
GATE_CATALOG: Final = (
    {
        "gate": "K500",
        "meaning": "declared digest reference for whole-grid K=500 execution",
    },
    {
        "gate": "CAL",
        "meaning": "declared digest reference for calibration evidence and policy",
    },
    {
        "gate": "COAUTH",
        "meaning": "declared digest reference for a coauthor presentation decision",
    },
    {
        "gate": "COMP",
        "meaning": "declared digest reference for comparator-method evidence",
    },
    {
        "gate": "MSK",
        "meaning": "declared digest reference for MSK validation evidence",
    },
    {
        "gate": "SIM",
        "meaning": "declared digest reference for simulation design and evidence",
    },
)
TRUST_MODEL: Final = {
    "gate_receipts": "declared SHA-256 references; receipt payloads are not opened",
    "source_data": "declared metadata references; row-bearing sources are not opened",
    "render_inputs": (
        "builder, renderer, and output bytes are opened and SHA-256 checked"
    ),
    "scientific_scope": "no result interpretation or scientific approval is inferred",
}
_BUILDER_SCRIPT_MEMBER: Final = (
    "analysis/build_tcga_revision_artifact_registry.py"
)

_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_TOKEN_PATTERN: Final = re.compile(r"[a-z0-9][a-z0-9._:/-]{1,127}")
_RELEASE_ID_PATTERN: Final = re.compile(r"[a-z0-9][a-z0-9._-]{2,127}")
_READ_CHUNK_BYTES: Final = 1024 * 1024
_MAX_RECONCILIATION_BYTES: Final = 1024 * 1024

_OUTPUT_MEDIA_SUFFIXES: Final = {
    "application/json": (".json",),
    "application/pdf": (".pdf",),
    "application/x-tex": (".tex",),
    "image/png": (".png",),
    "image/svg+xml": (".svg",),
    "image/tiff": (".tif", ".tiff"),
    "text/csv": (".csv",),
}
_REQUIRED_MEDIA_BY_KIND: Final = {
    "figure": frozenset(
        {
            "application/pdf",
            "image/png",
            "image/svg+xml",
            "image/tiff",
        },
    ),
    "table": frozenset({"application/pdf", "application/x-tex"}),
    "supplementary-data": frozenset({"application/json", "text/csv"}),
    "provenance-record": frozenset({"application/json"}),
}
_SOURCE_ROLES: Final = frozenset(
    {
        "primary",
        "calibration",
        "comparison",
        "simulation",
        "validation",
        "runtime",
        "provenance",
    },
)
_UPSTREAM_MANIFEST_MEMBERS: Final = {
    "postprocess-release-manifest": "postprocess/release_manifest.json",
    "source-data-manifest": "source-data/source_data_manifest.json",
}


class ArtifactRegistryError(ValueError):
    """Raised when an artifact reconciliation cannot be validated."""


@dataclass(frozen=True, slots=True)
class ClaimSpec:
    """A result-blind semantic claim bound to an artifact."""

    claim_id: str
    scope: str

    def as_record(self) -> dict[str, str]:
        """Return the canonical claim record."""
        return {"claim_id": self.claim_id, "scope": self.scope}


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    """Frozen semantic identity and release gates for one artifact."""

    semantic_id: str
    title: str
    kind: str
    required_gates: tuple[str, ...]
    required_source_roles: tuple[str, ...]
    claims: tuple[ClaimSpec, ...]
    source_requirement: str = "required"

    def catalog_record(self) -> dict[str, object]:
        """Return this artifact's canonical catalog record."""
        return {
            "semantic_id": self.semantic_id,
            "title": self.title,
            "kind": self.kind,
            "required_gates": list(self.required_gates),
            "required_source_roles": list(self.required_source_roles),
            "claims": [claim.as_record() for claim in self.claims],
            "source_requirement": self.source_requirement,
        }


ARTIFACT_SPECS: Final = (
    ArtifactSpec(
        semantic_id="cross_cancer_bmr_co_sensitivity",
        title="Cross-cancer BMR and co-occurrence sensitivity",
        kind="figure",
        required_gates=("K500", "CAL", "COAUTH"),
        required_source_roles=("primary", "calibration"),
        claims=(
            ClaimSpec(
                "cross_cancer_bmr_robustness",
                "provider sensitivity across the TCGA cohort grid",
            ),
            ClaimSpec(
                "co_occurrence_burden_context",
                "co-occurrence behavior under per-sample burden control",
            ),
        ),
    ),
    ArtifactSpec(
        semantic_id="interaction_model_diagnostic_panels",
        title="Interaction-model diagnostic panels",
        kind="figure",
        required_gates=("K500", "CAL", "COAUTH"),
        required_source_roles=("calibration",),
        claims=(
            ClaimSpec(
                "interaction_null_calibration",
                "null calibration of the interaction test",
            ),
            ClaimSpec(
                "interaction_recovery_behavior",
                "recovery behavior under declared diagnostic settings",
            ),
        ),
    ),
    ArtifactSpec(
        semantic_id="selected_pair_biological_validation",
        title="Selected-pair biological validation",
        kind="figure",
        required_gates=("K500", "CAL", "COAUTH", "MSK"),
        required_source_roles=("primary", "calibration", "validation"),
        claims=(
            ClaimSpec(
                "selected_pair_biological_context",
                "biological context for a coauthor-declared selected pair",
            ),
            ClaimSpec(
                "selected_pair_independent_validation",
                "independent validation status for the selected pair",
            ),
        ),
    ),
    ArtifactSpec(
        semantic_id="simulation_method_comparison",
        title="Simulation method comparison",
        kind="figure",
        required_gates=("CAL", "COAUTH", "COMP", "SIM"),
        required_source_roles=("calibration", "comparison", "simulation"),
        claims=(
            ClaimSpec(
                "simulation_error_control",
                "error-control behavior in the declared simulation design",
            ),
            ClaimSpec(
                "simulation_method_performance",
                "comparative method behavior in the declared simulation design",
            ),
        ),
    ),
    ArtifactSpec(
        semantic_id="interaction_summary",
        title="Interaction summary",
        kind="table",
        required_gates=("K500", "CAL", "COAUTH"),
        required_source_roles=("primary", "calibration"),
        claims=(
            ClaimSpec(
                "reported_interaction_inventory",
                "coauthor-declared inventory of reported interactions",
            ),
        ),
    ),
    ArtifactSpec(
        semantic_id="raw_supplementary_inventory",
        title="Raw supplementary interaction inventory",
        kind="supplementary-data",
        required_gates=("K500", "COAUTH"),
        required_source_roles=("primary",),
        claims=(
            ClaimSpec(
                "complete_k500_family_availability",
                "availability of complete K=500 tested families",
            ),
        ),
    ),
    ArtifactSpec(
        semantic_id="provider_conjunction_summary",
        title="Provider and conjunction summary",
        kind="table",
        required_gates=("K500", "CAL", "COAUTH"),
        required_source_roles=("primary", "calibration"),
        claims=(
            ClaimSpec(
                "provider_robustness_context",
                "relationship among declared BMR-provider analyses",
            ),
            ClaimSpec(
                "conjunction_decision_rule",
                "declared conjunction and multiplicity rule",
            ),
        ),
    ),
    ArtifactSpec(
        semantic_id="comparator_benchmark",
        title="Comparator benchmark",
        kind="table",
        required_gates=("K500", "COAUTH", "COMP"),
        required_source_roles=("comparison",),
        claims=(
            ClaimSpec(
                "comparator_method_context",
                "declared comparison with external interaction methods",
            ),
        ),
    ),
    ArtifactSpec(
        semantic_id="calibration_diagnostics",
        title="Calibration diagnostics",
        kind="figure",
        required_gates=("CAL", "COAUTH"),
        required_source_roles=("calibration",),
        claims=(
            ClaimSpec(
                "calibration_diagnostic_accounting",
                "diagnostic accounting for the declared calibration analysis",
            ),
        ),
    ),
    ArtifactSpec(
        semantic_id="runtime_failure_summary",
        title="Runtime and failure summary",
        kind="table",
        required_gates=("K500",),
        required_source_roles=("runtime",),
        claims=(
            ClaimSpec(
                "execution_completeness",
                "whole-grid task completion and runtime accounting",
            ),
            ClaimSpec(
                "failure_accounting",
                "explicit accounting of task failures and exclusions",
            ),
        ),
    ),
    ArtifactSpec(
        semantic_id="msk_validation",
        title="MSK validation",
        kind="table",
        required_gates=("K500", "CAL", "COAUTH", "MSK"),
        required_source_roles=("primary", "calibration", "validation"),
        claims=(
            ClaimSpec(
                "independent_msk_validation",
                "independent MSK validation under its declared target scope",
            ),
        ),
    ),
    ArtifactSpec(
        semantic_id="method_overview",
        title="DIALECT method overview",
        kind="figure",
        required_gates=("COAUTH",),
        required_source_roles=(),
        claims=(
            ClaimSpec(
                "latent_driver_background_decomposition",
                "DIALECT decomposition of observed counts into background and "
                "latent-driver components",
            ),
            ClaimSpec(
                "interaction_testing_workflow",
                "DIALECT workflow for testing pairwise interaction modes",
            ),
        ),
        source_requirement="none",
    ),
    ArtifactSpec(
        semantic_id="release_provenance",
        title="Release provenance",
        kind="provenance-record",
        required_gates=("COAUTH",),
        required_source_roles=("provenance",),
        claims=(
            ClaimSpec(
                "release_provenance_and_reproducibility",
                "source-to-render traceability for the revision release",
            ),
        ),
        source_requirement="upstream-manifest",
    ),
)

_SPEC_BY_ID: Final = {spec.semantic_id: spec for spec in ARTIFACT_SPECS}

_OMISSION_REASONS: Final = {
    "required_gate_not_satisfied": (
        "One or more required release gates lack declared receipts."
    ),
    "coauthor_decision_to_omit": (
        "All required gates have declared receipts and the coauthors elected to omit "
        "this artifact from the release."
    ),
    "release_scope_exclusion": (
        "All required gates have declared receipts, but this artifact is outside "
        "the selected release scope."
    ),
}


@dataclass(frozen=True, slots=True)
class ArtifactRegistryReceipt:
    """Digest-only receipt for a validated or newly published registry."""

    manifest_path: str
    manifest_sha256: str
    ready_count: int
    omitted_count: int


@dataclass(slots=True)
class _PinnedRoot:
    path: Path
    descriptor: int = field(repr=False)
    identity: os.stat_result = field(repr=False)

    def close(self) -> None:
        """Close the pinned root descriptor."""
        os.close(self.descriptor)


@dataclass(slots=True)
class _PinnedMember:
    root: _PinnedRoot
    member: str
    descriptor: int = field(repr=False)
    device: int
    inode: int
    size_bytes: int
    modified_ns: int
    sha256: str

    def close(self) -> None:
        """Close the pinned member descriptor."""
        os.close(self.descriptor)


@dataclass(slots=True)
class _PinnedMetadata:
    path: Path
    descriptor: int = field(repr=False)
    device: int
    inode: int
    size_bytes: int
    modified_ns: int
    sha256: str
    raw: bytes = field(repr=False)

    def close(self) -> None:
        """Close the pinned metadata descriptor."""
        os.close(self.descriptor)


@dataclass(frozen=True, slots=True)
class _PublishedDestinationExpectation:
    staged_identity: tuple[int, int]
    sha256: str
    size_bytes: int
    link_count: int


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _expect_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ArtifactRegistryError(f"{context} must be a JSON object")
    return value


def _expect_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ArtifactRegistryError(
            f"{context} has a non-closed schema; missing={missing}, extra={extra}",
        )


def _expect_sequence(value: object, *, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ArtifactRegistryError(f"{context} must be a JSON array")
    return value


def _expect_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactRegistryError(f"{context} must be a nonempty string")
    return value


def _expect_sha256(value: object, *, context: str) -> str:
    digest = _expect_string(value, context=context)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ArtifactRegistryError(f"{context} must be a lowercase SHA-256 digest")
    return digest


def _expect_size(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactRegistryError(f"{context} must be a nonnegative integer")
    return value


def _expect_token(value: object, *, context: str) -> str:
    token = _expect_string(value, context=context)
    if _TOKEN_PATTERN.fullmatch(token) is None:
        raise ArtifactRegistryError(f"{context} is not a valid semantic token")
    return token


def _expect_gate(value: object, *, context: str) -> str:
    gate = _expect_string(value, context=context)
    if gate not in GATE_ORDER:
        raise ArtifactRegistryError(f"{context} is not a recognized gate: {gate!r}")
    return gate


def _expect_relative_member(value: object, *, context: str) -> str:
    member = _expect_string(value, context=context)
    if (
        not member.isascii()
        or "\\" in member
        or any(ord(character) < 32 or ord(character) == 127 for character in member)
    ):
        raise ArtifactRegistryError(f"{context} is not a canonical POSIX member")
    path = PurePosixPath(member)
    if path.is_absolute() or path.as_posix() != member:
        raise ArtifactRegistryError(f"{context} is not a canonical relative member")
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ArtifactRegistryError(f"{context} escapes its declared release root")
    return member


def _catalog_payload() -> dict[str, object]:
    return {
        "schema": ARTIFACT_CATALOG_SCHEMA,
        "gates": [dict(record) for record in GATE_CATALOG],
        "artifacts": [spec.catalog_record() for spec in ARTIFACT_SPECS],
    }


def artifact_catalog_sha256() -> str:
    """Return the frozen semantic catalog digest."""
    return _sha256(_canonical_json(_catalog_payload()))


def _pin_root(path: Path, *, context: str) -> _PinnedRoot:
    absolute = path.absolute()
    try:
        entry = os.lstat(absolute)
    except OSError as exc:
        raise ArtifactRegistryError(f"cannot inspect {context}: {absolute}") from exc
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
        raise ArtifactRegistryError(f"{context} must be a non-symlink directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise ArtifactRegistryError(f"cannot pin {context}: {absolute}") from exc
    try:
        identity = os.fstat(descriptor)
    except OSError as exc:
        os.close(descriptor)
        raise ArtifactRegistryError(
            f"cannot inspect pinned {context}: {absolute}",
        ) from exc
    if (identity.st_dev, identity.st_ino) != (entry.st_dev, entry.st_ino):
        os.close(descriptor)
        raise ArtifactRegistryError(f"{context} changed while it was pinned")
    return _PinnedRoot(path=absolute, descriptor=descriptor, identity=identity)


def _digest_descriptor(descriptor: int) -> tuple[str, int, os.stat_result]:
    before = os.fstat(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = os.read(descriptor, _READ_CHUNK_BYTES)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    after = os.fstat(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or size != after.st_size:
        raise ArtifactRegistryError("a pinned file changed while it was hashed")
    return digest.hexdigest(), size, after


def _open_member_descriptor(root: _PinnedRoot, member: str) -> int:
    parts = PurePosixPath(member).parts
    directory_descriptor = os.dup(root.descriptor)
    try:
        for part in parts[:-1]:
            flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            next_descriptor = os.open(part, flags, dir_fd=directory_descriptor)
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        return os.open(parts[-1], flags, dir_fd=directory_descriptor)
    except OSError as exc:
        raise ArtifactRegistryError(
            f"cannot open pinned member {member!r} under {root.path}",
        ) from exc
    finally:
        os.close(directory_descriptor)


def _pin_member(root: _PinnedRoot, member: str, *, context: str) -> _PinnedMember:
    descriptor = _open_member_descriptor(root, member)
    try:
        identity = os.fstat(descriptor)
        if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
            raise ArtifactRegistryError(
                f"{context} must be a singly linked regular file",
            )
        digest, size, identity = _digest_descriptor(descriptor)
        return _PinnedMember(
            root=root,
            member=member,
            descriptor=descriptor,
            device=identity.st_dev,
            inode=identity.st_ino,
            size_bytes=size,
            modified_ns=identity.st_mtime_ns,
            sha256=digest,
        )
    except Exception:
        os.close(descriptor)
        raise


def _pin_metadata(path: Path, *, context: str) -> _PinnedMetadata:
    absolute = path.absolute()
    try:
        entry = os.lstat(absolute)
    except OSError as exc:
        raise ArtifactRegistryError(f"cannot inspect {context}: {absolute}") from exc
    if (
        stat.S_ISLNK(entry.st_mode)
        or not stat.S_ISREG(entry.st_mode)
        or entry.st_nlink != 1
    ):
        raise ArtifactRegistryError(f"{context} must be a singly linked regular file")
    if entry.st_size > _MAX_RECONCILIATION_BYTES:
        raise ArtifactRegistryError(f"{context} exceeds the metadata size limit")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise ArtifactRegistryError(f"cannot pin {context}: {absolute}") from exc
    try:
        raw = bytearray()
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > _MAX_RECONCILIATION_BYTES:
                raise ArtifactRegistryError(
                    f"{context} exceeds the metadata size limit",
                )
        identity = os.fstat(descriptor)
        if (
            (identity.st_dev, identity.st_ino) != (entry.st_dev, entry.st_ino)
            or identity.st_size != len(raw)
            or identity.st_mtime_ns != entry.st_mtime_ns
        ):
            raise ArtifactRegistryError(
                f"{context} changed while it was read",
            )
        raw_bytes = bytes(raw)
        return _PinnedMetadata(
            path=absolute,
            descriptor=descriptor,
            device=identity.st_dev,
            inode=identity.st_ino,
            size_bytes=identity.st_size,
            modified_ns=identity.st_mtime_ns,
            sha256=_sha256(raw_bytes),
            raw=raw_bytes,
        )
    except Exception:
        os.close(descriptor)
        raise


def _revalidate_root(root: _PinnedRoot, *, context: str) -> None:
    current = os.fstat(root.descriptor)
    try:
        entry = os.lstat(root.path)
    except OSError as exc:
        raise ArtifactRegistryError(f"{context} disappeared during validation") from exc
    expected = (root.identity.st_dev, root.identity.st_ino)
    if (
        not stat.S_ISDIR(entry.st_mode)
        or (current.st_dev, current.st_ino) != expected
        or (entry.st_dev, entry.st_ino) != expected
    ):
        raise ArtifactRegistryError(f"{context} changed during validation")


def _revalidate_member(member: _PinnedMember, *, context: str) -> None:
    digest, size, identity = _digest_descriptor(member.descriptor)
    if (
        (identity.st_dev, identity.st_ino) != (member.device, member.inode)
        or identity.st_mtime_ns != member.modified_ns
        or size != member.size_bytes
        or digest != member.sha256
    ):
        raise ArtifactRegistryError(f"{context} changed during validation")
    replacement = _pin_member(member.root, member.member, context=context)
    try:
        if (
            (replacement.device, replacement.inode) != (member.device, member.inode)
            or replacement.modified_ns != member.modified_ns
            or replacement.size_bytes != member.size_bytes
            or replacement.sha256 != member.sha256
        ):
            raise ArtifactRegistryError(f"{context} entry changed during validation")
    finally:
        replacement.close()


def _revalidate_metadata(metadata: _PinnedMetadata, *, context: str) -> None:
    digest, size, identity = _digest_descriptor(metadata.descriptor)
    try:
        entry = os.lstat(metadata.path)
    except OSError as exc:
        raise ArtifactRegistryError(f"{context} disappeared during validation") from exc
    if (
        (identity.st_dev, identity.st_ino) != (metadata.device, metadata.inode)
        or (entry.st_dev, entry.st_ino) != (metadata.device, metadata.inode)
        or identity.st_mtime_ns != metadata.modified_ns
        or entry.st_mtime_ns != metadata.modified_ns
        or size != metadata.size_bytes
        or digest != metadata.sha256
    ):
        raise ArtifactRegistryError(f"{context} changed during validation")


def _parse_canonical_metadata(
    metadata: _PinnedMetadata,
    *,
    context: str,
) -> Mapping[str, object]:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        parsed: dict[str, object] = {}
        for key, item in pairs:
            if key in parsed:
                raise ArtifactRegistryError(
                    f"{context} contains duplicate JSON key {key!r}",
                )
            parsed[key] = item
        return parsed

    def reject_nonfinite_constant(constant: str) -> object:
        raise ArtifactRegistryError(
            f"{context} contains non-finite JSON constant {constant!r}",
        )

    try:
        value = json.loads(
            metadata.raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ArtifactRegistryError(f"{context} is not valid UTF-8 JSON") from exc
    mapping = _expect_mapping(value, context=context)
    if metadata.raw != _canonical_json(mapping) + b"\n":
        raise ArtifactRegistryError(f"{context} is not canonical JSON with one newline")
    return mapping


def _normalize_release(value: object, *, context: str) -> dict[str, str]:
    release = _expect_mapping(value, context=context)
    _expect_keys(
        release,
        {
            "release_id",
            "postprocess_release_sha256",
            "source_data_manifest_sha256",
        },
        context=context,
    )
    release_id = _expect_string(release["release_id"], context=f"{context}.release_id")
    if _RELEASE_ID_PATTERN.fullmatch(release_id) is None:
        raise ArtifactRegistryError(f"{context}.release_id is not canonical")
    return {
        "release_id": release_id,
        "postprocess_release_sha256": _expect_sha256(
            release["postprocess_release_sha256"],
            context=f"{context}.postprocess_release_sha256",
        ),
        "source_data_manifest_sha256": _expect_sha256(
            release["source_data_manifest_sha256"],
            context=f"{context}.source_data_manifest_sha256",
        ),
    }


def _normalize_gate_receipts(
    value: object,
    *,
    permitted_gates: tuple[str, ...],
    require_all: bool,
    context: str,
) -> list[dict[str, str]]:
    records = _expect_sequence(value, context=context)
    by_gate: dict[str, dict[str, str]] = {}
    for index, raw_record in enumerate(records):
        record_context = f"{context}[{index}]"
        record = _expect_mapping(raw_record, context=record_context)
        _expect_keys(
            record,
            {"gate", "receipt_id", "sha256"},
            context=record_context,
        )
        gate = _expect_gate(record["gate"], context=f"{record_context}.gate")
        if gate not in permitted_gates:
            raise ArtifactRegistryError(
                f"{record_context} supplies non-required gate {gate!r}",
            )
        if gate in by_gate:
            raise ArtifactRegistryError(f"{context} duplicates gate {gate!r}")
        by_gate[gate] = {
            "gate": gate,
            "receipt_id": _expect_token(
                record["receipt_id"],
                context=f"{record_context}.receipt_id",
            ),
            "sha256": _expect_sha256(
                record["sha256"],
                context=f"{record_context}.sha256",
            ),
        }
    if require_all and set(by_gate) != set(permitted_gates):
        missing = [gate for gate in permitted_gates if gate not in by_gate]
        raise ArtifactRegistryError(f"{context} is missing required gates: {missing}")
    return [by_gate[gate] for gate in GATE_ORDER if gate in by_gate]


def _artifact_gate_receipts(
    ledger: Sequence[Mapping[str, str]],
    spec: ArtifactSpec,
) -> list[dict[str, str]]:
    required = set(spec.required_gates)
    return [dict(receipt) for receipt in ledger if receipt["gate"] in required]


def _normalize_sources(
    value: object,
    *,
    source_requirement: str,
    required_source_roles: tuple[str, ...],
    upstream_manifest_sha256: tuple[str, str] | None,
    context: str,
) -> list[dict[str, object]]:
    records = _expect_sequence(value, context=context)
    if source_requirement == "none":
        if records:
            raise ArtifactRegistryError(
                f"{context} must be empty for a conceptual artifact",
            )
        return []
    if source_requirement not in {"required", "upstream-manifest"}:
        raise ArtifactRegistryError(f"{context} has an unknown source requirement")
    if not records:
        raise ArtifactRegistryError(
            f"{context} must contain at least one source and its required source roles",
        )
    normalized: list[dict[str, object]] = []
    source_ids: set[str] = set()
    members: set[str] = set()
    for index, raw_record in enumerate(records):
        record_context = f"{context}[{index}]"
        record = _expect_mapping(raw_record, context=record_context)
        _expect_keys(
            record,
            {"source_id", "release_member", "role", "sha256", "bytes"},
            context=record_context,
        )
        source_id = _expect_token(
            record["source_id"],
            context=f"{record_context}.source_id",
        )
        member = _expect_relative_member(
            record["release_member"],
            context=f"{record_context}.release_member",
        )
        role = _expect_string(record["role"], context=f"{record_context}.role")
        if role not in _SOURCE_ROLES:
            raise ArtifactRegistryError(f"{record_context}.role is not recognized")
        if source_id in source_ids or member in members:
            raise ArtifactRegistryError(f"{context} contains duplicate source identity")
        source_ids.add(source_id)
        members.add(member)
        normalized.append(
            {
                "source_id": source_id,
                "release_member": member,
                "role": role,
                "sha256": _expect_sha256(
                    record["sha256"],
                    context=f"{record_context}.sha256",
                ),
                "bytes": _expect_size(
                    record["bytes"],
                    context=f"{record_context}.bytes",
                ),
            },
        )
    normalized.sort(key=lambda record: str(record["source_id"]))
    actual_roles = {str(record["role"]) for record in normalized}
    missing_roles = [role for role in required_source_roles if role not in actual_roles]
    if missing_roles:
        raise ArtifactRegistryError(
            f"{context} is missing artifact-specific source roles: {missing_roles}",
        )
    if source_requirement == "upstream-manifest":
        if upstream_manifest_sha256 is None:
            raise ArtifactRegistryError(
                f"{context} lacks its release-manifest digest contract",
            )
        expected_identities = {
            "postprocess-release-manifest": (
                _UPSTREAM_MANIFEST_MEMBERS["postprocess-release-manifest"],
                "provenance",
                upstream_manifest_sha256[0],
            ),
            "source-data-manifest": (
                _UPSTREAM_MANIFEST_MEMBERS["source-data-manifest"],
                "provenance",
                upstream_manifest_sha256[1],
            ),
        }
        actual_identities = {
            str(record["source_id"]): (
                str(record["release_member"]),
                str(record["role"]),
                str(record["sha256"]),
            )
            for record in normalized
        }
        if actual_identities != expected_identities:
            raise ArtifactRegistryError(
                f"{context} must exactly bind the canonical postprocess-release "
                "and source-data manifests, members, roles, and digests",
            )
    return normalized


def _normalize_renderer_input(
    value: object,
    *,
    renderer_root: _PinnedRoot,
    context: str,
) -> tuple[dict[str, object], _PinnedMember]:
    renderer = _expect_mapping(value, context=context)
    _expect_keys(renderer, {"script", "sha256"}, context=context)
    script = _expect_relative_member(renderer["script"], context=f"{context}.script")
    if not script.startswith("analysis/") or not script.endswith(".py"):
        raise ArtifactRegistryError(
            f"{context}.script must be an analysis Python module",
        )
    expected_sha256 = _expect_sha256(renderer["sha256"], context=f"{context}.sha256")
    pinned = _pin_member(renderer_root, script, context=context)
    if pinned.sha256 != expected_sha256:
        pinned.close()
        raise ArtifactRegistryError(f"{context}.sha256 does not match the script")
    return {
        "script": script,
        "sha256": pinned.sha256,
        "bytes": pinned.size_bytes,
    }, pinned


def _normalize_renderer_registry(
    value: object,
    *,
    renderer_root: _PinnedRoot,
    context: str,
) -> tuple[dict[str, object], _PinnedMember]:
    renderer = _expect_mapping(value, context=context)
    _expect_keys(renderer, {"script", "sha256", "bytes"}, context=context)
    script = _expect_relative_member(renderer["script"], context=f"{context}.script")
    if not script.startswith("analysis/") or not script.endswith(".py"):
        raise ArtifactRegistryError(
            f"{context}.script must be an analysis Python module",
        )
    expected_sha256 = _expect_sha256(renderer["sha256"], context=f"{context}.sha256")
    expected_bytes = _expect_size(renderer["bytes"], context=f"{context}.bytes")
    pinned = _pin_member(renderer_root, script, context=context)
    if pinned.sha256 != expected_sha256 or pinned.size_bytes != expected_bytes:
        pinned.close()
        raise ArtifactRegistryError(f"{context} does not match the renderer script")
    return dict(renderer), pinned


def _bind_live_builder(
    renderer_root: _PinnedRoot,
    *,
    live_builder: _PinnedMetadata,
) -> tuple[dict[str, object], _PinnedMember]:
    pinned = _pin_member(
        renderer_root,
        _BUILDER_SCRIPT_MEMBER,
        context="artifact registry builder",
    )
    if (
        pinned.sha256 != live_builder.sha256
        or pinned.size_bytes != live_builder.size_bytes
    ):
        pinned.close()
        raise ArtifactRegistryError(
            "renderer root does not contain the live artifact registry builder",
        )
    return {
        "script": _BUILDER_SCRIPT_MEMBER,
        "sha256": pinned.sha256,
        "bytes": pinned.size_bytes,
    }, pinned


def _normalize_builder_registry(
    value: object,
    *,
    renderer_root: _PinnedRoot,
) -> tuple[dict[str, object], _PinnedMember]:
    builder, pinned = _normalize_renderer_registry(
        value,
        renderer_root=renderer_root,
        context="registry.builder",
    )
    if builder["script"] != _BUILDER_SCRIPT_MEMBER:
        pinned.close()
        raise ArtifactRegistryError("registry.builder changes the frozen builder path")
    return builder, pinned


def _normalize_outputs(
    value: object,
    *,
    artifact_id: str,
    artifact_kind: str,
    output_root: _PinnedRoot,
    context: str,
) -> tuple[list[dict[str, object]], list[_PinnedMember]]:
    records = _expect_sequence(value, context=context)
    if not records:
        raise ArtifactRegistryError(f"{context} must contain at least one output")
    normalized: list[dict[str, object]] = []
    pinned_members: list[_PinnedMember] = []
    output_ids: set[str] = set()
    members: set[str] = set()
    try:
        for index, raw_record in enumerate(records):
            record_context = f"{context}[{index}]"
            record = _expect_mapping(raw_record, context=record_context)
            _expect_keys(
                record,
                {"output_id", "release_member", "media_type", "sha256", "bytes"},
                context=record_context,
            )
            output_id = _expect_token(
                record["output_id"],
                context=f"{record_context}.output_id",
            )
            member = _expect_relative_member(
                record["release_member"],
                context=f"{record_context}.release_member",
            )
            required_prefix = f"rendered/{artifact_id}/"
            if not member.startswith(required_prefix):
                raise ArtifactRegistryError(
                    f"{record_context}.release_member must start with "
                    f"{required_prefix!r}",
                )
            media_type = _expect_string(
                record["media_type"],
                context=f"{record_context}.media_type",
            )
            expected_suffixes = _OUTPUT_MEDIA_SUFFIXES.get(media_type)
            if expected_suffixes is None or not member.endswith(expected_suffixes):
                raise ArtifactRegistryError(
                    f"{record_context} has an unsupported media type or suffix",
                )
            if output_id in output_ids or member in members:
                raise ArtifactRegistryError(
                    f"{context} contains duplicate output identity",
                )
            output_ids.add(output_id)
            members.add(member)
            expected_sha256 = _expect_sha256(
                record["sha256"],
                context=f"{record_context}.sha256",
            )
            expected_bytes = _expect_size(
                record["bytes"],
                context=f"{record_context}.bytes",
            )
            pinned = _pin_member(output_root, member, context=record_context)
            if pinned.sha256 != expected_sha256 or pinned.size_bytes != expected_bytes:
                pinned.close()
                raise ArtifactRegistryError(
                    f"{record_context} does not match the rendered output",
                )
            pinned_members.append(pinned)
            normalized.append(
                {
                    "output_id": output_id,
                    "release_member": member,
                    "media_type": media_type,
                    "sha256": expected_sha256,
                    "bytes": expected_bytes,
                },
            )
    except Exception:
        for pinned in pinned_members:
            pinned.close()
        raise
    normalized.sort(key=lambda record: str(record["output_id"]))
    required_media = _REQUIRED_MEDIA_BY_KIND.get(artifact_kind)
    if required_media is None:
        for pinned in pinned_members:
            pinned.close()
        raise ArtifactRegistryError(f"{context} has an unsupported artifact kind")
    if not any(record["media_type"] in required_media for record in normalized):
        for pinned in pinned_members:
            pinned.close()
        raise ArtifactRegistryError(
            f"{context} lacks a presentation output compatible with {artifact_kind!r}",
        )
    return normalized, pinned_members


def _normalize_omission(  # noqa: PLR0913
    value: object,
    *,
    spec: ArtifactSpec,
    receipts: list[dict[str, str]],
    global_receipt_gates: frozenset[str],
    context: str,
    include_reason: bool = False,
) -> dict[str, object]:
    omission = _expect_mapping(value, context=context)
    expected_keys = {"reason_code", "unsatisfied_gates"}
    if include_reason:
        expected_keys.add("reason")
    _expect_keys(
        omission,
        expected_keys,
        context=context,
    )
    reason_code = _expect_string(
        omission["reason_code"],
        context=f"{context}.reason_code",
    )
    if reason_code not in _OMISSION_REASONS:
        raise ArtifactRegistryError(f"{context}.reason_code is not recognized")
    expected_reason = _OMISSION_REASONS[reason_code]
    if include_reason and omission["reason"] != expected_reason:
        raise ArtifactRegistryError(f"{context}.reason is not canonical")
    raw_unsatisfied = _expect_sequence(
        omission["unsatisfied_gates"],
        context=f"{context}.unsatisfied_gates",
    )
    unsatisfied: list[str] = []
    for index, raw_gate in enumerate(raw_unsatisfied):
        gate = _expect_gate(
            raw_gate,
            context=f"{context}.unsatisfied_gates[{index}]",
        )
        if gate in unsatisfied:
            raise ArtifactRegistryError(
                f"{context} duplicates unsatisfied gate {gate!r}",
            )
        unsatisfied.append(gate)
    unsatisfied = [gate for gate in GATE_ORDER if gate in unsatisfied]
    satisfied = {str(record["gate"]) for record in receipts}
    expected_unsatisfied = [
        gate for gate in spec.required_gates if gate not in satisfied
    ]
    expected_unsatisfied = [
        gate for gate in GATE_ORDER if gate in expected_unsatisfied
    ]
    if unsatisfied != expected_unsatisfied:
        raise ArtifactRegistryError(
            f"{context}.unsatisfied_gates does not exactly complement receipts",
        )
    if reason_code == "required_gate_not_satisfied" and not unsatisfied:
        raise ArtifactRegistryError(
            f"{context} requires at least one unsatisfied gate",
        )
    if (
        reason_code == "coauthor_decision_to_omit"
        and (unsatisfied or "COAUTH" not in global_receipt_gates)
    ):
        raise ArtifactRegistryError(
            f"{context} requires complete gates including COAUTH",
        )
    if reason_code == "release_scope_exclusion" and (
        unsatisfied or "COAUTH" not in global_receipt_gates
    ):
        raise ArtifactRegistryError(
            f"{context} requires complete artifact gates and global COAUTH",
        )
    return {
        "reason_code": reason_code,
        "reason": expected_reason,
        "unsatisfied_gates": unsatisfied,
    }


def _base_artifact_record(spec: ArtifactSpec) -> dict[str, object]:
    return {
        "semantic_id": spec.semantic_id,
        "title": spec.title,
        "kind": spec.kind,
        "required_gates": list(spec.required_gates),
        "required_source_roles": list(spec.required_source_roles),
        "source_requirement": spec.source_requirement,
    }


def _normalize_input_artifacts(
    value: object,
    *,
    release: Mapping[str, str],
    gate_ledger: list[dict[str, str]],
    renderer_root: _PinnedRoot,
    output_root: _PinnedRoot,
) -> tuple[list[dict[str, object]], list[_PinnedMember]]:
    records = _expect_sequence(value, context="reconciliation.artifacts")
    by_id: dict[str, Mapping[str, object]] = {}
    for index, raw_record in enumerate(records):
        context = f"reconciliation.artifacts[{index}]"
        record = _expect_mapping(raw_record, context=context)
        semantic_id = _expect_string(
            record.get("semantic_id"),
            context=f"{context}.semantic_id",
        )
        if semantic_id in by_id:
            raise ArtifactRegistryError(
                f"reconciliation duplicates {semantic_id!r}",
            )
        by_id[semantic_id] = record
    if set(by_id) != set(_SPEC_BY_ID):
        missing = sorted(set(_SPEC_BY_ID) - set(by_id))
        extra = sorted(set(by_id) - set(_SPEC_BY_ID))
        raise ArtifactRegistryError(
            "reconciliation artifact inventory mismatch; "
            f"missing={missing}, extra={extra}",
        )

    normalized: list[dict[str, object]] = []
    pins: list[_PinnedMember] = []
    global_receipt_gates = frozenset(receipt["gate"] for receipt in gate_ledger)
    try:
        for spec in ARTIFACT_SPECS:
            record = by_id[spec.semantic_id]
            context = f"reconciliation.artifacts[{spec.semantic_id}]"
            status = _expect_string(record.get("status"), context=f"{context}.status")
            if status == "ready":
                _expect_keys(
                    record,
                    {
                        "semantic_id",
                        "status",
                        "gate_receipts",
                        "source_data",
                        "renderer",
                        "outputs",
                    },
                    context=context,
                )
                receipts = _normalize_gate_receipts(
                    record["gate_receipts"],
                    permitted_gates=spec.required_gates,
                    require_all=True,
                    context=f"{context}.gate_receipts",
                )
                if receipts != _artifact_gate_receipts(gate_ledger, spec):
                    raise ArtifactRegistryError(
                        f"{context}.gate_receipts does not match the global ledger",
                    )
                sources = _normalize_sources(
                    record["source_data"],
                    source_requirement=spec.source_requirement,
                    required_source_roles=spec.required_source_roles,
                    upstream_manifest_sha256=(
                        release["postprocess_release_sha256"],
                        release["source_data_manifest_sha256"],
                    ),
                    context=f"{context}.source_data",
                )
                renderer, renderer_pin = _normalize_renderer_input(
                    record["renderer"],
                    renderer_root=renderer_root,
                    context=f"{context}.renderer",
                )
                pins.append(renderer_pin)
                outputs, output_pins = _normalize_outputs(
                    record["outputs"],
                    artifact_id=spec.semantic_id,
                    artifact_kind=spec.kind,
                    output_root=output_root,
                    context=f"{context}.outputs",
                )
                pins.extend(output_pins)
                normalized.append(
                    {
                        **_base_artifact_record(spec),
                        "status": "ready",
                        "gate_receipts": receipts,
                        "source_data": sources,
                        "renderer": renderer,
                        "outputs": outputs,
                        "claims": [claim.as_record() for claim in spec.claims],
                    },
                )
            elif status == "omitted":
                _expect_keys(
                    record,
                    {"semantic_id", "status", "gate_receipts", "omission"},
                    context=context,
                )
                receipts = _normalize_gate_receipts(
                    record["gate_receipts"],
                    permitted_gates=spec.required_gates,
                    require_all=False,
                    context=f"{context}.gate_receipts",
                )
                if receipts != _artifact_gate_receipts(gate_ledger, spec):
                    raise ArtifactRegistryError(
                        f"{context}.gate_receipts does not match the global ledger",
                    )
                omission = _normalize_omission(
                    record["omission"],
                    spec=spec,
                    receipts=receipts,
                    global_receipt_gates=global_receipt_gates,
                    context=f"{context}.omission",
                )
                normalized.append(
                    {
                        **_base_artifact_record(spec),
                        "status": "omitted",
                        "gate_receipts": receipts,
                        "omission": omission,
                        "planned_claims": [
                            claim.as_record() for claim in spec.claims
                        ],
                    },
                )
            else:
                raise ArtifactRegistryError(
                    f"{context}.status must be 'ready' or 'omitted'",
                )
        _validate_global_reconciliation(normalized, gate_ledger=gate_ledger)
    except Exception:
        for pinned in pins:
            pinned.close()
        raise
    return normalized, pins


def _validate_global_reconciliation(
    artifacts: Sequence[Mapping[str, object]],
    *,
    gate_ledger: list[dict[str, str]],
) -> None:
    sources_by_id: dict[str, tuple[str, str, int]] = {}
    sources_by_member: dict[str, tuple[str, str, int]] = {}
    output_ids: set[str] = set()
    output_members: set[str] = set()
    renderer_hashes: dict[str, tuple[str, int]] = {}
    for artifact in artifacts:
        semantic_id = str(artifact["semantic_id"])
        spec = _SPEC_BY_ID[semantic_id]
        actual_receipts = _expect_sequence(
            artifact["gate_receipts"],
            context="artifact.gate_receipts",
        )
        if list(actual_receipts) != _artifact_gate_receipts(gate_ledger, spec):
            raise ArtifactRegistryError(
                f"artifact {semantic_id!r} contradicts the global gate ledger",
            )
        if artifact["status"] != "ready":
            continue
        renderer = _expect_mapping(artifact["renderer"], context="artifact.renderer")
        script = str(renderer["script"])
        renderer_identity = (str(renderer["sha256"]), int(renderer["bytes"]))
        previous_renderer = renderer_hashes.setdefault(script, renderer_identity)
        if previous_renderer != renderer_identity:
            raise ArtifactRegistryError(
                f"renderer {script!r} has inconsistent identities",
            )
        for source_raw in _expect_sequence(
            artifact["source_data"],
            context="artifact.source_data",
        ):
            source = _expect_mapping(source_raw, context="source data")
            source_id = str(source["source_id"])
            member = str(source["release_member"])
            identity = (
                member,
                str(source["sha256"]),
                int(source["bytes"]),
            )
            previous_id = sources_by_id.setdefault(source_id, identity)
            if previous_id != identity:
                raise ArtifactRegistryError(
                    f"source_id {source_id!r} has inconsistent identities",
                )
            member_identity = (
                source_id,
                str(source["sha256"]),
                int(source["bytes"]),
            )
            previous_member = sources_by_member.setdefault(member, member_identity)
            if previous_member != member_identity:
                raise ArtifactRegistryError(
                    f"source member {member!r} has inconsistent identities",
                )
        for output_raw in _expect_sequence(
            artifact["outputs"],
            context="artifact.outputs",
        ):
            output = _expect_mapping(output_raw, context="rendered output")
            output_id = str(output["output_id"])
            member = str(output["release_member"])
            if output_id in output_ids or member in output_members:
                raise ArtifactRegistryError(
                    "a rendered output is assigned to multiple semantic artifacts",
                )
            output_ids.add(output_id)
            output_members.add(member)
    circular_members = set(sources_by_member) & output_members
    if circular_members:
        raise ArtifactRegistryError(
            "rendered outputs cannot also serve as their registry source data: "
            f"{sorted(circular_members)}",
        )


def _normalize_reconciliation(
    value: Mapping[str, object],
    *,
    live_builder: _PinnedMetadata,
    renderer_root: _PinnedRoot,
    output_root: _PinnedRoot,
) -> tuple[dict[str, object], list[_PinnedMember]]:
    _expect_keys(
        value,
        {"schema", "release", "gate_ledger", "artifacts"},
        context="reconciliation",
    )
    if value["schema"] != RECONCILIATION_INPUT_SCHEMA:
        raise ArtifactRegistryError("reconciliation has the wrong schema")
    release = _normalize_release(value["release"], context="reconciliation.release")
    gate_ledger = _normalize_gate_receipts(
        value["gate_ledger"],
        permitted_gates=GATE_ORDER,
        require_all=False,
        context="reconciliation.gate_ledger",
    )
    builder, builder_pin = _bind_live_builder(
        renderer_root,
        live_builder=live_builder,
    )
    try:
        artifacts, artifact_pins = _normalize_input_artifacts(
            value["artifacts"],
            release=release,
            gate_ledger=gate_ledger,
            renderer_root=renderer_root,
            output_root=output_root,
        )
    except Exception:
        builder_pin.close()
        raise
    pins = [builder_pin, *artifact_pins]
    payload = {
        "schema": ARTIFACT_REGISTRY_SCHEMA,
        "contract": ARTIFACT_REGISTRY_CONTRACT,
        "trust_model": dict(TRUST_MODEL),
        "release": release,
        "builder": builder,
        "gate_catalog": [dict(record) for record in GATE_CATALOG],
        "gate_ledger": gate_ledger,
        "artifact_catalog_sha256": artifact_catalog_sha256(),
        "artifacts": artifacts,
    }
    return {
        **payload,
        "registry_payload_sha256": _sha256(_canonical_json(payload)),
    }, pins


def _validate_catalog_fields(
    artifact: Mapping[str, object],
    *,
    spec: ArtifactSpec,
    context: str,
) -> None:
    if artifact["semantic_id"] != spec.semantic_id:
        raise ArtifactRegistryError(f"{context}.semantic_id is not canonical")
    if artifact["title"] != spec.title or artifact["kind"] != spec.kind:
        raise ArtifactRegistryError(f"{context} changes the frozen semantic catalog")
    if artifact["required_gates"] != list(spec.required_gates):
        raise ArtifactRegistryError(f"{context}.required_gates is not canonical")
    if artifact["required_source_roles"] != list(spec.required_source_roles):
        raise ArtifactRegistryError(
            f"{context}.required_source_roles is not canonical",
        )
    if artifact["source_requirement"] != spec.source_requirement:
        raise ArtifactRegistryError(f"{context}.source_requirement is not canonical")


def _validate_registry_artifacts(
    value: object,
    *,
    release: Mapping[str, str],
    gate_ledger: list[dict[str, str]],
    renderer_root: _PinnedRoot,
    output_root: _PinnedRoot,
) -> tuple[list[Mapping[str, object]], list[_PinnedMember]]:
    records = _expect_sequence(value, context="registry.artifacts")
    if len(records) != len(ARTIFACT_SPECS):
        raise ArtifactRegistryError("registry has the wrong artifact count")
    validated: list[Mapping[str, object]] = []
    pins: list[_PinnedMember] = []
    global_receipt_gates = frozenset(receipt["gate"] for receipt in gate_ledger)
    try:
        for index, spec in enumerate(ARTIFACT_SPECS):
            context = f"registry.artifacts[{index}]"
            artifact = _expect_mapping(records[index], context=context)
            status = _expect_string(
                artifact.get("status"),
                context=f"{context}.status",
            )
            if status == "ready":
                _expect_keys(
                    artifact,
                    {
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
                    },
                    context=context,
                )
                _validate_catalog_fields(artifact, spec=spec, context=context)
                receipts = _normalize_gate_receipts(
                    artifact["gate_receipts"],
                    permitted_gates=spec.required_gates,
                    require_all=True,
                    context=f"{context}.gate_receipts",
                )
                if artifact["gate_receipts"] != receipts:
                    raise ArtifactRegistryError(
                        f"{context}.gate_receipts is not canonically ordered",
                    )
                if receipts != _artifact_gate_receipts(gate_ledger, spec):
                    raise ArtifactRegistryError(
                        f"{context}.gate_receipts contradicts the global ledger",
                    )
                sources = _normalize_sources(
                    artifact["source_data"],
                    source_requirement=spec.source_requirement,
                    required_source_roles=spec.required_source_roles,
                    upstream_manifest_sha256=(
                        release["postprocess_release_sha256"],
                        release["source_data_manifest_sha256"],
                    ),
                    context=f"{context}.source_data",
                )
                if artifact["source_data"] != sources:
                    raise ArtifactRegistryError(
                        f"{context}.source_data is not canonically ordered",
                    )
                expected_claims = [claim.as_record() for claim in spec.claims]
                if artifact["claims"] != expected_claims:
                    raise ArtifactRegistryError(f"{context}.claims is not canonical")
                renderer, renderer_pin = _normalize_renderer_registry(
                    artifact["renderer"],
                    renderer_root=renderer_root,
                    context=f"{context}.renderer",
                )
                pins.append(renderer_pin)
                if artifact["renderer"] != renderer:
                    raise ArtifactRegistryError(f"{context}.renderer is not canonical")
                outputs, output_pins = _normalize_outputs(
                    artifact["outputs"],
                    artifact_id=spec.semantic_id,
                    artifact_kind=spec.kind,
                    output_root=output_root,
                    context=f"{context}.outputs",
                )
                pins.extend(output_pins)
                if artifact["outputs"] != outputs:
                    raise ArtifactRegistryError(
                        f"{context}.outputs is not canonically ordered",
                    )
            elif status == "omitted":
                _expect_keys(
                    artifact,
                    {
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
                    },
                    context=context,
                )
                _validate_catalog_fields(artifact, spec=spec, context=context)
                receipts = _normalize_gate_receipts(
                    artifact["gate_receipts"],
                    permitted_gates=spec.required_gates,
                    require_all=False,
                    context=f"{context}.gate_receipts",
                )
                if artifact["gate_receipts"] != receipts:
                    raise ArtifactRegistryError(
                        f"{context}.gate_receipts is not canonically ordered",
                    )
                if receipts != _artifact_gate_receipts(gate_ledger, spec):
                    raise ArtifactRegistryError(
                        f"{context}.gate_receipts contradicts the global ledger",
                    )
                omission = _normalize_omission(
                    artifact["omission"],
                    spec=spec,
                    receipts=receipts,
                    global_receipt_gates=global_receipt_gates,
                    context=f"{context}.omission",
                    include_reason=True,
                )
                if artifact["omission"] != omission:
                    raise ArtifactRegistryError(f"{context}.omission is not canonical")
                expected_claims = [claim.as_record() for claim in spec.claims]
                if artifact["planned_claims"] != expected_claims:
                    raise ArtifactRegistryError(
                        f"{context}.planned_claims is not canonical",
                    )
            else:
                raise ArtifactRegistryError(
                    f"{context}.status must be 'ready' or 'omitted'",
                )
            validated.append(artifact)
        _validate_global_reconciliation(validated, gate_ledger=gate_ledger)
    except Exception:
        for pinned in pins:
            pinned.close()
        raise
    return validated, pins


def _validate_registry_object(
    registry: Mapping[str, object],
    *,
    renderer_root: _PinnedRoot,
    output_root: _PinnedRoot,
) -> tuple[list[Mapping[str, object]], list[_PinnedMember]]:
    _expect_keys(
        registry,
        {
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
        },
        context="registry",
    )
    if registry["schema"] != ARTIFACT_REGISTRY_SCHEMA:
        raise ArtifactRegistryError("registry has the wrong schema")
    if registry["contract"] != ARTIFACT_REGISTRY_CONTRACT:
        raise ArtifactRegistryError("registry has the wrong contract")
    if registry["trust_model"] != TRUST_MODEL:
        raise ArtifactRegistryError("registry changes the explicit trust model")
    if registry["gate_catalog"] != [dict(record) for record in GATE_CATALOG]:
        raise ArtifactRegistryError("registry changes the frozen gate catalog")
    gate_ledger = _normalize_gate_receipts(
        registry["gate_ledger"],
        permitted_gates=GATE_ORDER,
        require_all=False,
        context="registry.gate_ledger",
    )
    if registry["gate_ledger"] != gate_ledger:
        raise ArtifactRegistryError("registry gate ledger is not canonically ordered")
    expected_catalog_sha256 = artifact_catalog_sha256()
    if registry["artifact_catalog_sha256"] != expected_catalog_sha256:
        raise ArtifactRegistryError("registry changes the frozen artifact catalog")
    release = _normalize_release(registry["release"], context="registry.release")
    expected_payload = dict(registry)
    declared_payload_sha256 = _expect_sha256(
        expected_payload.pop("registry_payload_sha256"),
        context="registry.registry_payload_sha256",
    )
    if _sha256(_canonical_json(expected_payload)) != declared_payload_sha256:
        raise ArtifactRegistryError("registry payload digest does not match")
    builder, builder_pin = _normalize_builder_registry(
        registry["builder"],
        renderer_root=renderer_root,
    )
    if registry["builder"] != builder:
        builder_pin.close()
        raise ArtifactRegistryError("registry.builder is not canonical")
    try:
        artifacts, artifact_pins = _validate_registry_artifacts(
            registry["artifacts"],
            release=release,
            gate_ledger=gate_ledger,
            renderer_root=renderer_root,
            output_root=output_root,
        )
    except Exception:
        builder_pin.close()
        raise
    return artifacts, [builder_pin, *artifact_pins]


def _revalidate_all(
    *,
    metadata: _PinnedMetadata,
    renderer_root: _PinnedRoot,
    output_root: _PinnedRoot,
    members: Sequence[_PinnedMember],
    metadata_context: str,
) -> None:
    _revalidate_metadata(metadata, context=metadata_context)
    _revalidate_root(renderer_root, context="renderer root")
    _revalidate_root(output_root, context="rendered-output root")
    for member in members:
        _revalidate_member(member, context=member.member)


def _ensure_destination_parent(destination: Path) -> tuple[Path, int]:
    absolute = destination.absolute()
    if absolute.suffix != ".json":
        raise ArtifactRegistryError("registry destination must end in .json")
    try:
        parent_entry = os.lstat(absolute.parent)
    except OSError as exc:
        raise ArtifactRegistryError(
            "registry destination parent does not exist",
        ) from exc
    if stat.S_ISLNK(parent_entry.st_mode) or not stat.S_ISDIR(parent_entry.st_mode):
        raise ArtifactRegistryError(
            "registry destination parent must be a non-symlink directory",
        )
    try:
        resolved_parent = absolute.parent.resolve(strict=True)
    except OSError as exc:
        raise ArtifactRegistryError(
            "registry destination parent cannot be resolved",
        ) from exc
    if resolved_parent != absolute.parent:
        raise ArtifactRegistryError(
            "registry destination parent has symlinked ancestors",
        )
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_descriptor = os.open(absolute.parent, flags)
    try:
        pinned_parent = os.fstat(parent_descriptor)
    except OSError as exc:
        os.close(parent_descriptor)
        raise ArtifactRegistryError(
            "cannot inspect pinned registry destination parent",
        ) from exc
    if (pinned_parent.st_dev, pinned_parent.st_ino) != (
        parent_entry.st_dev,
        parent_entry.st_ino,
    ):
        os.close(parent_descriptor)
        raise ArtifactRegistryError(
            "registry destination parent changed while it was pinned",
        )
    try:
        os.stat(absolute.name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return absolute, parent_descriptor
    except OSError:
        os.close(parent_descriptor)
        raise
    os.close(parent_descriptor)
    raise ArtifactRegistryError("registry destination already exists")


def _revalidate_destination_parent(absolute: Path, parent_descriptor: int) -> None:
    try:
        path_entry = os.lstat(absolute.parent)
        resolved_parent = absolute.parent.resolve(strict=True)
    except OSError as exc:
        raise ArtifactRegistryError(
            "registry destination parent disappeared during publication",
        ) from exc
    pinned_parent = os.fstat(parent_descriptor)
    if (
        not stat.S_ISDIR(path_entry.st_mode)
        or stat.S_ISLNK(path_entry.st_mode)
        or resolved_parent != absolute.parent
        or (path_entry.st_dev, path_entry.st_ino)
        != (pinned_parent.st_dev, pinned_parent.st_ino)
    ):
        raise ArtifactRegistryError(
            "registry destination parent changed during publication",
        )


def _validate_published_destination(
    absolute: Path,
    parent_descriptor: int,
    *,
    expected: _PublishedDestinationExpectation,
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute.name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        raise ArtifactRegistryError(
            "published registry destination cannot be pinned",
        ) from exc
    try:
        digest, size, identity = _digest_descriptor(descriptor)
        if (
            not stat.S_ISREG(identity.st_mode)
            or stat.S_IMODE(identity.st_mode) != 0o400
            or (identity.st_dev, identity.st_ino) != expected.staged_identity
            or identity.st_nlink != expected.link_count
            or size != expected.size_bytes
            or digest != expected.sha256
        ):
            raise ArtifactRegistryError(
                "published registry destination does not match the staged file",
            )
        try:
            destination_entry = os.stat(
                absolute.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ArtifactRegistryError(
                "published registry destination changed during readback",
            ) from exc
        if (
            not stat.S_ISREG(destination_entry.st_mode)
            or stat.S_IMODE(destination_entry.st_mode) != 0o400
            or (destination_entry.st_dev, destination_entry.st_ino)
            != expected.staged_identity
            or destination_entry.st_nlink != expected.link_count
            or destination_entry.st_size != expected.size_bytes
        ):
            raise ArtifactRegistryError(
                "published registry destination changed during readback",
            )
        _revalidate_destination_parent(absolute, parent_descriptor)
    finally:
        os.close(descriptor)


def _publish_no_replace(
    destination: Path,
    raw: bytes,
    *,
    link_boundary_check: Callable[[], None] | None = None,
) -> None:
    absolute, parent_descriptor = _ensure_destination_parent(destination)
    staging_name = f".{absolute.name}.staging-{uuid.uuid4().hex}"
    descriptor = -1
    staging_present = False
    destination_linked = False
    published = False
    staged_identity: tuple[int, int] | None = None
    try:
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(staging_name, flags, 0o600, dir_fd=parent_descriptor)
        staging_present = True
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        staged_sha256, staged_size, staged = _digest_descriptor(descriptor)
        if (
            not stat.S_ISREG(staged.st_mode)
            or staged_size != len(raw)
            or staged_sha256 != _sha256(raw)
        ):
            raise ArtifactRegistryError("staged registry failed readback accounting")
        staged_identity = (staged.st_dev, staged.st_ino)
        if link_boundary_check is not None:
            link_boundary_check()
        _revalidate_destination_parent(absolute, parent_descriptor)
        os.link(
            staging_name,
            absolute.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        destination_linked = True
        os.fsync(parent_descriptor)
        _validate_published_destination(
            absolute,
            parent_descriptor,
            expected=_PublishedDestinationExpectation(
                staged_identity=staged_identity,
                sha256=staged_sha256,
                size_bytes=staged_size,
                link_count=2,
            ),
        )
        if link_boundary_check is not None:
            link_boundary_check()
        _revalidate_destination_parent(absolute, parent_descriptor)
        os.unlink(staging_name, dir_fd=parent_descriptor)
        staging_present = False
        os.fsync(parent_descriptor)
        _validate_published_destination(
            absolute,
            parent_descriptor,
            expected=_PublishedDestinationExpectation(
                staged_identity=staged_identity,
                sha256=staged_sha256,
                size_bytes=staged_size,
                link_count=1,
            ),
        )
        if link_boundary_check is not None:
            link_boundary_check()
        _revalidate_destination_parent(absolute, parent_descriptor)
        _validate_published_destination(
            absolute,
            parent_descriptor,
            expected=_PublishedDestinationExpectation(
                staged_identity=staged_identity,
                sha256=staged_sha256,
                size_bytes=staged_size,
                link_count=1,
            ),
        )
        published = True
    except FileExistsError as exc:
        raise ArtifactRegistryError("registry destination already exists") from exc
    finally:
        try:
            if descriptor >= 0:
                os.close(descriptor)
        finally:
            try:
                if (
                    not published
                    and destination_linked
                    and staged_identity is not None
                ):
                    with contextlib.suppress(FileNotFoundError):
                        destination_entry = os.stat(
                            absolute.name,
                            dir_fd=parent_descriptor,
                            follow_symlinks=False,
                        )
                        destination_identity = (
                            destination_entry.st_dev,
                            destination_entry.st_ino,
                        )
                        if destination_identity == staged_identity:
                            os.unlink(absolute.name, dir_fd=parent_descriptor)
                            os.fsync(parent_descriptor)
            finally:
                try:
                    if staging_present:
                        with contextlib.suppress(FileNotFoundError):
                            os.unlink(staging_name, dir_fd=parent_descriptor)
                            os.fsync(parent_descriptor)
                finally:
                    os.close(parent_descriptor)


def build_artifact_registry(
    reconciliation_manifest: Path,
    renderer_root: Path,
    rendered_output_root: Path,
    destination: Path,
    *,
    expected_reconciliation_sha256: str,
) -> ArtifactRegistryReceipt:
    """Validate reconciliation metadata and publish a sealed registry.

    The function has deliberately no source-data-root parameter.  Source-data
    members are reconciled by metadata only and are never opened here.
    """
    # Fail closed on an existing destination before opening any result-adjacent
    # output.  The publication helper repeats this check to close the race.
    absolute_destination, parent_descriptor = _ensure_destination_parent(destination)
    os.close(parent_descriptor)

    metadata = _pin_metadata(reconciliation_manifest, context="reconciliation manifest")
    live_builder: _PinnedMetadata | None = None
    renderer: _PinnedRoot | None = None
    outputs: _PinnedRoot | None = None
    pins: list[_PinnedMember] = []
    try:
        expected_reconciliation = _expect_sha256(
            expected_reconciliation_sha256,
            context="expected_reconciliation_sha256",
        )
        if metadata.sha256 != expected_reconciliation:
            raise ArtifactRegistryError(
                "reconciliation manifest SHA-256 does not match its independent anchor",
            )
        live_builder = _pin_metadata(
            Path(__file__),
            context="live artifact registry builder",
        )
        # The independent metadata anchor is checked before either result-adjacent
        # root is opened.  This is a precondition, not a scientific approval.
        renderer = _pin_root(renderer_root, context="renderer root")
        outputs = _pin_root(rendered_output_root, context="rendered-output root")
        reconciliation = _parse_canonical_metadata(
            metadata,
            context="reconciliation manifest",
        )
        registry, pins = _normalize_reconciliation(
            reconciliation,
            live_builder=live_builder,
            renderer_root=renderer,
            output_root=outputs,
        )
        raw = _canonical_json(registry) + b"\n"
        _revalidate_metadata(
            live_builder,
            context="live artifact registry builder",
        )
        _revalidate_all(
            metadata=metadata,
            renderer_root=renderer,
            output_root=outputs,
            members=pins,
            metadata_context="reconciliation manifest",
        )

        def revalidate_at_link_boundary() -> None:
            _revalidate_metadata(
                live_builder,
                context="live artifact registry builder",
            )
            _revalidate_all(
                metadata=metadata,
                renderer_root=renderer,
                output_root=outputs,
                members=pins,
                metadata_context="reconciliation manifest",
            )

        _publish_no_replace(
            absolute_destination,
            raw,
            link_boundary_check=revalidate_at_link_boundary,
        )
        ready_count = sum(
            artifact["status"] == "ready"
            for artifact in _expect_sequence(
                registry["artifacts"],
                context="registry.artifacts",
            )
        )
        return ArtifactRegistryReceipt(
            manifest_path=str(absolute_destination),
            manifest_sha256=_sha256(raw),
            ready_count=ready_count,
            omitted_count=len(ARTIFACT_SPECS) - ready_count,
        )
    finally:
        for pinned in pins:
            pinned.close()
        if outputs is not None:
            outputs.close()
        if renderer is not None:
            renderer.close()
        if live_builder is not None:
            live_builder.close()
        metadata.close()


def validate_artifact_registry(
    manifest_path: Path,
    renderer_root: Path,
    rendered_output_root: Path,
    *,
    expected_manifest_sha256: str,
) -> ArtifactRegistryReceipt:
    """Validate an independently anchored registry and opaque render bindings."""
    metadata = _pin_metadata(manifest_path, context="artifact registry")
    renderer: _PinnedRoot | None = None
    outputs: _PinnedRoot | None = None
    pins: list[_PinnedMember] = []
    try:
        expected = _expect_sha256(
            expected_manifest_sha256,
            context="expected_manifest_sha256",
        )
        if metadata.sha256 != expected:
            raise ArtifactRegistryError(
                "artifact registry SHA-256 does not match its independent anchor",
            )
        # Identity is established before either result-adjacent root is opened.
        renderer = _pin_root(renderer_root, context="renderer root")
        outputs = _pin_root(rendered_output_root, context="rendered-output root")
        registry = _parse_canonical_metadata(metadata, context="artifact registry")
        artifacts, pins = _validate_registry_object(
            registry,
            renderer_root=renderer,
            output_root=outputs,
        )
        _revalidate_all(
            metadata=metadata,
            renderer_root=renderer,
            output_root=outputs,
            members=pins,
            metadata_context="artifact registry",
        )
        ready_count = sum(artifact["status"] == "ready" for artifact in artifacts)
        return ArtifactRegistryReceipt(
            manifest_path=str(manifest_path.absolute()),
            manifest_sha256=metadata.sha256,
            ready_count=ready_count,
            omitted_count=len(artifacts) - ready_count,
        )
    finally:
        for pinned in pins:
            pinned.close()
        if outputs is not None:
            outputs.close()
        if renderer is not None:
            renderer.close()
        metadata.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="publish a no-replace registry")
    build.add_argument("--reconciliation", type=Path, required=True)
    build.add_argument("--renderer-root", type=Path, required=True)
    build.add_argument("--rendered-output-root", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--expected-reconciliation-sha256", required=True)

    validate = subparsers.add_parser("validate", help="validate a registry")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--renderer-root", type=Path, required=True)
    validate.add_argument("--rendered-output-root", type=Path, required=True)
    validate.add_argument("--expected-sha256", required=True)
    return parser


def main() -> None:
    """Run the artifact-registry command-line interface."""
    args = _parser().parse_args()
    if args.command == "build":
        receipt = build_artifact_registry(
            args.reconciliation,
            args.renderer_root,
            args.rendered_output_root,
            args.out,
            expected_reconciliation_sha256=args.expected_reconciliation_sha256,
        )
    else:
        receipt = validate_artifact_registry(
            args.manifest,
            args.renderer_root,
            args.rendered_output_root,
            expected_manifest_sha256=args.expected_sha256,
        )
    print(json.dumps(asdict(receipt), sort_keys=True))


if __name__ == "__main__":
    main()
