"""Canonical row handling for TCGA-style mutation annotation files.

The functions in this module are deliberately pure: they do not read or write a
MAF and they are not wired into either BMR implementation.  This keeps the
deduplication policy testable without silently changing a scientific run.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

FULL_VARIANT_KEY_COLUMNS = (
    "Tumor_Sample_Barcode",
    "Chromosome",
    "Start_Position",
    "Reference_Allele",
    "Tumor_Seq_Allele2",
)
"""Columns defining one observed full-variant event."""

REQUIRED_SEMANTIC_COLUMNS = (
    "Entrez_Gene_Id",
    "Hugo_Symbol",
    "End_Position",
    "Variant_Classification",
    "Variant_Type",
    "NCBI_Build",
    "Tumor_Seq_Allele1",
)
"""Required non-key fields consumed by the duplicate-resolution contract."""

OPTIONAL_SEMANTIC_COLUMNS = (
    "context65",
    "firehose_patient_id",
)
"""Optional input columns that MutSig reads when they are present."""

_X_CHROMOSOME = 23
_Y_CHROMOSOME = 24
_CHROMOSOME_NUMBER_COLUMN = "__dialect_chromosome_number"
_EFFECTIVE_NEWBASE_COLUMN = "__dialect_effective_newbase"
_ROW_SORT_COLUMN = "__dialect_row_sort_value"
_INTERNAL_COLUMNS = frozenset(
    {
        _CHROMOSOME_NUMBER_COLUMN,
        _EFFECTIVE_NEWBASE_COLUMN,
        _ROW_SORT_COLUMN,
    },
)

_IGNORED_CONFLICT_COLUMNS = (
    "Entrez_Gene_Id",
    "Hugo_Symbol",
)
_EFFECT_CONFLICT_COLUMNS = (
    "Variant_Classification",
    "Variant_Type",
)
_STRICT_SEMANTIC_COLUMNS = (
    "End_Position",
    "NCBI_Build",
    "Tumor_Seq_Allele1",
    _EFFECTIVE_NEWBASE_COLUMN,
)

_MUTSIG_EFFECT_TABLE = """\
SNP\t3'Promoter\tncd
DEL\t3'UTR\tindel_ncd
INS\t3'UTR\tindel_ncd
DNP\t3'UTR\tncd
SNP\t3'UTR\tncd
TNP\t3'UTR\tncd
DEL\t3'-UTR\tindel_ncd
INS\t3'-UTR\tindel_ncd
DNP\t3'-UTR\tncd
SNP\t3'-UTR\tncd
DEL\t5'Flank\tindel_ncd
INS\t5'Flank\tindel_ncd
DNP\t5'Flank\tncd
SNP\t5'Flank\tncd
TNP\t5'Flank\tncd
DEL\t5'-Flank\tindel_ncd
INS\t5'-Flank\tindel_ncd
DNP\t5'-Flank\tncd
SNP\t5'-Flank\tncd
TNP\t5'-Flank\tncd
SNP\t5'Promoter\tncd
DEL\t5'UTR\tindel_ncd
INS\t5'UTR\tindel_ncd
DNP\t5'UTR\tncd
ONP\t5'UTR\tncd
SNP\t5'UTR\tncd
TNP\t5'UTR\tncd
DEL\t5'-UTR\tindel_ncd
INS\t5'-UTR\tindel_ncd
DNP\t5'-UTR\tncd
SNP\t5'-UTR\tncd
SNP\tDe_novo_Start\tncd
DEL\tDe_novo_Start_InFrame\tindel_ncd
DNP\tDe_novo_Start_InFrame\tncd
SNP\tDe_novo_Start_InFrame\tncd
INS\tDe_novo_Start_OutOfFrame\tindel_ncd
SNP\tDe_novo_Start_OutOfFrame\tncd
SNP\tdownstream\tncd
DEL\tFrame_Shift_Del\tindel_cod
INS\tFrame_Shift_Ins\tindel_cod
DEL\tIGR\tindel_ncd
INS\tIGR\tindel_ncd
DNP\tIGR\tncd
SNP\tIGR\tncd
TNP\tIGR\tncd
DEL\tIn_Frame_Del\tindel_cod
DEL\tIn_frame_Del\tindel_cod
INS\tIn_Frame_Ins\tindel_cod
INS\tIn_frame_Ins\tindel_cod
DEL\tIntron\tindel_ncd
INS\tIntron\tindel_ncd
DNP\tIntron\tncd
SNP\tIntron\tncd
TNP\tIntron\tncd
SNP\tmiRNA\tncd
DNP\tMissense\tmis
SNP\tMissense\tmis
TNP\tMissense\tmis
DNP\tMissense_Mutation\tmis
ONP\tMissense_Mutation\tmis
SNP\tMissense_Mutation\tmis
TNP\tMissense_Mutation\tmis
SNP\tNCSD\tncd
DEL\tNon-coding_Transcript\tindel_ncd
INS\tNon-coding_Transcript\tindel_ncd
DNP\tNon-coding_Transcript\tncd
SNP\tNon-coding_Transcript\tncd
DNP\tNonsense\tnon
SNP\tNonsense\tnon
DNP\tNonsense_Mutation\tnon
ONP\tNonsense_Mutation\tnon
SNP\tNonsense_Mutation\tnon
TNP\tNonsense_Mutation\tnon
DNP\tNonstop_Mutation\tnon
SNP\tNonstop_Mutation\tnon
DNP\tPromoter\tncd
SNP\tPromoter\tncd
TNP\tPromoter\tncd
DNP\tRead-through\tnon
SNP\tRead-through\tnon
DEL\tRNA\tindel_ncd
INS\tRNA\tindel_ncd
DNP\tRNA\tncd
SNP\tRNA\tncd
TNP\tRNA\tncd
DEL\tSilent\tsyn
DNP\tSilent\tsyn
INS\tSilent\tsyn
SNP\tSilent\tsyn
SNP\tSplice_Region\tspl
DEL\tSplice_Site\tindel_spl
DEL\tSplice_site\tindel_spl
INS\tSplice_Site\tindel_spl
INS\tSplice_site\tindel_spl
DNP\tSplice_Site\tspl
DNP\tSplice_site\tspl
SNP\tSplice_Site\tspl
SNP\tSplice_site\tspl
TNP\tSplice_Site\tspl
TNP\tSplice_site\tspl
DEL\tSplice_Site_Del\tindel_spl
DNP\tSplice_Site_DNP\tspl
INS\tSplice_Site_Ins\tindel_spl
SNP\tSplice_Site_SNP\tspl
SNP\tSplice_site_SNP\tspl
DEL\tStart_Codon_Del\tindel_cod
DNP\tSynonymous\tsyn
SNP\tSynonymous\tsyn
INS\tTranslation_Start_Site\tindel_ncd
DNP\tTranslation_Start_Site\tncd
SNP\tTranslation_Start_Site\tncd
SNP\tupstream\tncd
SNP\tupstream;downstream\tncd
ONP\tSplice_Site_ONP\tspl
Complex_substitution\tMissense\tmis
DEL\tDe_novo_Start_OutOfFrame\tncd
Complex_substitution\tSplice_site\tspl
DEL\tStop_Codon_Del\tindel_cod
INS\tStart_Codon_Ins\tindel_cod
DEL\tStart_Codon_Del\tindel_cod
INS\tStop_Codon_Ins\tindel_cod
"""
_MUTSIG_DICTIONARY_SHA256 = (
    "aeb7171cc22ac298fb0b8b635afc4ecbfb7eb030240d72365d14bcc2d0551780"
)


def _load_frozen_mutsig_effects() -> dict[tuple[str, str], str]:
    frozen_dictionary = f"classification\ttype\teffect\n{_MUTSIG_EFFECT_TABLE}"
    digest = hashlib.sha256(frozen_dictionary.encode()).hexdigest()
    if digest != _MUTSIG_DICTIONARY_SHA256:
        msg = "Embedded MutSig effect table does not match its frozen SHA-256."
        raise RuntimeError(msg)
    mapping: dict[tuple[str, str], str] = {}
    for line in _MUTSIG_EFFECT_TABLE.splitlines():
        variant_type, variant_classification, effect = line.split("\t")
        key = (variant_type, variant_classification)
        previous = mapping.setdefault(key, effect)
        if previous != effect:
            msg = f"Ambiguous frozen MutSig effect mapping for {key!r}."
            raise RuntimeError(msg)
    return mapping


_MUTSIG_EFFECT_BY_TYPE_AND_CLASSIFICATION = _load_frozen_mutsig_effects()


@dataclass(frozen=True, slots=True)
class DuplicateResolutionPolicy:
    """Immutable description of the frozen TCGA duplicate resolver."""

    version: str
    full_variant_key: tuple[str, ...]
    ignored_conflict_columns: tuple[str, ...]
    effect_conflict_columns: tuple[str, ...]
    strict_semantic_columns: tuple[str, ...]
    representative_rule: str
    effect_resolution_rule: str
    cbase_rule: str
    mutsig_dictionary_sha256: str


TCGA_DUPLICATE_RESOLUTION_POLICY = DuplicateResolutionPolicy(
    version="tcga-full-variant-frozen-effects-v1",
    full_variant_key=FULL_VARIANT_KEY_COLUMNS,
    ignored_conflict_columns=_IGNORED_CONFLICT_COLUMNS,
    effect_conflict_columns=_EFFECT_CONFLICT_COLUMNS,
    strict_semantic_columns=(
        "End_Position",
        "NCBI_Build",
        "Tumor_Seq_Allele1",
        "effective_newbase",
        *OPTIONAL_SEMANTIC_COLUMNS,
    ),
    representative_rule=(
        "select the lexicographically smallest framed complete-row token; never "
        "use input or column order"
    ),
    effect_resolution_rule=(
        "when Variant_Type or Variant_Classification conflicts, retain rows for "
        "the sole effect surviving the frozen MutSig mapping; fail closed on zero "
        "or multiple surviving effects"
    ),
    cbase_rule=(
        "raw type, classification, Entrez ID, and Hugo symbol are not projected; "
        "CBaSE reannotates eligible full-key SNVs and excludes non-SNV events"
    ),
    mutsig_dictionary_sha256=_MUTSIG_DICTIONARY_SHA256,
)
"""Exact result-blind policy applied by the canonical TCGA resolver."""


@dataclass(frozen=True, slots=True)
class VariantResolutionAudit:
    """Aggregate, identifier-free audit of one successful canonicalization."""

    policy_version: str
    input_row_count: int
    output_row_count: int
    duplicate_group_count: int
    collapsed_row_count: int
    semantic_agreement_group_count: int
    ignored_conflict_group_count: int
    frozen_effect_resolution_group_count: int
    resolved_conflict_groups_by_column: tuple[tuple[str, int], ...]
    selected_mutsig_effect_groups: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class _GroupResolution:
    ignored_conflicts: tuple[str, ...]
    effect_conflicts: tuple[str, ...]
    selected_mutsig_effect: str | None


def canonicalize_tcga_full_variants(maf: pd.DataFrame) -> pd.DataFrame:
    """Return one deterministic row per normalized full-variant key.

    The key is exact tumor sample, normalized chromosome and position, and
    normalized reference and alternate alleles.  Distinct reference or alternate
    alleles therefore remain distinct events. Ignored gene-label conflicts use a
    deterministic complete-row representative. Type/classification conflicts use
    the sole effect surviving the frozen MutSig mapping and fail closed when zero
    or multiple effects survive. Every other semantic conflict fails closed. The
    returned rows are sorted by normalized key, independent of input row order.

    Args:
        maf: TCGA-style MAF rows. The input frame is not mutated.

    Returns:
        A new frame with normalized key fields and one row per full-variant key.

    Raises:
        ValueError: If required columns or key values are invalid, duplicate
            columns exist, or duplicate events disagree in a downstream- or
            contract-relevant semantic field.
    """
    return canonicalize_tcga_full_variants_with_audit(maf)[0]


def canonicalize_tcga_full_variants_with_audit(
    maf: pd.DataFrame,
) -> tuple[pd.DataFrame, VariantResolutionAudit]:
    """Canonicalize TCGA variants and return an identifier-free resolution audit.

    This is the auditable form of :func:`canonicalize_tcga_full_variants`; the
    original function remains a backward-compatible frame-only wrapper.

    Args:
        maf: TCGA-style MAF rows. The input frame is not mutated.

    Returns:
        The canonical frame and aggregate resolution counts bound to the frozen
        policy version.

    Raises:
        ValueError: If the MAF violates the key, semantic, or frozen-effect
            resolution contract.
    """
    _validate_columns(maf)
    canonical = maf.copy().reset_index(drop=True)
    canonical["Tumor_Sample_Barcode"] = _normalize_samples(
        canonical["Tumor_Sample_Barcode"],
    )
    chromosome_numbers = canonical["Chromosome"].map(_chromosome_number)
    if chromosome_numbers.isna().any():
        msg = "Chromosome contains a value outside chromosomes 1-22, X, and Y."
        raise ValueError(msg)
    chromosome_numbers = chromosome_numbers.astype("int64")
    canonical["Chromosome"] = chromosome_numbers.map(_chromosome_label)
    canonical["Start_Position"] = _normalize_positions(
        canonical["Start_Position"],
        column="Start_Position",
    )
    canonical["End_Position"] = _normalize_positions(
        canonical["End_Position"],
        column="End_Position",
    )
    for column in ("Reference_Allele", "Tumor_Seq_Allele2"):
        canonical[column] = _normalize_alleles(canonical[column], column=column)
    canonical["Tumor_Seq_Allele1"] = _normalize_alleles(
        canonical["Tumor_Seq_Allele1"],
        column="Tumor_Seq_Allele1",
    )
    effective_newbases = _effective_newbases(canonical)
    _validate_effective_alternate(canonical, effective_newbases)
    _validate_snv_end_positions(canonical)

    original_columns = canonical.columns.tolist()
    canonical[_CHROMOSOME_NUMBER_COLUMN] = chromosome_numbers
    canonical[_EFFECTIVE_NEWBASE_COLUMN] = effective_newbases
    duplicate_mask = canonical.duplicated(
        subset=list(FULL_VARIANT_KEY_COLUMNS),
        keep=False,
    )
    duplicate_rows = canonical.loc[duplicate_mask].copy()
    conflict_counts: Counter[str] = Counter()
    effect_counts: Counter[str] = Counter()
    semantic_agreement_groups = 0
    ignored_conflict_groups = 0
    effect_resolution_groups = 0
    if not duplicate_rows.empty:
        resolved_rows: list[pd.DataFrame] = []
        duplicate_groups = duplicate_rows.groupby(
            list(FULL_VARIANT_KEY_COLUMNS),
            dropna=False,
            observed=True,
            sort=False,
        )
        for _, duplicate_group in duplicate_groups:
            representative, resolution = _resolve_duplicate_group(
                duplicate_group,
                original_columns=original_columns,
            )
            resolved_rows.append(representative)
            conflict_counts.update(resolution.ignored_conflicts)
            conflict_counts.update(resolution.effect_conflicts)
            if resolution.ignored_conflicts:
                ignored_conflict_groups += 1
            if resolution.selected_mutsig_effect is not None:
                effect_resolution_groups += 1
                effect_counts.update([resolution.selected_mutsig_effect])
            if not resolution.ignored_conflicts and not resolution.effect_conflicts:
                semantic_agreement_groups += 1
        canonical = pd.concat(
            [canonical.loc[~duplicate_mask], *resolved_rows],
            axis="index",
        )

    canonical = canonical.sort_values(
        [
            "Tumor_Sample_Barcode",
            _CHROMOSOME_NUMBER_COLUMN,
            "Start_Position",
            "Reference_Allele",
            "Tumor_Seq_Allele2",
        ],
        kind="stable",
    )
    result = canonical.loc[:, original_columns].reset_index(drop=True)
    duplicate_group_count = int(
        duplicate_rows.groupby(
            list(FULL_VARIANT_KEY_COLUMNS),
            dropna=False,
            observed=True,
            sort=False,
        ).ngroups,
    )
    audit = VariantResolutionAudit(
        policy_version=TCGA_DUPLICATE_RESOLUTION_POLICY.version,
        input_row_count=len(maf),
        output_row_count=len(result),
        duplicate_group_count=duplicate_group_count,
        collapsed_row_count=len(maf) - len(result),
        semantic_agreement_group_count=semantic_agreement_groups,
        ignored_conflict_group_count=ignored_conflict_groups,
        frozen_effect_resolution_group_count=effect_resolution_groups,
        resolved_conflict_groups_by_column=tuple(sorted(conflict_counts.items())),
        selected_mutsig_effect_groups=tuple(sorted(effect_counts.items())),
    )
    return result, audit


def _resolve_duplicate_group(
    duplicate_group: pd.DataFrame,
    *,
    original_columns: Sequence[str],
) -> tuple[pd.DataFrame, _GroupResolution]:
    strict_columns = [
        *_STRICT_SEMANTIC_COLUMNS,
        *(
            column
            for column in OPTIONAL_SEMANTIC_COLUMNS
            if column in duplicate_group.columns
        ),
    ]
    strict_conflicts = _conflicting_columns(duplicate_group, strict_columns)
    if strict_conflicts:
        msg = (
            "Duplicate full-variant events disagree in downstream-relevant "
            f"columns: {', '.join(strict_conflicts)}"
        )
        raise ValueError(msg)

    ignored_conflicts = _conflicting_columns(
        duplicate_group,
        _IGNORED_CONFLICT_COLUMNS,
    )
    effect_conflicts = _conflicting_columns(
        duplicate_group,
        _EFFECT_CONFLICT_COLUMNS,
    )
    candidates = duplicate_group
    selected_effect: str | None = None
    if effect_conflicts:
        mapped_effects = [
            _mutsig_effect(row["Variant_Type"], row["Variant_Classification"])
            for _, row in duplicate_group.iterrows()
        ]
        surviving_effects = sorted(
            {effect for effect in mapped_effects if effect is not None},
        )
        if not surviving_effects:
            msg = (
                "Duplicate full-variant type/classification conflict has zero "
                "surviving frozen MutSig effect mappings."
            )
            raise ValueError(msg)
        if len(surviving_effects) > 1:
            msg = (
                "Duplicate full-variant type/classification conflict has multiple "
                "surviving frozen MutSig effect mappings: "
                f"{', '.join(surviving_effects)}"
            )
            raise ValueError(msg)
        selected_effect = surviving_effects[0]
        keep = [effect == selected_effect for effect in mapped_effects]
        candidates = duplicate_group.loc[keep]

    token_columns = sorted(original_columns)
    row_tokens = candidates[token_columns].apply(_row_sort_token, axis="columns")
    representative_offset = min(
        range(len(candidates)),
        key=lambda offset: row_tokens.iloc[offset],
    )
    representative = candidates.iloc[[representative_offset]].copy()
    return representative, _GroupResolution(
        ignored_conflicts=tuple(ignored_conflicts),
        effect_conflicts=tuple(effect_conflicts),
        selected_mutsig_effect=selected_effect,
    )


def _mutsig_effect(
    variant_type: object,
    variant_classification: object,
) -> str | None:
    if not isinstance(variant_type, str) or not isinstance(
        variant_classification,
        str,
    ):
        return None
    return _MUTSIG_EFFECT_BY_TYPE_AND_CLASSIFICATION.get(
        (variant_type, variant_classification),
    )


def _conflicting_columns(
    rows: pd.DataFrame,
    columns: Sequence[str],
) -> list[str]:
    return sorted(
        column
        for column in columns
        if len({_scalar_sort_token(value) for value in rows[column]}) > 1
    )


def _validate_columns(maf: pd.DataFrame) -> None:
    if not maf.columns.is_unique:
        msg = "MAF contains duplicate column names."
        raise ValueError(msg)
    reserved = sorted(_INTERNAL_COLUMNS.intersection(maf.columns))
    if reserved:
        msg = f"MAF contains reserved internal columns: {', '.join(reserved)}"
        raise ValueError(msg)
    required = {*FULL_VARIANT_KEY_COLUMNS, *REQUIRED_SEMANTIC_COLUMNS}
    missing = sorted(required.difference(maf.columns))
    if missing:
        msg = f"MAF is missing required columns: {', '.join(missing)}"
        raise ValueError(msg)


def _normalize_samples(samples: pd.Series) -> pd.Series:
    normalized: list[str] = []
    for value in samples:
        if not isinstance(value, str) or not value or value != value.strip():
            msg = "Tumor_Sample_Barcode must contain exact, non-blank text."
            raise ValueError(msg)
        normalized.append(value)
    return pd.Series(normalized, index=samples.index, dtype="string")


def _chromosome_number(value: object) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower().startswith("chr"):
        text = text[3:]
    upper = text.upper()
    if upper == "X":
        return _X_CHROMOSOME
    if upper == "Y":
        return _Y_CHROMOSOME
    if not text.isdigit():
        return None
    number = int(text)
    return number if 1 <= number <= _Y_CHROMOSOME else None


def _chromosome_label(number: int) -> str:
    if number == _X_CHROMOSOME:
        return "X"
    if number == _Y_CHROMOSOME:
        return "Y"
    return str(number)


def _normalize_positions(positions: pd.Series, *, column: str) -> pd.Series:
    numeric = pd.to_numeric(positions, errors="coerce")
    invalid = numeric.isna() | (numeric <= 0) | (numeric % 1 != 0)
    if invalid.any():
        msg = f"{column} must contain positive integers."
        raise ValueError(msg)
    return numeric.astype("int64")


def _normalize_alleles(alleles: pd.Series, *, column: str) -> pd.Series:
    normalized: list[str] = []
    for value in alleles:
        if not isinstance(value, str) or not value.strip():
            msg = f"{column} must contain non-blank text."
            raise ValueError(msg)
        normalized.append(value.strip().upper())
    return pd.Series(normalized, index=alleles.index, dtype="string")


def _validate_snv_end_positions(maf: pd.DataFrame) -> None:
    snv_mask = (
        maf["Reference_Allele"].isin(["A", "C", "G", "T"])
        & maf["Tumor_Seq_Allele2"].isin(["A", "C", "G", "T"])
        & (maf["Reference_Allele"] != maf["Tumor_Seq_Allele2"])
    )
    if (snv_mask & (maf["Start_Position"] != maf["End_Position"])).any():
        msg = "SNV rows must have matching Start_Position and End_Position."
        raise ValueError(msg)


def _validate_effective_alternate(
    maf: pd.DataFrame,
    effective_newbases: pd.Series,
) -> None:
    if not effective_newbases.equals(maf["Tumor_Seq_Allele2"]):
        msg = (
            "MutSig effective_newbase (effective alternate alleles derived from "
            "Tumor_Seq_Allele1 or newbase) must equal Tumor_Seq_Allele2 under the "
            "frozen full-variant key contract."
        )
        raise ValueError(msg)


def _effective_newbases(maf: pd.DataFrame) -> pd.Series:
    if "newbase" in maf and _contains_only_nonblank_text(maf["newbase"]):
        maf["newbase"] = _normalize_alleles(maf["newbase"], column="newbase")
        return maf["newbase"]
    allele1 = maf["Tumor_Seq_Allele1"]
    fallback = maf["Tumor_Seq_Allele2"].where(
        maf["Reference_Allele"] == allele1,
        allele1,
    )
    if "newbase" in maf:
        # MutSig falls back for the entire input when any newbase is blank. Persist
        # that effective value so deduplication cannot remove the blank row and
        # accidentally switch the downstream file back to its supplied-newbase path.
        maf["newbase"] = fallback
    return fallback


def _contains_only_nonblank_text(values: pd.Series) -> bool:
    return all(isinstance(value, str) and bool(value.strip()) for value in values)


def _row_sort_token(row: pd.Series) -> str:
    return "".join(_scalar_sort_token(value) for value in row)


def _scalar_sort_token(value: object) -> str:
    if pd.isna(value):
        return "M;"
    kind = f"{type(value).__module__}.{type(value).__qualname__}"
    text = str(value)
    return f"V{len(kind)}:{kind}{len(text)}:{text};"
