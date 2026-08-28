"""Canonical row handling for TCGA-style mutation annotation files.

The functions in this module are deliberately pure: they do not read or write a
MAF and they are not wired into either BMR implementation.  This keeps the
deduplication policy testable without silently changing a scientific run.
"""

from __future__ import annotations

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
"""Required non-key fields that must agree under the fixed-build contract."""

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


def canonicalize_tcga_full_variants(maf: pd.DataFrame) -> pd.DataFrame:
    """Return one deterministic row per normalized full-variant key.

    The key is exact tumor sample, normalized chromosome and position, and
    normalized reference and alternate alleles.  Distinct reference or alternate
    alleles therefore remain distinct events.  Duplicate keys must agree on every
    downstream- or contract-relevant semantic field; a conflict fails closed
    instead of selecting an input-order-dependent biological interpretation. When
    duplicates differ only in ignored metadata, the lexicographically smallest
    complete row is the representative. The returned rows are sorted by the
    normalized key, making the result independent of input row order.

    Args:
        maf: TCGA-style MAF rows. The input frame is not mutated.

    Returns:
        A new frame with normalized key fields and one row per full-variant key.

    Raises:
        ValueError: If required columns or key values are invalid, duplicate
            columns exist, or duplicate events disagree in a downstream- or
            contract-relevant semantic field.
    """
    _validate_columns(maf)
    canonical = maf.copy()
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
    semantic_columns = [
        *REQUIRED_SEMANTIC_COLUMNS,
        _EFFECTIVE_NEWBASE_COLUMN,
        *(
            column
            for column in OPTIONAL_SEMANTIC_COLUMNS
            if column in canonical.columns
        ),
    ]
    _reject_semantic_conflicts(
        duplicate_rows,
        semantic_columns=semantic_columns,
    )
    if not duplicate_rows.empty:
        token_columns = sorted(original_columns)
        duplicate_rows[_ROW_SORT_COLUMN] = duplicate_rows[token_columns].apply(
            _row_sort_token,
            axis="columns",
        )
        duplicate_rows = duplicate_rows.sort_values(
            [*FULL_VARIANT_KEY_COLUMNS, _ROW_SORT_COLUMN],
            kind="stable",
        ).drop_duplicates(
            subset=list(FULL_VARIANT_KEY_COLUMNS),
            keep="first",
        )
        canonical = pd.concat(
            [canonical.loc[~duplicate_mask], duplicate_rows],
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
    return canonical.loc[:, original_columns].reset_index(drop=True)


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


def _reject_semantic_conflicts(
    maf: pd.DataFrame,
    *,
    semantic_columns: Sequence[str],
) -> None:
    if maf.empty:
        return
    grouped = maf.groupby(
        list(FULL_VARIANT_KEY_COLUMNS),
        dropna=False,
        observed=True,
        sort=False,
    )[list(semantic_columns)].nunique(dropna=False)
    conflicts = sorted(grouped.columns[(grouped > 1).any(axis="index")].tolist())
    if conflicts:
        conflicts = [
            "effective_newbase" if column == _EFFECTIVE_NEWBASE_COLUMN else column
            for column in conflicts
        ]
        msg = (
            "Duplicate full-variant events disagree in downstream-relevant "
            f"columns: {', '.join(conflicts)}"
        )
        raise ValueError(msg)


def _row_sort_token(row: pd.Series) -> str:
    return "".join(_scalar_sort_token(value) for value in row)


def _scalar_sort_token(value: object) -> str:
    if pd.isna(value):
        return "M;"
    kind = f"{type(value).__module__}.{type(value).__qualname__}"
    text = str(value)
    return f"V{len(kind)}:{kind}{len(text)}:{text};"
