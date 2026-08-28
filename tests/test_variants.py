import pandas as pd
import pytest

from dialect.data.variants import (
    TCGA_DUPLICATE_RESOLUTION_POLICY,
    canonicalize_tcga_full_variants,
    canonicalize_tcga_full_variants_with_audit,
)


def _maf(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "Hugo_Symbol",
            "Entrez_Gene_Id",
            "Chromosome",
            "Start_Position",
            "End_Position",
            "Reference_Allele",
            "Tumor_Seq_Allele1",
            "Tumor_Seq_Allele2",
            "Tumor_Sample_Barcode",
            "Variant_Classification",
            "Variant_Type",
            "NCBI_Build",
            "Center",
        ],
    )


def test_canonical_full_variant_deduplication_is_order_independent():
    rows = [
        [
            "B",
            "2",
            "chr2",
            "20",
            20,
            "a",
            "A",
            "t",
            "S2",
            "Missense",
            "SNP",
            "hg19",
            "Z",
        ],
        ["B", "2", "2", 20, 20, "A", "A", "T", "S2", "Missense", "SNP", "hg19", "A"],
        ["A", "1", "X", 10, 10, "C", "C", "G", "S1", "Nonsense", "SNP", "hg19", "Q"],
    ]

    forward = canonicalize_tcga_full_variants(_maf(rows))
    reverse = canonicalize_tcga_full_variants(_maf(rows[::-1]))

    pd.testing.assert_frame_equal(forward, reverse)
    assert len(forward) == 2
    assert forward.loc[0, "Tumor_Sample_Barcode"] == "S1"
    assert forward.loc[1, "Chromosome"] == "2"
    assert forward.loc[1, "Reference_Allele"] == "A"
    assert forward.loc[1, "Tumor_Seq_Allele2"] == "T"
    assert forward.loc[1, "Center"] == "A"


def test_canonical_representative_is_independent_of_column_order():
    rows = [
        ["B", "2", "2", 20, 20, "A", "A", "T", "S2", "Missense", "SNP", "hg19", "Z"],
        ["B", "2", "2", 20, 20, "A", "A", "T", "S2", "Missense", "SNP", "hg19", "A"],
    ]
    maf = _maf(rows)
    reordered_columns = list(reversed(maf.columns))

    forward = canonicalize_tcga_full_variants(maf)
    reordered = canonicalize_tcga_full_variants(maf[reordered_columns])

    pd.testing.assert_frame_equal(forward, reordered[forward.columns])


def test_canonical_full_variant_key_preserves_distinct_alleles():
    maf = _maf(
        [
            [
                "A",
                "1",
                "1",
                10,
                10,
                "A",
                "A",
                "C",
                "S1",
                "Missense",
                "SNP",
                "hg19",
                "X",
            ],
            [
                "A",
                "1",
                "1",
                10,
                10,
                "A",
                "A",
                "G",
                "S1",
                "Missense",
                "SNP",
                "hg19",
                "X",
            ],
            [
                "A",
                "1",
                "1",
                10,
                10,
                "G",
                "G",
                "C",
                "S1",
                "Missense",
                "SNP",
                "hg19",
                "X",
            ],
        ],
    )

    result = canonicalize_tcga_full_variants(maf)

    assert len(result) == 3
    observed_alleles = zip(
        result["Reference_Allele"],
        result["Tumor_Seq_Allele2"],
        strict=True,
    )
    assert set(observed_alleles) == {
        ("A", "C"),
        ("A", "G"),
        ("G", "C"),
    }


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("NCBI_Build", "hg38"),
        ("Tumor_Seq_Allele1", "G"),
    ],
)
def test_canonical_full_variant_deduplication_rejects_semantic_conflicts(
    column,
    replacement,
):
    rows = [
        ["A", "1", "1", 10, 10, "A", "A", "C", "S1", "Missense", "SNP", "hg19", "X"],
        ["A", "1", "1", 10, 10, "A", "A", "C", "S1", "Missense", "SNP", "hg19", "Y"],
    ]
    maf = _maf(rows)
    maf.loc[1, column] = replacement

    with pytest.raises(ValueError, match=column):
        canonicalize_tcga_full_variants(maf)


def test_ignored_gene_conflicts_use_deterministic_token_representative():
    maf = _maf(
        [
            [
                "B",
                "2",
                "1",
                10,
                10,
                "A",
                "A",
                "C",
                "S1",
                "Missense_Mutation",
                "SNP",
                "hg19",
                "X",
            ],
            [
                "A",
                "1",
                "1",
                10,
                10,
                "A",
                "A",
                "C",
                "S1",
                "Missense_Mutation",
                "SNP",
                "hg19",
                "X",
            ],
        ],
    )

    result, audit = canonicalize_tcga_full_variants_with_audit(maf)

    assert len(result) == 1
    assert result.loc[0, "Entrez_Gene_Id"] == "1"
    assert result.loc[0, "Hugo_Symbol"] == "A"
    assert audit.ignored_conflict_group_count == 1
    assert audit.frozen_effect_resolution_group_count == 0
    assert audit.resolved_conflict_groups_by_column == (
        ("Entrez_Gene_Id", 1),
        ("Hugo_Symbol", 1),
    )


def test_lihc_non_snv_effect_conflict_retains_frame_shift_deletion():
    maf = _maf(
        [
            [
                "A",
                "1",
                "1",
                10,
                11,
                "AT",
                "AT",
                "A",
                "S1",
                "Missense_Mutation",
                "DEL",
                "hg19",
                "X",
            ],
            [
                "A",
                "1",
                "1",
                10,
                11,
                "AT",
                "AT",
                "A",
                "S1",
                "Frame_Shift_Del",
                "DEL",
                "hg19",
                "Y",
            ],
        ],
    )
    reordered_columns = list(reversed(maf.columns))

    forward, forward_audit = canonicalize_tcga_full_variants_with_audit(maf)
    shuffled, shuffled_audit = canonicalize_tcga_full_variants_with_audit(
        maf.iloc[::-1][reordered_columns],
    )

    pd.testing.assert_frame_equal(forward, shuffled[forward.columns])
    assert forward_audit == shuffled_audit
    assert forward.loc[0, "Variant_Classification"] == "Frame_Shift_Del"
    assert forward_audit.frozen_effect_resolution_group_count == 1
    assert forward_audit.selected_mutsig_effect_groups == (("indel_cod", 1),)
    assert forward_audit.resolved_conflict_groups_by_column == (
        ("Variant_Classification", 1),
    )


def test_same_frozen_effect_allows_classification_synonyms():
    maf = _maf(
        [
            [
                "A",
                "1",
                "1",
                10,
                10,
                "A",
                "A",
                "C",
                "S1",
                "Missense",
                "SNP",
                "hg19",
                "X",
            ],
            [
                "A",
                "1",
                "1",
                10,
                10,
                "A",
                "A",
                "C",
                "S1",
                "Missense_Mutation",
                "SNP",
                "hg19",
                "Y",
            ],
        ],
    )

    result, audit = canonicalize_tcga_full_variants_with_audit(maf)

    assert len(result) == 1
    assert audit.selected_mutsig_effect_groups == (("mis", 1),)
    assert audit.frozen_effect_resolution_group_count == 1


def test_unique_effect_resolution_drops_unmapped_variant_type():
    maf = _maf(
        [
            [
                "A",
                "1",
                "1",
                10,
                10,
                "A",
                "A",
                "C",
                "S1",
                "Missense_Mutation",
                "DEL",
                "hg19",
                "X",
            ],
            [
                "A",
                "1",
                "1",
                10,
                10,
                "A",
                "A",
                "C",
                "S1",
                "Missense_Mutation",
                "SNP",
                "hg19",
                "Y",
            ],
        ],
    )

    result, audit = canonicalize_tcga_full_variants_with_audit(maf)

    assert result.loc[0, "Variant_Type"] == "SNP"
    assert audit.selected_mutsig_effect_groups == (("mis", 1),)


def test_effect_conflict_fails_closed_when_no_mapping_survives():
    maf = _maf(
        [
            [
                "A",
                "1",
                "1",
                10,
                11,
                "AT",
                "AT",
                "A",
                "S1",
                "Unknown_A",
                "DEL",
                "hg19",
                "X",
            ],
            [
                "A",
                "1",
                "1",
                10,
                11,
                "AT",
                "AT",
                "A",
                "S1",
                "Unknown_B",
                "DEL",
                "hg19",
                "Y",
            ],
        ],
    )

    with pytest.raises(ValueError, match="zero surviving"):
        canonicalize_tcga_full_variants(maf)


def test_effect_conflict_fails_closed_when_multiple_mappings_survive():
    maf = _maf(
        [
            [
                "A",
                "1",
                "1",
                10,
                10,
                "A",
                "A",
                "C",
                "S1",
                "Missense_Mutation",
                "SNP",
                "hg19",
                "X",
            ],
            [
                "A",
                "1",
                "1",
                10,
                10,
                "A",
                "A",
                "C",
                "S1",
                "Nonsense_Mutation",
                "SNP",
                "hg19",
                "Y",
            ],
        ],
    )

    with pytest.raises(ValueError, match="multiple surviving"):
        canonicalize_tcga_full_variants(maf)


def test_duplicate_resolution_policy_is_explicit_and_pinned():
    policy = TCGA_DUPLICATE_RESOLUTION_POLICY

    assert policy.full_variant_key == (
        "Tumor_Sample_Barcode",
        "Chromosome",
        "Start_Position",
        "Reference_Allele",
        "Tumor_Seq_Allele2",
    )
    assert policy.ignored_conflict_columns == ("Entrez_Gene_Id", "Hugo_Symbol")
    assert policy.effect_conflict_columns == (
        "Variant_Classification",
        "Variant_Type",
    )
    assert policy.mutsig_dictionary_sha256 == (
        "aeb7171cc22ac298fb0b8b635afc4ecbfb7eb030240d72365d14bcc2d0551780"
    )


def test_duplicate_indels_must_agree_on_end_position():
    maf = _maf(
        [
            [
                "A",
                "1",
                "1",
                10,
                10,
                "AT",
                "AT",
                "A",
                "S1",
                "Frame_Shift_Del",
                "DEL",
                "hg19",
                "X",
            ],
            [
                "A",
                "1",
                "1",
                10,
                11,
                "AT",
                "AT",
                "A",
                "S1",
                "Frame_Shift_Del",
                "DEL",
                "hg19",
                "Y",
            ],
        ],
    )

    with pytest.raises(ValueError, match="End_Position"):
        canonicalize_tcga_full_variants(maf)


def test_single_base_substitution_requires_matching_end_position():
    maf = _maf(
        [
            [
                "A",
                "1",
                "1",
                10,
                11,
                "A",
                "A",
                "C",
                "S1",
                "Missense",
                "SNP",
                "hg19",
                "X",
            ],
        ],
    )

    with pytest.raises(ValueError, match="matching Start_Position and End_Position"):
        canonicalize_tcga_full_variants(maf)


def test_canonical_full_variant_deduplication_checks_optional_mutsig_fields():
    maf = _maf(
        [
            [
                "A",
                "1",
                "1",
                10,
                10,
                "A",
                "A",
                "C",
                "S1",
                "Missense",
                "SNP",
                "hg19",
                "X",
            ],
            [
                "A",
                "1",
                "1",
                10,
                10,
                "A",
                "A",
                "C",
                "S1",
                "Missense",
                "SNP",
                "hg19",
                "Y",
            ],
        ],
    )
    maf["context65"] = [1, 2]

    with pytest.raises(ValueError, match="context65"):
        canonicalize_tcga_full_variants(maf)


def test_canonical_full_variant_deduplication_checks_effective_newbase():
    maf = _maf(
        [
            [
                "A",
                "1",
                "1",
                10,
                10,
                "A",
                "A",
                "C",
                "S1",
                "Missense",
                "SNP",
                "hg19",
                "X",
            ],
            [
                "A",
                "1",
                "1",
                10,
                10,
                "A",
                "A",
                "C",
                "S1",
                "Missense",
                "SNP",
                "hg19",
                "Y",
            ],
        ],
    )
    maf["newbase"] = ["C", "G"]

    with pytest.raises(ValueError, match="effective_newbase"):
        canonicalize_tcga_full_variants(maf)


def test_incomplete_newbase_column_uses_mutsig_allele_fallback():
    maf = _maf(
        [
            [
                "A",
                "1",
                "1",
                10,
                10,
                "A",
                "A",
                "C",
                "S1",
                "Missense",
                "SNP",
                "hg19",
                "X",
            ],
            [
                "A",
                "1",
                "1",
                10,
                10,
                "A",
                "A",
                "C",
                "S1",
                "Missense",
                "SNP",
                "hg19",
                "Y",
            ],
        ],
    )
    maf["newbase"] = ["", "G"]

    result = canonicalize_tcga_full_variants(maf)

    assert len(result) == 1
    assert result.loc[0, "newbase"] == "C"


def test_canonical_full_variant_rejects_effective_alternate_mismatch():
    maf = _maf(
        [
            [
                "A",
                "1",
                "1",
                10,
                11,
                "A",
                "C",
                "A",
                "S1",
                "Missense",
                "SNP",
                "hg19",
                "X",
            ],
        ],
    )

    with pytest.raises(ValueError, match="effective alternate alleles"):
        canonicalize_tcga_full_variants(maf)


def test_canonical_full_variant_rejects_swapped_tumor_allele_encoding():
    maf = _maf(
        [
            [
                "A",
                "1",
                "1",
                10,
                10,
                "A",
                "A",
                "C",
                "S1",
                "Missense",
                "SNP",
                "hg19",
                "X",
            ],
            [
                "A",
                "1",
                "1",
                10,
                10,
                "A",
                "C",
                "A",
                "S1",
                "Missense",
                "SNP",
                "hg19",
                "Y",
            ],
        ],
    )

    with pytest.raises(ValueError, match="effective alternate alleles"):
        canonicalize_tcga_full_variants(maf)


@pytest.mark.parametrize(
    "column",
    [
        "__dialect_chromosome_number",
        "__dialect_effective_newbase",
        "__dialect_row_sort_value",
    ],
)
def test_canonical_full_variant_rejects_reserved_internal_columns(column):
    maf = _maf(
        [
            [
                "A",
                "1",
                "1",
                10,
                10,
                "A",
                "A",
                "C",
                "S1",
                "Missense",
                "SNP",
                "hg19",
                "X",
            ],
        ],
    )
    maf[column] = "user supplied"

    with pytest.raises(ValueError, match="reserved internal columns"):
        canonicalize_tcga_full_variants(maf)


def test_canonical_full_variant_allows_multibase_end_position():
    maf = _maf(
        [
            [
                "A",
                "1",
                "1",
                10,
                12,
                "ATC",
                "ATC",
                "A",
                "S1",
                "Frame_Shift_Del",
                "DEL",
                "hg19",
                "X",
            ],
        ],
    )

    result = canonicalize_tcga_full_variants(maf)

    assert result.loc[0, "End_Position"] == 12


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("Tumor_Sample_Barcode", " S1", "Tumor_Sample_Barcode"),
        ("Chromosome", "MT", "Chromosome"),
        ("Start_Position", 1.5, "Start_Position"),
        ("End_Position", 1.5, "End_Position"),
        ("Reference_Allele", "", "Reference_Allele"),
        ("Tumor_Seq_Allele2", None, "Tumor_Seq_Allele2"),
    ],
)
def test_canonical_full_variant_deduplication_rejects_invalid_keys(
    column,
    value,
    match,
):
    maf = _maf(
        [
            [
                "A",
                "1",
                "1",
                10,
                10,
                "A",
                "A",
                "C",
                "S1",
                "Missense",
                "SNP",
                "hg19",
                "X",
            ],
        ],
    )
    maf[column] = maf[column].astype("object")
    maf.loc[0, column] = value

    with pytest.raises(ValueError, match=match):
        canonicalize_tcga_full_variants(maf)
