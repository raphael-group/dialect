import pandas as pd
import pytest

from dialect.data.variants import canonicalize_tcga_full_variants


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
        ("Entrez_Gene_Id", "2"),
        ("Hugo_Symbol", "B"),
        ("Variant_Classification", "Silent"),
        ("Variant_Type", "DEL"),
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
