from __future__ import annotations

import hashlib
from types import MappingProxyType

import pytest

import dialect.data.tcga as tcga_module
from dialect.data.tcga import (
    PRIMARY_DISEASE_SAMPLE_TYPE_CODES,
    TCGA_CASE_LIST_RECEIPTS,
    TCGA_COHORTS,
    TCGA_DATAHUB_COMMIT,
    TCGA_DATAHUB_TREE,
    TCGA_MAF_SHA256,
    TCGA_SELECTED_SAMPLE_AXIS_SHA256,
    TCGACaseListReceipt,
    TCGASampleBarcode,
    build_tcga_selected_sample_axis,
    parse_tcga_participant_id,
    parse_tcga_sample_barcode,
    parse_tcga_sequenced_case_list,
    select_one_sample_per_participant,
    tcga_datahub_case_list_path,
    tcga_datahub_public_path,
    tcga_datahub_study_id,
)


def test_primary_disease_codes_are_exact_and_immutable():
    assert frozenset({"01", "03", "09"}) == PRIMARY_DISEASE_SAMPLE_TYPE_CODES


def test_revision_datahub_source_and_cohort_family_are_frozen():
    assert TCGA_DATAHUB_COMMIT == "64392efc82b38655f67188a4e95e44ca22e030c0"
    assert TCGA_DATAHUB_TREE == "199590867e8780b05b9873fb0867241a1f4fbea0"
    assert len(TCGA_COHORTS) == len(set(TCGA_COHORTS)) == 32
    assert tuple(sorted(TCGA_COHORTS)) == TCGA_COHORTS
    assert tuple(sorted(TCGA_MAF_SHA256)) == TCGA_COHORTS
    assert tuple(sorted(TCGA_CASE_LIST_RECEIPTS)) == TCGA_COHORTS
    assert tuple(sorted(TCGA_SELECTED_SAMPLE_AXIS_SHA256)) == TCGA_COHORTS
    assert (
        sum(
            receipt.sample_count for receipt in TCGA_CASE_LIST_RECEIPTS.values()
        )
        == 10443
    )
    assert (
        sum(
            receipt.participant_count
            for receipt in TCGA_CASE_LIST_RECEIPTS.values()
        )
        == 10433
    )
    assert all(
        len(digest) == 64 and set(digest) <= set("0123456789abcdef")
        for digest in TCGA_MAF_SHA256.values()
    )
    assert all(
        len(digest) == 64 and set(digest) <= set("0123456789abcdef")
        for digest in TCGA_SELECTED_SAMPLE_AXIS_SHA256.values()
    )


def test_datahub_study_and_public_paths_bind_crad_to_combined_study():
    assert tcga_datahub_study_id("CHOL") == "chol_tcga_pan_can_atlas_2018"
    assert tcga_datahub_study_id("CRAD") == "coadread_tcga_pan_can_atlas_2018"
    assert tcga_datahub_public_path(
        "CRAD",
        "data_mutations.txt",
    ).as_posix() == (
        "public/coadread_tcga_pan_can_atlas_2018/data_mutations.txt"
    )
    assert tcga_datahub_case_list_path("CRAD").as_posix() == (
        "public/coadread_tcga_pan_can_atlas_2018/"
        "case_lists/cases_sequenced.txt"
    )


def _synthetic_case_list(sample_ids):
    joined_ids = "\t".join(sample_ids)
    description = f"Samples with mutation data ({len(sample_ids)} samples)"
    return (
        "cancer_study_identifier: chol_tcga_pan_can_atlas_2018\n"
        "stable_id: chol_tcga_pan_can_atlas_2018_sequenced\n"
        "case_list_name: Samples with mutation data\n"
        f"case_list_description: {description}\n"
        "case_list_category: all_cases_with_mutation_data\n"
        f"case_list_ids: {joined_ids}\n"
    ).encode()


def _install_synthetic_case_list_receipt(monkeypatch, content, participant_count):
    sample_count = len(
        content.decode().splitlines()[-1].removeprefix("case_list_ids: ").split("\t"),
    )
    monkeypatch.setattr(
        tcga_module,
        "TCGA_CASE_LIST_RECEIPTS",
        MappingProxyType(
            {
                "CHOL": TCGACaseListReceipt(
                    hashlib.sha256(content).hexdigest(),
                    participant_count,
                    sample_count,
                ),
            },
        ),
    )


def _install_synthetic_selected_axis_receipt(monkeypatch, sample_ids):
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        encoded = sample_id.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    monkeypatch.setattr(
        tcga_module,
        "TCGA_SELECTED_SAMPLE_AXIS_SHA256",
        MappingProxyType({"CHOL": digest.hexdigest()}),
    )


def test_case_list_parser_binds_receipt_metadata_and_aggregate_counts(monkeypatch):
    sample_ids = ("TCGA-AA-0001-01", "TCGA-BB-0002-01")
    content = _synthetic_case_list(sample_ids)
    _install_synthetic_case_list_receipt(monkeypatch, content, participant_count=2)

    assert parse_tcga_sequenced_case_list(content, "CHOL") == sample_ids


def test_selected_axis_parser_projects_participants_and_checks_digest(monkeypatch):
    sample_ids = (
        "TCGA-ZZ-0002-06",
        "TCGA-AA-0001-02",
        "TCGA-ZZ-0002-01",
    )
    selected = ("TCGA-AA-0001-02", "TCGA-ZZ-0002-01")
    content = _synthetic_case_list(sample_ids)
    _install_synthetic_case_list_receipt(monkeypatch, content, participant_count=2)
    _install_synthetic_selected_axis_receipt(monkeypatch, selected)

    assert build_tcga_selected_sample_axis(content, "CHOL") == selected


def test_case_list_parser_rejects_receipt_mismatch():
    content = _synthetic_case_list(("TCGA-AA-0001-01",))

    with pytest.raises(ValueError, match="case-list SHA-256 mismatch"):
        parse_tcga_sequenced_case_list(content, "CHOL")


def test_case_list_parser_rejects_duplicate_sample_ids(monkeypatch):
    sample_ids = ("TCGA-AA-0001-01", "TCGA-AA-0001-01")
    content = _synthetic_case_list(sample_ids)
    _install_synthetic_case_list_receipt(monkeypatch, content, participant_count=1)

    with pytest.raises(ValueError, match="duplicate sample IDs"):
        parse_tcga_sequenced_case_list(content, "CHOL")


@pytest.mark.parametrize("cohort", ["COAD", "READ", "crad", "", "../CRAD"])
def test_datahub_study_id_rejects_cohorts_outside_the_frozen_family(cohort):
    with pytest.raises(ValueError, match="Unknown TCGA revision cohort"):
        tcga_datahub_study_id(cohort)


@pytest.mark.parametrize("filename", ["", ".", "..", "case/list.txt", "/x"])
def test_datahub_public_path_rejects_unsafe_filenames(filename):
    with pytest.raises(ValueError, match="one nonempty path basename"):
        tcga_datahub_public_path("CHOL", filename)


def test_parse_tcga_participant_id_preserves_valid_identifier():
    participant_id = "TCGA-A8-A06X"

    assert parse_tcga_participant_id(participant_id) == participant_id


@pytest.mark.parametrize(
    "participant_id",
    [
        None,
        "TCGA-A8-A06",
        "TCGA-A8-A06XX",
        "tcga-A8-A06X",
        "TCGA-a8-A06X",
        "TCGA-A8-A06_",
        " TCGA-A8-A06X",
        "TCGA-A8-A06X\n",
    ],
)
def test_parse_tcga_participant_id_rejects_malformed_values(participant_id):
    with pytest.raises(ValueError, match="12-character"):
        parse_tcga_participant_id(participant_id)


def test_parse_tcga_sample_barcode_returns_validated_components():
    assert parse_tcga_sample_barcode("TCGA-A8-A06X-01") == TCGASampleBarcode(
        sample_id="TCGA-A8-A06X-01",
        participant_id="TCGA-A8-A06X",
        sample_type_code="01",
    )


@pytest.mark.parametrize(
    "sample_id",
    [
        None,
        "TCGA-A8-A06X-1",
        "TCGA-A8-A06X-001",
        "TCGA-A8-A06X-AA",
        "tcga-A8-A06X-01",
        "TCGA-a8-A06X-01",
        "TCGA-A8-A06X-01A",
        "TCGA-A8-A06X-01 ",
        "TCGA-A8-A06X-01\n",
    ],
)
def test_parse_tcga_sample_barcode_rejects_malformed_values(sample_id):
    with pytest.raises(ValueError, match="15-character"):
        parse_tcga_sample_barcode(sample_id)


def test_projection_is_order_independent_and_lexicographically_sorted():
    sample_ids = [
        "TCGA-ZZ-0002-06",
        "TCGA-AA-0001-02",
        "TCGA-ZZ-0002-01",
        "TCGA-BB-0003-09",
        "TCGA-BB-0003-11",
    ]

    forward = select_one_sample_per_participant(sample_ids)
    reverse = select_one_sample_per_participant(reversed(sample_ids))

    expected = (
        "TCGA-AA-0001-02",
        "TCGA-BB-0003-09",
        "TCGA-ZZ-0002-01",
    )
    assert forward == expected
    assert reverse == expected


@pytest.mark.parametrize("singleton_type", ["02", "06", "11", "99"])
def test_projection_retains_a_singleton_regardless_of_sample_type(singleton_type):
    sample_id = f"TCGA-AA-0001-{singleton_type}"

    assert select_one_sample_per_participant([sample_id]) == (sample_id,)


def test_projection_accepts_exactly_one_primary_among_multiple_samples():
    sample_ids = [
        "TCGA-AA-0001-06",
        "TCGA-AA-0001-03",
        "TCGA-AA-0001-11",
    ]

    assert select_one_sample_per_participant(sample_ids) == (
        "TCGA-AA-0001-03",
    )


def test_projection_rejects_repeated_participant_without_a_primary():
    sample_ids = ["TCGA-AA-0001-02", "TCGA-AA-0001-06"]

    with pytest.raises(ValueError, match="exactly one primary-disease sample"):
        select_one_sample_per_participant(sample_ids)


def test_projection_rejects_repeated_participant_with_multiple_primaries():
    sample_ids = ["TCGA-AA-0001-01", "TCGA-AA-0001-03"]

    with pytest.raises(ValueError, match="found 2 among 2"):
        select_one_sample_per_participant(sample_ids)


def test_projection_rejects_duplicate_sample_ids():
    sample_id = "TCGA-AA-0001-01"

    with pytest.raises(ValueError, match="Duplicate TCGA sample barcode"):
        select_one_sample_per_participant([sample_id, sample_id])


def test_projection_supports_an_explicit_primary_type_contract():
    sample_ids = ["TCGA-AA-0001-01", "TCGA-AA-0001-06"]

    assert select_one_sample_per_participant(
        sample_ids,
        primary_sample_type_codes={"06"},
    ) == ("TCGA-AA-0001-06",)


@pytest.mark.parametrize("primary_codes", [set(), {"1"}, {"AA"}, {" 01"}])
def test_projection_rejects_invalid_primary_type_contract(primary_codes):
    with pytest.raises(ValueError, match="nonempty two-digit strings"):
        select_one_sample_per_participant(
            ["TCGA-AA-0001-01"],
            primary_sample_type_codes=primary_codes,
        )


def test_projection_accepts_a_generator_and_an_empty_axis():
    sample_ids = (sample_id for sample_id in ["TCGA-AA-0001-01"])

    assert select_one_sample_per_participant(sample_ids) == (
        "TCGA-AA-0001-01",
    )
    assert select_one_sample_per_participant(iter(())) == ()
