"""Pure TCGA barcode parsing and participant-axis projection.

This module contains no file I/O and is deliberately not wired into the analysis
pipeline.  It makes the proposed one-observation-per-participant contract explicit
and independently testable before any result-bearing input rebuild is authorized.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping

PRIMARY_DISEASE_SAMPLE_TYPE_CODES: Final[frozenset[str]] = frozenset(
    {"01", "03", "09"},
)
"""TCGA sample-type codes treated as primary disease specimens."""

TCGA_DATAHUB_COMMIT: Final = "64392efc82b38655f67188a4e95e44ca22e030c0"
"""Pinned cBioPortal DataHub commit for the revision input contract."""

TCGA_DATAHUB_TREE: Final = "199590867e8780b05b9873fb0867241a1f4fbea0"
"""Git tree bound to :data:`TCGA_DATAHUB_COMMIT`."""

TCGA_COHORTS: Final[tuple[str, ...]] = (
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
"""Canonical 32-cohort TCGA PanCancer Atlas revision family."""


@dataclass(frozen=True, slots=True)
class TCGACaseListReceipt:
    """Frozen aggregate receipt for one sequenced-sample case list."""

    sha256: str
    participant_count: int
    sample_count: int


TCGA_CASE_LIST_RECEIPTS: Final[Mapping[str, TCGACaseListReceipt]] = MappingProxyType(
    {
        "ACC": TCGACaseListReceipt(
            "227c6cf8de2246ecc369f38b00f6785ebca672f48af61b021b98d57a522a71ee",
            91,
            91,
        ),
        "BLCA": TCGACaseListReceipt(
            "7740539eb1477585744dfab420102b11fc7381a72aef904fd8e9bb09373652b9",
            410,
            410,
        ),
        "BRCA": TCGACaseListReceipt(
            "ae464df8154970f13b1cfe34e5538f9286abbe8bd13e1105474638ff2b45c19a",
            1066,
            1066,
        ),
        "CESC": TCGACaseListReceipt(
            "b59d6f01fa71bf18b74162fe60dea3c18949af4161887696097cffc64a44d07b",
            291,
            291,
        ),
        "CHOL": TCGACaseListReceipt(
            "607dd5ba07255231dac1ce3384351d0777ca1fdec69c5f60db131addf5630ce7",
            36,
            36,
        ),
        "CRAD": TCGACaseListReceipt(
            "03e55fba68c9aac8ce44f6c2f1e0f981cdb8130d5713b899c25aefa844f0da99",
            534,
            534,
        ),
        "DLBC": TCGACaseListReceipt(
            "8b1ca5c9cd2070a86810c8d3942167ec074edc868721404dac0cf45aeeba0bf6",
            41,
            41,
        ),
        "ESCA": TCGACaseListReceipt(
            "9ac79a47561fec1a8f8990c6f040ebeb7cad4dd3250195f326bcc51a7885b229",
            182,
            182,
        ),
        "GBM": TCGACaseListReceipt(
            "2b6158e9013e6a4c65c6125e9782f41d855e574066a44b7459a7cf4758a69043",
            390,
            397,
        ),
        "HNSC": TCGACaseListReceipt(
            "831603548eb6bc6cac9688db2df021f653c20f69d2f2887e1867032da32e94bc",
            515,
            515,
        ),
        "KICH": TCGACaseListReceipt(
            "0740f16dd12198c1eff2b27862c80d0c16764874cd0245ee195d791d3797aa20",
            65,
            65,
        ),
        "KIRC": TCGACaseListReceipt(
            "c024c74a755636c00ec11829d8ed797a7164f75b2bcb124d0b53e8d852bcb263",
            402,
            402,
        ),
        "KIRP": TCGACaseListReceipt(
            "c3984c1b1849133da6c8e3bdddf5c6da27a36f0d48aa743fee37f02b0f17ab33",
            276,
            276,
        ),
        "LAML": TCGACaseListReceipt(
            "ba3e5830d6df8785b9fe97934ec160f7f3733a2971aff38bbeaeab74a99cdeb9",
            200,
            200,
        ),
        "LGG": TCGACaseListReceipt(
            "32d4b65bdae8e33a7e981ba4476076b0183728bb80be559a016cca943fd9cb80",
            514,
            514,
        ),
        "LIHC": TCGACaseListReceipt(
            "15927e2551e5f7f133b0ab816230538755c6a2212b18e822c9baedac79a79719",
            366,
            366,
        ),
        "LUAD": TCGACaseListReceipt(
            "0a804793afbbf753b4ade84f86884430692e62dd41951b093ca585c59252d263",
            566,
            566,
        ),
        "LUSC": TCGACaseListReceipt(
            "e088a1d7ae386488befeac646366f3bd1df54f37e95ce7452709cf361985d1fd",
            484,
            484,
        ),
        "MESO": TCGACaseListReceipt(
            "379528a6573a951a74fb8de5cc705f454845a33b2244d62d78d2bc1aaf08e57a",
            86,
            86,
        ),
        "OV": TCGACaseListReceipt(
            "03ab65fdadde077cbcfa2a220ccc5188b79daf7a54939c30f6b4f91059a22f33",
            523,
            523,
        ),
        "PAAD": TCGACaseListReceipt(
            "161fda79bbcbc2984b8b4ded75fdf4103997d84746735bb507fd6a7ede06ae07",
            179,
            179,
        ),
        "PCPG": TCGACaseListReceipt(
            "9d7b6d0f9d4277bb9973d11597468d80300dc38d82b898b820c1461e27cdc946",
            178,
            178,
        ),
        "PRAD": TCGACaseListReceipt(
            "908ee334c771b2ad8d6f9234e8ccdd036fe21fda24d943bedffd6454436a61e1",
            494,
            494,
        ),
        "SARC": TCGACaseListReceipt(
            "3c8cabefe19bfaabf7a6065c16d98ce5a8e00240c4374a55982e1b9733b95c60",
            255,
            255,
        ),
        "SKCM": TCGACaseListReceipt(
            "359e57cf014511197bf8dfae9d3f3718f50c8cd7fa87a254a99d17d40e3e7094",
            438,
            440,
        ),
        "STAD": TCGACaseListReceipt(
            "ea288371d3011c4d5fdf94371307d0c9c66290c9f6bfee6718b02d61c3c5bd87",
            436,
            436,
        ),
        "TGCT": TCGACaseListReceipt(
            "85f670c9f903b398bd5a77173b014ddb5229f19d7085699fd7367e5c6900bd12",
            149,
            149,
        ),
        "THCA": TCGACaseListReceipt(
            "9db20d7c7251eeebd48ad647964de8eee5558ec32ad448ae358b978b8be5bf92",
            489,
            490,
        ),
        "THYM": TCGACaseListReceipt(
            "b2581fca63a3f8a24b6c80b25a7ab3ce7b0256461626a9ac3155adec9b330b7a",
            123,
            123,
        ),
        "UCEC": TCGACaseListReceipt(
            "b5202a9c7ddd1a66a32500f1f784a49f8ea623a4b83a4cad4d145b457537ac41",
            517,
            517,
        ),
        "UCS": TCGACaseListReceipt(
            "9c2f3b73dd7b969c18010a7e283c115fc9b7d15306e02dfa03a597195ee248bf",
            57,
            57,
        ),
        "UVM": TCGACaseListReceipt(
            "fa30cefdfa02c9f3a8b1c3018590af6cc84e1fd6e52060e8d93ec86f37fc0c53",
            80,
            80,
        ),
    },
)
"""Commit-matched case-list hashes and aggregate sample/participant counts."""

TCGA_SELECTED_SAMPLE_AXIS_SHA256: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ACC": "67a0442904016c7507a347a177bdcc06f63880ae1319f3b139bbbf7d67b50ebd",
        "BLCA": "fe2314ec8465dd6ab914b4f99de7365bc940c1253a9d1c83efad675f596d9cfa",
        "BRCA": "532da76666344100d6808ee005b541b92bfce52788e71fe54635c22d133bc554",
        "CESC": "f1ce106800dcfa30cc8d891b26fce9710dca459ff3e420e53f615ca0d805f5ee",
        "CHOL": "4c0fccb7692398dc236f4c8a5377d887e0b239996d85e120a44f715363225c3b",
        "CRAD": "9f5a4da82cb2839b52a2b84cc906ac7bfed970f7e849d79519ffac9fbe44990f",
        "DLBC": "ec9e7688186a373286314849c6a9555f254a5f13bd7e677a5f49185fd5304d7c",
        "ESCA": "cf0c0d98f3b196c8e0d57190a8ce2a793935282eb77989bded1d0b44d61214e3",
        "GBM": "219f120a91050e480d256c5a446b3a19338551e183a8822db4d23c183c97eec3",
        "HNSC": "0cf7e2d4c30713f56a00ac6bc8102d00bf063f7458da55d087dc0cc5d4da0c8a",
        "KICH": "eb37e3983191e2e4666425bf120016cf5e19571bb2efdb6ac1b83c4ebab4f1fa",
        "KIRC": "f17e28c1564303f78a4264f65ccbb07726db73bff3de9a4547b0cf1c048b06d3",
        "KIRP": "2f32f9dd2528eb9edf25b5f3189755fbddfc0bf5cd4b67290f8187e5c78e2761",
        "LAML": "007f225448343f9112e18aa89b6a667efcc41e722dac00e4b985b3fbc4eab62e",
        "LGG": "ffd25471da9b1cb8389038979dd8884ae8fdc49f457adbe77489d902014c98ba",
        "LIHC": "2f2bc389956019755bb843897c01acb65d7e210364edb32e61bc1aed1b73c9d3",
        "LUAD": "bba000b1abdb0d34b791ec00298157946f41ca073b4d981ff2c88f2b1542386f",
        "LUSC": "cf2807c2e5360172676aad69e909a6b2d78973674a77c2feaa167a95014fb48c",
        "MESO": "72de92c7f8eff46ade26099b42c0a766545c90c79e5d1563dc7502dc27e75189",
        "OV": "79a5da8a7ecde8b73c955c41103c6ec6663b3b319a7453a54297bce3170fad73",
        "PAAD": "b2003667293c3fc1d8c83171326034424a9ea1a1c3f691e9a96ae3c1156f7eaf",
        "PCPG": "5a1439e9b57e3569b8565980b0188bd1eb7d456a6bbe2be0cd38ed58233aa1b4",
        "PRAD": "cf64638d25bb14c4c3cbc299f03100896f13949ecb9acb52b559b2185d1d5bcd",
        "SARC": "2def7b9fbeba6ec213d185e0d5c1bd189d329c98946b26a69ba0891d74859e91",
        "SKCM": "49ee190d5ee5c6d07f57fb677056bd07f68f09791e8445c8d5253b89de47026b",
        "STAD": "8763c3a64b4f13dafc7d6ae942a4bb519e40332592c996173d65713e45d9995b",
        "TGCT": "03df1a6eae14cf3912c9fd723020a96aa6fcf142997f20e2dd500f8bef7ddab4",
        "THCA": "77fcbc75705e4c806e62c8bd63322b2ca915788cfee7d6dc528bf58d86d80cfd",
        "THYM": "067ec50bab2fe1426fd82a732097db2a8370dc20168fbf9f670b7362da0056f2",
        "UCEC": "9d6b4899c1b446062e002fdffc6096ae9e9ca97776afd0a33d4c67d5dea291fc",
        "UCS": "187ecca73b1e12cb893f35f7c961a1cc40ade58dc66800725a946e357ba99ff4",
        "UVM": "919e92ab058fb987602e5819bfb617abd5c180c6eb6bd63a1582bdb4c6b947f7",
    },
)
"""Length-framed SHA-256 of each selected, lexicographically ordered sample axis."""

TCGA_MAF_SHA256: Final = MappingProxyType(
    {
        "ACC": "3912bbe4b5cceac75eb3e4293f5061d2f7c2dc5bda55bcccb474acb337317d64",
        "BLCA": "9a0db2407fe54796d991c970bb8a1122a2ec1a8af757003506813b36b48c270c",
        "BRCA": "9cb009b18e3ea2efbd7ae124c7a78490135e9a9e4288b1a5ee904f5546fa7d02",
        "CESC": "111f199942c2e01208cff807ada28b70016c73b8f47a74021ea7d014d34af692",
        "CHOL": "8a0b57668d83d48bfd6b0d64bee28fa702384312866e574cf830e91df912b118",
        "CRAD": "f3e27a06e4497767bdae99e993a4f627493b6cce1f7842658b102bd6dcc7091a",
        "DLBC": "d1cdc7f01b998189c535a4433185b79f958b5ab4a247eb44d0362b651873e91e",
        "ESCA": "772657b704413cfaaedc7c3b9987a9f085ca071eaa8ef724aa2c07046924af2d",
        "GBM": "da880391ed9f5d1d68138ee3d1e6563d14ae2b198c4a08eb59f06bf2e0d69f11",
        "HNSC": "3a36b5fa05f666b2b40f0af38a04ff94f7c3b24189f02f1998f4fba567088916",
        "KICH": "1f4b833a7e448cb9abe9e8928ab72f81a3524dab15f267aae447b78f5d0148a2",
        "KIRC": "82a230cd1d234065385a7ce1444f3d749c9df5e9d2bb767576a04b872dc3cff7",
        "KIRP": "ade1cc182b0b511f06774d7663bc81d5a5ad1bf451672f5fb7ff38c0662855a2",
        "LAML": "b77ffe65a403978060185ffb29926585465adf0f6f3a65251417cf49dcdf9c6e",
        "LGG": "5c28ab7231560ee9d2835312c3c18574c03f67a65547b6297b22c1e4d3caff73",
        "LIHC": "8f231d1e5e669d1b65c55fec50ae494ce86b0adf8687bb1a3931fe56f1811fff",
        "LUAD": "a08847f9536886be6de4399caabf87202afacf19ee79f0500284020d58827079",
        "LUSC": "c2f267ad1961df75f805bedfa07a987c629a7d272c7755a79aee0cba9c200a05",
        "MESO": "a319054a8dc1ecdd3dbc59343bf3ba249cf0273e097df558c5881875d13b20b8",
        "OV": "f2039f96207b55c6871ca1025fe933ef4ba884a0514a5835698ad3acbf44b9ec",
        "PAAD": "e2c557f3a3d9d6e0b7539df124cb7b3d31f1ce160b380abb807717837e87cf49",
        "PCPG": "00ae4fe0323ba9e7c3f7f7c06556fd80c759838ad7040b3534e4bdc2e0385060",
        "PRAD": "a33181d3a337196fd5349b07720828dacf0d352413f8dc2f99e62af37c292ad9",
        "SARC": "0191ae403a954235200ee990dc878246212a842459660a9c063c66684a02c1e7",
        "SKCM": "00deb86dbab6686ac7abaf7b30172bb599e2c4ccaf9a2d4d60fb49e5864912ee",
        "STAD": "84da71c9c20598ebc5e62a8ee078466d780abe31d5d2415918058a14c77bf808",
        "TGCT": "55bd31dc3c49eca52e2a310e326b0835b2242af4d021fafca84e7228c96cdb46",
        "THCA": "5c002df5d3ae23640317a67441582a8b4d4bd01913abc5ac07e7d7045cc73039",
        "THYM": "f92c02add946866e92dd581e1d775d1cd3008dc19e226520ca35c3993b00816b",
        "UCEC": "1c7ea2ed1015dfae1987c2c0d3dd8180348d69b912c065b6673d5faa2d2ee4cd",
        "UCS": "79e267d6a08c78fc3c64903b1bc9208cffc37b597f931e9ad1aa6310a6245175",
        "UVM": "fd4b4cf70ca3ced50c0b925e59ce2f480e1f6fc221303da255e6eb39ee6d2d95",
    },
)
"""SHA-256 of each raw MAF at the pinned DataHub commit."""

_STUDY_CODE_OVERRIDES: Final = {"CRAD": "coadread"}
_PANCAN_STUDY_SUFFIX: Final = "_tcga_pan_can_atlas_2018"
_SEQUENCED_CASE_LIST_CATEGORY: Final = "all_cases_with_mutation_data"
_CASE_LIST_FIELDS: Final = (
    "cancer_study_identifier",
    "stable_id",
    "case_list_name",
    "case_list_description",
    "case_list_category",
    "case_list_ids",
)

_PARTICIPANT_ID_PATTERN = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}")
_SAMPLE_BARCODE_PATTERN = re.compile(
    r"(?P<participant_id>TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-"
    r"(?P<sample_type_code>[0-9]{2})",
)
_SAMPLE_TYPE_CODE_PATTERN = re.compile(r"[0-9]{2}")


@dataclass(frozen=True, slots=True)
class TCGASampleBarcode:
    """The validated components of a 15-character TCGA sample barcode.

    Attributes:
        sample_id: Complete 15-character tumor-sample barcode.
        participant_id: Corresponding 12-character participant identifier.
        sample_type_code: Two-digit TCGA specimen-type code.
    """

    sample_id: str
    participant_id: str
    sample_type_code: str


def tcga_datahub_study_id(cohort: str) -> str:
    """Return the pinned DataHub study identifier for one DIALECT cohort.

    Args:
        cohort: Exact uppercase member of :data:`TCGA_COHORTS`.

    Returns:
        cBioPortal PanCancer Atlas study identifier. ``CRAD`` maps to the
        combined COAD/READ study.

    Raises:
        ValueError: If ``cohort`` is not in the frozen 32-cohort family.
    """
    if cohort not in TCGA_COHORTS:
        msg = f"Unknown TCGA revision cohort: {cohort!r}."
        raise ValueError(msg)
    code = _STUDY_CODE_OVERRIDES.get(cohort, cohort.lower())
    return f"{code}{_PANCAN_STUDY_SUFFIX}"


def tcga_datahub_public_path(cohort: str, filename: str) -> PurePosixPath:
    """Return a validated repository-relative path in the pinned study.

    Args:
        cohort: Exact uppercase member of :data:`TCGA_COHORTS`.
        filename: One basename beneath the study's ``public`` directory.

    Returns:
        Repository-relative POSIX path.

    Raises:
        ValueError: If the cohort is unknown or ``filename`` is not one safe
            nonempty basename.
    """
    if (
        not isinstance(filename, str)
        or not filename
        or filename != PurePosixPath(filename).name
        or filename in {".", ".."}
    ):
        msg = "DataHub filename must be one nonempty path basename."
        raise ValueError(msg)
    return PurePosixPath("public", tcga_datahub_study_id(cohort), filename)


def tcga_datahub_case_list_path(cohort: str) -> PurePosixPath:
    """Return the frozen sequenced-sample case-list path for one cohort."""
    return PurePosixPath(
        "public",
        tcga_datahub_study_id(cohort),
        "case_lists",
        "cases_sequenced.txt",
    )


def parse_tcga_sequenced_case_list(content: bytes, cohort: str) -> tuple[str, ...]:
    """Validate and parse one exact commit-matched sequenced-sample case list.

    Args:
        content: Raw Git blob bytes.
        cohort: Exact uppercase member of :data:`TCGA_COHORTS`.

    Returns:
        Sample barcodes in their source order.

    Raises:
        TypeError: If ``content`` is not bytes.
        ValueError: If the receipt, metadata, or identifier contract is violated.
    """
    study_id = tcga_datahub_study_id(cohort)
    if not isinstance(content, bytes):
        msg = "TCGA case-list content must be raw bytes."
        raise TypeError(msg)
    receipt = TCGA_CASE_LIST_RECEIPTS[cohort]
    observed_sha256 = hashlib.sha256(content).hexdigest()
    if observed_sha256 != receipt.sha256:
        msg = (
            f"TCGA {cohort} case-list SHA-256 mismatch: expected {receipt.sha256}, "
            f"observed {observed_sha256}."
        )
        raise ValueError(msg)
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        msg = f"TCGA {cohort} case list is not valid UTF-8."
        raise ValueError(msg) from error
    if len(lines) != len(_CASE_LIST_FIELDS):
        msg = f"TCGA {cohort} case list must contain exactly six metadata rows."
        raise ValueError(msg)

    fields: dict[str, str] = {}
    for expected_field, line in zip(_CASE_LIST_FIELDS, lines, strict=True):
        field, separator, value = line.partition(": ")
        if not separator or field != expected_field or not value:
            msg = f"TCGA {cohort} case list has an invalid {expected_field} row."
            raise ValueError(msg)
        fields[field] = value

    expected_stable_id = f"{study_id}_sequenced"
    if (
        fields["cancer_study_identifier"] != study_id
        or fields["stable_id"] != expected_stable_id
        or fields["case_list_category"] != _SEQUENCED_CASE_LIST_CATEGORY
    ):
        msg = f"TCGA {cohort} case-list metadata does not match its frozen study."
        raise ValueError(msg)

    sample_ids = fields["case_list_ids"].split("\t")
    if len(sample_ids) != len(set(sample_ids)):
        msg = f"TCGA {cohort} case list contains duplicate sample IDs."
        raise ValueError(msg)
    parsed_barcodes = [
        parse_tcga_sample_barcode(sample_id) for sample_id in sample_ids
    ]
    parsed = tuple(barcode.sample_id for barcode in parsed_barcodes)
    participant_count = len(
        {barcode.participant_id for barcode in parsed_barcodes},
    )
    if (
        len(parsed) != receipt.sample_count
        or participant_count != receipt.participant_count
    ):
        msg = f"TCGA {cohort} case-list aggregate counts violate the frozen receipt."
        raise ValueError(msg)
    return parsed


def parse_tcga_participant_id(participant_id: object) -> str:
    """Validate and return an exact 12-character TCGA participant identifier.

    Accepted identifiers have the uppercase form ``TCGA-XX-XXXX``, where each
    ``X`` is an ASCII letter or digit. No whitespace, case normalization, or
    truncation is performed.

    Args:
        participant_id: Candidate participant identifier.

    Returns:
        The unchanged, validated participant identifier.

    Raises:
        ValueError: If ``participant_id`` is not an exact TCGA participant ID.
    """
    if not isinstance(participant_id, str) or _PARTICIPANT_ID_PATTERN.fullmatch(
        participant_id,
    ) is None:
        msg = (
            "TCGA participant ID must match the exact 12-character form "
            "'TCGA-XX-XXXX'."
        )
        raise ValueError(msg)
    return participant_id


def parse_tcga_sample_barcode(sample_id: object) -> TCGASampleBarcode:
    """Parse an exact 15-character TCGA sample barcode.

    Accepted barcodes have the uppercase form ``TCGA-XX-XXXX-NN``. The final
    two digits are the TCGA sample-type code. Longer aliquot, portion, analyte,
    or plate barcodes are intentionally rejected.

    Args:
        sample_id: Candidate tumor-sample barcode.

    Returns:
        Validated barcode components.

    Raises:
        ValueError: If ``sample_id`` is not an exact TCGA sample barcode.
    """
    if not isinstance(sample_id, str):
        match = None
    else:
        match = _SAMPLE_BARCODE_PATTERN.fullmatch(sample_id)
    if match is None:
        msg = (
            "TCGA sample barcode must match the exact 15-character form "
            "'TCGA-XX-XXXX-NN'."
        )
        raise ValueError(msg)

    participant_id = parse_tcga_participant_id(match.group("participant_id"))
    return TCGASampleBarcode(
        sample_id=sample_id,
        participant_id=participant_id,
        sample_type_code=match.group("sample_type_code"),
    )


def select_one_sample_per_participant(
    sample_ids: Iterable[str],
    *,
    primary_sample_type_codes: Collection[str] = PRIMARY_DISEASE_SAMPLE_TYPE_CODES,
) -> tuple[str, ...]:
    """Project TCGA samples onto a deterministic participant-unique axis.

    A participant represented by one sample retains that sample regardless of
    its sample-type code. A participant represented by multiple samples must
    have exactly one sample whose type belongs to ``primary_sample_type_codes``;
    that unique primary sample is retained. Any ambiguous collision fails closed.
    Duplicate sample IDs are invalid even if their rows would otherwise agree.

    Args:
        sample_ids: Exact 15-character TCGA sample barcodes.
        primary_sample_type_codes: Two-digit sample types considered primary
            disease specimens. Defaults to TCGA codes 01, 03, and 09.

    Returns:
        Selected sample IDs in lexicographic order, one per participant.

    Raises:
        ValueError: If a barcode or primary code is malformed, a sample ID is
            duplicated, or a repeated participant lacks exactly one primary.
    """
    primary_codes = _validate_primary_sample_type_codes(
        primary_sample_type_codes,
    )
    parsed_samples: list[TCGASampleBarcode] = []
    observed_ids: set[str] = set()
    for sample_id in sample_ids:
        parsed = parse_tcga_sample_barcode(sample_id)
        if parsed.sample_id in observed_ids:
            msg = f"Duplicate TCGA sample barcode: {parsed.sample_id}."
            raise ValueError(msg)
        observed_ids.add(parsed.sample_id)
        parsed_samples.append(parsed)

    by_participant: dict[str, list[TCGASampleBarcode]] = defaultdict(list)
    for sample in parsed_samples:
        by_participant[sample.participant_id].append(sample)

    selected: list[str] = []
    for participant_id, participant_samples in by_participant.items():
        if len(participant_samples) == 1:
            selected.append(participant_samples[0].sample_id)
            continue

        primary_samples = [
            sample
            for sample in participant_samples
            if sample.sample_type_code in primary_codes
        ]
        if len(primary_samples) != 1:
            msg = (
                f"Repeated participant {participant_id} must have exactly one "
                "primary-disease sample; "
                f"found {len(primary_samples)} among {len(participant_samples)}."
            )
            raise ValueError(msg)
        selected.append(primary_samples[0].sample_id)

    return tuple(sorted(selected))


def build_tcga_selected_sample_axis(content: bytes, cohort: str) -> tuple[str, ...]:
    """Build and attest the frozen participant-unique axis for one cohort."""
    case_list_samples = parse_tcga_sequenced_case_list(content, cohort)
    selected = select_one_sample_per_participant(case_list_samples)
    digest = hashlib.sha256()
    for sample_id in selected:
        encoded = sample_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    observed_sha256 = digest.hexdigest()
    expected_sha256 = TCGA_SELECTED_SAMPLE_AXIS_SHA256[cohort]
    if observed_sha256 != expected_sha256:
        msg = (
            f"TCGA {cohort} selected sample-axis SHA-256 mismatch: expected "
            f"{expected_sha256}, observed {observed_sha256}."
        )
        raise ValueError(msg)
    return selected


def _validate_primary_sample_type_codes(
    primary_sample_type_codes: Collection[str],
) -> frozenset[str]:
    codes = frozenset(primary_sample_type_codes)
    if not codes or any(
        not isinstance(code, str)
        or _SAMPLE_TYPE_CODE_PATTERN.fullmatch(code) is None
        for code in codes
    ):
        msg = "Primary TCGA sample-type codes must be nonempty two-digit strings."
        raise ValueError(msg)
    return codes
