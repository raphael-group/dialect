"""Verify the public CBaSE lineage, license, and release-file boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CBASE_ROOT = _REPO_ROOT / "external" / "CBaSE"
_NOTICE = _CBASE_ROOT / "NOTICE"
_LICENSE = _REPO_ROOT / "LICENSE"
_RECORD = (
    _REPO_ROOT / "provenance" / "dependencies" / "cbase-v1.2-dialect-fork.json"
)
_UPSTREAM_ARCHIVE_SHA256 = (
    "d2453bc208c0cb97fbbb530fe015c42c631564df9119fe59cb86b7a0902c5aae"
)
_COMPOSITE_LICENSE = "LicenseRef-CBaSE-Public-Domain AND BSD-3-Clause"
_BSD_LICENSE_SHA256 = (
    "3c900e08a49b06523f496c107d6d1548d3da2e45fba6972e628cde67f2251d16"
)
_RELEASE_FILES = {
    "external/CBaSE/CBaSE_params_v1.2.py": (
        "d3721916eaffbef7ce138f703a219849780fbcc1171b6e8d6b3833d9fd51474c"
    ),
    "external/CBaSE/CBaSE_qvals_v1.2.py": (
        "c1f0eb723f71f7206d9abd76fe81168204b6673bc2c8cdf805b2c72c3e1e81cd"
    ),
    "external/CBaSE/cbase_cohort_size.py": (
        "381a91aa575c90868ec088b21267c9320a364e2b17ff7fcaeac50c1c527d1d69"
    ),
    "external/CBaSE/NOTICE": (
        "b37c74f3f333fa3efdc5c81b72395cbada205e3fa427a03a5fabed2eb694fe80"
    ),
}
_OFFICIAL_REFERENCE_MEMBERS = {
    "CBaSE_v1.2/CBaSE_v1.2.py": {
        "bytes": 2085,
        "sha256": "8e19d8505c3bd36002065395627c9a266576e6c1a11ab999713c8f77beff58cd",
    },
    "CBaSE_v1.2/Auxiliary/CBaSE_v1.2_parameters.py": {
        "bytes": 37113,
        "sha256": "2923cacfbf6317b5b859a8e82c686f05d119bf95a1a78cfab32a4b4075a1f0e9",
    },
    "CBaSE_v1.2/Auxiliary/CBaSE_v1.2_qvalues.py": {
        "bytes": 30398,
        "sha256": "2e209740a15bb34e0fcba08f6da0eccc02c2ef1d2d1146c2162b93763d3998b3",
    },
}
_HISTORICAL_SNAPSHOT = {
    "commit": "d4c195fafcdcd18a0f066b139754f280ca068b66",
    "files": {
        "archive/bmrs/cbase/cbase_params_v1.2.py": {
            "bytes": 56121,
            "sha256": (
                "d6b3583d118603e802ae173d2593784004357bde35d83487f9587f658159cf08"
            ),
        },
        "archive/bmrs/cbase/cbase_qvals_v1.2.py": {
            "bytes": 42135,
            "sha256": (
                "327048957e576133c2a70be353bbf4e5f87ef9fea6f9aca33e30622bc84a720b"
            ),
        },
    },
}


def _sha256(path: Path) -> str:
    """Return the SHA-256 of one repository file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cbase_notice_binds_lineage_and_current_release_files() -> None:
    """Require the notice and record to close distinct lineage byte boundaries."""
    notice = _NOTICE.read_text(encoding="utf-8")
    record = json.loads(_RECORD.read_text(encoding="utf-8"))
    identity = record["identity"]

    assert _UPSTREAM_ARCHIVE_SHA256 in notice
    assert "https://weghornlab.org/Software/CBaSE_v1.2.zip" in notice
    assert "not represented as the exact parent" in notice
    assert "byte-identical upstream distribution snapshot" in notice
    assert identity["official_archive_reference_members"] == (
        _OFFICIAL_REFERENCE_MEMBERS
    )
    assert identity["historical_repository_snapshot"] == _HISTORICAL_SNAPSHOT
    for path, item in _OFFICIAL_REFERENCE_MEMBERS.items():
        assert path in notice
        assert item["sha256"] in notice
    for path, item in _HISTORICAL_SNAPSHOT["files"].items():
        assert path in notice
        assert item["sha256"] in notice
    for relative, expected_sha256 in _RELEASE_FILES.items():
        assert _sha256(_REPO_ROOT / relative) == expected_sha256
        assert identity["release_files"][relative] == expected_sha256
        assert Path(relative).name in notice


def test_cbase_record_separates_upstream_and_dialect_licenses() -> None:
    """Do not apply the upstream Public Domain label to DIALECT-authored files."""
    record = json.loads(_RECORD.read_text(encoding="utf-8"))
    identity = record["identity"]

    assert record["license_id"] == _COMPOSITE_LICENSE
    assert _sha256(_LICENSE) == _BSD_LICENSE_SHA256
    assert identity["dialect_license_file_sha256"] == _BSD_LICENSE_SHA256
    assert identity["release_file_licenses"] == {
        "external/CBaSE/CBaSE_params_v1.2.py": [
            "LicenseRef-CBaSE-Public-Domain",
            "BSD-3-Clause",
        ],
        "external/CBaSE/CBaSE_qvals_v1.2.py": [
            "LicenseRef-CBaSE-Public-Domain",
            "BSD-3-Clause",
        ],
        "external/CBaSE/cbase_cohort_size.py": ["BSD-3-Clause"],
        "external/CBaSE/NOTICE": ["BSD-3-Clause"],
    }
    assert identity["release_file_roles"][
        "external/CBaSE/cbase_cohort_size.py"
    ] == "dialect-authored-helper"
    assert identity["release_file_roles"][
        "external/CBaSE/NOTICE"
    ] == "dialect-authored-provenance-notice"


def test_cbase_forks_preserve_upstream_headers() -> None:
    """Keep authorship, version, and Public Domain headers in derived scripts."""
    for filename in ("CBaSE_params_v1.2.py", "CBaSE_qvals_v1.2.py"):
        header = (_CBASE_ROOT / filename).read_text(encoding="utf-8")[:1200]
        assert "Author:      Donate Weghorn" in header
        assert "License:     Public Domain" in header
        assert "Version:     1.2" in header
