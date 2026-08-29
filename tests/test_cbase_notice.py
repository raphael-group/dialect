"""Verify that the public CBaSE modification notice matches vendored bytes."""

from __future__ import annotations

import hashlib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CBASE_ROOT = _REPO_ROOT / "external" / "CBaSE"
_NOTICE = _CBASE_ROOT / "NOTICE"
_UPSTREAM_ARCHIVE_SHA256 = (
    "d2453bc208c0cb97fbbb530fe015c42c631564df9119fe59cb86b7a0902c5aae"
)
_VENDORED_FILES = {
    "CBaSE_params_v1.2.py": (
        "d3721916eaffbef7ce138f703a219849780fbcc1171b6e8d6b3833d9fd51474c"
    ),
    "CBaSE_qvals_v1.2.py": (
        "f626d9857db455eecc7b6093b9c23396d22aac92308a3475767871733bb154ff"
    ),
    "cbase_cohort_size.py": (
        "381a91aa575c90868ec088b21267c9320a364e2b17ff7fcaeac50c1c527d1d69"
    ),
}


def _sha256(path: Path) -> str:
    """Return the SHA-256 of one repository file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cbase_notice_binds_current_vendored_files() -> None:
    """Require the notice to identify the exact modified CBaSE files."""
    notice = _NOTICE.read_text(encoding="utf-8")

    assert _UPSTREAM_ARCHIVE_SHA256 in notice
    assert "https://weghornlab.org/Software/CBaSE_v1.2.zip" in notice
    assert '"License: Public Domain"' in notice
    assert "has not reviewed or endorsed DIALECT's modifications" in notice
    for filename, expected_sha256 in _VENDORED_FILES.items():
        assert _sha256(_CBASE_ROOT / filename) == expected_sha256
        assert filename in notice
        assert expected_sha256 in notice


def test_cbase_notice_preserves_upstream_headers() -> None:
    """Keep authorship, version, and public-domain headers in both upstream forks."""
    for filename in ("CBaSE_params_v1.2.py", "CBaSE_qvals_v1.2.py"):
        header = (_CBASE_ROOT / filename).read_text(encoding="utf-8")[:1200]
        assert "Author:      Donate Weghorn" in header
        assert "License:     Public Domain" in header
        assert "Version:     1.2" in header
