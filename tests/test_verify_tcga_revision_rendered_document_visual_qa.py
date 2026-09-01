"""Synthetic adversarial tests for signed visual-QA authentication."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import stat
import struct
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from analysis import verify_tcga_revision_rendered_document_visual_qa as verifier

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

# The suite intentionally tests private process, parser, and pinning seams with
# synthetic bytes and ephemeral Ed25519 keys only.
# ruff: noqa: D101, D102, D107, S603, SLF001

_PRINCIPAL = "synthetic.visual.reviewer@example.test"
_RELEASE_ID = "synthetic-visual-qa-release"
_NAMESPACE = "dialect-revision-rendered-document-visual-qa-v2"
_DOMAIN_SEPARATOR = b"DIALECT-RENDERED-DOCUMENT-VISUAL-QA-V2\n"
_RECEIPT_SCHEMA = "dialect-revision-rendered-document-visual-qa-v2"
_REVIEW_KIND = "human-visual-qa-attestation"
_REVIEWER_ROLE = "independent-visual-reviewer"
_ATTESTATION_STATEMENT = (
    "I attest that I inspected every bound rendered page and found none of the "
    "listed visual defects."
)
_CRITERIA = [
    "page-render-present",
    "no-obvious-render-failure",
    "no-clipping",
    "no-overlap",
]
_NON_INFERENCE_LIMITS = {
    "human_identity_to_key_binding": "caller-authority-not-inferred",
    "actual_visual_inspection": (
        "caller-authority-signed-claim-not-independently-observed"
    ),
    "current_private_key_control_or_signature_freshness": "not-inferred",
    "reviewer_independence": "signed-claim-not-independently-verified",
    "review_timestamp_accuracy": "signed-claim-not-independently-verified",
    "upstream_manifests_and_pdf_png_bytes": "not-opened-or-verified-here",
    "independent_anchor_provenance": "caller-authority-not-inferred",
    "scientific_correctness": "not-inferred",
    "coauthor_approval": "not-inferred",
    "journal_submission_or_acceptance": "not-inferred",
    "loaded_python_and_ssh_keygen_dependency_closure": "not-native-attested",
    "detached_process_descendants": "outside-process-group-containment",
}
_PDF_ORDER = (
    ("clean", "manuscript-clean.pdf", "clean revised manuscript PDF"),
    ("marked", "manuscript-marked.pdf", "marked manuscript PDF"),
    ("s1", "s1-appendix.pdf", "S1 Appendix PDF"),
    ("rebuttal", "response-to-reviewers.pdf", "response to reviewers PDF"),
)
_EXPECTED_PDF_SET_SHA256 = (
    "aaebd8fbfd738fd47da2e31fdd7f8d6423582cd2de14cab77122be5083c6e1e1"
)
_EXPECTED_RENDER_SET_SHA256 = (
    "de0cafc2033d72a55edcabb42493ba2a45a926e98708574ce34fc7c20d70619f"
)
_EXPECTED_RECEIPT_SHA256 = (
    "866facd45ff338c0c52705d2ab48c923ecfc6d8c2c347454142a986a23ce0b2f"
)
_GOLDEN_RECEIPT_BASE64 = (
    "eyJhdHRlc3RhdGlvbl9zdGF0ZW1lbnQiOiJJIGF0dGVzdCB0aGF0IEkgaW5zcGVjdGVkIGV2ZXJ5IGJvdW5kIHJl"
    "bmRlcmVkIHBhZ2UgYW5kIGZvdW5kIG5vbmUgb2YgdGhlIGxpc3RlZCB2aXN1YWwgZGVmZWN0cy4iLCJjcml0ZXJp"
    "YSI6WyJwYWdlLXJlbmRlci1wcmVzZW50Iiwibm8tb2J2aW91cy1yZW5kZXItZmFpbHVyZSIsIm5vLWNsaXBwaW5n"
    "Iiwibm8tb3ZlcmxhcCJdLCJkZXJpdmF0aW9uX21hbmlmZXN0X3NoYTI1NiI6IjhhYmYwMWQzZDRhNzkzOWYyMmNh"
    "MTE1ZDg5ZTg2ZDVmMDNhNDFmNjEyZDdiZTgzN2UxOThlZjVlYjk3NWYxMWYiLCJkb2N1bWVudHMiOlt7InBhZ2Vf"
    "Y291bnQiOjEsInBhZ2VzIjpbeyJkZWNpc2lvbiI6InBhc3MiLCJoZWlnaHRfcGl4ZWxzIjoxNjUwLCJpc3N1ZV9j"
    "b2RlcyI6W10sInBhZ2UiOjEsInBuZ19ieXRlcyI6MjAwMSwicG5nX21lbWJlciI6InBhZ2VzL2NsZWFuL3BhZ2Ut"
    "MDAwMS5wbmciLCJwbmdfc2hhMjU2IjoiNmU0NzRmYmNjZTE0MmQ0NzBhM2YwMDBhMDg3N2UxNmY2MmMyNTA1M2E5"
    "YzZhMjczOWFlODhiMTU1MDJlOTg5NyIsIndpZHRoX3BpeGVscyI6MTI3NX1dLCJwZGZfYnl0ZXMiOjEwMDEsInBk"
    "Zl9pZCI6ImNsZWFuIiwicGRmX21lbWJlciI6Im1hbnVzY3JpcHQtY2xlYW4ucGRmIiwicGRmX3JvbGUiOiJjbGVh"
    "biByZXZpc2VkIG1hbnVzY3JpcHQgUERGIiwicGRmX3NoYTI1NiI6IjRmOGM3ZjRmNTBjNjg4OWM1N2ViMjZjNjFj"
    "ZWVjZDllM2U5NWQxM2RhZDVjZDkwY2ZiYTBhYTc1NjM3OGEyMTcifSx7InBhZ2VfY291bnQiOjEsInBhZ2VzIjpb"
    "eyJkZWNpc2lvbiI6InBhc3MiLCJoZWlnaHRfcGl4ZWxzIjoxNjUwLCJpc3N1ZV9jb2RlcyI6W10sInBhZ2UiOjEs"
    "InBuZ19ieXRlcyI6MjAwMiwicG5nX21lbWJlciI6InBhZ2VzL21hcmtlZC9wYWdlLTAwMDEucG5nIiwicG5nX3No"
    "YTI1NiI6IjUzODdjY2MxNWUzNzEwNzI3YmQ0MTE3MDE3ODY5YTQwYzA1OGI0MTE0NzU1NWViNjI2MTgyMmZkMWYx"
    "Mjg2Y2MiLCJ3aWR0aF9waXhlbHMiOjEyNzV9XSwicGRmX2J5dGVzIjoxMDAyLCJwZGZfaWQiOiJtYXJrZWQiLCJw"
    "ZGZfbWVtYmVyIjoibWFudXNjcmlwdC1tYXJrZWQucGRmIiwicGRmX3JvbGUiOiJtYXJrZWQgbWFudXNjcmlwdCBQ"
    "REYiLCJwZGZfc2hhMjU2IjoiYTY1MDljNWI5ZDM1MTIwOWYyMzIzYTc3NTUyZDliYTcxNTQ2MGNkMTE1NmU2NWNi"
    "ODA1NTAyODA5MWU0ZWU2NyJ9LHsicGFnZV9jb3VudCI6MSwicGFnZXMiOlt7ImRlY2lzaW9uIjoicGFzcyIsImhl"
    "aWdodF9waXhlbHMiOjE2NTAsImlzc3VlX2NvZGVzIjpbXSwicGFnZSI6MSwicG5nX2J5dGVzIjoyMDAzLCJwbmdf"
    "bWVtYmVyIjoicGFnZXMvczEvcGFnZS0wMDAxLnBuZyIsInBuZ19zaGEyNTYiOiJjNTdjNGNmMTUzNjIzMDU4Mjdj"
    "OWEyNTcyNWIyZjBiMmVjMTQyOGJhYzU5MjBhNjEwZjMxYzYxMjM3YjYzMTNmIiwid2lkdGhfcGl4ZWxzIjoxMjc1"
    "fV0sInBkZl9ieXRlcyI6MTAwMywicGRmX2lkIjoiczEiLCJwZGZfbWVtYmVyIjoiczEtYXBwZW5kaXgucGRmIiwi"
    "cGRmX3JvbGUiOiJTMSBBcHBlbmRpeCBQREYiLCJwZGZfc2hhMjU2IjoiMjQ0Yzk5Y2QyNWQ2ZjZjOTZiOTM0MmU3"
    "MDBiOWFiZjBiMjQzYmZlZDY1ZGFkNDhjZjJlMmU3NjYzMDg1OWM3NCJ9LHsicGFnZV9jb3VudCI6MSwicGFnZXMi"
    "Olt7ImRlY2lzaW9uIjoicGFzcyIsImhlaWdodF9waXhlbHMiOjE2NTAsImlzc3VlX2NvZGVzIjpbXSwicGFnZSI6"
    "MSwicG5nX2J5dGVzIjoyMDA0LCJwbmdfbWVtYmVyIjoicGFnZXMvcmVidXR0YWwvcGFnZS0wMDAxLnBuZyIsInBu"
    "Z19zaGEyNTYiOiJiYjFlYzZjM2ViZjlmMDMwYzFiZjcwYTk4MTVhNGYzZDU5OWNhZThhYThhMjNjYTdiNTdmNTY2"
    "NTJiNTg0MTQ4Iiwid2lkdGhfcGl4ZWxzIjoxMjc1fV0sInBkZl9ieXRlcyI6MTAwNCwicGRmX2lkIjoicmVidXR0"
    "YWwiLCJwZGZfbWVtYmVyIjoicmVzcG9uc2UtdG8tcmV2aWV3ZXJzLnBkZiIsInBkZl9yb2xlIjoicmVzcG9uc2Ug"
    "dG8gcmV2aWV3ZXJzIFBERiIsInBkZl9zaGEyNTYiOiJhYmQ0N2M0N2I0Nzc3ZjI2ZDM1YjdhZjZkNDA5ODA1Y2Qy"
    "NmJkZmRmYWUxMTYwMjBiZjhiNjhkN2EwNmM2M2E3In1dLCJpbmRlcGVuZGVudF9yZXZpZXciOnRydWUsIm1hY2hp"
    "bmVfbWFuaWZlc3Rfc2hhMjU2IjoiYmMwMjBhMzViN2Y5Y2IxMzgyZTdiNTM0YzY4ZTNjNTMxZDg0OWIxMTliZjE0"
    "Zjc1ZGRlYWQ2Y2M0NWMzY2NjMSIsIm1vZGUiOiJmaW5hbCIsIm5vbl9pbmZlcmVuY2VfbGltaXRzIjp7ImFjdHVh"
    "bF92aXN1YWxfaW5zcGVjdGlvbiI6ImNhbGxlci1hdXRob3JpdHktc2lnbmVkLWNsYWltLW5vdC1pbmRlcGVuZGVu"
    "dGx5LW9ic2VydmVkIiwiY29hdXRob3JfYXBwcm92YWwiOiJub3QtaW5mZXJyZWQiLCJjdXJyZW50X3ByaXZhdGVf"
    "a2V5X2NvbnRyb2xfb3Jfc2lnbmF0dXJlX2ZyZXNobmVzcyI6Im5vdC1pbmZlcnJlZCIsImRldGFjaGVkX3Byb2Nl"
    "c3NfZGVzY2VuZGFudHMiOiJvdXRzaWRlLXByb2Nlc3MtZ3JvdXAtY29udGFpbm1lbnQiLCJodW1hbl9pZGVudGl0"
    "eV90b19rZXlfYmluZGluZyI6ImNhbGxlci1hdXRob3JpdHktbm90LWluZmVycmVkIiwiaW5kZXBlbmRlbnRfYW5j"
    "aG9yX3Byb3ZlbmFuY2UiOiJjYWxsZXItYXV0aG9yaXR5LW5vdC1pbmZlcnJlZCIsImpvdXJuYWxfc3VibWlzc2lv"
    "bl9vcl9hY2NlcHRhbmNlIjoibm90LWluZmVycmVkIiwibG9hZGVkX3B5dGhvbl9hbmRfc3NoX2tleWdlbl9kZXBl"
    "bmRlbmN5X2Nsb3N1cmUiOiJub3QtbmF0aXZlLWF0dGVzdGVkIiwicmV2aWV3X3RpbWVzdGFtcF9hY2N1cmFjeSI6"
    "InNpZ25lZC1jbGFpbS1ub3QtaW5kZXBlbmRlbnRseS12ZXJpZmllZCIsInJldmlld2VyX2luZGVwZW5kZW5jZSI6"
    "InNpZ25lZC1jbGFpbS1ub3QtaW5kZXBlbmRlbnRseS12ZXJpZmllZCIsInNjaWVudGlmaWNfY29ycmVjdG5lc3Mi"
    "OiJub3QtaW5mZXJyZWQiLCJ1cHN0cmVhbV9tYW5pZmVzdHNfYW5kX3BkZl9wbmdfYnl0ZXMiOiJub3Qtb3BlbmVk"
    "LW9yLXZlcmlmaWVkLWhlcmUifSwicGRmX3NldF9zaGEyNTYiOiJhYWViZDhmYmZkNzM4ZmQ0N2RhMmUzMWZkZDdm"
    "OGQ2NDIzNTgyY2QyZGUxNGNhYjc3MTIyYmU1MDgzYzZlMWUxIiwicmVidXR0YWxfcmVuZGVyZXJfbWFuaWZlc3Rf"
    "c2hhMjU2IjoiNmJkNTJiMjA0ZjViNGNmZmIyNjc1OTdmMzdkMGZhNjJiYWUyMjkzNDEzOTRkZmVjMGU1ZDQyNDM5"
    "ZDhiNzIyYyIsInJlbGVhc2VfaWQiOiJzeW50aGV0aWMtdmlzdWFsLXFhLXJlbGVhc2UiLCJyZW5kZXJfc2V0X3No"
    "YTI1NiI6ImRlMGNhZmMyMDMzZDcyYTU1ZWRjYWJiNDI0OTNiYTJhNDVhOTI2ZTk4NzA4NTc0Y2UzNGZjN2MyMGQ3"
    "MDYxOWYiLCJyZXZpZXdfaWQiOiJzeW50aGV0aWMtYWxsLXBhZ2UtcmV2aWV3IiwicmV2aWV3X2tpbmQiOiJodW1h"
    "bi12aXN1YWwtcWEtYXR0ZXN0YXRpb24iLCJyZXZpZXdlZF9hdF91dGMiOiIyMDI2LTA4LTMxVDEyOjAwOjAwWiIs"
    "InJldmlld2VyX3ByaW5jaXBhbCI6InN5bnRoZXRpYy52aXN1YWwucmV2aWV3ZXJAZXhhbXBsZS50ZXN0IiwicmV2"
    "aWV3ZXJfcm9sZSI6ImluZGVwZW5kZW50LXZpc3VhbC1yZXZpZXdlciIsInNjaGVtYSI6ImRpYWxlY3QtcmV2aXNp"
    "b24tcmVuZGVyZWQtZG9jdW1lbnQtdmlzdWFsLXFhLXYyIn0K"
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _ssh_string(raw: bytes, offset: int) -> tuple[bytes, int]:
    length = struct.unpack_from(">I", raw, offset)[0]
    start = offset + 4
    end = start + length
    return raw[start:end], end


def _pack_ssh_string(raw: bytes) -> bytes:
    return struct.pack(">I", len(raw)) + raw


def _decode_signature_armor(raw: bytes) -> bytes:
    lines = raw.decode("ascii").splitlines()
    return base64.b64decode("".join(lines[1:-1]), validate=True)


def _armor_signature(raw: bytes) -> bytes:
    encoded = base64.b64encode(raw).decode("ascii")
    lines = [encoded[index : index + 70] for index in range(0, len(encoded), 70)]
    return (
        "-----BEGIN SSH SIGNATURE-----\n"
        + "\n".join(lines)
        + "\n-----END SSH SIGNATURE-----\n"
    ).encode("ascii")


def _signature_parts(raw: bytes) -> dict[str, object]:
    decoded = _decode_signature_armor(raw)
    assert decoded.startswith(b"SSHSIG")
    version = struct.unpack_from(">I", decoded, 6)[0]
    public_key, offset = _ssh_string(decoded, 10)
    namespace, offset = _ssh_string(decoded, offset)
    reserved, offset = _ssh_string(decoded, offset)
    hash_algorithm, offset = _ssh_string(decoded, offset)
    signature_blob, offset = _ssh_string(decoded, offset)
    assert offset == len(decoded)
    algorithm, inner_offset = _ssh_string(signature_blob, 0)
    signature, inner_offset = _ssh_string(signature_blob, inner_offset)
    assert inner_offset == len(signature_blob)
    return {
        "version": version,
        "public_key": public_key,
        "namespace": namespace,
        "reserved": reserved,
        "hash_algorithm": hash_algorithm,
        "signature_algorithm": algorithm,
        "signature": signature,
        "trailing": b"",
    }


def _encode_signature_parts(parts: dict[str, object]) -> bytes:
    signature_blob = _pack_ssh_string(parts["signature_algorithm"]) + _pack_ssh_string(
        parts["signature"],
    )
    decoded = (
        b"SSHSIG"
        + struct.pack(">I", parts["version"])
        + _pack_ssh_string(parts["public_key"])
        + _pack_ssh_string(parts["namespace"])
        + _pack_ssh_string(parts["reserved"])
        + _pack_ssh_string(parts["hash_algorithm"])
        + _pack_ssh_string(signature_blob)
        + parts["trailing"]
    )
    return _armor_signature(decoded)


def _tool_path() -> Path:
    raw = shutil.which("ssh-keygen")
    if raw is None:
        pytest.skip("OpenSSH ssh-keygen is required for this native sshsig canary")
    return Path(raw).resolve(strict=True)


def _independent_pdf_projection(
    documents: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "pdf_id": document["pdf_id"],
            "pdf_member": document["pdf_member"],
            "pdf_bytes": document["pdf_bytes"],
            "pdf_sha256": document["pdf_sha256"],
        }
        for document in documents
    ]


def _independent_render_projection(
    documents: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "pdf_id": document["pdf_id"],
            "pdf_sha256": document["pdf_sha256"],
            "pages": [
                {
                    "page": page["page"],
                    "member": page["png_member"],
                    "sha256": page["png_sha256"],
                    "bytes": page["png_bytes"],
                    "width_pixels": page["width_pixels"],
                    "height_pixels": page["height_pixels"],
                }
                for page in document["pages"]
            ],
        }
        for document in documents
    ]


def _run(argv: list[str], *, cwd: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        argv,
        cwd=cwd,
        env={"LANG": "C", "LC_ALL": "C", "TZ": "UTC"},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
        timeout=10,
    )


class SyntheticCase:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.key_path = root / "synthetic-ed25519"
        self.receipt_path = root / "visual-qa-receipt.json"
        self.authority_path = root / "allowed-signers"
        self.payload_path = root / "signed-payload.bin"
        self.signature_path = root / "visual-qa.sig"
        self.output_path = root / "verification"
        self.replay_path = root / "verification-replay"
        self.tool_path = _tool_path()
        self.signing_index = 0
        self.receipt = self._receipt()
        self._generate_key()
        self.rewrite_receipt()
        self.rewrite_authority()
        self.resign()

    @staticmethod
    def _documents() -> list[dict[str, object]]:
        documents: list[dict[str, object]] = []
        for index, (pdf_id, member, role) in enumerate(_PDF_ORDER, start=1):
            documents.append(
                {
                    "pdf_id": pdf_id,
                    "pdf_member": member,
                    "pdf_role": role,
                    "pdf_sha256": hashlib.sha256(f"pdf-{index}".encode()).hexdigest(),
                    "pdf_bytes": 1000 + index,
                    "page_count": 1,
                    "pages": [
                        {
                            "page": 1,
                            "png_member": f"pages/{pdf_id}/page-0001.png",
                            "png_sha256": hashlib.sha256(
                                f"png-{index}".encode(),
                            ).hexdigest(),
                            "png_bytes": 2000 + index,
                            "width_pixels": 1275,
                            "height_pixels": 1650,
                            "decision": "pass",
                            "issue_codes": [],
                        },
                    ],
                },
            )
        return documents

    def _receipt(self) -> dict[str, object]:
        documents = self._documents()
        return {
            "schema": _RECEIPT_SCHEMA,
            "mode": "final",
            "release_id": _RELEASE_ID,
            "review_kind": _REVIEW_KIND,
            "review_id": "synthetic-all-page-review",
            "reviewer_principal": _PRINCIPAL,
            "reviewer_role": _REVIEWER_ROLE,
            "reviewed_at_utc": "2026-08-31T12:00:00Z",
            "independent_review": True,
            "criteria": _CRITERIA,
            "attestation_statement": _ATTESTATION_STATEMENT,
            "derivation_manifest_sha256": hashlib.sha256(b"derivation").hexdigest(),
            "machine_manifest_sha256": hashlib.sha256(b"machine").hexdigest(),
            "rebuttal_renderer_manifest_sha256": hashlib.sha256(
                b"renderer",
            ).hexdigest(),
            "pdf_set_sha256": _sha(
                _canonical(_independent_pdf_projection(documents)),
            ),
            "render_set_sha256": _sha(
                _canonical(_independent_render_projection(documents)),
            ),
            "documents": documents,
            "non_inference_limits": _NON_INFERENCE_LIMITS,
        }

    def _generate_key(self) -> None:
        _run(
            [
                str(self.tool_path),
                "-q",
                "-t",
                "ed25519",
                "-N",
                "",
                "-C",
                "",
                "-f",
                str(self.key_path),
            ],
            cwd=self.root,
        )

    def rewrite_receipt(self, *, canonical: bool = True) -> None:
        raw = _canonical(self.receipt)
        if not canonical:
            raw = json.dumps(self.receipt, indent=2).encode() + b"\n"
        self.receipt_path.write_bytes(raw)

    def rewrite_authority(
        self,
        *,
        principal: str = _PRINCIPAL,
        namespace: str = _NAMESPACE,
    ) -> None:
        fields = self.key_path.with_suffix(".pub").read_text(encoding="ascii").split()
        assert fields[0] == "ssh-ed25519"
        self.authority_path.write_text(
            f'{principal} namespaces="{namespace}" ssh-ed25519 {fields[1]}\n',
            encoding="ascii",
        )

    def resign(self, *, namespace: str = _NAMESPACE) -> None:
        payload = _DOMAIN_SEPARATOR + self.receipt_path.read_bytes()
        self.payload_path.write_bytes(payload)
        self.signing_index += 1
        signing_input = self.root / f"signed-payload-{self.signing_index:04d}.bin"
        signing_input.write_bytes(payload)
        generated = signing_input.with_suffix(signing_input.suffix + ".sig")
        _run(
            [
                str(self.tool_path),
                "-Y",
                "sign",
                "-f",
                str(self.key_path),
                "-n",
                namespace,
                str(signing_input),
            ],
            cwd=self.root,
        )
        self.signature_path.write_bytes(generated.read_bytes())

    def config(self) -> verifier.VisualQaVerificationInputs:
        return verifier.VisualQaVerificationInputs(
            receipt_path=self.receipt_path,
            allowed_signers_path=self.authority_path,
            signature_path=self.signature_path,
            ssh_keygen_path=self.tool_path,
            expected_receipt_sha256=_sha(self.receipt_path.read_bytes()),
            expected_allowed_signers_sha256=_sha(self.authority_path.read_bytes()),
            expected_signature_sha256=_sha(self.signature_path.read_bytes()),
            expected_ssh_keygen_sha256=_sha(self.tool_path.read_bytes()),
            expected_principal=_PRINCIPAL,
        )


@pytest.fixture
def case(tmp_path: Path) -> SyntheticCase:
    return SyntheticCase(tmp_path)


@pytest.fixture(autouse=True)
def _restore_test_owned_tree_modes(tmp_path: Path) -> Iterator[None]:
    """Let pytest later manage only the synthetic trees that this suite creates."""
    yield
    for path in sorted(tmp_path.rglob("*"), reverse=True):
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o700)


def _refresh_set_digests(case: SyntheticCase) -> None:
    documents = case.receipt["documents"]
    assert isinstance(documents, list)
    case.receipt["pdf_set_sha256"] = _sha(
        _canonical(_independent_pdf_projection(documents)),
    )
    case.receipt["render_set_sha256"] = _sha(
        _canonical(_independent_render_projection(documents)),
    )


def _resign_after_receipt_change(case: SyntheticCase) -> None:
    _refresh_set_digests(case)
    case.rewrite_receipt()
    case.resign()


def test_literal_golden_v2_receipt_contract_is_frozen_independently() -> None:
    synthetic = SyntheticCase.__new__(SyntheticCase)
    receipt = synthetic._receipt()
    raw = _canonical(receipt)
    assert raw == base64.b64decode(_GOLDEN_RECEIPT_BASE64, validate=True)
    assert receipt["schema"] == _RECEIPT_SCHEMA
    assert receipt["criteria"] == _CRITERIA
    assert receipt["non_inference_limits"] == _NON_INFERENCE_LIMITS
    assert [
        (
            document["pdf_id"],
            document["pdf_member"],
            document["pdf_role"],
        )
        for document in receipt["documents"]
    ] == list(_PDF_ORDER)
    assert receipt["pdf_set_sha256"] == _EXPECTED_PDF_SET_SHA256
    assert receipt["render_set_sha256"] == _EXPECTED_RENDER_SET_SHA256
    assert _sha(raw) == _EXPECTED_RECEIPT_SHA256
    normalized, page_count = verifier._normalize_visual_receipt(raw)
    assert normalized == receipt
    assert page_count == 4


def test_build_and_retained_replay_authenticate_exact_anchored_public_key(
    case: SyntheticCase,
) -> None:
    receipt = verifier.build_visual_qa_verification(case.config(), case.output_path)
    assert receipt.authentication_status == "verified"
    assert receipt.promotable is False
    assert receipt.page_count == 4
    assert receipt.release_id == _RELEASE_ID
    assert case.output_path.stat().st_mode & 0o7777 == 0o500
    assert sorted(path.name for path in case.output_path.iterdir()) == sorted(
        verifier.OUTPUT_MEMBERS,
    )
    assert all(
        path.stat().st_mode & 0o7777 == 0o400 for path in case.output_path.iterdir()
    )
    manifest_raw = (case.output_path / verifier.MANIFEST_MEMBER).read_bytes()
    manifest = json.loads(manifest_raw)
    assert manifest_raw == _canonical(manifest)
    assert manifest["authentication"]["status"] == "verified"
    assert manifest["promotion"] == {
        "authority": "none-this-is-an-isolated-authentication-seam",
        "promotable": False,
    }
    assert manifest["non_inference_limits"] == _NON_INFERENCE_LIMITS
    assert manifest["authority"]["principal"] == _PRINCIPAL
    assert manifest["authority"]["namespace"] == _NAMESPACE
    assert manifest["signature"]["signature_algorithm"] == "ssh-ed25519"
    replay = verifier.validate_visual_qa_verification(
        case.config(),
        case.output_path,
        case.replay_path,
        expected_manifest_sha256=receipt.manifest_sha256,
    )
    assert replay.replay_root == str(case.replay_path)
    assert replay.manifest_sha256 == receipt.manifest_sha256
    assert case.replay_path.stat().st_mode & 0o7777 == 0o500
    for member in verifier.OUTPUT_MEMBERS:
        assert (case.output_path / member).read_bytes() == (
            case.replay_path / member
        ).read_bytes()


def test_two_independent_builds_are_byte_deterministic(
    case: SyntheticCase,
) -> None:
    second = case.root / "verification-second"
    first_receipt = verifier.build_visual_qa_verification(
        case.config(),
        case.output_path,
    )
    second_receipt = verifier.build_visual_qa_verification(case.config(), second)
    assert first_receipt.manifest_sha256 == second_receipt.manifest_sha256
    for member in verifier.OUTPUT_MEMBERS:
        assert (case.output_path / member).read_bytes() == (
            second / member
        ).read_bytes()


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("schema", "wrong-schema", "wrong schema"),
        ("mode", "draft", "mode must be final"),
        ("review_kind", "robot", "wrong review kind"),
        ("reviewer_role", "author", "wrong reviewer role"),
        ("independent_review", False, "independent_review must be true"),
        ("attestation_statement", "I looked.", "statement drifted"),
        ("reviewed_at_utc", "2026-02-30T12:00:00Z", "not a real date"),
    ],
)
def test_receipt_fixed_semantics_fail_closed(
    case: SyntheticCase,
    field: str,
    replacement: object,
    match: str,
) -> None:
    case.receipt[field] = replacement
    case.rewrite_receipt()
    case.resign()
    with pytest.raises(verifier.VisualQaVerificationError, match=match):
        verifier.build_visual_qa_verification(case.config(), case.output_path)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda value: value["documents"].reverse(),
            "exact fixed document order",
        ),
        (
            lambda value: value["documents"][0]["pages"][0].__setitem__(
                "page",
                2,
            ),
            "not sequential",
        ),
        (
            lambda value: value["documents"][0]["pages"][0].__setitem__(
                "decision",
                "pending",
            ),
            "must be pass",
        ),
        (
            lambda value: value["documents"][0]["pages"][0].__setitem__(
                "issue_codes",
                ["clipping"],
            ),
            "must be pass",
        ),
        (
            lambda value: value["documents"][0]["pages"][0].__setitem__(
                "png_member",
                "pages/clean/other.png",
            ),
            "not canonical",
        ),
        (
            lambda value: value["documents"][0]["pages"][0].__setitem__(
                "width_pixels",
                verifier.MAX_PAGE_EDGE_PIXELS + 1,
            ),
            "outside the bound",
        ),
    ],
)
def test_all_page_inventory_is_exact_and_all_pass(
    case: SyntheticCase,
    mutate: Callable[[dict[str, object]], object],
    match: str,
) -> None:
    mutate(case.receipt)
    case.rewrite_receipt()
    case.resign()
    with pytest.raises(verifier.VisualQaVerificationError, match=match):
        verifier.build_visual_qa_verification(case.config(), case.output_path)


def test_pdf_and_render_set_digests_are_self_consistent(case: SyntheticCase) -> None:
    case.receipt["pdf_set_sha256"] = "0" * 64
    case.rewrite_receipt()
    case.resign()
    with pytest.raises(verifier.VisualQaVerificationError, match="PDF set digest"):
        verifier.build_visual_qa_verification(case.config(), case.output_path)
    case.receipt = case._receipt()
    case.receipt["render_set_sha256"] = "0" * 64
    case.rewrite_receipt()
    case.resign()
    with pytest.raises(verifier.VisualQaVerificationError, match="render set digest"):
        verifier.build_visual_qa_verification(case.config(), case.root / "other")


def test_receipt_must_be_exact_canonical_json(case: SyntheticCase) -> None:
    case.rewrite_receipt(canonical=False)
    case.resign()
    with pytest.raises(verifier.VisualQaVerificationError, match="canonical JSON"):
        verifier.build_visual_qa_verification(case.config(), case.output_path)


def test_duplicate_json_keys_are_rejected(case: SyntheticCase) -> None:
    raw = case.receipt_path.read_bytes()
    assert raw.startswith(b"{")
    case.receipt_path.write_bytes(b'{"schema":"duplicate",' + raw[1:])
    case.resign()
    with pytest.raises(verifier.VisualQaVerificationError, match="duplicate key"):
        verifier.build_visual_qa_verification(case.config(), case.output_path)


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ("expected_receipt_sha256", "visual-QA receipt does not match"),
        ("expected_allowed_signers_sha256", "allowed-signers authority does not match"),
        ("expected_signature_sha256", "sshsig signature does not match"),
        ("expected_ssh_keygen_sha256", "ssh-keygen executable does not match"),
    ],
)
def test_every_external_input_requires_an_independent_sha_anchor(
    case: SyntheticCase,
    field: str,
    match: str,
) -> None:
    config = replace(case.config(), **{field: "0" * 64})
    with pytest.raises(verifier.VisualQaVerificationError, match=match):
        verifier.build_visual_qa_verification(config, case.output_path)


@pytest.mark.parametrize(
    ("principal", "namespace", "match"),
    [
        ("other@example.test", _NAMESPACE, "exact principal"),
        ("*", _NAMESPACE, "exact principal"),
        (_PRINCIPAL, "other-namespace", "exact sshsig namespace"),
    ],
)
def test_allowed_signers_authority_is_one_exact_namespace_restricted_record(
    case: SyntheticCase,
    principal: str,
    namespace: str,
    match: str,
) -> None:
    case.rewrite_authority(principal=principal, namespace=namespace)
    with pytest.raises(verifier.VisualQaVerificationError, match=match):
        verifier.build_visual_qa_verification(case.config(), case.output_path)


def test_allowed_signers_comments_and_multiple_records_are_rejected(
    case: SyntheticCase,
) -> None:
    case.authority_path.write_bytes(case.authority_path.read_bytes() + b"# comment\n")
    with pytest.raises(verifier.VisualQaVerificationError, match="exactly one line"):
        verifier.build_visual_qa_verification(case.config(), case.output_path)


def test_signature_namespace_and_embedded_key_are_parsed_before_execution(
    case: SyntheticCase,
) -> None:
    case.resign(namespace="other-namespace")
    with pytest.raises(verifier.VisualQaVerificationError, match="namespace differs"):
        verifier.build_visual_qa_verification(case.config(), case.output_path)


def test_valid_signature_over_different_receipt_is_rejected(
    case: SyntheticCase,
) -> None:
    signed_signature = case.signature_path.read_bytes()
    case.receipt["review_id"] = "different-review"
    case.rewrite_receipt()
    case.signature_path.write_bytes(signed_signature)
    with pytest.raises(
        verifier.VisualQaVerificationError,
        match="rejected the signed receipt",
    ):
        verifier.build_visual_qa_verification(case.config(), case.output_path)


def test_expected_principal_must_match_receipt_and_authority(
    case: SyntheticCase,
) -> None:
    config = replace(case.config(), expected_principal="other@example.test")
    with pytest.raises(
        verifier.VisualQaVerificationError,
        match="receipt principal differs",
    ):
        verifier.build_visual_qa_verification(config, case.output_path)


def test_output_is_no_replace_and_existing_bytes_are_unchanged(
    case: SyntheticCase,
) -> None:
    case.output_path.mkdir()
    sentinel = case.output_path / "sentinel"
    sentinel.write_bytes(b"keep")
    with pytest.raises(verifier.VisualQaVerificationError, match="already exists"):
        verifier.build_visual_qa_verification(case.config(), case.output_path)
    assert sentinel.read_bytes() == b"keep"


def test_failure_preserves_private_candidate(case: SyntheticCase) -> None:
    case.receipt["review_id"] = "changed-after-signing"
    case.rewrite_receipt()
    with pytest.raises(
        verifier.VisualQaVerificationError,
        match="partial-private-candidate-do-not-auto-delete",
    ):
        verifier.build_visual_qa_verification(case.config(), case.output_path)
    candidates = list(case.root.glob(".verification.private-candidate-*"))
    assert len(candidates) == 1
    assert candidates[0].is_dir()


def test_replay_requires_anchored_manifest_and_new_destination(
    case: SyntheticCase,
) -> None:
    built = verifier.build_visual_qa_verification(case.config(), case.output_path)
    with pytest.raises(verifier.VisualQaVerificationError, match="independent SHA-256"):
        verifier.validate_visual_qa_verification(
            case.config(),
            case.output_path,
            case.replay_path,
            expected_manifest_sha256="0" * 64,
        )
    case.replay_path.mkdir()
    sentinel = case.replay_path / "sentinel"
    sentinel.write_bytes(b"keep")
    with pytest.raises(verifier.VisualQaVerificationError, match="already exists"):
        verifier.validate_visual_qa_verification(
            case.config(),
            case.output_path,
            case.replay_path,
            expected_manifest_sha256=built.manifest_sha256,
        )
    assert sentinel.read_bytes() == b"keep"


def test_symlinked_inputs_are_rejected(case: SyntheticCase) -> None:
    symlink = case.root / "receipt-link.json"
    symlink.symlink_to(case.receipt_path)
    config = replace(case.config(), receipt_path=symlink)
    with pytest.raises(verifier.VisualQaVerificationError, match="without symlinks"):
        verifier.build_visual_qa_verification(config, case.output_path)


def test_signature_armor_rejects_crlf(case: SyntheticCase) -> None:
    case.signature_path.write_bytes(
        case.signature_path.read_bytes().replace(b"\n", b"\r\n"),
    )
    with pytest.raises(verifier.VisualQaVerificationError, match="canonical LF"):
        verifier.build_visual_qa_verification(case.config(), case.output_path)


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("version", 2, "version must be 1"),
        ("reserved", b"x", "reserved field must be empty"),
        ("hash_algorithm", b"sha256", "hash algorithm must be sha512"),
        ("signature_algorithm", b"ssh-ed25518", "canonical Ed25519"),
        ("signature", b"x" * 63, "canonical Ed25519"),
        ("trailing", b"x", "trailing bytes"),
    ],
)
def test_sshsig_binary_fields_fail_closed(
    case: SyntheticCase,
    field: str,
    replacement: object,
    match: str,
) -> None:
    parts = _signature_parts(case.signature_path.read_bytes())
    parts[field] = replacement
    case.signature_path.write_bytes(_encode_signature_parts(parts))
    with pytest.raises(verifier.VisualQaVerificationError, match=match):
        verifier.build_visual_qa_verification(case.config(), case.output_path)


def test_signature_armor_wrapping_and_base64_are_canonical(case: SyntheticCase) -> None:
    lines = case.signature_path.read_text(encoding="ascii").splitlines()
    body = "".join(lines[1:-1])
    case.signature_path.write_text(
        f"{lines[0]}\n{body}\n{lines[-1]}\n",
        encoding="ascii",
    )
    with pytest.raises(verifier.VisualQaVerificationError, match="line wrapping"):
        verifier.build_visual_qa_verification(case.config(), case.output_path)
    case.resign()
    raw = bytearray(case.signature_path.read_bytes())
    marker_end = raw.index(b"\n") + 1
    raw[marker_end] = ord("!")
    case.signature_path.write_bytes(bytes(raw))
    with pytest.raises(verifier.VisualQaVerificationError, match="base64"):
        verifier.build_visual_qa_verification(case.config(), case.root / "other")


def test_bounded_runner_rejects_stdout_overflow() -> None:
    with pytest.raises(verifier.VisualQaVerificationError, match="stdout exceeds"):
        verifier._run_bounded(
            Path(sys.executable),
            ["-c", f"import os; os.write(1, b'x' * {verifier.MAX_STDOUT_BYTES + 1})"],
            b"",
            inherited_fds=(),
        )


def test_bounded_runner_rejects_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verifier, "TOOL_TIMEOUT_SECONDS", 0.05)
    with pytest.raises(verifier.VisualQaVerificationError, match="timed out"):
        verifier._run_bounded(
            Path(sys.executable),
            ["-c", "import time; time.sleep(10)"],
            b"",
            inherited_fds=(),
        )


def test_bounded_runner_kills_same_process_group_descendant(tmp_path: Path) -> None:
    pid_path = tmp_path / "same-group-child.pid"
    script = (
        "import os,pathlib,subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)']);"
        f"pathlib.Path({str(pid_path)!r}).write_text(str(child.pid));"
        f"os.write(1,b'x'*{verifier.MAX_STDOUT_BYTES + 1});"
        "time.sleep(10)"
    )
    with pytest.raises(verifier.VisualQaVerificationError, match="stdout exceeds"):
        verifier._run_bounded(
            Path(sys.executable),
            ["-c", script],
            b"",
            inherited_fds=(),
        )
    child_pid = int(pid_path.read_text(encoding="ascii"))
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("same-process-group descendant survived bounded-runner cleanup")


def test_cli_help_lists_explicit_build_and_validate_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        verifier.main(["--help"])
    assert raised.value.code == 0
    output = capsys.readouterr().out
    assert "build" in output
    assert "validate" in output


def test_signature_public_key_must_match_allowed_signers(case: SyntheticCase) -> None:
    other = case.root / "other-key"
    _run(
        [
            str(case.tool_path),
            "-q",
            "-t",
            "ed25519",
            "-N",
            "",
            "-C",
            "",
            "-f",
            str(other),
        ],
        cwd=case.root,
    )
    fields = other.with_suffix(".pub").read_text(encoding="ascii").split()
    case.authority_path.write_text(
        f'{_PRINCIPAL} namespaces="{_NAMESPACE}" ssh-ed25519 {fields[1]}\n',
        encoding="ascii",
    )
    with pytest.raises(verifier.VisualQaVerificationError, match="differs from"):
        verifier.build_visual_qa_verification(case.config(), case.output_path)


def test_payload_domain_separator_is_fixed_and_published(case: SyntheticCase) -> None:
    verifier.build_visual_qa_verification(case.config(), case.output_path)
    payload = (case.output_path / verifier.PAYLOAD_MEMBER).read_bytes()
    assert payload == _DOMAIN_SEPARATOR + case.receipt_path.read_bytes()
    assert payload.startswith(b"DIALECT-RENDERED-DOCUMENT-VISUAL-QA-V2\n")


def test_output_manifest_never_claims_human_identity_or_inspection_proof(
    case: SyntheticCase,
) -> None:
    verifier.build_visual_qa_verification(case.config(), case.output_path)
    manifest = json.loads(
        (case.output_path / verifier.MANIFEST_MEMBER).read_text(encoding="ascii"),
    )
    assert manifest["non_inference_limits"]["human_identity_to_key_binding"] == (
        "caller-authority-not-inferred"
    )
    assert manifest["non_inference_limits"]["actual_visual_inspection"] == (
        "caller-authority-signed-claim-not-independently-observed"
    )
    assert manifest["authentication"]["claim"] == (
        "valid-ed25519-signature-under-the-anchored-public-key"
    )
    assert (
        manifest["non_inference_limits"][
            "current_private_key_control_or_signature_freshness"
        ]
        == "not-inferred"
    )
    assert manifest["invocation"]["process_containment"] == {
        "new_session": True,
        "same_process_group_kill_on_exit": True,
        "detached_descendants": "not-contained",
    }
    assert manifest["promotion"]["promotable"] is False


def test_tool_must_be_root_owned_and_not_writable(
    case: SyntheticCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = verifier._pin_file

    def wrapped(*args: object, **kwargs: object) -> tuple[verifier._PinnedFile, bytes]:
        pin, raw = original(*args, **kwargs)
        if kwargs.get("executable"):
            pin.uid = os.getuid() + 1
        return pin, raw

    monkeypatch.setattr(verifier, "_pin_file", wrapped)
    with pytest.raises(
        verifier.VisualQaVerificationError,
        match="descriptor identity changed",
    ):
        verifier.build_visual_qa_verification(case.config(), case.output_path)


def test_user_owned_ssh_keygen_copy_is_not_an_executable_authority(
    case: SyntheticCase,
) -> None:
    if os.getuid() == 0:
        pytest.skip("the root-owned negative control requires an unprivileged uid")
    copied_tool = case.root / "ssh-keygen-copy"
    shutil.copyfile(case.tool_path, copied_tool)
    copied_tool.chmod(0o755)
    config = replace(
        case.config(),
        ssh_keygen_path=copied_tool,
        expected_ssh_keygen_sha256=_sha(copied_tool.read_bytes()),
    )
    with pytest.raises(verifier.VisualQaVerificationError, match="owned by root"):
        verifier.build_visual_qa_verification(config, case.output_path)


def test_published_tree_member_tampering_is_detected_before_replay(
    case: SyntheticCase,
) -> None:
    built = verifier.build_visual_qa_verification(case.config(), case.output_path)
    receipt_member = case.output_path / verifier.RECEIPT_MEMBER
    case.output_path.chmod(0o700)
    receipt_member.chmod(0o600)
    original = receipt_member.read_bytes()
    receipt_member.write_bytes(b"X" + original[1:])
    receipt_member.chmod(0o400)
    case.output_path.chmod(0o500)
    with pytest.raises(
        verifier.VisualQaVerificationError,
        match=r"canonical|receipt|manifest",
    ):
        verifier.validate_visual_qa_verification(
            case.config(),
            case.output_path,
            case.replay_path,
            expected_manifest_sha256=built.manifest_sha256,
        )


def test_parent_name_swap_cannot_publish_under_an_unreported_root(
    case: SyntheticCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication_parent = case.root / "publication-parent"
    publication_parent.mkdir()
    destination = publication_parent / "verification"
    moved_parent = case.root / "publication-parent-moved"
    original_rename = verifier._rename_no_replace

    def swap_then_rename(source: str, target: str, parent_descriptor: int) -> None:
        publication_parent.rename(moved_parent)
        publication_parent.mkdir()
        original_rename(source, target, parent_descriptor)

    monkeypatch.setattr(verifier, "_rename_no_replace", swap_then_rename)
    with pytest.raises(
        verifier.VisualQaVerificationError,
        match="published-destination-may-exist-do-not-auto-delete",
    ):
        verifier.build_visual_qa_verification(case.config(), destination)
    assert not destination.exists()
    preserved = moved_parent / destination.name
    assert preserved.is_dir()
    assert preserved.stat().st_mode & 0o7777 == 0o500


def test_page_count_must_match_exact_page_inventory(case: SyntheticCase) -> None:
    documents = case.receipt["documents"]
    assert isinstance(documents, list)
    first = documents[0]
    assert isinstance(first, dict)
    first["page_count"] = 2
    case.rewrite_receipt()
    case.resign()
    with pytest.raises(verifier.VisualQaVerificationError, match="cover every page"):
        verifier.build_visual_qa_verification(case.config(), case.output_path)


def test_unknown_fields_are_rejected(case: SyntheticCase) -> None:
    case.receipt["comment"] = "not signed protocol data"
    case.rewrite_receipt()
    case.resign()
    with pytest.raises(
        verifier.VisualQaVerificationError,
        match=r"extra=\['comment'\]",
    ):
        verifier.build_visual_qa_verification(case.config(), case.output_path)


@pytest.mark.parametrize("level", ["document", "page"])
def test_unknown_nested_receipt_fields_are_rejected(
    case: SyntheticCase,
    level: str,
) -> None:
    documents = case.receipt["documents"]
    assert isinstance(documents, list)
    first = documents[0]
    assert isinstance(first, dict)
    target = first if level == "document" else first["pages"][0]
    target["unexpected"] = "drift"
    case.rewrite_receipt()
    case.resign()
    with pytest.raises(
        verifier.VisualQaVerificationError,
        match=r"extra=\['unexpected'\]",
    ):
        verifier.build_visual_qa_verification(case.config(), case.output_path)


def test_duplicate_pdf_digests_are_rejected(case: SyntheticCase) -> None:
    documents = case.receipt["documents"]
    assert isinstance(documents, list)
    documents[1]["pdf_sha256"] = documents[0]["pdf_sha256"]
    _resign_after_receipt_change(case)
    with pytest.raises(verifier.VisualQaVerificationError, match="byte-distinct PDFs"):
        verifier.build_visual_qa_verification(case.config(), case.output_path)


def test_page_pixel_product_bound_is_independent_of_edge_bound(
    case: SyntheticCase,
) -> None:
    documents = case.receipt["documents"]
    assert isinstance(documents, list)
    page = documents[0]["pages"][0]
    page["width_pixels"] = 3000
    page["height_pixels"] = 3000
    case.rewrite_receipt()
    case.resign()
    with pytest.raises(verifier.VisualQaVerificationError, match="page-pixel bound"):
        verifier.build_visual_qa_verification(case.config(), case.output_path)


def test_aggregate_page_bound_rejects_individually_bounded_documents() -> None:
    synthetic = SyntheticCase.__new__(SyntheticCase)
    receipt = synthetic._receipt()
    documents = receipt["documents"]
    assert isinstance(documents, list)
    for document_index, document in enumerate(documents):
        pdf_id = document["pdf_id"]
        pages = [
            {
                "page": page_number,
                "png_member": f"pages/{pdf_id}/page-{page_number:04d}.png",
                "png_sha256": hashlib.sha256(
                    f"{document_index}-{page_number}".encode(),
                ).hexdigest(),
                "png_bytes": 100,
                "width_pixels": 10,
                "height_pixels": 10,
                "decision": "pass",
                "issue_codes": [],
            }
            for page_number in range(1, 130)
        ]
        document["page_count"] = len(pages)
        document["pages"] = pages
    receipt["pdf_set_sha256"] = _sha(
        _canonical(_independent_pdf_projection(documents)),
    )
    receipt["render_set_sha256"] = _sha(
        _canonical(_independent_render_projection(documents)),
    )
    with pytest.raises(
        verifier.VisualQaVerificationError,
        match="aggregate page bound",
    ):
        verifier._normalize_visual_receipt(_canonical(receipt))


def test_mutation_during_tree_validation_is_detected(
    case: SyntheticCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = verifier.build_visual_qa_verification(case.config(), case.output_path)
    target = case.output_path / verifier.STDOUT_MEMBER
    original_validate = verifier._validate_tree_payload

    def mutate_after_semantic_validation(
        raw_by_member: dict[str, bytes],
        manifest_raw: bytes,
    ) -> dict[str, object]:
        result = original_validate(raw_by_member, manifest_raw)
        target.chmod(0o600)
        target.write_bytes(b"tampered-after-read\n")
        target.chmod(0o400)
        return result

    monkeypatch.setattr(
        verifier,
        "_validate_tree_payload",
        mutate_after_semantic_validation,
    )
    with pytest.raises(
        verifier.VisualQaVerificationError,
        match=r"changed during validation|identity changed",
    ):
        verifier.validate_visual_qa_verification(
            case.config(),
            case.output_path,
            case.replay_path,
            expected_manifest_sha256=built.manifest_sha256,
        )


def test_noncanonical_principal_metacharacters_are_rejected(
    case: SyntheticCase,
) -> None:
    config = replace(case.config(), expected_principal="reviewer,*")
    with pytest.raises(
        verifier.VisualQaVerificationError,
        match="exact OpenSSH principal",
    ):
        verifier.build_visual_qa_verification(config, case.output_path)


def test_receipt_input_remains_unchanged_by_build(case: SyntheticCase) -> None:
    modes = {
        "receipt": stat.S_IMODE(case.receipt_path.stat().st_mode),
        "authority": stat.S_IMODE(case.authority_path.stat().st_mode),
        "signature": stat.S_IMODE(case.signature_path.stat().st_mode),
    }
    bytes_before = {
        "receipt": case.receipt_path.read_bytes(),
        "authority": case.authority_path.read_bytes(),
        "signature": case.signature_path.read_bytes(),
    }
    verifier.build_visual_qa_verification(case.config(), case.output_path)
    assert case.receipt_path.read_bytes() == bytes_before["receipt"]
    assert case.authority_path.read_bytes() == bytes_before["authority"]
    assert case.signature_path.read_bytes() == bytes_before["signature"]
    assert stat.S_IMODE(case.receipt_path.stat().st_mode) == modes["receipt"]
    assert stat.S_IMODE(case.authority_path.stat().st_mode) == modes["authority"]
    assert stat.S_IMODE(case.signature_path.stat().st_mode) == modes["signature"]
