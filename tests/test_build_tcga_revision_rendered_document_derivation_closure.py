"""Synthetic adversarial tests for the native PDF-derivation closure."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from analysis import (
    build_tcga_revision_rendered_document_derivation_closure as closure,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

# The suite intentionally exercises private fail-closed publication and pinning
# seams with synthetic bytes only.
# ruff: noqa: EM101, S603, SLF001, TRY003

RELEASE_ID = "synthetic-derivation-v1"


@pytest.fixture(autouse=True)
def _restore_synthetic_tree_modes(tmp_path: Path) -> Iterator[None]:
    """Allow pytest to manage only its own synthetic sealed directories."""
    yield
    for directory, child_directories, _files in os.walk(tmp_path):
        path = Path(directory)
        if not path.is_symlink():
            path.chmod(0o700)
        for child in child_directories:
            child_path = path / child
            if not child_path.is_symlink():
                child_path.chmod(0o700)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(64 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _pdf(pdf_id: str) -> bytes:
    return f"%PDF-1.7\nsynthetic derivation {pdf_id}\n%%EOF\n".encode("ascii")


def _mapping(value: object) -> dict[str, object]:
    return cast("dict[str, object]", value)


def _array(value: object) -> list[object]:
    return cast("list[object]", value)


def _rewrite_sealed_manifest(
    root: Path,
    mutate: Callable[[dict[str, object]], None],
) -> str:
    """Rewrite only a current test's synthetic manifest with a valid self-digest."""
    root.chmod(0o700)
    path = root / closure.MANIFEST_MEMBER
    path.chmod(0o600)
    manifest = json.loads(path.read_text(encoding="ascii"))
    mutate(manifest)
    unsigned = dict(manifest)
    unsigned.pop("payload_sha256")
    manifest["payload_sha256"] = _sha256(_canonical(unsigned))
    raw = _canonical(manifest)
    path.write_bytes(raw)
    path.chmod(0o400)
    root.chmod(0o500)
    return _sha256(raw)


@dataclass(slots=True)
class SyntheticCase:
    """Own one exact synthetic four-document derivation input set."""

    root: Path
    plan_path: Path
    source_root: Path
    producer_root: Path
    authority_root: Path
    destination: Path
    replay_root: Path
    upstream_sha256: dict[str, str]
    authority_sha256: dict[str, str]
    source_sha256: dict[str, str]
    producer_sha256: dict[str, str]
    pdf_sha256: dict[str, str]
    mode: str = closure.MODE_SYNTHETIC
    release_id: str = RELEASE_ID

    def plan(self) -> dict[str, object]:
        """Read the synthetic canonical plan."""
        return json.loads(self.plan_path.read_text(encoding="ascii"))

    def write_plan(self, value: object) -> None:
        """Replace only this case's synthetic plan bytes."""
        self.plan_path.write_bytes(_canonical(value) + b"\n")

    def kwargs(self) -> dict[str, object]:
        """Return every independent caller trust anchor."""
        return {
            "release_id": self.release_id,
            "expected_plan_sha256": _file_sha256(self.plan_path),
            "expected_builder_sha256": _file_sha256(Path(closure.__file__)),
            "expected_machine_runner_sha256": _file_sha256(
                Path(closure._machine.__file__),
            ),
            "expected_upstream_sha256": self.upstream_sha256,
            "expected_authority_sha256": self.authority_sha256,
            "expected_source_sha256": self.source_sha256,
            "expected_producer_sha256": self.producer_sha256,
            "expected_pdf_sha256": self.pdf_sha256,
        }


def _profile(pdf_id: str, mode: str) -> str:
    if mode == closure.MODE_SYNTHETIC:
        return "synthetic-native-copy-v1"
    baseline = closure.BUILDER_RELEASE_PROFILE_BY_ID[pdf_id]
    return str(baseline) if baseline is not None else "reviewed-rebuttal-renderer-v1"


def _authority_receipt(
    document: dict[str, object],
    *,
    mode: str,
    release_id: str,
) -> dict[str, object]:
    pdf_id = str(document["pdf_id"])
    adapter_authority = _mapping(document["adapter_authority"])
    return {
        "schema": closure.PRODUCER_AUTHORITY_SCHEMA,
        "mode": mode,
        "release_id": release_id,
        "pdf_id": pdf_id,
        "authority_id": adapter_authority["authority_id"],
        "status": adapter_authority["status"],
        "authentication": adapter_authority["authentication"],
        "source_bundle_member": document["source_bundle_member"],
        "source_bundle_sha256": document["source_bundle_sha256"],
        "producer_member": document["producer_member"],
        "producer_bytes": document["producer_bytes"],
        "producer_sha256": document["producer_sha256"],
        "toolchain_profile": document["toolchain_profile"],
        "producer_arguments": document["producer_arguments"],
        "review_scope": closure.PRODUCER_AUTHORITY_REVIEW_SCOPE,
    }


def _make_case(
    tmp_path: Path,
    *,
    mode: str = closure.MODE_SYNTHETIC,
) -> SyntheticCase:
    source_root = tmp_path / "source-bundles"
    producer_root = tmp_path / "producers"
    authority_root = tmp_path / "producer-authorities"
    source_root.mkdir()
    producer_root.mkdir()
    authority_root.mkdir()
    upstream = {
        key: _sha256(f"upstream {key}".encode("ascii"))
        for key in closure.UPSTREAM_BINDING_KEYS
    }
    authority: dict[str, str] = {}
    source_sha: dict[str, str] = {}
    producer_sha: dict[str, str] = {}
    pdf_sha: dict[str, str] = {}
    for pdf_id in closure.PDF_IDS:
        source = source_root / closure.SOURCE_MEMBER_BY_ID[pdf_id]
        source.write_bytes(_pdf(pdf_id))
        producer = producer_root / closure.PRODUCER_MEMBER_BY_ID[pdf_id]
        producer.write_bytes(f"synthetic executable {pdf_id}\n".encode("ascii"))
        producer.chmod(0o500)
        source_sha[pdf_id] = _file_sha256(source)
        producer_sha[pdf_id] = _file_sha256(producer)
        pdf_sha[pdf_id] = _sha256(_pdf(pdf_id))
    release_id = (
        RELEASE_ID if mode == closure.MODE_SYNTHETIC else "revision-authorized-v1"
    )
    documents: list[dict[str, object]] = []
    for pdf_id, pdf_member in closure.PDF_ORDER:
        source = source_root / closure.SOURCE_MEMBER_BY_ID[pdf_id]
        producer = producer_root / closure.PRODUCER_MEMBER_BY_ID[pdf_id]
        documents.append(
            {
                "pdf_id": pdf_id,
                "pdf_member": pdf_member,
                "source_bundle_member": source.name,
                "source_bundle_bytes": source.stat().st_size,
                "source_bundle_sha256": source_sha[pdf_id],
                "producer_member": producer.name,
                "producer_bytes": producer.stat().st_size,
                "producer_sha256": producer_sha[pdf_id],
                "expected_pdf_sha256": pdf_sha[pdf_id],
                "toolchain_profile": _profile(pdf_id, mode),
                "adapter_authority": {
                    "status": (
                        "synthetic-test-only"
                        if mode == closure.MODE_SYNTHETIC
                        else "caller-authorized"
                    ),
                    "authority_id": f"authority-{pdf_id}-v1",
                    "authentication": "caller-sha-anchor-only",
                    "authority_receipt_sha256": "0" * 64,
                    "reviewed_producer_sha256": producer_sha[pdf_id],
                },
                "producer_arguments": closure._expected_arguments(pdf_id),
            },
        )
    for document in documents:
        pdf_id = str(document["pdf_id"])
        adapter_authority = _mapping(document["adapter_authority"])
        authority_receipt = _authority_receipt(
            document,
            mode=mode,
            release_id=release_id,
        )
        authority_path = authority_root / closure.AUTHORITY_MEMBER_BY_ID[pdf_id]
        authority_path.write_bytes(_canonical(authority_receipt) + b"\n")
        authority[pdf_id] = _file_sha256(authority_path)
        adapter_authority["authority_receipt_sha256"] = authority[pdf_id]
    plan = {
        "schema": closure.DERIVATION_PLAN_SCHEMA,
        "mode": mode,
        "release_id": release_id,
        "upstream_bindings": upstream,
        "real_producer_toolchain_authority": authority,
        "execution_contract": closure.EXECUTION_CONTRACT,
        "builder_release_toolchain_baseline": (
            closure.BUILDER_RELEASE_TOOLCHAIN_BASELINE
        ),
        "documents": documents,
        "non_inference_limits": closure.NON_INFERENCE_LIMITS,
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(_canonical(plan) + b"\n")
    return SyntheticCase(
        root=tmp_path,
        plan_path=plan_path,
        source_root=source_root,
        producer_root=producer_root,
        authority_root=authority_root,
        destination=tmp_path / "derivation-closure",
        replay_root=tmp_path / "derivation-replay",
        upstream_sha256=upstream,
        authority_sha256=authority,
        source_sha256=source_sha,
        producer_sha256=producer_sha,
        pdf_sha256=pdf_sha,
        mode=mode,
        release_id=release_id,
    )


def _sync_authority_receipts(
    case: SyntheticCase,
    plan: dict[str, object],
) -> None:
    """Rewrite only this fixture's authorities after an intentional input update."""
    authority_bindings = _mapping(plan["real_producer_toolchain_authority"])
    for raw_document in _array(plan["documents"]):
        document = _mapping(raw_document)
        pdf_id = str(document["pdf_id"])
        authority_path = case.authority_root / closure.AUTHORITY_MEMBER_BY_ID[pdf_id]
        authority_path.write_bytes(
            _canonical(
                _authority_receipt(
                    document,
                    mode=case.mode,
                    release_id=case.release_id,
                ),
            )
            + b"\n",
        )
        authority_sha256 = _file_sha256(authority_path)
        case.authority_sha256[pdf_id] = authority_sha256
        authority_bindings[pdf_id] = authority_sha256
        _mapping(document["adapter_authority"])["authority_receipt_sha256"] = (
            authority_sha256
        )
    case.write_plan(plan)


def _install_fake_native_runner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mutate: Callable[[str, bytes], bytes] | None = None,
    stderr: bytes = b"",
    return_code: int = 0,
) -> None:
    call_count = 0

    def invoke(  # noqa: PLR0913
        producer: object,
        arguments: list[str],
        *,
        inherited_fds: tuple[int],
        budget: object,
        before: Callable[[], None],
        after: Callable[[], None],
    ) -> tuple[int, bytes, bytes, dict[str, object]]:
        nonlocal call_count
        call_count += 1
        budget.consume()
        before()
        pdf_id = arguments[3]
        (source_fd,) = inherited_fds
        source_size = os.fstat(source_fd).st_size
        raw = os.pread(source_fd, source_size, 0)
        if mutate is not None:
            raw = mutate(f"{pdf_id}-{call_count}", raw)
        after()
        return (
            return_code,
            raw,
            stderr,
            {
                "protocol": "synthetic-attested-main-executable-v1",
                "producer_sha256": producer.sha256,
                "execution_binding_scope": "main_executable",
                "non_system_dylib_closure": "not_attested",
            },
        )

    monkeypatch.setattr(closure, "_invoke_producer", invoke)


def _build(case: SyntheticCase) -> closure.DerivationClosureReceipt:
    return closure.build_derivation_closure(
        case.plan_path,
        case.source_root,
        case.producer_root,
        case.authority_root,
        case.destination,
        **case.kwargs(),
    )


def _validate(
    case: SyntheticCase,
    receipt: closure.DerivationClosureReceipt,
) -> closure.DerivationClosureReceipt:
    return closure.validate_derivation_closure(
        case.plan_path,
        case.source_root,
        case.producer_root,
        case.authority_root,
        case.destination,
        case.replay_root,
        expected_manifest_sha256=receipt.manifest_sha256,
        **case.kwargs(),
    )


def test_synthetic_build_and_retained_replay_close_exact_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    _install_fake_native_runner(monkeypatch)
    built = _build(case)
    assert built.pdf_count == 4
    assert built.rebuild_count == 8
    assert built.mode == closure.MODE_SYNTHETIC
    assert built.promotable is False
    assert built.replay_root is None
    assert stat.S_IMODE(case.destination.stat().st_mode) == 0o500

    validated = _validate(case, built)
    assert validated.manifest_sha256 == built.manifest_sha256
    assert validated.pdf_set_sha256 == built.pdf_set_sha256
    assert validated.source_bundle_set_sha256 == built.source_bundle_set_sha256
    assert validated.producer_set_sha256 == built.producer_set_sha256
    assert (
        validated.producer_toolchain_authority_set_sha256
        == built.producer_toolchain_authority_set_sha256
    )
    assert validated.replay_root == str(case.replay_root)
    assert case.replay_root.is_dir()
    assert stat.S_IMODE(case.replay_root.stat().st_mode) == 0o500


def test_manifest_exposes_ordered_content_only_pdf_set_and_exact_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    _install_fake_native_runner(monkeypatch)
    built = _build(case)
    manifest = json.loads(
        (case.destination / closure.MANIFEST_MEMBER).read_text(encoding="ascii"),
    )
    expected_pdf_set = [
        {
            "pdf_id": pdf_id,
            "pdf_member": member,
            "pdf_bytes": len(_pdf(pdf_id)),
            "pdf_sha256": case.pdf_sha256[pdf_id],
        }
        for pdf_id, member in closure.PDF_ORDER
    ]
    assert manifest["pdf_set"] == expected_pdf_set
    assert manifest["pdf_set_sha256"] == _sha256(_canonical(expected_pdf_set))
    assert manifest["inputs"]["upstream_bindings"] == case.upstream_sha256
    assert (
        manifest["inputs"]["real_producer_toolchain_authority"] == case.authority_sha256
    )
    assert manifest["status"] == "synthetic-canary-only"
    assert manifest["wrapper_integration"]["native_derivation_candidate"] is False
    assert (
        manifest["execution_contract"]["producer_filesystem_side_effects"]
        == "ambient-same-uid-access-not-contained"
    )
    assert (
        "already loaded" in manifest["execution_contract"]["loaded_python_code_binding"]
    )
    assert manifest["execution_contract"]["pdf_structural_validity"].startswith(
        "not-inferred",
    )
    assert manifest["non_inference_limits"]["pdf_structural_validity"].startswith(
        "not inferred",
    )
    assert manifest["non_inference_limits"]["text_legibility_or_visual_quality"] == (
        "not inferred"
    )
    first_run = manifest["documents"][0]["runs"][0]
    invocation_receipt = json.loads(
        (case.destination / first_run["invocation_receipt_member"]).read_text(
            encoding="ascii",
        ),
    )
    assert invocation_receipt["source_bundle"]["sha256"] == case.source_sha256["clean"]
    assert invocation_receipt["producer"]["sha256"] == case.producer_sha256["clean"]
    assert (
        invocation_receipt["producer_toolchain_authority"]["sha256"]
        == (case.authority_sha256["clean"])
    )
    assert invocation_receipt["producer_toolchain_authority"]["authentication"] == (
        "caller-sha-anchor-only"
    )
    assert invocation_receipt["pdf"]["sha256"] == case.pdf_sha256["clean"]
    assert built.plan_sha256 == _file_sha256(case.plan_path)


def test_revision_mode_does_not_admit_unopened_or_mismatched_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path, mode=closure.MODE_REVISION)
    authority_path = case.authority_root / closure.AUTHORITY_MEMBER_BY_ID["rebuttal"]
    authority = json.loads(authority_path.read_text(encoding="ascii"))
    authority["toolchain_profile"] = "unreviewed-profile-v1"
    authority_path.write_bytes(_canonical(authority) + b"\n")
    case.authority_sha256["rebuttal"] = _file_sha256(authority_path)
    plan = case.plan()
    plan["real_producer_toolchain_authority"]["rebuttal"] = case.authority_sha256[
        "rebuttal"
    ]
    _mapping(
        _mapping(_array(plan["documents"])[3])["adapter_authority"],
    )["authority_receipt_sha256"] = case.authority_sha256["rebuttal"]
    case.write_plan(plan)
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(closure, "_invoke_producer", forbidden)
    with pytest.raises(
        closure.DerivationClosureError,
        match="does not authorize the exact selected",
    ):
        _build(case)
    assert called is False
    assert closure._mode_status(closure.MODE_REVISION) == (
        "native-revision-derivation-candidate",
        ["requires-separate-downstream-promotion-closure"],
    )


def test_caller_authorized_revision_fixture_reaches_nonpromotable_candidate_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path, mode=closure.MODE_REVISION)
    _install_fake_native_runner(monkeypatch)
    built = _build(case)
    replayed = _validate(case, built)
    manifest = json.loads(
        (case.destination / closure.MANIFEST_MEMBER).read_text(encoding="ascii"),
    )
    assert built.mode == closure.MODE_REVISION
    assert replayed.manifest_sha256 == built.manifest_sha256
    assert built.promotable is False
    assert manifest["status"] == "native-revision-derivation-candidate"
    assert manifest["promotion_blockers"] == [
        "requires-separate-downstream-promotion-closure",
    ]
    assert manifest["non_inference_limits"]["human_identity_or_approval"] == (
        "not authenticated"
    )
    assert all(
        document["producer"]["adapter_authority"]["status"] == "caller-authorized"
        and document["producer"]["adapter_authority"]["authentication"]
        == "caller-sha-anchor-only"
        for document in manifest["documents"]
    )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("status", "status"),
        ("authority_receipt_sha256", "authority binding"),
        ("reviewed_producer_sha256", "reviews different bytes"),
    ],
)
def test_revision_authority_drift_fails_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    message: str,
) -> None:
    case = _make_case(tmp_path, mode=closure.MODE_REVISION)
    value = case.plan()
    authority = _mapping(_mapping(_array(value["documents"])[3])["adapter_authority"])
    authority[field] = "synthetic-test-only" if field == "status" else "f" * 64
    case.write_plan(value)
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(closure, "_invoke_producer", forbidden)
    with pytest.raises(closure.DerivationClosureError, match=message):
        _build(case)
    assert called is False


def test_revision_rejects_synthetic_rebuttal_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path, mode=closure.MODE_REVISION)
    value = case.plan()
    _mapping(_array(value["documents"])[3])["toolchain_profile"] = (
        "synthetic-native-copy-v1"
    )
    case.write_plan(value)
    monkeypatch.setattr(
        closure,
        "_invoke_producer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError),
    )
    with pytest.raises(closure.DerivationClosureError, match="real producer"):
        _build(case)


@pytest.mark.parametrize(
    ("anchor", "message"),
    [
        ("expected_plan_sha256", "plan"),
        ("expected_builder_sha256", "builder"),
        ("expected_machine_runner_sha256", "execution dependency"),
    ],
)
def test_live_code_and_plan_anchors_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    anchor: str,
    message: str,
) -> None:
    case = _make_case(tmp_path)
    kwargs = case.kwargs()
    kwargs[anchor] = "f" * 64
    monkeypatch.setattr(
        closure,
        "_invoke_producer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError),
    )
    with pytest.raises(closure.DerivationClosureError, match=message):
        closure.build_derivation_closure(
            case.plan_path,
            case.source_root,
            case.producer_root,
            case.authority_root,
            case.destination,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("mapping_name", "message"),
    [
        ("upstream_sha256", "upstream"),
        ("authority_sha256", "authority"),
        ("source_sha256", "source-bundle"),
        ("producer_sha256", "producer"),
        ("pdf_sha256", "PDF"),
    ],
)
def test_each_independent_role_map_is_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mapping_name: str,
    message: str,
) -> None:
    case = _make_case(tmp_path)
    mapping = getattr(case, mapping_name)
    first = next(iter(mapping))
    mapping[first] = "f" * 64
    monkeypatch.setattr(
        closure,
        "_invoke_producer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError),
    )
    with pytest.raises(closure.DerivationClosureError, match=message):
        _build(case)


@pytest.mark.parametrize("kind", ["extra", "missing", "symlink", "hardlink", "fifo"])
def test_source_inventory_and_member_type_attacks_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    case = _make_case(tmp_path)
    original_root = case.source_root
    attacked_root = case.root / f"attacked-source-{kind}"
    attacked_root.mkdir()
    for pdf_id in closure.PDF_IDS:
        member = closure.SOURCE_MEMBER_BY_ID[pdf_id]
        original = original_root / member
        target = attacked_root / member
        if pdf_id != "clean":
            target.write_bytes(original.read_bytes())
        elif kind == "missing":
            continue
        elif kind == "symlink":
            target.symlink_to(original)
        elif kind == "hardlink":
            os.link(original, target)
        elif kind == "fifo":
            os.mkfifo(target)
        else:
            target.write_bytes(original.read_bytes())
    if kind == "extra":
        (attacked_root / "extra.bundle").write_bytes(b"extra")
    case.source_root = attacked_root
    monkeypatch.setattr(
        closure,
        "_invoke_producer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError),
    )
    with pytest.raises(closure.DerivationClosureError):
        _build(case)


def test_producer_must_be_single_link_executable_and_not_writable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    producer = case.producer_root / closure.PRODUCER_MEMBER_BY_ID["clean"]
    producer.chmod(0o720)
    monkeypatch.setattr(
        closure,
        "_invoke_producer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError),
    )
    with pytest.raises(closure.DerivationClosureError, match="executable"):
        _build(case)


@pytest.mark.parametrize(
    ("runner_options", "message"),
    [
        ({"return_code": 4}, "exited with 4"),
        ({"stderr": b"warning"}, "wrote stderr"),
    ],
)
def test_adapter_exit_and_stderr_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner_options: dict[str, object],
    message: str,
) -> None:
    case = _make_case(tmp_path)
    _install_fake_native_runner(monkeypatch, **runner_options)
    with pytest.raises(closure.DerivationClosureError, match=message):
        _build(case)
    assert (case.root / ".derivation-closure.private-candidate").is_dir()


def test_output_must_be_pdf_and_match_caller_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    _install_fake_native_runner(
        monkeypatch,
        mutate=lambda _name, raw: b"not-a-pdf" + raw,
    )
    with pytest.raises(closure.DerivationClosureError, match="signature"):
        _build(case)
    assert not (case.root / ".derivation-closure.private-candidate" / "runs").exists()


def test_oversized_and_partial_stdout_never_materialize_a_pdf_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for label, mutate, message in (
        ("oversized", lambda _name, raw: raw, "bounded byte contract"),
        ("partial", lambda _name, raw: raw[:8], "signature/EOF"),
    ):
        case_root = tmp_path / label
        case_root.mkdir()
        case = _make_case(case_root)
        if label == "oversized":
            monkeypatch.setattr(closure, "MAX_PDF_BYTES", 8)
        else:
            monkeypatch.setattr(closure, "MAX_PDF_BYTES", 128 * 1024 * 1024)
        _install_fake_native_runner(monkeypatch, mutate=mutate)
        with pytest.raises(closure.DerivationClosureError, match=message):
            _build(case)
        stage = case.root / ".derivation-closure.private-candidate"
        assert stage.is_dir()
        assert not (stage / "runs").exists()


def test_nondeterministic_second_rebuild_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)

    def mutate(name: str, raw: bytes) -> bytes:
        return raw if name.endswith("-1") else raw.replace(b"%%EOF", b"drift\n%%EOF")

    _install_fake_native_runner(monkeypatch, mutate=mutate)
    with pytest.raises(closure.DerivationClosureError, match="authority anchor"):
        _build(case)


def test_same_vnode_source_mutation_is_detected_after_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    changed = False

    def invoke(
        _producer: object,
        _arguments: list[str],
        *,
        inherited_fds: tuple[int],
        budget: object,
        before: Callable[[], None],
        after: Callable[[], None],
    ) -> tuple[int, bytes, bytes, dict[str, object]]:
        nonlocal changed
        budget.consume()
        before()
        (source_fd,) = inherited_fds
        raw = os.pread(source_fd, os.fstat(source_fd).st_size, 0)
        if not changed:
            source_path = case.source_root / closure.SOURCE_MEMBER_BY_ID["clean"]
            source_path.write_bytes(raw + b"drift")
            changed = True
        after()
        return 0, raw, b"", {"protocol": "synthetic-attestation-v1"}

    monkeypatch.setattr(closure, "_invoke_producer", invoke)
    with pytest.raises(closure.DerivationClosureError, match="changed"):
        _build(case)


@pytest.mark.parametrize("attack", ["same-vnode-mutation", "inventory-growth"])
def test_post_parse_authority_mutation_or_inventory_growth_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    case = _make_case(tmp_path)
    changed = False

    def invoke(
        _producer: object,
        _arguments: list[str],
        *,
        inherited_fds: tuple[int],
        budget: object,
        before: Callable[[], None],
        after: Callable[[], None],
    ) -> tuple[int, bytes, bytes, dict[str, object]]:
        nonlocal changed
        budget.consume()
        before()
        (source_fd,) = inherited_fds
        raw = os.pread(source_fd, os.fstat(source_fd).st_size, 0)
        if not changed:
            if attack == "same-vnode-mutation":
                authority_path = (
                    case.authority_root / closure.AUTHORITY_MEMBER_BY_ID["clean"]
                )
                authority_path.write_bytes(authority_path.read_bytes() + b" ")
            else:
                (case.authority_root / "unexpected-authority.json").write_bytes(b"{}")
            changed = True
        after()
        return 0, raw, b"", {"protocol": "synthetic-attestation-v1"}

    monkeypatch.setattr(closure, "_invoke_producer", invoke)
    with pytest.raises(closure.DerivationClosureError):
        _build(case)


@pytest.mark.parametrize("kind", ["extra", "missing", "symlink", "hardlink", "fifo"])
def test_authority_inventory_and_member_type_attacks_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    case = _make_case(tmp_path)
    original_root = case.authority_root
    attacked_root = case.root / f"attacked-authority-{kind}"
    attacked_root.mkdir()
    for pdf_id in closure.PDF_IDS:
        member = closure.AUTHORITY_MEMBER_BY_ID[pdf_id]
        original = original_root / member
        target = attacked_root / member
        if pdf_id != "clean":
            target.write_bytes(original.read_bytes())
        elif kind == "missing":
            continue
        elif kind == "symlink":
            target.symlink_to(original)
        elif kind == "hardlink":
            os.link(original, target)
        elif kind == "fifo":
            os.mkfifo(target)
        else:
            target.write_bytes(original.read_bytes())
    if kind == "extra":
        (attacked_root / "extra-authority.json").write_bytes(b"{}")
    case.authority_root = attacked_root
    monkeypatch.setattr(
        closure,
        "_invoke_producer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError),
    )
    with pytest.raises(closure.DerivationClosureError):
        _build(case)


def test_destination_preexistence_and_concurrent_publish_are_no_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    case.destination.mkdir()
    _install_fake_native_runner(monkeypatch)
    with pytest.raises(closure.DerivationClosureError, match="already exists"):
        _build(case)
    assert case.destination.is_dir()


def test_publish_race_preserves_competitor_and_private_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    _install_fake_native_runner(monkeypatch)
    original = closure._rename_no_replace

    def race(source: str, destination: str, parent_fd: int) -> None:
        case.destination.mkdir()
        original(source, destination, parent_fd)

    monkeypatch.setattr(closure, "_rename_no_replace", race)
    with pytest.raises(closure.DerivationClosureError, match="candidate_paths"):
        _build(case)
    assert case.destination.is_dir()
    assert (case.root / ".derivation-closure.private-candidate").is_dir()


def test_base_exception_retains_stage_and_closes_owned_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor_root = Path("/dev/fd")
    if not descriptor_root.exists():
        pytest.skip("descriptor inventory unavailable")
    case = _make_case(tmp_path)

    def interrupt(*_args: object, **_kwargs: object) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(closure, "_invoke_producer", interrupt)
    before = len(tuple(descriptor_root.iterdir()))
    with pytest.raises(closure.DerivationClosureError, match="candidate_path"):
        _build(case)
    after = len(tuple(descriptor_root.iterdir()))
    assert after <= before + 2
    assert (case.root / ".derivation-closure.private-candidate").is_dir()


@pytest.mark.parametrize("fail_call", [2, 3, 4])
def test_late_member_open_failure_closes_every_previously_owned_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_call: int,
) -> None:
    descriptor_root = Path("/dev/fd")
    if not descriptor_root.exists():
        pytest.skip("descriptor inventory unavailable")
    case = _make_case(tmp_path)
    original = closure._open_root_member
    calls = 0

    def fail_late(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == fail_call:
            message = f"synthetic open failure {fail_call}"
            raise KeyboardInterrupt(message)
        return original(*args, **kwargs)

    monkeypatch.setattr(closure, "_open_root_member", fail_late)
    before = len(tuple(descriptor_root.iterdir()))
    with pytest.raises(closure.DerivationClosureError, match="synthetic open failure"):
        _build(case)
    after = len(tuple(descriptor_root.iterdir()))
    assert after <= before + 2


def test_attempt_all_cleanup_closes_later_fd_after_first_close_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "owned-descriptor.txt"
    path.write_bytes(b"synthetic")
    descriptor = os.open(path, os.O_RDONLY)

    class FailingClose:
        def close(self) -> None:
            raise KeyboardInterrupt("first close failed")

    class DescriptorClose:
        def close(self) -> None:
            os.close(descriptor)

    with pytest.raises(
        closure.DerivationClosureError,
        match="first close failed",
    ):
        closure._close_resources(
            [("first", FailingClose()), ("later descriptor", DescriptorClose())],
            context="synthetic cleanup",
        )
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(descriptor)


def test_anchored_manifest_tamper_and_wrong_anchor_fail_before_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    _install_fake_native_runner(monkeypatch)
    built = _build(case)
    with pytest.raises(closure.DerivationClosureError, match="independent"):
        closure.validate_derivation_closure(
            case.plan_path,
            case.source_root,
            case.producer_root,
            case.authority_root,
            case.destination,
            case.replay_root,
            expected_manifest_sha256="f" * 64,
            **case.kwargs(),
        )
    assert not case.replay_root.exists()

    case.destination.chmod(0o700)
    manifest_path = case.destination / closure.MANIFEST_MEMBER
    manifest_path.chmod(0o600)
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
    manifest_path.chmod(0o400)
    case.destination.chmod(0o500)
    with pytest.raises(closure.DerivationClosureError, match="anchor"):
        closure._read_anchored_manifest(
            case.destination,
            expected_manifest_sha256=built.manifest_sha256,
        )


@pytest.mark.parametrize(
    ("record_key", "input_key"),
    [
        ("builder", "builder_sha256"),
        ("native_execution_dependency", "machine_runner_sha256"),
    ],
)
def test_manifest_false_code_provenance_fails_before_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_key: str,
    input_key: str,
) -> None:
    case = _make_case(tmp_path)
    _install_fake_native_runner(monkeypatch)
    _build(case)

    def mutate(manifest: dict[str, object]) -> None:
        _mapping(manifest[record_key])["sha256"] = "f" * 64
        assert _mapping(manifest["inputs"])[input_key] != "f" * 64

    forged_sha256 = _rewrite_sealed_manifest(case.destination, mutate)
    with pytest.raises(closure.DerivationClosureError, match="pinned input binding"):
        closure._read_anchored_manifest(
            case.destination,
            expected_manifest_sha256=forged_sha256,
        )


def test_consistent_declared_receipt_forgery_fails_against_opened_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    _install_fake_native_runner(monkeypatch)
    _build(case)

    def mutate(manifest: dict[str, object]) -> None:
        document = _mapping(_array(manifest["documents"])[0])
        runs = _array(document["runs"])
        for raw_run in runs:
            run = _mapping(raw_run)
            run["invocation_receipt_bytes"] = 64
            run["invocation_receipt_sha256"] = "f" * 64
        projection = _mapping(document["native_closure_projection"])
        projection["invocation_receipt_bytes"] = 64
        projection["invocation_receipt_sha256"] = "f" * 64

    forged_sha256 = _rewrite_sealed_manifest(case.destination, mutate)
    with pytest.raises(
        closure.DerivationClosureError,
        match="invocation receipt digest or byte count",
    ):
        closure._read_anchored_manifest(
            case.destination,
            expected_manifest_sha256=forged_sha256,
        )


def test_replay_failure_is_retained_and_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    _install_fake_native_runner(monkeypatch)
    built = _build(case)
    calls = 0

    def fail_replay(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise RuntimeError("synthetic replay failure")

    monkeypatch.setattr(closure, "_invoke_producer", fail_replay)
    with pytest.raises(closure.DerivationClosureError, match="replay_candidate_path"):
        _validate(case, built)
    assert calls == 1
    assert case.replay_root.is_dir()


def test_fd_headroom_and_process_budget_are_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    monkeypatch.setattr(closure.resource, "getrlimit", lambda _key: (8, 8))
    _install_fake_native_runner(monkeypatch)
    with pytest.raises(closure.DerivationClosureError, match="RLIMIT_NOFILE"):
        _build(case)
    budget = closure._InvocationBudget()
    for _ in range(closure.MAX_ADAPTER_INVOCATIONS):
        budget.consume()
    with pytest.raises(closure.DerivationClosureError, match="invocation count"):
        budget.consume()


def test_cli_help_surfaces_are_available() -> None:
    script = Path(closure.__file__)
    for argv in (["--help"], ["build", "--help"], ["validate", "--help"]):
        result = subprocess.run(
            [sys.executable, str(script), *argv],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "derivation" in result.stdout.lower()
        if argv[0] == "build":
            assert "authorized plan" in result.stdout.lower()
            assert "sealed synthetic" not in result.stdout.lower()


def test_source_contains_no_destructive_or_shell_helpers() -> None:
    source = Path(closure.__file__).read_text(encoding="utf-8")
    assert "os.unlink(" not in source
    assert "shutil.rmtree" not in source
    assert "subprocess.run(" not in source
    assert "shell=True" not in source
    assert "research/notes/18_rebuttal_draft.md" not in source


def test_real_darwin_stdout_overflow_and_descendant_timeout_are_killed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sys.platform != "darwin" or os.uname().machine != "arm64":
        pytest.skip("Darwin arm64 suspended executable attestation is required")
    compiler = shutil.which("clang") or shutil.which("cc")
    if compiler is None:
        pytest.skip("a local C compiler is unavailable")
    programs = {
        "overflow": r"""#include <unistd.h>
int main(void) {
  char bytes[4096] = {0};
  for (;;) if (write(STDOUT_FILENO, bytes, sizeof(bytes)) <= 0) return 2;
}
""",
        "descendant": r"""#include <unistd.h>
int main(void) {
  pid_t child = fork();
  if (child < 0) return 2;
  if (child == 0) { for (;;) pause(); }
  const char pdf[] = "%PDF-1.7\nsynthetic\n%%EOF\n";
  return write(STDOUT_FILENO, pdf, sizeof(pdf) - 1) < 0 ? 3 : 0;
}
""",
    }
    for name, source in programs.items():
        source_path = tmp_path / f"{name}.c"
        source_path.write_text(source, encoding="ascii")
        executable = tmp_path / f"native-{name}"
        result = subprocess.run(
            [compiler, "-O2", "-o", str(executable), str(source_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip(f"synthetic native helper compilation failed: {result.stderr}")
        pin = closure._pin_file(
            executable,
            maximum=closure.MAX_PRODUCER_BYTES,
            context=f"synthetic {name} executable",
        )
        monkeypatch.setattr(
            closure,
            "MAX_PDF_BYTES",
            1024 if name == "overflow" else 4096,
        )
        monkeypatch.setattr(
            closure,
            "PRODUCER_TIMEOUT_SECONDS",
            2.0 if name == "overflow" else 0.75,
        )
        expected = "stdout exceeds" if name == "overflow" else "timeout"
        started = time.monotonic()
        try:
            with pytest.raises(closure.DerivationClosureError, match=expected):
                closure._invoke_producer(
                    pin,
                    [],
                    inherited_fds=(),
                    budget=closure._InvocationBudget(),
                    before=lambda: None,
                    after=lambda: None,
                )
        finally:
            closure._close_resources(
                [(f"synthetic {name} executable", pin)],
                context="synthetic bounded-child canary",
            )
        assert time.monotonic() - started < 5


def test_real_darwin_native_four_pdf_canary_builds_and_replays(
    tmp_path: Path,
) -> None:
    if sys.platform != "darwin":
        pytest.skip("Darwin suspended native-executable attestation is required")
    compiler = shutil.which("clang") or shutil.which("cc")
    if compiler is None:
        pytest.skip("a local C compiler is unavailable")
    case = _make_case(tmp_path)
    source_code = tmp_path / "synthetic_adapter.c"
    source_code.write_text(
        r"""#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
  if (argc != 9 || strcmp(argv[1], "--dialect-derivation-protocol") != 0 ||
      strcmp(argv[2], "dialect-pdf-derivation-fd-protocol-v1") != 0 ||
      strcmp(argv[3], "--pdf-id") != 0 || strcmp(argv[5], "--source-fd") != 0 ||
      strcmp(argv[7], "--pdf-output") != 0 || strcmp(argv[8], "stdout") != 0)
    return 20;
  int input = atoi(argv[6]);
  char buffer[4096];
  for (;;) {
    ssize_t got = read(input, buffer, sizeof(buffer));
    if (got < 0) return 21;
    if (got == 0) break;
    ssize_t written = 0;
    while (written < got) {
      ssize_t step = write(STDOUT_FILENO, buffer + written,
                           (size_t)(got - written));
      if (step <= 0) return 22;
      written += step;
    }
  }
  return 0;
}
""",
        encoding="ascii",
    )
    compiled = tmp_path / "synthetic-native-adapter"
    result = subprocess.run(
        [compiler, "-O2", "-o", str(compiled), str(source_code)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"synthetic native adapter compilation failed: {result.stderr}")
    for pdf_id in closure.PDF_IDS:
        target = case.producer_root / closure.PRODUCER_MEMBER_BY_ID[pdf_id]
        target.chmod(0o700)
        target.write_bytes(compiled.read_bytes())
        target.chmod(0o500)
        case.producer_sha256[pdf_id] = _file_sha256(target)
    plan = case.plan()
    for raw_document in _array(plan["documents"]):
        document = _mapping(raw_document)
        pdf_id = str(document["pdf_id"])
        producer = case.producer_root / closure.PRODUCER_MEMBER_BY_ID[pdf_id]
        document["producer_bytes"] = producer.stat().st_size
        document["producer_sha256"] = case.producer_sha256[pdf_id]
        _mapping(document["adapter_authority"])["reviewed_producer_sha256"] = (
            case.producer_sha256[pdf_id]
        )
    _sync_authority_receipts(case, plan)
    built = _build(case)
    replayed = _validate(case, built)
    assert replayed.manifest_sha256 == built.manifest_sha256
    assert replayed.pdf_count == 4
