"""Tests for the immutable revision-document role registry."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from analysis import build_tcga_revision_rebuttal_native_producer as native
from analysis import render_tcga_revision_rebuttal as renderer
from analysis import tcga_revision_document_roles as roles

EXPECTED_ROLES = (
    {
        "role_id": "clean",
        "pdf_member": "manuscript-clean.pdf",
        "source_bundle_member": "clean-source.bundle",
        "producer_member": "derive-clean",
        "authority_member": "clean-producer-toolchain-authority.json",
        "render_manifest_schema": "dialect-revision-clean-render-v1",
        "derivation_bundle_schema": "dialect-revision-clean-derivation-bundle-v1",
        "native_producer_authority_schema": (
            "dialect-revision-clean-native-producer-authority-v2"
        ),
        "native_launcher_config_schema": (
            "dialect-revision-clean-native-launcher-config-v2"
        ),
        "input_members": (),
        "input_inventory_frozen": False,
        "source_bundle_max_bytes": None,
        "pdf_max_bytes": None,
        "renderer_locator": "current-renderer",
        "renderer_member": "analysis/render_tcga_revision_clean.py",
        "launcher_source_member": None,
        "native_producer_builder_member": (
            "analysis/build_tcga_revision_clean_native_producer.py"
        ),
        "derivation_bundle_builder_member": (
            "analysis/build_tcga_revision_clean_derivation_bundle.py"
        ),
        "toolchain_profile": "latexpand-pdflatex-bibtex-inlined-double-build",
    },
    {
        "role_id": "marked",
        "pdf_member": "manuscript-marked.pdf",
        "source_bundle_member": "marked-source.bundle",
        "producer_member": "derive-marked",
        "authority_member": "marked-producer-toolchain-authority.json",
        "render_manifest_schema": "dialect-revision-marked-render-v1",
        "derivation_bundle_schema": "dialect-revision-marked-derivation-bundle-v1",
        "native_producer_authority_schema": (
            "dialect-revision-marked-native-producer-authority-v2"
        ),
        "native_launcher_config_schema": (
            "dialect-revision-marked-native-launcher-config-v2"
        ),
        "input_members": (),
        "input_inventory_frozen": False,
        "source_bundle_max_bytes": None,
        "pdf_max_bytes": None,
        "renderer_locator": "current-renderer",
        "renderer_member": "analysis/render_tcga_revision_marked.py",
        "launcher_source_member": None,
        "native_producer_builder_member": (
            "analysis/build_tcga_revision_marked_native_producer.py"
        ),
        "derivation_bundle_builder_member": (
            "analysis/build_tcga_revision_marked_derivation_bundle.py"
        ),
        "toolchain_profile": (
            "latexpand-latexdiff-latexrevise-pdflatex-bibtex-inlined-double-build"
        ),
    },
    {
        "role_id": "s1",
        "pdf_member": "s1-appendix.pdf",
        "source_bundle_member": "s1-source.bundle",
        "producer_member": "derive-s1",
        "authority_member": "s1-producer-toolchain-authority.json",
        "render_manifest_schema": "dialect-revision-s1-render-v1",
        "derivation_bundle_schema": "dialect-revision-s1-derivation-bundle-v1",
        "native_producer_authority_schema": (
            "dialect-revision-s1-native-producer-authority-v2"
        ),
        "native_launcher_config_schema": (
            "dialect-revision-s1-native-launcher-config-v2"
        ),
        "input_members": (),
        "input_inventory_frozen": False,
        "source_bundle_max_bytes": None,
        "pdf_max_bytes": None,
        "renderer_locator": "current-renderer",
        "renderer_member": "analysis/render_tcga_revision_s1.py",
        "launcher_source_member": None,
        "native_producer_builder_member": (
            "analysis/build_tcga_revision_s1_native_producer.py"
        ),
        "derivation_bundle_builder_member": (
            "analysis/build_tcga_revision_s1_derivation_bundle.py"
        ),
        "toolchain_profile": "latexmk-pdf-pdflatex",
    },
    {
        "role_id": "rebuttal",
        "pdf_member": "response-to-reviewers.pdf",
        "source_bundle_member": "rebuttal-source.bundle",
        "producer_member": "derive-rebuttal",
        "authority_member": "rebuttal-producer-toolchain-authority.json",
        "render_manifest_schema": "dialect-revision-rebuttal-render-v1",
        "derivation_bundle_schema": "dialect-revision-rebuttal-derivation-bundle-v1",
        "native_producer_authority_schema": (
            "dialect-revision-rebuttal-native-producer-authority-v2"
        ),
        "native_launcher_config_schema": (
            "dialect-revision-rebuttal-native-launcher-config-v2"
        ),
        "input_members": (
            "source.canonical.md",
            "template.canonical.json",
            "config.canonical.json",
        ),
        "input_inventory_frozen": True,
        "source_bundle_max_bytes": 8 * 1024 * 1024,
        "pdf_max_bytes": 32 * 1024 * 1024,
        "renderer_locator": "current-renderer",
        "renderer_member": "analysis/render_tcga_revision_rebuttal.py",
        "launcher_source_member": "analysis/native/rebuttal_derivation_launcher.c",
        "native_producer_builder_member": (
            "analysis/build_tcga_revision_rebuttal_native_producer.py"
        ),
        "derivation_bundle_builder_member": (
            "analysis/build_tcga_revision_rebuttal_derivation_bundle.py"
        ),
        "toolchain_profile": ("descriptor-rooted-reportlab-invariant-double-build-v1"),
    },
)


@pytest.mark.parametrize(
    ("spec", "expected"),
    zip(roles.DOCUMENT_ROLE_SPECS, EXPECTED_ROLES, strict=True),
)
def test_role_spec_matches_exact_contract(
    spec: roles.DocumentRoleSpec,
    expected: dict[str, object],
) -> None:
    assert {
        field: getattr(spec, field)
        for field in roles.DocumentRoleSpec.__dataclass_fields__
    } == expected


def test_registry_has_exact_order_and_lookup_identity() -> None:
    assert roles.DOCUMENT_ROLE_IDS == ("clean", "marked", "s1", "rebuttal")
    assert tuple(roles.DOCUMENT_ROLE_BY_ID) == roles.DOCUMENT_ROLE_IDS
    assert tuple(roles.DOCUMENT_ROLE_BY_ID.values()) == roles.DOCUMENT_ROLE_SPECS


def test_role_specs_and_lookup_are_immutable() -> None:
    spec = roles.DOCUMENT_ROLE_BY_ID["clean"]
    with pytest.raises(FrozenInstanceError):
        spec.pdf_member = "changed.pdf"  # type: ignore[misc]
    with pytest.raises(TypeError):
        roles.DOCUMENT_ROLE_BY_ID["clean"] = spec  # type: ignore[index]


def test_role_bindings_are_unique_and_source_only() -> None:
    unique_fields = (
        "pdf_member",
        "source_bundle_member",
        "producer_member",
        "authority_member",
        "render_manifest_schema",
        "derivation_bundle_schema",
        "native_producer_authority_schema",
        "native_launcher_config_schema",
        "renderer_member",
        "native_producer_builder_member",
        "derivation_bundle_builder_member",
        "toolchain_profile",
    )
    for field in unique_fields:
        values = tuple(getattr(spec, field) for spec in roles.DOCUMENT_ROLE_SPECS)
        assert len(set(values)) == len(roles.DOCUMENT_ROLE_SPECS)
    for spec in roles.DOCUMENT_ROLE_SPECS:
        assert len(set(spec.input_members)) == len(spec.input_members)
        assert spec.pdf_member not in spec.input_members
        assert all(not member.endswith(".pdf") for member in spec.input_members)
        assert spec.input_inventory_frozen is bool(spec.input_members)
        if spec.input_inventory_frozen:
            assert spec.source_bundle_max_bytes is not None
            assert spec.source_bundle_max_bytes > 0
            assert spec.pdf_max_bytes is not None
            assert spec.pdf_max_bytes > 0
        else:
            assert spec.source_bundle_max_bytes is None
            assert spec.pdf_max_bytes is None

    assert {
        spec.role_id: spec.launcher_source_member for spec in roles.DOCUMENT_ROLE_SPECS
    } == {
        "clean": None,
        "marked": None,
        "s1": None,
        "rebuttal": "analysis/native/rebuttal_derivation_launcher.c",
    }


def test_only_existing_rebuttal_input_inventory_is_frozen() -> None:
    assert {spec.role_id: spec.input_members for spec in roles.DOCUMENT_ROLE_SPECS} == {
        "clean": (),
        "marked": (),
        "s1": (),
        "rebuttal": (
            "source.canonical.md",
            "template.canonical.json",
            "config.canonical.json",
        ),
    }


def test_shared_contract_constants_are_exact() -> None:
    assert roles.DERIVATION_PROTOCOL == "dialect-pdf-derivation-fd-protocol-v1"
    assert roles.DERIVATION_BUNDLE_CONTRACT == (
        "canonical-inputs-pinned-renderer-fresh-pdf-derivation-v1"
    )
    assert roles.NATIVE_PRODUCER_AUTHORITY_CONTRACT == (
        "host-bound-thin-arm64-double-build-adhoc-codesign-two-member-package-v2"
    )
    assert roles.SHARED_LAUNCHER_SOURCE_MEMBER == (
        "analysis/native/document_derivation_launcher.c"
    )


def test_rebuttal_registry_cross_binds_current_released_implementation() -> None:
    spec = roles.DOCUMENT_ROLE_BY_ID["rebuttal"]
    assert spec.role_id == native.PDF_ID == renderer.DERIVATION_ROLE
    assert spec.pdf_member == native.PDF_MEMBER == renderer.PDF_MEMBER
    assert spec.producer_member == native.PRODUCER_MEMBER
    assert spec.authority_member == native.AUTHORITY_MEMBER
    assert spec.render_manifest_schema == renderer.SCHEMA
    assert spec.derivation_bundle_schema == renderer.DERIVATION_BUNDLE_SCHEMA
    assert spec.native_producer_authority_schema == native.AUTHORITY_SCHEMA
    assert spec.native_launcher_config_schema == native.CONFIG_SCHEMA
    assert spec.input_members == (
        renderer.SOURCE_MEMBER,
        renderer.TEMPLATE_MEMBER,
        renderer.CONFIG_MEMBER,
    )
    assert spec.source_bundle_max_bytes == renderer.MAX_DERIVATION_BUNDLE_BYTES
    assert spec.pdf_max_bytes == renderer.MAX_PDF_BYTES
    assert spec.launcher_source_member == native.LAUNCHER_SOURCE_MEMBER
    assert roles.DERIVATION_PROTOCOL == native.PROTOCOL == renderer.DERIVATION_PROTOCOL
    assert roles.DERIVATION_BUNDLE_CONTRACT == renderer.DERIVATION_BUNDLE_CONTRACT
    assert roles.NATIVE_PRODUCER_AUTHORITY_CONTRACT == native.AUTHORITY_CONTRACT
