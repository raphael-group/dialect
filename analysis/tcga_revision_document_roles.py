"""Define the immutable four-role revision-document contract registry.

The registry contains names and build profiles only.  It does not open source
bundles, PDFs, manuscript text, or scientific result files, and it does not
authorize any document for release.  Future bundle, renderer, and native
producer entry points can share these exact role bindings without duplicating
role-specific constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

DocumentRoleId: TypeAlias = Literal["clean", "marked", "s1", "rebuttal"]

DERIVATION_PROTOCOL: Final = "dialect-pdf-derivation-fd-protocol-v1"
DERIVATION_BUNDLE_CONTRACT: Final = (
    "canonical-inputs-pinned-renderer-fresh-pdf-derivation-v1"
)
NATIVE_PRODUCER_AUTHORITY_CONTRACT: Final = (
    "host-bound-thin-arm64-double-build-adhoc-codesign-two-member-package-v2"
)
RENDERER_LOCATOR: Final = "current-renderer"
SHARED_LAUNCHER_SOURCE_MEMBER: Final = "analysis/native/document_derivation_launcher.c"


@dataclass(frozen=True, slots=True)
class DocumentRoleSpec:
    """Hold the exact public names and build profile for one PDF role.

    Attributes:
        role_id: Canonical role token used by the descriptor protocol.
        pdf_member: Final PDF filename produced on standard output.
        source_bundle_member: Opaque source-bundle filename used by the closure.
        producer_member: Native launcher filename inside its sealed package.
        authority_member: Canonical producer-authority filename.
        render_manifest_schema: Schema of the role's renderer receipt.
        derivation_bundle_schema: Schema of the role's source-only bundle.
        native_producer_authority_schema: Schema of its native authority capsule.
        native_launcher_config_schema: Schema of its embedded launcher contract.
        input_members: Ordered canonical source-only inputs; never a PDF.  An
            empty tuple is a fail-closed declaration that the inventory has not
            yet been designed or frozen.
        input_inventory_frozen: Whether ``input_members`` is production-defined.
        source_bundle_max_bytes: Frozen positive bundle bound, or ``None`` while
            the role's source inventory remains unresolved.
        pdf_max_bytes: Frozen positive output bound, or ``None`` while the role's
            production renderer contract remains unresolved.
        renderer_locator: Stable logical locator recorded for the renderer.
        renderer_member: Tracked role-specific renderer entry point.
        launcher_source_member: Tracked launcher source currently bound by the
            role's released producer, or ``None`` while no production producer
            exists.  This intentionally does not project the unintegrated shared
            launcher as current authority.
        native_producer_builder_member: Tracked role-specific producer CLI.
        derivation_bundle_builder_member: Tracked role-specific bundle CLI.
        toolchain_profile: Canonical role-specific document build profile.
    """

    role_id: DocumentRoleId
    pdf_member: str
    source_bundle_member: str
    producer_member: str
    authority_member: str
    render_manifest_schema: str
    derivation_bundle_schema: str
    native_producer_authority_schema: str
    native_launcher_config_schema: str
    input_members: tuple[str, ...]
    input_inventory_frozen: bool
    source_bundle_max_bytes: int | None
    pdf_max_bytes: int | None
    renderer_locator: str
    renderer_member: str
    launcher_source_member: str | None
    native_producer_builder_member: str
    derivation_bundle_builder_member: str
    toolchain_profile: str


DOCUMENT_ROLE_SPECS: Final[tuple[DocumentRoleSpec, ...]] = (
    DocumentRoleSpec(
        role_id="clean",
        pdf_member="manuscript-clean.pdf",
        source_bundle_member="clean-source.bundle",
        producer_member="derive-clean",
        authority_member="clean-producer-toolchain-authority.json",
        render_manifest_schema="dialect-revision-clean-render-v1",
        derivation_bundle_schema="dialect-revision-clean-derivation-bundle-v1",
        native_producer_authority_schema=(
            "dialect-revision-clean-native-producer-authority-v2"
        ),
        native_launcher_config_schema=(
            "dialect-revision-clean-native-launcher-config-v2"
        ),
        input_members=(),
        input_inventory_frozen=False,
        source_bundle_max_bytes=None,
        pdf_max_bytes=None,
        renderer_locator=RENDERER_LOCATOR,
        renderer_member="analysis/render_tcga_revision_clean.py",
        launcher_source_member=None,
        native_producer_builder_member=(
            "analysis/build_tcga_revision_clean_native_producer.py"
        ),
        derivation_bundle_builder_member=(
            "analysis/build_tcga_revision_clean_derivation_bundle.py"
        ),
        toolchain_profile="latexpand-pdflatex-bibtex-inlined-double-build",
    ),
    DocumentRoleSpec(
        role_id="marked",
        pdf_member="manuscript-marked.pdf",
        source_bundle_member="marked-source.bundle",
        producer_member="derive-marked",
        authority_member="marked-producer-toolchain-authority.json",
        render_manifest_schema="dialect-revision-marked-render-v1",
        derivation_bundle_schema="dialect-revision-marked-derivation-bundle-v1",
        native_producer_authority_schema=(
            "dialect-revision-marked-native-producer-authority-v2"
        ),
        native_launcher_config_schema=(
            "dialect-revision-marked-native-launcher-config-v2"
        ),
        input_members=(),
        input_inventory_frozen=False,
        source_bundle_max_bytes=None,
        pdf_max_bytes=None,
        renderer_locator=RENDERER_LOCATOR,
        renderer_member="analysis/render_tcga_revision_marked.py",
        launcher_source_member=None,
        native_producer_builder_member=(
            "analysis/build_tcga_revision_marked_native_producer.py"
        ),
        derivation_bundle_builder_member=(
            "analysis/build_tcga_revision_marked_derivation_bundle.py"
        ),
        toolchain_profile=(
            "latexpand-latexdiff-latexrevise-pdflatex-bibtex-inlined-double-build"
        ),
    ),
    DocumentRoleSpec(
        role_id="s1",
        pdf_member="s1-appendix.pdf",
        source_bundle_member="s1-source.bundle",
        producer_member="derive-s1",
        authority_member="s1-producer-toolchain-authority.json",
        render_manifest_schema="dialect-revision-s1-render-v1",
        derivation_bundle_schema="dialect-revision-s1-derivation-bundle-v1",
        native_producer_authority_schema=(
            "dialect-revision-s1-native-producer-authority-v2"
        ),
        native_launcher_config_schema=("dialect-revision-s1-native-launcher-config-v2"),
        input_members=(),
        input_inventory_frozen=False,
        source_bundle_max_bytes=None,
        pdf_max_bytes=None,
        renderer_locator=RENDERER_LOCATOR,
        renderer_member="analysis/render_tcga_revision_s1.py",
        launcher_source_member=None,
        native_producer_builder_member=(
            "analysis/build_tcga_revision_s1_native_producer.py"
        ),
        derivation_bundle_builder_member=(
            "analysis/build_tcga_revision_s1_derivation_bundle.py"
        ),
        toolchain_profile="latexmk-pdf-pdflatex",
    ),
    DocumentRoleSpec(
        role_id="rebuttal",
        pdf_member="response-to-reviewers.pdf",
        source_bundle_member="rebuttal-source.bundle",
        producer_member="derive-rebuttal",
        authority_member="rebuttal-producer-toolchain-authority.json",
        render_manifest_schema="dialect-revision-rebuttal-render-v1",
        derivation_bundle_schema="dialect-revision-rebuttal-derivation-bundle-v1",
        native_producer_authority_schema=(
            "dialect-revision-rebuttal-native-producer-authority-v2"
        ),
        native_launcher_config_schema=(
            "dialect-revision-rebuttal-native-launcher-config-v2"
        ),
        input_members=(
            "source.canonical.md",
            "template.canonical.json",
            "config.canonical.json",
        ),
        input_inventory_frozen=True,
        source_bundle_max_bytes=8 * 1024 * 1024,
        pdf_max_bytes=32 * 1024 * 1024,
        renderer_locator=RENDERER_LOCATOR,
        renderer_member="analysis/render_tcga_revision_rebuttal.py",
        launcher_source_member="analysis/native/rebuttal_derivation_launcher.c",
        native_producer_builder_member=(
            "analysis/build_tcga_revision_rebuttal_native_producer.py"
        ),
        derivation_bundle_builder_member=(
            "analysis/build_tcga_revision_rebuttal_derivation_bundle.py"
        ),
        toolchain_profile="descriptor-rooted-reportlab-invariant-double-build-v1",
    ),
)

DOCUMENT_ROLE_IDS: Final[tuple[DocumentRoleId, ...]] = tuple(
    spec.role_id for spec in DOCUMENT_ROLE_SPECS
)
DOCUMENT_ROLE_BY_ID: Final = MappingProxyType(
    {spec.role_id: spec for spec in DOCUMENT_ROLE_SPECS},
)
