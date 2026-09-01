"""Tests for the deterministic response-to-reviewers PDF producer."""

# Verbatim synthetic reviewer quotations intentionally exceed normal line length.
# Tests directly exercise private fail-closed seams by design.
# ruff: noqa: COM812, E501, PLR0915, PT011, PT018, RUF001, RUF043, S108, SLF001

from __future__ import annotations

import builtins
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import types
from pathlib import Path

import pytest

from analysis import render_tcga_revision_rebuttal as renderer

REQUIRES_NATIVE_DARWIN = pytest.mark.skipif(
    sys.platform != "darwin" or os.uname().machine != "arm64",
    reason="native execution attestation and no-replace publication require arm64 Darwin",
)

BUNDLED_PYTHON = Path(
    os.environ.get(
        "DIALECT_REBUTTAL_PYTHON",
        Path.home()
        / ".cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12",
    ),
)
REPORTLAB_ROOT = Path(
    os.environ.get(
        "DIALECT_REBUTTAL_REPORTLAB_ROOT",
        BUNDLED_PYTHON.parent.parent / "lib/python3.12/site-packages/reportlab",
    ),
)
REGULAR_FONT = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
BOLD_FONT = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
MACHINE_RUNNER = renderer._machine_runner_path()

SYNTHETIC_SOURCE = """# Response to Reviewers

We thank the editor and reviewers for their careful assessment. This fixture
contains no scientific result and exists only to exercise document packaging.

## Reviewer #1

### R1-1 - Clarify the model description

<!-- SOURCE-COMMENT:r1-1:BEGIN -->
```text
1. Please distinguish the observed count from the latent driver state and explain the background input.
```
<!-- SOURCE-COMMENT:r1-1:END -->

We agree. The revised description separates the observed count, passenger
background, and latent driver state in one concise paragraph.

- The observed count is recorded directly.
- The background distribution is supplied independently.
- The latent driver state is estimated by the model.

### R1-2 - State the claim boundary

> 2. Please state which conclusions follow from the statistical analysis and which require later biological validation.

We agree and now label the statistical association separately from any mechanistic
interpretation. The response also identifies the independent validation needed for
a biological claim.

## Reviewer #2

### R2-1 - Improve reproducibility

<!-- SOURCE-COMMENT:r2-1:BEGIN -->
```text
1) Please provide enough implementation detail for another group to reproduce the analysis.
```
<!-- SOURCE-COMMENT:r2-1:END -->

Thank you. We now report the software release, fixed inputs, deterministic settings,
and machine-readable provenance. The source and PDF are bound in one receipt.

1. The source is normalized before rendering.
2. Two independent builds must be byte-identical.
3. A separate replay must reproduce the complete receipt.
"""

TEMPLATE = {
    "schema": renderer.TEMPLATE_SCHEMA,
    "page_size": "letter",
    "margins": {"top": 60, "right": 58, "bottom": 54, "left": 58},
    "type": {
        "title": 24,
        "heading_2": 16,
        "heading_3": 12,
        "heading_4": 10,
        "body": 10.5,
        "leading": 15,
    },
    "colors": {
        "ink": "#17212B",
        "muted": "#5E6872",
        "accent": "#245C73",
        "quote_background": "#F2F5F6",
        "rule": "#CBD4D8",
    },
}

CONFIG = {
    "schema": renderer.CONFIG_SCHEMA,
    "release_id": "synthetic-rebuttal-v1",
    "manuscript_id": "PCOMPBIOL-SYNTHETIC-00000",
    "manuscript_title": "A synthetic manuscript used only for renderer QA",
    "response_title": "Response to Reviewers",
    "authors": ["Ada Éxample", "Ben Example"],
    "source_date_epoch": renderer.REPORTLAB_INVARIANT_EPOCH,
}


def _multipage_source() -> str:
    long_quote = " ".join(f"reviewer{i:04d}" for i in range(360))
    long_response = " ".join(f"response{i:04d}" for i in range(500))
    long_list_item = " ".join(f"listitem{i:04d}" for i in range(500))
    return rf"""# Response to Reviewers

This synthetic stress fixture contains no scientific result.

## Reviewer #1

### R1-1 - Long literal reviewer quotation

<!-- SOURCE-COMMENT:stress-quote:BEGIN -->
```text
TODO [K500+FIG: pending] `code` *literal* [paper](https://example.org/a_(b)) $x$ RESPONSE REVIEWER COMMENT PCOMPBIOL-SYNTHETIC-00000 two  spaces
{long_quote}
```
<!-- SOURCE-COMMENT:stress-quote:END -->

We acknowledge this synthetic comment and provide a bounded response.

### R1-2 - Long response paragraph

> Please verify that a long response can cross page boundaries.

The exact scientific typography χ² R² x₂ non‑breaking µ μ and symbols $\rho$ and $\tau$ remain visible. {long_response}

### R1-3 - Long list item

> Please verify that a long list item can cross page boundaries.

- This synthetic list item remains one ordered ledger entry across pages. {long_list_item}
"""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tool_paths() -> dict[str, Path]:
    def poppler(name: str) -> Path:
        override = os.environ.get(f"DIALECT_REBUTTAL_{name.upper()}")
        discovered = override or shutil.which(name)
        if discovered is None:
            pytest.skip(f"{name} is unavailable")
        return Path(discovered).resolve(strict=True)

    return {
        "python": BUNDLED_PYTHON.resolve(strict=True),
        "pdfinfo": poppler("pdfinfo"),
        "pdffonts": poppler("pdffonts"),
        "pdftotext": poppler("pdftotext"),
    }


def _require_real_dependencies() -> None:
    paths = [
        BUNDLED_PYTHON,
        REPORTLAB_ROOT,
        REGULAR_FONT,
        BOLD_FONT,
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        pytest.skip(f"real synthetic-render dependencies unavailable: {missing}")
    _tool_paths()


def _write_inputs(
    root: Path,
    *,
    source_text: str = SYNTHETIC_SOURCE,
    template_value: object = TEMPLATE,
    config_value: object = CONFIG,
) -> tuple[Path, Path, Path]:
    source = root / "response.md"
    template = root / "template.json"
    config = root / "config.json"
    source.write_text(source_text, encoding="utf-8")
    template.write_text(json.dumps(template_value, indent=2) + "\n", encoding="utf-8")
    config.write_text(json.dumps(config_value, indent=2) + "\n", encoding="utf-8")
    return source, template, config


def _build_kwargs(source: Path, template: Path, config: Path) -> dict[str, object]:
    _require_real_dependencies()
    tools = _tool_paths()
    dependency = renderer.reportlab_dependency_digest(REPORTLAB_ROOT)
    return {
        "regular_font": REGULAR_FONT,
        "bold_font": BOLD_FONT,
        "reportlab_root": REPORTLAB_ROOT,
        "release_id": str(CONFIG["release_id"]),
        "tool_paths": tools,
        "expected_source_sha256": _sha(source),
        "expected_template_sha256": _sha(template),
        "expected_config_sha256": _sha(config),
        "expected_regular_font_sha256": _sha(REGULAR_FONT),
        "expected_bold_font_sha256": _sha(BOLD_FONT),
        "expected_reportlab_tree_sha256": str(dependency["tree_sha256"]),
        "expected_machine_runner_sha256": _sha(MACHINE_RUNNER),
        "expected_builder_sha256": _sha(Path(renderer.__file__).resolve(strict=True)),
        "expected_tool_sha256": {name: _sha(path) for name, path in tools.items()},
    }


def _make_writable(root: Path) -> None:
    """Allow pytest to remove only its fresh synthetic temporary tree."""
    if not root.exists():
        return
    for directory, names, files in os.walk(root, topdown=False):
        for name in files:
            path = Path(directory) / name
            if path.exists() and not path.is_symlink():
                path.chmod(0o600)
        for name in names:
            path = Path(directory) / name
            if path.exists() and not path.is_symlink():
                path.chmod(0o700)
        Path(directory).chmod(0o700)


def _fake_production() -> tuple[renderer._Production, dict[str, bytes]]:
    canonical = {
        renderer.SOURCE_MEMBER: b"# source\n",
        renderer.TEMPLATE_MEMBER: b"{}\n",
        renderer.CONFIG_MEMBER: b"{}\n",
    }
    production = renderer._Production(
        manifest={},
        manifest_raw=b"{}\n",
        pdf_raw=b"%PDF-1.4\n%%EOF\n",
        page_count=1,
        quote_count=1,
        response_count=1,
    )
    return production, canonical


def test_canonical_markdown_and_comment_audit_preserve_verbatim_quotes() -> None:
    raw = SYNTHETIC_SOURCE.replace("\n", "\r\n").encode()
    canonical = renderer._canonicalize_markdown(raw)
    audit = renderer._parse_markdown(canonical)

    assert canonical.endswith(b"\n")
    assert b"\r" not in canonical
    assert audit.reviewer_count == 2
    assert audit.response_count == 3
    assert [quote_id for quote_id, _ in audit.quotes] == [
        "r1-1",
        "blockquote-2",
        "r2-1",
    ]
    assert audit.quotes[0][1] == (
        "1. Please distinguish the observed count from the latent driver state and "
        "explain the background input."
    )
    assert audit.quotes[2][1].startswith("1) Please provide enough implementation")


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("TODO: add final answer", "TODO/TBD/FIXME"),
        ("TBD after analysis", "TODO/TBD/FIXME"),
        ("FIXME this", "TODO/TBD/FIXME"),
        ("**[K500+CAL: insert result.]**", "DIALECT gate"),
        ("**[COMP: insert comparator result.]**", "DIALECT gate"),
        ("**[NONE: unresolved decision.]**", "DIALECT gate"),
        ("[fig: insert final panel]", "DIALECT gate"),
        ("[K500+TABLE: insert values]", "DIALECT gate"),
        ("[REL+msk: reconcile cohort]", "DIALECT gate"),
        ("**TODO: fill**", "TODO/TBD/FIXME"),
        ("(TBD)", "TODO/TBD/FIXME"),
        ("`FIXME`", "TODO/TBD/FIXME"),
        ("TBD.", "TODO/TBD/FIXME"),
        ("ＴＯＤＯ: add result", "TODO/TBD/FIXME"),
        ("[ＣＯＭＰ: insert result]", "DIALECT gate"),
        ("RECONCILIATION-PENDING:x", "reconciliation or gate"),
        (r"\input{secret}", "raw TeX"),
        ("<script>bad</script>", "raw HTML|private absolute"),
        ("![plot](https://example.org/plot.png)", "Markdown image"),
    ],
)
def test_unsafe_or_unresolved_source_fails_closed(payload: str, expected: str) -> None:
    source = SYNTHETIC_SOURCE.replace(
        "We agree. The revised description",
        f"{payload}\n\nWe agree. The revised description",
        1,
    )
    with pytest.raises(renderer.RebuttalRenderError, match=expected):
        renderer._parse_markdown(renderer._canonicalize_markdown(source.encode()))


@pytest.mark.parametrize(
    "payload",
    [
        "<!-- unterminated",
        "<!DOCTYPE html",
        "<![CDATA[unterminated",
        "<?xml unterminated",
        "<b malformed",
        "&alpha;",
        "&#945;",
        "&#x3B1;",
        "&copy;",
    ],
)
def test_malformed_html_and_entity_openers_fail_closed(payload: str) -> None:
    source = SYNTHETIC_SOURCE.replace(
        "We agree. The revised description",
        f"{payload}\n\nWe agree. The revised description",
        1,
    )
    with pytest.raises(renderer.RebuttalRenderError, match="raw HTML|entity"):
        renderer._parse_markdown(renderer._canonicalize_markdown(source.encode()))


def test_gate_vocabulary_avoids_scientific_and_editorial_false_positives() -> None:
    source = SYNTHETIC_SOURCE.replace(
        "We agree. The revised description",
        "[DNA: repair pathway] and [Note: concise wording]. A & B and x<y.\n\n"
        "We agree. The revised description",
        1,
    )
    audit = renderer._parse_markdown(renderer._canonicalize_markdown(source.encode()))
    assert audit.response_count == 3


def test_reviewer_quote_is_literal_while_author_tokens_fail() -> None:
    quoted = (
        "TODO TBD [K500+FIG: pending] \\input{x} <b malformed "
        "![plot](https://example.org/plot.png) `code` *word* "
        "[paper](https://example.org/paper) $x$"
    )
    source = SYNTHETIC_SOURCE.replace(
        "1. Please distinguish the observed count from the latent driver state and explain the background input.",
        quoted,
        1,
    )
    audit = renderer._parse_markdown(renderer._canonicalize_markdown(source.encode()))
    assert audit.quotes[0][1] == quoted

    author = source.replace("We agree. The revised description", "TODO author", 1)
    with pytest.raises(renderer.RebuttalRenderError, match="TODO/TBD/FIXME"):
        renderer._parse_markdown(renderer._canonicalize_markdown(author.encode()))


@pytest.mark.parametrize(
    "payload",
    [
        "/Users/alice/secret.csv",
        "path=/opt/private/file.txt",
        "file:///private/tmp/file.txt",
        "path=C:\\Users\\alice\\secret.txt",
        "path=..\\secret.txt",
        "path:/Users/alice/secret.csv",
        "source:/opt/private/file.txt",
        "home:~alice/Documents/private.txt",
        "~alice/Documents/private.txt",
        "<~alice/Documents/private.txt>",
        "</Users/alice/private.txt>",
        "</Users>",
        "</credentials.json>",
        "<$HOME/secret.txt>",
        "<$TMPDIR/secret.txt>",
        "</workspace>",
        "$HOME/secret.txt",
        "${HOME}/secret.txt",
        r"%USERPROFILE%\secret.txt",
        r"%APPDATA%\secret.txt",
        r"%TEMP%\secret.txt",
        r"<%USERPROFILE%\secret.txt>",
        r"home:%USERPROFILE%\secret.txt",
        "../../research/secret.md",
        "research/rebuttal.md",
        r"research\secret.md",
        ".codex/attachments/item.txt",
        r".codex\attachments\item.txt",
        "output/cohort/results.csv",
        r"output\cohort\results.csv",
        "data/private.csv",
        r"data\private.csv",
        "data/.env",
        r"data\.env",
        "attachments/.env",
        "results/run/output.tsv",
        "%2FUsers%2Falice%2Fsecret.txt",
        "%252FUsers%252Falice%252Fsecret.txt",
        "％２ＦUsers％２ｆalice％２ｆsecret.txt",
        "%EF%BC%852FUsers%EF%BC%852Falice%EF%BC%852Fsecret.txt",
        "C%3A%5CUsers%5Calice%5Csecret.txt",
        "file%3A%2F%2Fprivate%2Ftmp%2Fsecret.txt",
        "%2e%2e%2fresearch%2fsecret.md",
        "∕Users∕alice∕secret.txt",
        "research⁄secret.md",
        "research⧵secret.md",
        "https://example.org＜/Users/alice/secret.txt",
        "https://example.org＞/Users/alice/secret.txt",
        "https://example.org,/Users/alice/secret.txt",
        "https://example.org|/Users/alice/secret.txt",
        "https://example.org:/Users/alice/secret.txt",
        "https://example.org—/Users/alice/secret.txt",
        "https://example.org）/Users/alice/secret.txt",
        "ｆｉｌｅ：／／Users／alice／x",
        "research／secret.md",
    ],
)
def test_private_paths_fail_across_complete_source(payload: str) -> None:
    source = SYNTHETIC_SOURCE.replace(
        "1. Please distinguish the observed count from the latent driver state and explain the background input.",
        payload,
        1,
    )
    with pytest.raises(renderer.RebuttalRenderError, match="private absolute"):
        renderer._parse_markdown(renderer._canonicalize_markdown(source.encode()))


@pytest.mark.parametrize(
    "payload",
    [
        "https://github.com/example/repo/blob/main/analysis/build.py",
        "https://example.org/file.pdf",
        "https://example.org/a_(b)/public.pdf",
        "src/dialect/api.py",
        "docs/usage.rst",
        "analysis/build.py",
        "data/code and data/results",
        "ratio 1.2/3.4",
        "ratio a∕b and ~approximate",
        "95% confidence and https://example.org/a%2Fb/public.pdf",
    ],
)
def test_public_links_and_intentional_repo_paths_are_allowed(payload: str) -> None:
    source = SYNTHETIC_SOURCE.replace(
        "We agree. The revised description",
        f"{payload}.\n\nWe agree. The revised description",
        1,
    )
    renderer._parse_markdown(renderer._canonicalize_markdown(source.encode()))


def test_private_paths_are_rejected_in_config_strings() -> None:
    for key, value in (
        ("manuscript_title", "Private /Users/alice/title.txt"),
        ("authors", ["Ada Example", "file:///private/name.txt"]),
        ("manuscript_title", r"Private research\secret.md"),
        ("manuscript_title", "Private %2FUsers%2Falice%2Fsecret.txt"),
        ("manuscript_title", "Private research⁄secret.md"),
        ("manuscript_title", "Private home:~alice/Documents/secret.txt"),
        ("manuscript_title", "Private <~alice/Documents/secret.txt>"),
        ("manuscript_title", "Private </Users/alice/secret.txt>"),
        ("manuscript_title", "Private </Users>"),
        ("manuscript_title", "Private </credentials.json>"),
        ("manuscript_title", "Private <$HOME/secret.txt>"),
        ("manuscript_title", "Private <$TMPDIR/secret.txt>"),
        ("manuscript_title", "Private </workspace>"),
        ("manuscript_title", r"Private %USERPROFILE%\secret.txt"),
        ("manuscript_title", r"Private <%APPDATA%\secret.txt>"),
        ("manuscript_title", r"Private home:%TEMP%\secret.txt"),
        ("manuscript_title", "Private data/.env"),
        ("manuscript_title", r"Private attachments\.env"),
        (
            "manuscript_title",
            "Private https://example.org＜/Users/alice/secret.txt",
        ),
        (
            "manuscript_title",
            "Private https://example.org,/Users/alice/secret.txt",
        ),
        ("manuscript_title", "Private %252FUsers%252Falice%252Fsecret.txt"),
        ("manuscript_title", "Private ％２ＦUsers％２ｆalice％２ｆsecret.txt"),
        (
            "manuscript_title",
            "Private %EF%BC%852FUsers%EF%BC%852Falice%EF%BC%852Fsecret.txt",
        ),
    ):
        config = {**CONFIG, key: value}
        with pytest.raises(renderer.RebuttalRenderError, match="private absolute"):
            renderer._normalize_config(config)


def test_no_image_stub_rejects_every_direct_image_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_before = sys.modules.get("_imaging")
    monkeypatch.delitem(sys.modules, "PIL", raising=False)
    monkeypatch.delitem(sys.modules, "PIL.Image", raising=False)
    pillow_stub, image_stub = renderer._install_no_image_pillow_stub()

    assert sys.modules["PIL"] is pillow_stub
    assert sys.modules["PIL.Image"] is image_stub
    with pytest.raises(renderer.RebuttalRenderError, match="image operation 'open'"):
        image_stub.open("forbidden.png")
    with pytest.raises(renderer.RebuttalRenderError, match="image operation 'Image'"):
        image_stub.Image.new("RGB", (1, 1))
    assert sys.modules.get("_imaging") is native_before


def test_reportlab_import_guards_reject_ambient_preloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guarded = (
        "reportlab_mods",
        "reportlab_settings",
        *renderer.REPORTLAB_BLOCKED_OPTIONAL_MODULES,
    )
    for name in guarded:
        monkeypatch.delitem(sys.modules, name, raising=False)
    fake = types.ModuleType("_rl_accel")
    monkeypatch.setitem(sys.modules, "_rl_accel", fake)
    with pytest.raises(renderer.RebuttalRenderError, match="already occupied"):
        renderer._install_reportlab_import_guards()


def test_reportlab_import_guards_install_inert_search_paths() -> None:
    guarded = (
        "reportlab_mods",
        "reportlab_settings",
        *renderer.REPORTLAB_BLOCKED_OPTIONAL_MODULES,
    )
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in guarded}
    for name in guarded:
        sys.modules.pop(name, None)
    try:
        mods, settings = renderer._install_reportlab_import_guards()
        assert sys.modules["reportlab_mods"] is mods
        assert sys.modules["reportlab_settings"] is settings
        assert settings.T1SearchPath == []
        assert settings.TTFSearchPath == []
        assert settings.CMapSearchPath == []
        assert all(
            sys.modules[name] is None
            for name in renderer.REPORTLAB_BLOCKED_OPTIONAL_MODULES
        )
    finally:
        for name, value in previous.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def test_reportlab_runtime_audit_rejects_native_or_substituted_accelerator() -> None:
    guarded = {
        "reportlab_mods",
        "reportlab_settings",
        *renderer.REPORTLAB_BLOCKED_OPTIONAL_MODULES,
    }
    relevant = {
        name: value
        for name, value in sys.modules.items()
        if name == "reportlab" or name.startswith("reportlab.") or name in guarded
    }
    for name in relevant:
        sys.modules.pop(name, None)
    bundle = "/dev/fd/99"
    try:
        mods, settings = renderer._install_reportlab_import_guards()
        config = types.ModuleType("reportlab.rl_config")
        config.__file__ = f"{bundle}/reportlab/rl_config.py"
        config.T1SearchPath = []
        config.TTFSearchPath = []
        config.CMapSearchPath = []
        accel = types.ModuleType("reportlab.lib.rl_accel")
        accel.__file__ = f"{bundle}/reportlab/lib/rl_accel.py"

        def fallback() -> None:
            return None

        accel.__all__ = ["fallback"]
        accel._c_funcs = {}
        accel._py_funcs = {"fallback": fallback}
        accel.fallback = fallback
        package = types.ModuleType("reportlab")
        package.__file__ = f"{bundle}/reportlab/__init__.py"
        sys.modules["reportlab"] = package
        sys.modules["reportlab.rl_config"] = config
        sys.modules["reportlab.lib.rl_accel"] = accel
        renderer._audit_reportlab_runtime(bundle, mods, settings)

        accel._c_funcs = {"fallback": fallback}
        with pytest.raises(renderer.RebuttalRenderError, match="pure-Python fallback"):
            renderer._audit_reportlab_runtime(bundle, mods, settings)
        accel._c_funcs = {}
        accel.fallback = lambda: None
        with pytest.raises(renderer.RebuttalRenderError, match="pure-Python functions"):
            renderer._audit_reportlab_runtime(bundle, mods, settings)
        accel.fallback = fallback
        package.__file__ = "/tmp/reportlab.so"
        with pytest.raises(renderer.RebuttalRenderError, match="escaped"):
            renderer._audit_reportlab_runtime(bundle, mods, settings)
    finally:
        for name in tuple(sys.modules):
            if name == "reportlab" or name.startswith("reportlab.") or name in guarded:
                sys.modules.pop(name, None)
        sys.modules.update(relevant)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (SYNTHETIC_SOURCE.replace("```text", "```python", 1), "text fence"),
        (
            SYNTHETIC_SOURCE.replace("```text\n1.", "```text\n\n1.", 1),
            "blank edge lines",
        ),
        (
            SYNTHETIC_SOURCE.replace(
                "SOURCE-COMMENT:r1-1:END", "SOURCE-COMMENT:wrong:END"
            ),
            "mismatched",
        ),
        (
            SYNTHETIC_SOURCE.replace(
                "We agree. The revised description", "### Next comment", 1
            ),
            "no response",
        ),
        (SYNTHETIC_SOURCE.replace("## Reviewer #1", "## General notes"), "Reviewer #N"),
        (SYNTHETIC_SOURCE + "\n# A second title\n", "exactly one"),
        (
            SYNTHETIC_SOURCE.replace("### R1-1", "```\nunsafe\n```\n\n### R1-1", 1),
            "code fence",
        ),
        (
            SYNTHETIC_SOURCE.replace("### R1-1", "| a | b |\n| - | - |\n\n### R1-1", 1),
            "tables",
        ),
        (
            SYNTHETIC_SOURCE.replace(
                "> 2. Please state which conclusions follow from the statistical analysis and which require later biological validation.\n\nWe agree",
                "> Quote line\nlazy continuation\n\nWe agree",
                1,
            ),
            "blank line",
        ),
    ],
)
def test_malformed_markdown_contract_is_rejected(source: str, expected: str) -> None:
    with pytest.raises(renderer.RebuttalRenderError, match=expected):
        renderer._parse_markdown(renderer._canonicalize_markdown(source.encode()))


@pytest.mark.parametrize(
    "inserted",
    [
        "+ silently flattened",
        "    indented code",
        "##### hidden heading",
        "=====",
        "----",
        "- - - -",
        "  ---",
        "  ### Claimed response heading",
        "###\tClaimed response heading",
        "  > Another quote",
        "  ~~~\ncode\n  ~~~",
        "-",
        "1.",
        "0. item",
        "-\titem",
        "1.\titem",
        "  - orphan nested",
        "  * orphan nested",
        "  1. orphan nested",
        "___",
        "***",
        "- - -",
    ],
)
def test_unsupported_blocks_fail_at_top_level_and_in_continuations(
    inserted: str,
) -> None:
    top_level = SYNTHETIC_SOURCE.replace("### R1-1", f"{inserted}\n\n### R1-1", 1)
    with pytest.raises(renderer.RebuttalRenderError):
        renderer._parse_markdown(
            renderer._canonicalize_markdown(top_level.encode()),
        )
    continuation = SYNTHETIC_SOURCE.replace(
        "contains no scientific result and exists only to exercise document packaging.",
        "contains no scientific result\n" + inserted,
        1,
    )
    with pytest.raises(renderer.RebuttalRenderError):
        renderer._parse_markdown(
            renderer._canonicalize_markdown(continuation.encode()),
        )


@pytest.mark.parametrize(
    "claimed_response",
    [
        "----",
        "- - - -",
        "  ---",
        "  ### Claimed response heading",
        "###\tClaimed response heading",
        "  > Another quote",
        "  ~~~\ncode\n  ~~~",
        "-",
        "1.",
        "0. item",
        "-\titem",
        "1.\titem",
    ],
)
def test_unsupported_block_cannot_satisfy_response_gate(
    claimed_response: str,
) -> None:
    source = (
        "# Response to Reviewers\n\n"
        "## Reviewer #1\n\n"
        "> Synthetic reviewer comment.\n\n"
        f"{claimed_response}\n"
    )
    with pytest.raises(renderer.RebuttalRenderError):
        renderer._parse_markdown(renderer._canonicalize_markdown(source.encode()))


@pytest.mark.parametrize(
    "payload",
    [
        "**broken",
        "`broken",
        "**bold *nested* tail**",
        "**see [paper](https://example.org)**",
        "*italic is not pinned*",
        "[paper](https://example.org/a_(b))",
        "[paper](https://example.org/missing",
        "_Response_",
        "__Response__",
        "[Response][ref]",
        r"\cite{paper}",
        r"\ref{figure}",
        r"\emph{text}",
        r"$\binom{500}{2}$",
        "$p_i$",
        "$10^{-3}$",
        r"$\epsilon$",
    ],
)
def test_ambiguous_inline_and_unsupported_tex_fail_closed(payload: str) -> None:
    source = SYNTHETIC_SOURCE.replace(
        "We agree. The revised description",
        f"{payload}\n\nWe agree. The revised description",
        1,
    )
    with pytest.raises(renderer.RebuttalRenderError):
        renderer._parse_markdown(renderer._canonicalize_markdown(source.encode()))


def test_reference_definition_and_lazy_list_continuation_fail_closed() -> None:
    reference = SYNTHETIC_SOURCE.replace(
        "We agree. The revised description",
        "[ref]: https://example.org/paper\n\nWe agree. The revised description",
        1,
    )
    with pytest.raises(renderer.RebuttalRenderError, match="reference-link"):
        renderer._parse_markdown(renderer._canonicalize_markdown(reference.encode()))
    multiline_reference = SYNTHETIC_SOURCE.replace(
        "We agree. The revised description",
        "See [paper].\n\n[paper]:\n  https://example.org/paper\n\n"
        "We agree. The revised description",
        1,
    )
    with pytest.raises(renderer.RebuttalRenderError, match="reference-link"):
        renderer._parse_markdown(
            renderer._canonicalize_markdown(multiline_reference.encode()),
        )
    reference_image = SYNTHETIC_SOURCE.replace(
        "We agree. The revised description",
        "![paper]\n\nWe agree. The revised description",
        1,
    )
    with pytest.raises(renderer.RebuttalRenderError, match="reference image"):
        renderer._parse_markdown(
            renderer._canonicalize_markdown(reference_image.encode()),
        )
    lazy = SYNTHETIC_SOURCE.replace(
        "- The observed count is recorded directly.",
        "- The observed count is recorded directly.\nlazy continuation",
        1,
    )
    with pytest.raises(renderer.RebuttalRenderError, match="blank line"):
        renderer._parse_markdown(renderer._canonicalize_markdown(lazy.encode()))


def test_supported_inline_grammar_has_one_visible_transform() -> None:
    text = (
        r"**Strong** `code` [paper](https://example.org/paper) "
        r"$\rho$ $\tau$ $q < 0.01$ costs \$5"
    )
    assert renderer._plain_visible(text) == (
        "Strong code paper (https://example.org/paper) ρ τ q < 0.01 costs $5"
    )
    markup = renderer._safe_inline(text)
    assert "DialectBold" in markup
    assert "<b>" not in markup and "<i>" not in markup
    assert "ρ" in markup and "τ" in markup


@pytest.mark.parametrize(
    "title",
    [
        "Response **Title**",
        "Response [Title](https://example.org)",
        "Response $q$",
        r"Response \$500",
        "Response _to_ Reviewers",
        "Response __to__ Reviewers",
        "Response [revised]",
    ],
)
def test_literal_h1_rejects_markdown_delimiters(title: str) -> None:
    source = SYNTHETIC_SOURCE.replace(
        "# Response to Reviewers",
        f"# {title}",
        1,
    )
    config = {**CONFIG, "response_title": title}
    audit = renderer._parse_markdown(renderer._canonicalize_markdown(source.encode()))
    with pytest.raises(renderer.RebuttalRenderError, match="literal text"):
        renderer._validate_title_binding(audit, config)


def test_hard_break_and_closing_heading_hashes_fail_closed() -> None:
    hard_break = SYNTHETIC_SOURCE.replace(
        "careful assessment. This fixture",
        "careful assessment.  \nThis fixture",
        1,
    )
    with pytest.raises(renderer.RebuttalRenderError, match="hard line breaks"):
        renderer._canonicalize_markdown(hard_break.encode())
    closing = SYNTHETIC_SOURCE.replace(
        "### R1-1 - Clarify the model description",
        "### R1-1 - Clarify the model description ##",
        1,
    )
    with pytest.raises(renderer.RebuttalRenderError, match="closing heading hashes"):
        renderer._parse_markdown(renderer._canonicalize_markdown(closing.encode()))


def test_json_duplicate_keys_and_unknown_fields_fail_closed() -> None:
    duplicate = b'{"schema":"x","schema":"y"}\n'
    with pytest.raises(renderer.RebuttalRenderError, match="duplicates JSON key"):
        renderer._json_without_duplicates(duplicate, context="test")

    unknown_template = {**TEMPLATE, "extra": True}
    with pytest.raises(renderer.RebuttalRenderError, match="exactly keys"):
        renderer._normalize_template(unknown_template)

    invalid_config = {**CONFIG, "source_date_epoch": 1}
    with pytest.raises(renderer.RebuttalRenderError, match="fixed invariant epoch"):
        renderer._normalize_config(invalid_config)
    float_epoch = {**CONFIG, "source_date_epoch": 946684800.0}
    with pytest.raises(renderer.RebuttalRenderError, match="fixed invariant epoch"):
        renderer._normalize_config(float_epoch)


def test_structural_and_manifest_bounds_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = renderer._canonicalize_markdown(SYNTHETIC_SOURCE.encode())
    baseline = renderer._parse_markdown(canonical)
    source_lines = len(canonical.decode("utf-8")[:-1].split("\n"))
    list_items = sum(
        len(block.text.split("\n")) for block in baseline.blocks if block.kind == "list"
    )

    original_limits = {
        "MAX_SOURCE_LINES": renderer.MAX_SOURCE_LINES,
        "MAX_BLOCKS": renderer.MAX_BLOCKS,
        "MAX_QUOTES": renderer.MAX_QUOTES,
        "MAX_RESPONSES": renderer.MAX_RESPONSES,
        "MAX_REVIEWERS": renderer.MAX_REVIEWERS,
        "MAX_LIST_ITEMS": renderer.MAX_LIST_ITEMS,
        "MAX_LEDGER_ENTRIES": renderer.MAX_LEDGER_ENTRIES,
    }
    exact_limits = {
        "MAX_SOURCE_LINES": source_lines,
        "MAX_BLOCKS": len(baseline.blocks),
        "MAX_QUOTES": len(baseline.quotes),
        "MAX_RESPONSES": baseline.response_count,
        "MAX_REVIEWERS": baseline.reviewer_count,
        "MAX_LIST_ITEMS": list_items,
    }
    for name, limit in exact_limits.items():
        for reset_name, reset_value in original_limits.items():
            monkeypatch.setattr(renderer, reset_name, reset_value)
        monkeypatch.setattr(renderer, name, limit)
        renderer._parse_markdown(canonical)
        monkeypatch.setattr(renderer, name, limit - 1)
        with pytest.raises(renderer.RebuttalRenderError, match="limit"):
            renderer._parse_markdown(canonical)

    for reset_name, reset_value in original_limits.items():
        monkeypatch.setattr(renderer, reset_name, reset_value)
    ledger = renderer._visible_text_ledger(baseline, CONFIG)
    monkeypatch.setattr(renderer, "MAX_LEDGER_ENTRIES", len(ledger))
    assert len(renderer._visible_text_ledger(baseline, CONFIG)) == len(ledger)
    monkeypatch.setattr(renderer, "MAX_LEDGER_ENTRIES", len(ledger) - 1)
    with pytest.raises(renderer.RebuttalRenderError, match="ledger.*limit"):
        renderer._visible_text_ledger(baseline, CONFIG)

    monkeypatch.setattr(renderer, "MAX_MANIFEST_BYTES", 8)
    assert renderer._require_manifest_size(b"12345678") == b"12345678"
    with pytest.raises(renderer.RebuttalRenderError, match="manifest.*limit"):
        renderer._require_manifest_size(b"123456789")


@pytest.mark.parametrize(
    "bad",
    ["\ud800", "\u2028", "\u2029", "\u202e", "\u2066", "\u200b", "\x7f", "\x85"],
)
def test_config_and_source_reject_unsafe_unicode(bad: str) -> None:
    with pytest.raises(
        renderer.RebuttalRenderError, match="Unicode control|strict UTF-8"
    ):
        renderer._normalize_config({**CONFIG, "manuscript_title": f"Title{bad}"})
    with pytest.raises(
        renderer.RebuttalRenderError, match="Unicode control|strict UTF-8"
    ):
        renderer._canonicalize_markdown(
            SYNTHETIC_SOURCE.replace(
                "careful assessment", f"careful{bad} assessment"
            ).encode(
                "utf-8",
                errors="surrogatepass",
            ),
        )


def test_scientific_unicode_remains_codepoint_distinct() -> None:
    text = "χ² R² x₂ non‑breaking µ μ ① ﬁ"
    canonical = renderer._canonicalize_markdown(
        SYNTHETIC_SOURCE.replace(
            "We agree. The revised description",
            f"{text}.\n\nWe agree. The revised description",
            1,
        ).encode(),
    )
    audit = renderer._parse_markdown(canonical)
    ledger = renderer._visible_text_ledger(audit, CONFIG)
    joined = " ".join(record["text"] for record in ledger)
    assert text in joined
    assert "µ" in joined and "μ" in joined
    assert renderer._normalized_text("① ﬁ") == "① ﬁ"


def test_wrapped_list_items_are_joined_explicitly() -> None:
    source = SYNTHETIC_SOURCE.replace(
        "- The observed count is recorded directly.",
        "- The observed count is recorded directly and\n"
        "  remains attached to this wrapped list item.",
        1,
    )
    audit = renderer._parse_markdown(renderer._canonicalize_markdown(source.encode()))
    matching = [block for block in audit.blocks if block.kind == "list"]
    assert matching
    assert matching[0].text.split("\n")[0] == (
        "The observed count is recorded directly and remains attached to this "
        "wrapped list item."
    )


def test_exact_local_process_budget_rejects_extra_or_missing_children() -> None:
    budget = renderer._LocalProcessBudget()
    for _ in range(5):
        budget.consume()
    budget.assert_complete()
    with pytest.raises(renderer.RebuttalRenderError, match="exceeds"):
        budget.consume()

    incomplete = renderer._LocalProcessBudget()
    incomplete.consume()
    with pytest.raises(renderer.RebuttalRenderError, match="expected exactly 5"):
        incomplete.assert_complete()


def test_visible_ledger_is_exact_ordered_and_multiplicity_aware() -> None:
    ledger = (
        {"block_id": "a", "text": "RESPONSE"},
        {"block_id": "b", "text": "same"},
        {"block_id": "c", "text": "same"},
        {"block_id": "d", "text": "REVIEWER COMMENT"},
    )
    renderer._verify_visible_text_ledger(
        "RESPONSE same same REVIEWER COMMENT",
        ledger,
    )
    for altered in (
        "RESPONSE same REVIEWER COMMENT",
        "same RESPONSE same REVIEWER COMMENT",
        "RESPONSE same same REVIEWER COMMENT injected",
    ):
        with pytest.raises(renderer.RebuttalRenderError, match="complete ordered"):
            renderer._verify_visible_text_ledger(altered, ledger)


def test_footer_stripping_removes_only_exact_page_decorations() -> None:
    config = {**CONFIG, "manuscript_id": "SOURCE-WRAPPER"}
    extracted = (
        "SOURCE-WRAPPER\nRESPONSE\nSOURCE-WRAPPER Page 1\f"
        "REVIEWER COMMENT\nSOURCE-WRAPPER Page 2\f"
    )
    cleaned = renderer._strip_pdf_decorations(extracted, config, page_count=2)
    assert renderer._normalized_text(cleaned) == (
        "SOURCE-WRAPPER RESPONSE REVIEWER COMMENT"
    )


def test_footer_collision_boundary_is_exact() -> None:
    assert renderer._footer_width_fits(100.0, 50.0, 174.0)
    assert not renderer._footer_width_fits(100.0, 50.0, 173.999)


def test_snapshot_is_unlinked_read_only_and_binds_exact_bytes() -> None:
    snapshot = renderer._snapshot_bytes(b"anchored-bytes", context="synthetic")
    try:
        entry = os.fstat(snapshot.descriptor)
        assert entry.st_nlink == 0
        assert stat.S_IMODE(entry.st_mode) == 0o400
        assert (
            renderer._snapshot_payload(
                {"source": snapshot},
                "source",
                maximum=100,
            )
            == b"anchored-bytes"
        )
        with pytest.raises(OSError):
            os.write(snapshot.descriptor, b"x")
    finally:
        snapshot.close()
        snapshot.close()


def test_snapshot_preserves_consumed_bytes_after_original_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "input.txt"
    path.write_bytes(b"anchored")
    entry = path.stat()
    descriptor = os.open(path, os.O_RDONLY)
    pin = types.SimpleNamespace(
        path=path,
        descriptor=descriptor,
        device=entry.st_dev,
        inode=entry.st_ino,
        size=entry.st_size,
        mtime_ns=entry.st_mtime_ns,
        sha256=hashlib.sha256(b"anchored").hexdigest(),
    )
    snapshot = renderer._snapshot_bytes(
        renderer._read_anchored_pin(pin, maximum=100, context="input"),
        context="input",
    )
    try:
        path.write_bytes(b"mutated!")
        assert (
            renderer._snapshot_payload(
                {"builder": snapshot},
                "builder",
                maximum=100,
            )
            == b"anchored"
        )
        with pytest.raises(renderer.RebuttalRenderError, match="changed"):
            renderer._revalidate_pinned_file(pin, context="input")
    finally:
        snapshot.close()
        os.close(descriptor)


def test_pin_cleanup_attempts_every_close_and_preserves_primary_error() -> None:
    events: list[str] = []

    class Pin:
        def __init__(self, name: str, *, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def close(self) -> None:
            events.append(self.name)
            if self.fail:
                message = f"close-{self.name}"
                raise OSError(message)

    primary = ValueError("primary-failure")
    with pytest.raises(renderer.RebuttalRenderError, match=r"primary-failure.*close-b"):
        renderer._close_pins(
            {"a": Pin("a"), "b": Pin("b", fail=True), "c": Pin("c")},
            primary_error=primary,
        )
    assert events == ["a", "b", "c"]


def test_write_member_preserves_primary_and_close_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_descriptor = os.open(tmp_path, os.O_RDONLY)
    original_close = renderer.os.close

    def fail_write(*_args: object, **_kwargs: object) -> None:
        message = "primary-write-failure"
        raise ValueError(message)

    def close_then_fail(descriptor: int) -> None:
        original_close(descriptor)
        message = "member-close-failure"
        raise OSError(message)

    monkeypatch.setattr(renderer, "_write_all", fail_write)
    monkeypatch.setattr(renderer.os, "close", close_then_fail)
    try:
        with pytest.raises(
            renderer.RebuttalRenderError,
            match=r"primary-write-failure.*member-close-failure",
        ):
            renderer._write_member(root_descriptor, "synthetic.txt", b"payload")
    finally:
        original_close(root_descriptor)


def test_reportlab_inventory_preserves_primary_and_close_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "reportlab"
    package.mkdir()
    (package / "__init__.py").write_text("version = 1\n", encoding="utf-8")
    original_close = renderer.os.close

    def fail_read(*_args: object, **_kwargs: object) -> bytes:
        message = "primary-inventory-failure"
        raise ValueError(message)

    def close_then_fail(descriptor: int) -> None:
        original_close(descriptor)
        message = "inventory-close-failure"
        raise OSError(message)

    monkeypatch.setattr(renderer, "_read_fd", fail_read)
    monkeypatch.setattr(renderer.os, "close", close_then_fail)
    with pytest.raises(
        renderer.RebuttalRenderError,
        match=r"primary-inventory-failure.*inventory-close-failure",
    ):
        renderer.reportlab_dependency_digest(package)


def test_published_read_preserves_primary_and_close_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "published-cleanup"
    root.mkdir()
    for member in (*renderer.MEMBER_ORDER, renderer.MANIFEST_MEMBER):
        path = root / member
        path.write_bytes(b"synthetic")
        path.chmod(0o400)
    root.chmod(0o500)
    original_close = renderer.os.close

    def fail_read(*_args: object, **_kwargs: object) -> bytes:
        message = "primary-published-read-failure"
        raise ValueError(message)

    def close_then_fail(descriptor: int) -> None:
        original_close(descriptor)
        message = "published-close-failure"
        raise OSError(message)

    monkeypatch.setattr(renderer, "_read_fd", fail_read)
    monkeypatch.setattr(renderer.os, "close", close_then_fail)
    try:
        with pytest.raises(
            renderer.RebuttalRenderError,
            match=r"primary-published-read-failure.*published-close-failure",
        ):
            renderer._read_published(
                root,
                expected_manifest_sha256="0" * 64,
            )
    finally:
        _make_writable(root)


def _fake_machine_runner_source(*, side_effect: bool = False) -> str:
    prefix = (
        "import builtins\nbuiltins._dialect_fake_helper_executed = True\n"
        if side_effect
        else ""
    )
    return (
        prefix
        + """
DARWIN_SPAWN_FLAGS = 1
REQUIRED_CS_FLAGS = 1
REJECTED_CS_FLAGS = 2
VM_PROT_EXECUTE = 4
def _pin_file(*args, **kwargs):
    return None
def _run_bounded(*args, **kwargs):
    return None
"""
    )


def test_machine_helper_wrong_hash_never_executes_top_level(tmp_path: Path) -> None:
    helper = tmp_path / "helper.py"
    sentinel = tmp_path / "executed.sentinel"
    helper.write_text(
        "from pathlib import Path\n"
        f"Path({str(sentinel)!r}).write_text('executed', encoding='ascii')\n"
        + _fake_machine_runner_source(side_effect=True),
        encoding="utf-8",
    )
    if hasattr(builtins, "_dialect_fake_helper_executed"):
        del builtins._dialect_fake_helper_executed
    with pytest.raises(renderer.RebuttalRenderError, match="caller SHA-256"):
        renderer._load_machine_authority(helper, expected_sha256="0" * 64)
    assert not hasattr(builtins, "_dialect_fake_helper_executed")
    assert not sentinel.exists()


def test_machine_helper_ignores_sys_modules_shadow_and_close_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = _sha(MACHINE_RUNNER)
    module_name = f"_dialect_pinned_machine_runner_{digest}"
    shadow = types.ModuleType(module_name)
    monkeypatch.setitem(sys.modules, module_name, shadow)
    authority = renderer._load_machine_authority(
        MACHINE_RUNNER,
        expected_sha256=digest,
    )
    assert authority.module is not shadow
    assert sys.modules[module_name] is shadow
    authority.close()
    authority.close()


def test_machine_helper_named_drift_fails_revalidation(tmp_path: Path) -> None:
    helper = tmp_path / "helper.py"
    helper.write_text(_fake_machine_runner_source(), encoding="utf-8")
    authority = renderer._load_machine_authority(
        helper,
        expected_sha256=_sha(helper),
    )
    try:
        module_name = f"_dialect_pinned_machine_runner_{_sha(helper)}"
        assert module_name not in sys.modules
        assert callable(authority.module._pin_file)
        helper.write_text(_fake_machine_runner_source() + "# drift\n", encoding="utf-8")
        with pytest.raises(renderer.RebuttalRenderError, match="changed"):
            renderer._revalidate_pinned_file(authority, context="machine runner")
    finally:
        authority.close()


def test_zero_progress_write_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(renderer.os, "write", lambda _descriptor, _raw: 0)
    with pytest.raises(renderer.RebuttalRenderError, match="make progress"):
        renderer._write_all(99, b"payload", context="test")


def test_reportlab_dependency_digest_is_deterministic_and_excludes_bytecode() -> None:
    _require_real_dependencies()
    first = renderer.reportlab_dependency_digest(REPORTLAB_ROOT)
    second = renderer.reportlab_dependency_digest(REPORTLAB_ROOT)

    assert first == second
    assert first["schema"] == "dialect-reportlab-pure-python-tree-anchor-v1"
    assert first["file_count"] > 100
    assert first["directory_count"] > 10
    assert first["entry_count"] > first["file_count"]
    assert first["total_bytes"] > 1_000_000
    assert len(str(first["tree_sha256"])) == 64
    assert first["excluded"] == ["__pycache__", "*.pyc", "*.so", "*.dylib"]


def _pdffonts(rows: list[str]) -> bytes:
    return (
        renderer.PDF_FONTS_HEADER
        + "\n"
        + renderer.PDF_FONTS_SEPARATOR
        + "\n"
        + "\n".join(rows)
        + "\n"
    ).encode("ascii")


def test_pdffonts_parser_consumes_every_row_and_rejects_type3() -> None:
    safe = "AAAAAA+ArialMT TrueType WinAnsi yes yes yes 7 0"
    parsed = renderer._parse_pdffonts_output(_pdffonts([safe]))
    assert parsed[0]["name"] == "AAAAAA+ArialMT"
    with pytest.raises(renderer.RebuttalRenderError, match="unrecognized font row"):
        renderer._parse_pdffonts_output(_pdffonts([safe, "unknown row"]))
    unsafe = "BBBBBB+Unsafe Type 3 Custom yes yes yes 8 0"
    with pytest.raises(renderer.RebuttalRenderError, match="Type 3"):
        renderer._parse_pdffonts_output(_pdffonts([safe, unsafe]))
    with pytest.raises(renderer.RebuttalRenderError, match="unexpected header"):
        renderer._parse_pdffonts_output(b"bad\nheader\nrow\n")


def test_ttf_postscript_names_bind_distinct_font_roles() -> None:
    _require_real_dependencies()
    regular = REGULAR_FONT.read_bytes()
    bold = BOLD_FONT.read_bytes()
    names = renderer._validate_font_roles(regular, bold)
    assert len(set(names)) == 2
    rows = renderer._parse_pdffonts_output(
        _pdffonts(
            [
                f"AAAAAA+{names[0]} TrueType WinAnsi yes yes yes 7 0",
                f"BBBBBB+{names[1]} TrueType WinAnsi yes yes yes 8 0",
            ],
        ),
    )
    renderer._validate_pdf_font_roles(rows, names)
    with pytest.raises(renderer.RebuttalRenderError, match="pinned font roles"):
        renderer._validate_pdf_font_roles(rows, (names[0], "WrongFace"))
    with pytest.raises(renderer.RebuttalRenderError, match="PostScript names"):
        renderer._validate_font_roles(regular, regular + b"\0")


def test_reportlab_empty_directory_flood_is_bounded(tmp_path: Path) -> None:
    package = tmp_path / "reportlab"
    package.mkdir()
    (package / "__init__.py").write_text("version = 1\n", encoding="utf-8")
    for index in range(renderer.MAX_REPORTLAB_DIRECTORIES + 1):
        (package / f"empty-{index:03d}").mkdir()
    with pytest.raises(renderer.RebuttalRenderError, match="directory bound"):
        renderer.reportlab_dependency_digest(package)


def test_reportlab_excluded_cache_and_empty_directory_churn_is_noncanonical(
    tmp_path: Path,
) -> None:
    package = tmp_path / "reportlab"
    package.mkdir()
    (package / "__init__.py").write_text("version = 1\n", encoding="utf-8")
    (package / "module.py").write_text("value = 1\n", encoding="utf-8")
    before = renderer.reportlab_dependency_digest(package)
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "module.pyc").write_bytes(b"ignored")
    (package / "native.so").write_bytes(b"ignored")
    (package / "native.dylib").write_bytes(b"ignored")
    (package / "empty").mkdir()
    after = renderer.reportlab_dependency_digest(package)
    assert after == before


@pytest.mark.parametrize(
    "override",
    ["local_rl_mods.py", "local_rl_settings.py", "local_rl_mods", "local_rl_settings"],
)
def test_reportlab_local_customization_modules_are_rejected(
    tmp_path: Path,
    override: str,
) -> None:
    package = tmp_path / "reportlab"
    package.mkdir()
    (package / "__init__.py").write_text("version = 1\n", encoding="utf-8")
    path = package / override
    if "." in override:
        path.write_text("side_effect = True\n", encoding="utf-8")
    else:
        path.mkdir()
        (path / "__init__.py").write_text("side_effect = True\n", encoding="utf-8")
    with pytest.raises(renderer.RebuttalRenderError, match="customization module"):
        renderer.reportlab_dependency_digest(package)


def test_bounded_directory_names_stops_after_limit(tmp_path: Path) -> None:
    for index in range(8):
        (tmp_path / f"member-{index}").write_text("x", encoding="ascii")
    with pytest.raises(renderer.RebuttalRenderError, match="5-entry bound"):
        renderer._bounded_directory_names(
            tmp_path, maximum=5, context="synthetic flood"
        )


def test_reportlab_dependency_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    package = tmp_path / "reportlab"
    package.mkdir()
    (package / "__init__.py").write_text("version = 1\n", encoding="utf-8")
    target = package / "module.py"
    target.write_text("value = 1\n", encoding="utf-8")
    alias = package / "alias.py"
    os.link(target, alias)
    with pytest.raises(renderer.RebuttalRenderError, match="single-link"):
        renderer.reportlab_dependency_digest(package)

    second = tmp_path / "other" / "reportlab"
    second.parent.mkdir()
    second.symlink_to(package, target_is_directory=True)
    with pytest.raises(renderer.RebuttalRenderError, match="canonical directory"):
        renderer.reportlab_dependency_digest(second)


def test_reportlab_stat_to_open_fifo_swap_fails_without_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "reportlab"
    package.mkdir()
    (package / "__init__.py").write_text("version = 1\n", encoding="utf-8")
    member = package / "module.py"
    member.write_text("value = 1\n", encoding="utf-8")
    original_open = renderer.os.open
    swapped = False

    def swap_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if path == "module.py" and not swapped:
            swapped = True
            member.unlink()
            os.mkfifo(member)
            assert flags & getattr(os, "O_NONBLOCK", 0)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(renderer.os, "open", swap_open)
    with pytest.raises(renderer.RebuttalRenderError, match="changed while pinned"):
        renderer.reportlab_dependency_digest(package)
    assert swapped


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_source_special_files_fail_before_render(tmp_path: Path, kind: str) -> None:
    _require_real_dependencies()
    real_source, template, config = _write_inputs(tmp_path)
    special = tmp_path / f"special-{kind}.md"
    if kind == "symlink":
        special.symlink_to(real_source)
    elif kind == "hardlink":
        os.link(real_source, special)
    else:
        os.mkfifo(special)
    kwargs = _build_kwargs(real_source, template, config)
    kwargs["expected_source_sha256"] = _sha(real_source)
    with pytest.raises(
        renderer.RebuttalRenderError, match="canonical single-link regular file"
    ):
        renderer.build_rebuttal_pdf(
            special,
            template,
            config,
            tmp_path / "published",
            **kwargs,
        )
    assert not (tmp_path / "published").exists()


def test_wrong_source_and_tool_anchors_fail_before_publication(tmp_path: Path) -> None:
    source, template, config = _write_inputs(tmp_path)
    kwargs = _build_kwargs(source, template, config)
    kwargs["expected_source_sha256"] = "0" * 64
    with pytest.raises(renderer.RebuttalRenderError, match="caller SHA-256"):
        renderer.build_rebuttal_pdf(
            source,
            template,
            config,
            tmp_path / "source-mismatch",
            **kwargs,
        )
    assert not (tmp_path / "source-mismatch").exists()

    kwargs = _build_kwargs(source, template, config)
    kwargs["expected_tool_sha256"] = {
        **kwargs["expected_tool_sha256"],
        "pdfinfo": "0" * 64,
    }
    with pytest.raises(renderer.RebuttalRenderError, match="pdfinfo executable"):
        renderer.build_rebuttal_pdf(
            source,
            template,
            config,
            tmp_path / "tool-mismatch",
            **kwargs,
        )
    assert not (tmp_path / "tool-mismatch").exists()


def test_unresolved_gate_fails_before_any_destination_is_reserved(
    tmp_path: Path,
) -> None:
    source_text = SYNTHETIC_SOURCE.replace(
        "We agree. The revised description",
        "**[CAL+COAUTH: insert the final result.]**\n\nWe agree. The revised description",
        1,
    )
    source, template, config = _write_inputs(tmp_path, source_text=source_text)
    kwargs = _build_kwargs(source, template, config)
    destination = tmp_path / "published"
    with pytest.raises(renderer.RebuttalRenderError, match="DIALECT gate marker"):
        renderer.build_rebuttal_pdf(source, template, config, destination, **kwargs)
    assert not destination.exists()
    assert not (tmp_path / ".published.private-candidate").exists()


@REQUIRES_NATIVE_DARWIN
def test_existing_destination_is_never_replaced(tmp_path: Path) -> None:
    source, template, config = _write_inputs(tmp_path)
    kwargs = _build_kwargs(source, template, config)
    destination = tmp_path / "published"
    destination.mkdir()
    sentinel = destination / "sentinel"
    sentinel.write_text("original\n", encoding="utf-8")
    with pytest.raises(
        renderer.RebuttalRenderError, match="destination already exists"
    ):
        renderer.build_rebuttal_pdf(source, template, config, destination, **kwargs)
    assert sentinel.read_text(encoding="utf-8") == "original\n"


def test_publication_rejects_group_or_world_writable_parent(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o700)
    unsafe.chmod(0o777)
    production, canonical = _fake_production()
    try:
        with pytest.raises(renderer.RebuttalRenderError, match="group/world writable"):
            renderer._publish(unsafe / "published", production, canonical)
        assert not (unsafe / ".published.private-candidate").exists()
    finally:
        unsafe.chmod(0o700)


def test_publication_parent_swap_before_open_writes_no_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "authority"
    parent.mkdir()
    moved = tmp_path / "authority-original"
    production, canonical = _fake_production()
    original_open = renderer.os.open
    swapped = False

    def swap_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if not swapped and Path(path) == parent:
            swapped = True
            parent.rename(moved)
            parent.mkdir()
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(renderer.os, "open", swap_open)
    with pytest.raises(renderer.RebuttalRenderError, match="parent changed"):
        renderer._publish(parent / "published", production, canonical)
    assert swapped
    assert not (parent / ".published.private-candidate").exists()
    assert not (moved / ".published.private-candidate").exists()


def test_candidate_member_fifo_swap_fails_without_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production, canonical = _fake_production()
    original_open = renderer.os.open
    swapped = False

    def swap_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        access = flags & os.O_ACCMODE
        directory = kwargs.get("dir_fd")
        if path == renderer.SOURCE_MEMBER and access == os.O_RDONLY and not swapped:
            swapped = True
            assert isinstance(directory, int)
            os.unlink(path, dir_fd=directory)
            os.mkfifo(candidate / str(path))
            assert flags & getattr(os, "O_NONBLOCK", 0)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(renderer.os, "open", swap_open)
    candidate = tmp_path / ".published.private-candidate"
    try:
        with pytest.raises(renderer.RebuttalRenderError, match="changed type or mode"):
            renderer._publish(tmp_path / "published", production, canonical)
        assert swapped
        assert candidate.is_dir()
    finally:
        _make_writable(candidate)


def test_published_member_fifo_swap_fails_without_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "published"
    root.mkdir()
    member_raw = {
        renderer.SOURCE_MEMBER: b"# source\n",
        renderer.TEMPLATE_MEMBER: b"{}\n",
        renderer.CONFIG_MEMBER: b"{}\n",
        renderer.PDF_MEMBER: b"%PDF-1.4\n%%EOF\n",
    }
    manifest = {
        "schema": renderer.SCHEMA,
        "members": [
            {
                "member": member,
                "sha256": hashlib.sha256(member_raw[member]).hexdigest(),
                "size": len(member_raw[member]),
            }
            for member in renderer.MEMBER_ORDER
        ],
    }
    manifest_raw = renderer._canonical_json(manifest)
    for member, raw in {**member_raw, renderer.MANIFEST_MEMBER: manifest_raw}.items():
        path = root / member
        path.write_bytes(raw)
        path.chmod(0o400)
    root.chmod(0o500)
    original_open = renderer.os.open
    swapped = False

    def swap_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        directory = kwargs.get("dir_fd")
        if (
            path == renderer.SOURCE_MEMBER
            and isinstance(directory, int)
            and not swapped
        ):
            swapped = True
            os.fchmod(directory, 0o700)
            os.unlink(path, dir_fd=directory)
            os.mkfifo(root / str(path), 0o400)
            os.fchmod(directory, 0o500)
            assert flags & getattr(os, "O_NONBLOCK", 0)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(renderer.os, "open", swap_open)
    try:
        with pytest.raises(renderer.RebuttalRenderError, match="not sealed"):
            renderer._read_published(
                root,
                expected_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
            )
        assert swapped
    finally:
        _make_writable(root)


@REQUIRES_NATIVE_DARWIN
def test_post_rename_failure_reports_published_state_accurately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    pdf = b"%PDF-1.4\n%%EOF\n"
    manifest_raw = b"{}\n"
    production = renderer._Production(
        manifest={},
        manifest_raw=manifest_raw,
        pdf_raw=pdf,
        page_count=1,
        quote_count=1,
        response_count=1,
    )
    canonical = {
        renderer.SOURCE_MEMBER: b"# source\n",
        renderer.TEMPLATE_MEMBER: b"{}\n",
        renderer.CONFIG_MEMBER: b"{}\n",
    }
    original_rename = renderer._rename_no_replace
    renamed = False

    def wrapped_rename(parent: int, source: str, target: str) -> None:
        nonlocal renamed
        original_rename(parent, source, target)
        renamed = True

    original_fsync = renderer.os.fsync

    def fail_post_rename(descriptor: int) -> None:
        if renamed:
            message = "synthetic-parent-fsync-failure"
            raise OSError(message)
        original_fsync(descriptor)

    monkeypatch.setattr(renderer, "_rename_no_replace", wrapped_rename)
    monkeypatch.setattr(renderer.os, "fsync", fail_post_rename)
    try:
        with pytest.raises(
            renderer.RebuttalRenderError,
            match="rename-completed-post-publication-failure",
        ):
            renderer._publish(destination, production, canonical)
        assert destination.is_dir()
        assert not (tmp_path / ".published.private-candidate").exists()
    finally:
        _make_writable(destination)


@REQUIRES_NATIVE_DARWIN
def test_build_readback_failure_preserves_materialized_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, template, config = _write_inputs(tmp_path)
    kwargs = _build_kwargs(source, template, config)
    published = tmp_path / "published-readback-failure"
    original_read = renderer._read_published
    calls = 0

    def fail_first_read(
        root: Path,
        *,
        expected_manifest_sha256: str,
    ) -> tuple[bytes, dict[str, object], tuple[int, int]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            message = "synthetic-terminal-readback"
            raise renderer.RebuttalRenderError(message)
        return original_read(
            root,
            expected_manifest_sha256=expected_manifest_sha256,
        )

    monkeypatch.setattr(renderer, "_read_published", fail_first_read)
    try:
        with pytest.raises(
            renderer.RebuttalRenderError,
            match=(
                "synthetic-terminal-readback.*"
                "materialized-output-preserved-after-terminal-failure"
            ),
        ):
            renderer.build_rebuttal_pdf(
                source,
                template,
                config,
                published,
                **kwargs,
            )
        assert calls == 1
        assert published.is_dir()
        assert (published / renderer.PDF_MEMBER).is_file()
    finally:
        _make_writable(published)


@REQUIRES_NATIVE_DARWIN
def test_validation_terminal_identity_drift_preserves_original_and_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, template, config = _write_inputs(tmp_path)
    kwargs = _build_kwargs(source, template, config)
    published = tmp_path / "published-terminal-drift"
    replay = tmp_path / "replay-terminal-drift"
    try:
        receipt = renderer.build_rebuttal_pdf(
            source,
            template,
            config,
            published,
            **kwargs,
        )
        original_read = renderer._read_published
        calls = 0

        def drift_terminal_original(
            root: Path,
            *,
            expected_manifest_sha256: str,
        ) -> tuple[bytes, dict[str, object], tuple[int, int]]:
            nonlocal calls
            calls += 1
            raw, parsed, identity = original_read(
                root,
                expected_manifest_sha256=expected_manifest_sha256,
            )
            if calls == 3:
                return raw, parsed, (identity[0], identity[1] + 1)
            return raw, parsed, identity

        monkeypatch.setattr(renderer, "_read_published", drift_terminal_original)
        with pytest.raises(
            renderer.RebuttalRenderError,
            match=(
                "changed before terminal receipt.*"
                "materialized-output-preserved-after-terminal-failure"
            ),
        ):
            renderer.validate_rebuttal_pdf(
                source,
                template,
                config,
                published,
                replay,
                expected_manifest_sha256=receipt.manifest_sha256,
                **kwargs,
            )
        assert calls == 4
        assert published.is_dir()
        assert replay.is_dir()
    finally:
        _make_writable(published)
        _make_writable(replay)


def test_candidate_and_published_inventory_floods_are_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    production = renderer._Production(
        manifest={},
        manifest_raw=b"{}\n",
        pdf_raw=b"%PDF-1.4\n%%EOF\n",
        page_count=1,
        quote_count=1,
        response_count=1,
    )
    canonical = {
        renderer.SOURCE_MEMBER: b"# source\n",
        renderer.TEMPLATE_MEMBER: b"{}\n",
        renderer.CONFIG_MEMBER: b"{}\n",
    }
    original_write = renderer._write_member

    def flooding_write(root_descriptor: int, member: str, raw: bytes) -> None:
        original_write(root_descriptor, member, raw)
        if member == renderer.MANIFEST_MEMBER:
            for index in range(3):
                descriptor = os.open(
                    f"extra-{index}",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=root_descriptor,
                )
                os.close(descriptor)

    monkeypatch.setattr(renderer, "_write_member", flooding_write)
    candidate = tmp_path / ".published.private-candidate"
    try:
        with pytest.raises(renderer.RebuttalRenderError, match="entry bound"):
            renderer._publish(destination, production, canonical)
        assert candidate.is_dir()
        assert not destination.exists()
    finally:
        _make_writable(candidate)

    flooded = tmp_path / "flooded-published"
    flooded.mkdir()
    for index in range(len(renderer.MEMBER_ORDER) + 3):
        (flooded / f"member-{index}").write_text("x", encoding="ascii")
    flooded.chmod(0o500)
    try:
        with pytest.raises(renderer.RebuttalRenderError, match="entry bound"):
            renderer._read_published(
                flooded,
                expected_manifest_sha256="0" * 64,
            )
    finally:
        _make_writable(flooded)


@REQUIRES_NATIVE_DARWIN
def test_full_synthetic_build_and_independent_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, template, config = _write_inputs(tmp_path)
    kwargs = _build_kwargs(source, template, config)
    published = tmp_path / "published"
    replay = tmp_path / "replay"
    extra_descriptors: list[int] = []
    try:
        receipt = renderer.build_rebuttal_pdf(
            source,
            template,
            config,
            published,
            **kwargs,
        )
        assert receipt.replay_root is None
        assert receipt.page_count >= 2
        assert receipt.quote_count == 3
        assert receipt.response_count == 3
        assert Path(receipt.manifest_path) == published / renderer.MANIFEST_MEMBER
        assert Path(receipt.pdf_path) == published / renderer.PDF_MEMBER
        assert _sha(Path(receipt.pdf_path)) == receipt.pdf_sha256
        assert _sha(Path(receipt.manifest_path)) == receipt.manifest_sha256

        assert stat.S_IMODE(published.stat().st_mode) == 0o500
        assert sorted(path.name for path in published.iterdir()) == sorted(
            (*renderer.MEMBER_ORDER, renderer.MANIFEST_MEMBER),
        )
        assert all(
            stat.S_IMODE(path.stat().st_mode) == 0o400 for path in published.iterdir()
        )
        assert all(path.stat().st_nlink == 1 for path in published.iterdir())

        pdf = (published / renderer.PDF_MEMBER).read_bytes()
        assert pdf.startswith(b"%PDF-")
        assert pdf.endswith(b"%%EOF\n")
        assert pdf.count(b"%%EOF") == 1

        manifest = json.loads(
            (published / renderer.MANIFEST_MEMBER).read_text(encoding="ascii")
        )
        assert manifest["schema"] == renderer.SCHEMA
        assert manifest["contract"] == renderer.CONTRACT
        assert manifest["document"]["sha256"] == receipt.pdf_sha256
        assert manifest["document"]["reviewer_quote_text_blocks_verified"] == 3
        assert manifest["document"]["encrypted"] is False
        assert manifest["document"]["page_size"] == "letter"
        assert manifest["document"]["font_count"] >= 2
        assert manifest["document"]["visible_text_blocks_preserved"] == len(
            manifest["markdown"]["visible_text_ledger"],
        )
        assert manifest["determinism"]["independent_render_passes"] == 2
        assert manifest["determinism"]["byte_identical"] is True
        assert manifest["determinism"]["reportlab_invariant_epoch"] == 946684800
        assert manifest["determinism"]["direct_tool_invocation_budget"] == 5
        assert manifest["determinism"]["direct_tool_invocations_consumed"] == 5
        assert manifest["determinism"]["descendant_process_count"] == "not_attested"
        assert manifest["execution"]["content_derivation_authority"] == (
            "unlinked-read-only-snapshots-of-caller-sha-verified-bytes"
        )
        assert (
            manifest["execution"]["python_stdlib_and_dylib_closure"] == "not_attested"
        )
        assert (
            manifest["execution"]["ambient_same_uid_filesystem_containment"]
            == "not_provided"
        )
        assert manifest["execution"]["pillow_dependency"] == {
            "status": "not-loaded-fail-closed-no-image-stub",
            "stub_anchor": "builder-sha256",
            "native_imaging_loaded": False,
        }
        assert manifest["execution"]["file_descriptor_budget"] == {
            "required_headroom": 40,
            "status": "pass",
        }
        assert "/Users/" not in json.dumps(manifest, sort_keys=True)
        assert manifest["claims"]["human_visual_approval"] == "required-separately"
        assert manifest["integration"] == {
            "four_role_derivation_adapter_protocol": "not-implemented",
            "schema_compatibility_with_derivation_closure": False,
            "rebuttal_role_gate": "not-cleared-by-this-renderer",
            "promotion_gate": "not-cleared-by-this-renderer",
            "future_seam": (
                "reviewed native thin-arm64 adapter streams this renderer's "
                "deterministic PDF bytes into the fixed four-role derivation "
                "closure; downstream promotion separately binds derivation, "
                "machine replay, and authenticated page-complete visual approval"
            ),
        }

        # Equality-critical bytes must not encode unrelated live FD/RLIMIT state.
        extra_descriptors.extend(os.open("/dev/null", os.O_RDONLY) for _ in range(3))
        soft_limit, hard_limit = renderer.resource.getrlimit(
            renderer.resource.RLIMIT_NOFILE,
        )
        shifted_soft = (
            soft_limit - 1
            if soft_limit != renderer.resource.RLIM_INFINITY
            else 1_000_000
        )
        monkeypatch.setattr(
            renderer.resource,
            "getrlimit",
            lambda _limit: (shifted_soft, hard_limit),
        )
        validation = renderer.validate_rebuttal_pdf(
            source,
            template,
            config,
            published,
            replay,
            expected_manifest_sha256=receipt.manifest_sha256,
            **kwargs,
        )
        assert validation.replay_root == str(replay)
        assert validation.manifest_sha256 == receipt.manifest_sha256
        assert validation.pdf_sha256 == receipt.pdf_sha256
        assert (replay / renderer.MANIFEST_MEMBER).read_bytes() == (
            published / renderer.MANIFEST_MEMBER
        ).read_bytes()
        assert (replay / renderer.PDF_MEMBER).read_bytes() == pdf
        assert stat.S_IMODE(replay.stat().st_mode) == 0o500
    finally:
        for descriptor in extra_descriptors:
            os.close(descriptor)
        _make_writable(published)
        _make_writable(replay)


@REQUIRES_NATIVE_DARWIN
def test_forced_multipage_quote_response_and_list_replay(tmp_path: Path) -> None:
    source, template, config = _write_inputs(
        tmp_path,
        source_text=_multipage_source(),
    )
    kwargs = _build_kwargs(source, template, config)
    published = tmp_path / "published-multipage"
    replay = tmp_path / "replay-multipage"
    try:
        receipt = renderer.build_rebuttal_pdf(
            source,
            template,
            config,
            published,
            **kwargs,
        )
        assert receipt.page_count >= 6
        assert receipt.quote_count == 3
        assert receipt.response_count == 3
        validation = renderer.validate_rebuttal_pdf(
            source,
            template,
            config,
            published,
            replay,
            expected_manifest_sha256=receipt.manifest_sha256,
            **kwargs,
        )
        assert validation.page_count == receipt.page_count
        assert validation.pdf_sha256 == receipt.pdf_sha256
        assert (replay / renderer.PDF_MEMBER).read_bytes() == (
            published / renderer.PDF_MEMBER
        ).read_bytes()
        manifest = json.loads(
            (published / renderer.MANIFEST_MEMBER).read_text(encoding="ascii"),
        )
        assert manifest["document"]["text_equivalence"] == (
            "NFC-with-ASCII-whitespace-folding"
        )
        assert manifest["document"]["visible_text_blocks_preserved"] == len(
            manifest["markdown"]["visible_text_ledger"],
        )
    finally:
        _make_writable(published)
        _make_writable(replay)


@REQUIRES_NATIVE_DARWIN
def test_validation_rejects_wrong_manifest_anchor_before_replay(tmp_path: Path) -> None:
    source, template, config = _write_inputs(tmp_path)
    kwargs = _build_kwargs(source, template, config)
    published = tmp_path / "published"
    replay = tmp_path / "replay"
    try:
        receipt = renderer.build_rebuttal_pdf(
            source,
            template,
            config,
            published,
            **kwargs,
        )
        assert receipt.manifest_sha256 != "0" * 64
        with pytest.raises(
            renderer.RebuttalRenderError, match="independent SHA-256 anchor"
        ):
            renderer.validate_rebuttal_pdf(
                source,
                template,
                config,
                published,
                replay,
                expected_manifest_sha256="0" * 64,
                **kwargs,
            )
        assert not replay.exists()
    finally:
        _make_writable(published)


def test_cli_help_and_dependency_digest_are_bounded() -> None:
    _require_real_dependencies()
    script = Path(renderer.__file__).resolve(strict=True)
    help_result = subprocess.run(  # noqa: S603 - exact local interpreter/script.
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert "dependency-digest" in help_result.stdout
    assert "build" in help_result.stdout
    assert "validate" in help_result.stdout

    digest_result = subprocess.run(  # noqa: S603 - exact local interpreter/script.
        [
            sys.executable,
            str(script),
            "dependency-digest",
            "--reportlab-root",
            str(REPORTLAB_ROOT),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    parsed = json.loads(digest_result.stdout)
    assert parsed == renderer.reportlab_dependency_digest(REPORTLAB_ROOT)
