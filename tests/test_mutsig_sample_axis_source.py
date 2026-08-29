import hashlib
import os
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = REPO_ROOT / "external" / "mutsig2cv_octave_dialect.patch"
SOURCE_DOC = REPO_ROOT / "external" / "mutsig2cv_source.md"
MUTSIG_ROOT = REPO_ROOT / "external" / "MutSig2CV_src"
MUTSIG_SRC = MUTSIG_ROOT / "src"
PINNED_COMMIT = "0109e27e70478181695f31ca8dd281bb44f0b3af"


def _embedded_python_after(relative: str, anchor: str) -> str:
    source = (REPO_ROOT / relative).read_text(encoding="utf-8")
    scoped = source[source.index(anchor) :]
    marker = "run_python -c '\n"
    start = scoped.index(marker) + len(marker)
    return scoped[start : scoped.index("\n' \"", start)]


def _execute_embedded_python(code: str, arguments: list[str]) -> None:
    previous = sys.argv
    sys.argv = ["embedded-publication", *arguments]
    try:
        exec(compile(code, "<embedded-publication>", "exec"), {})  # noqa: S102
    finally:
        sys.argv = previous


def _require_source_clone() -> None:
    if not (MUTSIG_ROOT / ".git").is_dir():
        pytest.skip("ignored MutSig source clone is not present")


def _patch_postimage(path: str) -> str:
    """Return added/context lines for one file in the tracked unified patch."""
    wanted = f"diff --git a/{path} b/{path}"
    in_file = False
    lines: list[str] = []
    for line in PATCH_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("diff --git "):
            in_file = line == wanted
            continue
        if not in_file or line.startswith(("@@", "---", "+++")):
            continue
        if line.startswith((" ", "+")):
            lines.append(line[1:])
    if not lines:
        msg = f"tracked MutSig patch has no postimage for {path}"
        raise AssertionError(msg)
    return "\n".join(lines)


def _octave_eval(expression: str) -> subprocess.CompletedProcess[str]:
    _require_source_clone()
    octave = shutil.which("octave")
    if octave is None:
        pytest.skip("GNU Octave is not installed")
    escaped_src = str(MUTSIG_SRC).replace("'", "''")
    return subprocess.run(  # noqa: S603
        [
            octave,
            "--quiet",
            "--no-gui",
            "--eval",
            f"addpath('{escaped_src}'); {expression}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_sample_axis_loader_accepts_only_exact_sorted_unique_ids(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.txt"
    valid.write_text("s0\ns1\ns2\n", encoding="utf-8")
    result = _octave_eval(
        f"a=dialect_load_sample_axis('{valid}'); "
        "assert(isequal(a, {'s0';'s1';'s2'}));",
    )
    assert result.returncode == 0, result.stderr

    for name, contents in {
        "duplicate.txt": "s0\ns0\n",
        "unordered.txt": "s1\ns0\n",
        "blank.txt": "s0\n\ns2\n",
        "whitespace.txt": "s0\n s1\n",
    }.items():
        invalid = tmp_path / name
        invalid.write_text(contents, encoding="utf-8")
        result = _octave_eval(f"dialect_load_sample_axis('{invalid}');")
        assert result.returncode != 0, name


def test_normalized_full_allele_key_dedup_preserves_multiallelic_rows() -> None:
    expression = """
      m=[];
      m.patient={'s1';'s1';'s1';'s1';'s2'};
      m.chr=convert_chr({'1';'chr1';'1';'1';'1'});
      m.pos=[10;10;10;10;10];
      m.ref_allele={' a ';'A';'a';'T';'A'};
      m.newbase={' c ';'C';'g';'C';'C'};
      [d,k]=dialect_deduplicate_mutations(m);
      assert(isequal(k,[1;3;4;5]));
      assert(isequal(d.newbase,{'C';'G';'C';'C'}));
      assert(isequal(d.ref_allele,{'A';'A';'T';'A'}));
    """
    result = _octave_eval(expression)
    assert result.returncode == 0, result.stderr


def test_native_final_callscheme_is_not_overridden_for_zero_event_sample() -> None:
    expression = """
      M=[];
      M.pat.name=arrayfun(@(x) sprintf('s%d',x),0:9,'UniformOutput',false)';
      M.np=10;
      n=900;
      M.mut.patient=cell(n,1);
      M.mut.pat_idx=zeros(n,1);
      M.mut.type=repmat({'Missense_Mutation'},n,1);
      for p=2:10
        idx=((p-2)*100+1):((p-1)*100);
        M.mut.patient(idx)=M.pat.name(p);
        M.mut.pat_idx(idx)=p;
        M.mut.type(idx(end-4:end))={'Flank'};
      end
      M.mut.chr=ones(n,1);
      M.mut.start=(1:n)';
      M.mut.end=M.mut.start;
      M.mut.gene=repmat({'G'},n,1);
      M.mut.ref_allele=repmat({'A'},n,1);
      M.mut.newbase=repmat({'C'},n,1);
      M.mut.classification=repmat({'SNP'},n,1);
      M=impute_callschemes(M);
      assert(M.pat.nmut(1)==0);
      assert(M.pat.callscheme(1)==1);
    """
    result = _octave_eval(expression)
    assert result.returncode == 0, result.stderr


def test_lambda_validator_rejects_missing_invalid_or_negative_slices() -> None:
    expression = """
      x=single(zeros(2,3,2));
      written=true(2,2);
      dialect_validate_persample_lambda(x,written);
      failed=false;
      try dialect_validate_persample_lambda(x,[true false;true true]);
      catch failed=true; end
      assert(failed);
      failed=false; bad=x; bad(1)=NaN;
      try dialect_validate_persample_lambda(bad,written);
      catch failed=true; end
      assert(failed);
      failed=false; bad=x; bad(1)=-1;
      try dialect_validate_persample_lambda(bad,written);
      catch failed=true; end
      assert(failed);
    """
    result = _octave_eval(expression)
    assert result.returncode == 0, result.stderr


def test_tracked_patch_contains_fail_closed_source_contracts() -> None:
    core = _patch_postimage("src/MutSig_2CV_v3_11_core.m")
    runner = _patch_postimage("run_mutsig_persample.m")
    validator = _patch_postimage("src/dialect_validate_persample_lambda.m")

    assert "if nargin ~= 3" in runner
    assert "P.sample_axis_file = sample_axis_file" in runner
    assert "dialect_load_sample_axis(P.sample_axis_file)" in core
    assert "Every mutation sample must occur" in core
    assert "M.pat.name = dialect_sample_axis" in core
    assert "expected coding-only callscheme 0" not in core
    assert core.index("dialect_load_sample_axis") < core.index("% hg18 liftover")
    assert core.index("M.mut.chr = convert_chr") < core.index(
        "dialect_deduplicate_mutations",
    )
    deduplicator = _patch_postimage("src/dialect_deduplicate_mutations.m")
    assert "upper(strtrim(alleles{ii}))" in deduplicator
    assert deduplicator.index("normalize_alleles(mut.ref_allele") < (
        deduplicator.index("unique_combos")
    )
    assert "persample_lambda_written(g,d) = true" in core
    assert core.index("dialect_validate_persample_lambda") < core.index(
        "persample_lambda.f32'], 'w'",
    )
    assert "~all(written(:))" in validator
    assert "~isfinite(lambda(:))" in validator
    assert "lambda(:) < 0" in validator


def test_tracked_wrappers_bind_axis_source_and_receipts() -> None:
    shell = (REPO_ROOT / "scripts" / "run_mutsig_octave.sh").read_text(
        encoding="utf-8",
    )
    pipeline = (REPO_ROOT / "scripts" / "run_cohort_pipeline.sh").read_text(
        encoding="utf-8",
    )

    assert PINNED_COMMIT in shell
    assert '"$GIT_BIN" --no-pager -C "$MUTSIG_SOURCE" diff --cached --binary' in shell
    assert shell.count("--no-ext-diff --no-textconv") == 2
    assert "GIT_CONFIG_NOSYSTEM=1" in shell
    assert "getenv('DIALECT_MUTSIG_MAF')" in shell
    assert "tempfile.mkdtemp" in shell
    assert "persample_receipt.tsv" in shell
    assert "runner_sha256" in shell
    assert "runtime_sha256" in shell
    safe_octave_prefix = '"$OCTAVE_BIN" --no-init-all --no-history --no-gui'
    assert shell.count(safe_octave_prefix) == 2
    assert f"{safe_octave_prefix} --eval" in shell
    assert shell.index("persample_patients.txt") < shell.rindex(
        '"persample_receipt.tsv",',
    )
    assert 'run_python -m dialect "$@"' in pipeline
    assert 'PYTHONPATH="${REPO}:${REPO}/src"' in pipeline
    assert 'PYTHONPATH="${REPO}:${REPO}/src"' in shell
    assert "DIALECT import resolved outside the pinned repository source" in pipeline
    assert 'struct.pack(">Q", len(encoded))' in pipeline
    assert 'struct.pack(">Q", mode)' in pipeline
    assert 'struct.pack(">Q", before.st_size)' in pipeline
    assert 'if [ "${PREPARE_ONLY+x}" = "x" ]' in pipeline
    assert 'if [ "${PREPARE_ONLY:-}" = "1" ] && [ -n "${SKIP_MUTSIG:-}" ]' in pipeline
    assert '--cbase-sample-axis "$MUTSIG_SAMPLE_AXIS_FILE"' in pipeline
    assert "count_axis_matches" in pipeline
    assert "set -euo pipefail" in pipeline
    assert "directory_files_sha256" in pipeline
    assert "CBASE_INPUTS_SHA256" in pipeline
    assert "DIALECT_PROVIDER_CBASE_INPUTS_TREE_SHA256" in pipeline
    assert "DIALECT_PROVIDER_MUTSIG_SOURCE_TREE_SHA256" in shell
    assert "DIALECT_PROVIDER_MUTSIG_SOURCE_FILE_COUNT" in shell
    assert "source_tree_sha256" in shell
    assert "source_file_count" in shell
    assert 'files+=("$CB_Q")' in pipeline
    assert "cbase_outputs_sha256" in pipeline
    assert '|| log "STAGE-FAIL' not in pipeline
    assert "cbase_stage_receipt.tsv" in pipeline
    assert "dig_stage_receipt.tsv" in pipeline
    assert "identify_stage_receipt.tsv" in pipeline
    assert 'if [ -z "${PREPARE_ONLY:-}" ]; then' in pipeline
    assert "prepare-only: association identify stages remain sealed" in pipeline
    assert "prepare-only: provider inputs DONE" in pipeline
    assert pipeline.index('if [ -z "${PREPARE_ONLY:-}" ]; then') < pipeline.index(
        'log "identify cbase"',
    )
    assert pipeline.index("prepare-only: provider inputs DONE") < pipeline.index(
        "MUTSIG_ANALYSIS_SHA256",
    )


def test_publication_wrappers_use_dirfd_operations() -> None:
    shell = (REPO_ROOT / "scripts" / "run_mutsig_octave.sh").read_text(
        encoding="utf-8",
    )
    pipeline = (REPO_ROOT / "scripts" / "run_cohort_pipeline.sh").read_text(
        encoding="utf-8",
    )

    assert "src_dir_fd=source_parent" in shell
    assert "dst_dir_fd=destination_parent" in shell
    assert "os.unlink(name, dir_fd=parent)" in shell
    assert "src_dir_fd=parent_descriptor" in pipeline
    assert "dst_dir_fd=parent_descriptor" in pipeline
    assert "os.unlink(temporary_name, dir_fd=parent_descriptor)" in pipeline


def test_receipt_publication_rejects_parent_path_swap_after_dir_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = _embedded_python_after(
        "scripts/run_cohort_pipeline.sh",
        "publish_receipt() {",
    )
    parent = tmp_path / "receipt-parent"
    parent.mkdir()
    destination = parent / "stage_receipt.tsv"
    destination.write_bytes(b"stale\n")
    opened_parent = tmp_path / "opened-receipt-parent"
    expected = parent.stat()
    real_open = os.open
    swapped = False

    def racing_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        opened = os.fstat(descriptor)
        if (
            not swapped
            and stat.S_ISDIR(opened.st_mode)
            and opened.st_dev == expected.st_dev
            and opened.st_ino == expected.st_ino
        ):
            swapped = True
            parent.rename(opened_parent)
            parent.mkdir()
            (parent / destination.name).write_bytes(b"replacement-sentinel\n")
        return descriptor

    monkeypatch.setattr(os, "open", racing_open)
    with pytest.raises(
        SystemExit,
        match="receipt parent pathname changed during publication",
    ):
        _execute_embedded_python(code, [destination.as_posix(), "input", "output"])

    assert swapped
    assert (parent / destination.name).read_bytes() == b"replacement-sentinel\n"
    assert (opened_parent / destination.name).read_bytes() == (
        b"schema_version\t1\ninput_sha256\tinput\noutput_sha256\toutput\n"
    )


def test_mutsig_publication_rejects_destination_path_swap_after_dir_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = _embedded_python_after(
        "scripts/run_mutsig_octave.sh",
        '} > "${STAGING_DIR}/persample_receipt.tsv"',
    )
    staging = tmp_path / "staging"
    destination = tmp_path / "out" / "SYN"
    staging.mkdir()
    destination.mkdir(parents=True)
    artifacts = {
        "persample_lambda.f32": b"lambda",
        "persample_meta.txt": b"meta",
        "persample_genes.txt": b"genes",
        "persample_patients.txt": b"patients",
        "persample_receipt.tsv": b"receipt",
    }
    for name, payload in artifacts.items():
        (staging / name).write_bytes(payload)
    (destination / "persample_receipt.tsv").write_bytes(b"stale\n")
    opened_destination = tmp_path / "opened-destination"
    expected = destination.stat()
    real_open = os.open
    swapped = False

    def racing_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        opened = os.fstat(descriptor)
        if (
            not swapped
            and stat.S_ISDIR(opened.st_mode)
            and opened.st_dev == expected.st_dev
            and opened.st_ino == expected.st_ino
        ):
            swapped = True
            destination.rename(opened_destination)
            destination.mkdir()
            (destination / "persample_receipt.tsv").write_bytes(b"sentinel\n")
        return descriptor

    monkeypatch.setattr(os, "open", racing_open)
    with pytest.raises(
        SystemExit,
        match="MutSig destination pathname changed during publication",
    ):
        _execute_embedded_python(
            code,
            [staging.as_posix(), destination.as_posix()],
        )

    assert swapped
    assert (destination / "persample_receipt.tsv").read_bytes() == b"sentinel\n"
    assert sorted(path.name for path in destination.iterdir()) == [
        "persample_receipt.tsv",
    ]
    for name, payload in artifacts.items():
        assert (opened_destination / name).read_bytes() == payload


def test_patch_is_pinned_and_exactly_reconstructs_local_source(tmp_path: Path) -> None:
    _require_source_clone()
    git = shutil.which("git")
    assert git is not None
    head = subprocess.run(  # noqa: S603
        [git, "-C", str(MUTSIG_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert head == PINNED_COMMIT
    local_diff = subprocess.run(  # noqa: S603
        [git, "-C", str(MUTSIG_ROOT), "diff", "--cached", "--binary"],
        check=True,
        capture_output=True,
    ).stdout
    assert local_diff == PATCH_PATH.read_bytes()

    reconstructed = tmp_path / "MutSig2CV"
    subprocess.run(  # noqa: S603
        [git, "clone", "--no-local", str(MUTSIG_ROOT), str(reconstructed)],
        check=True,
        capture_output=True,
    )
    subprocess.run(  # noqa: S603
        [git, "-C", str(reconstructed), "apply", "--check", str(PATCH_PATH)],
        check=True,
    )
    subprocess.run(  # noqa: S603
        [
            git,
            "-C",
            str(reconstructed),
            "apply",
            "--index",
            str(PATCH_PATH),
        ],
        check=True,
    )
    rebuilt_diff = subprocess.run(  # noqa: S603
        [git, "-C", str(reconstructed), "diff", "--cached", "--binary"],
        check=True,
        capture_output=True,
    ).stdout
    assert rebuilt_diff == PATCH_PATH.read_bytes()


def test_source_documentation_pins_upstream_commit() -> None:
    documentation = SOURCE_DOC.read_text(encoding="utf-8")
    assert PINNED_COMMIT in documentation
    assert "apply --index" in documentation
    assert "diff --cached --binary" in documentation


def test_shell_runner_refuses_missing_axis_argument() -> None:
    bash = shutil.which("bash")
    assert bash is not None
    result = subprocess.run(  # noqa: S603
        [bash, str(REPO_ROOT / "scripts" / "run_mutsig_octave.sh")],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 64
    assert "<sample_axis_file>" in result.stderr


@pytest.mark.parametrize("value", ["", "0", "true"])
def test_pipeline_rejects_noncanonical_prepare_only_before_outputs(
    tmp_path: Path,
    value: str,
) -> None:
    bash = shutil.which("bash")
    assert bash is not None
    output = tmp_path / "provider-output"
    result = subprocess.run(  # noqa: S603
        [bash, str(REPO_ROOT / "scripts" / "run_cohort_pipeline.sh"), "SYN"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "PREPARE_ONLY": value, "ROOT": str(output)},
    )

    assert result.returncode == 64
    assert "PREPARE_ONLY must be unset or exactly 1" in result.stderr
    assert not output.exists()


def test_prepare_only_rejects_skip_mutsig_before_outputs(tmp_path: Path) -> None:
    bash = shutil.which("bash")
    assert bash is not None
    output = tmp_path / "provider-output"
    result = subprocess.run(  # noqa: S603
        [bash, str(REPO_ROOT / "scripts" / "run_cohort_pipeline.sh"), "SYN"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={
            **{
                key: value
                for key, value in os.environ.items()
                if key not in {"JAVA_HOME", "OCTAVE_BIN"}
            },
            "PREPARE_ONLY": "1",
            "ROOT": str(output),
            "SKIP_MUTSIG": "1",
        },
    )

    assert result.returncode == 64
    assert "PREPARE_ONLY=1 forbids SKIP_MUTSIG" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (
            {"OCTAVE_BIN": "/definitely-invalid/attacker-octave"},
            "OCTAVE_BIN override",
        ),
        (
            {"JAVA_HOME": "/definitely-invalid/attacker-java"},
            "JAVA_HOME override",
        ),
    ],
)
def test_prepare_mutsig_rejects_runtime_override_before_inputs(
    override: dict[str, str],
    message: str,
) -> None:
    bash = shutil.which("bash")
    assert bash is not None
    pinned = "/definitely-invalid/provider-runtime"
    result = subprocess.run(  # noqa: S603
        [
            bash,
            str(REPO_ROOT / "scripts" / "run_mutsig_octave.sh"),
            "SYN",
            "/definitely-invalid/missing-axis",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={
            **{
                key: value
                for key, value in os.environ.items()
                if key not in {"JAVA_HOME", "OCTAVE_BIN"}
            },
            "DIALECT_PROVIDER_GIT": pinned,
            "DIALECT_PROVIDER_GIT_SHA256": "1" * 64,
            "DIALECT_PROVIDER_JAVA": f"{pinned}/java",
            "DIALECT_PROVIDER_JAVA_HOME": f"{pinned}/jdk",
            "DIALECT_PROVIDER_JAVA_ID": "provider-java",
            "DIALECT_PROVIDER_JAVA_SHA256": "2" * 64,
            "DIALECT_PROVIDER_MUTSIG_RUNTIME_SHA256": "3" * 64,
            "DIALECT_PROVIDER_OCTAVE": f"{pinned}/octave",
            "DIALECT_PROVIDER_OCTAVE_ID": "provider-octave",
            "DIALECT_PROVIDER_OCTAVE_SHA256": "4" * 64,
            "DIALECT_PROVIDER_PYTHON": f"{pinned}/python",
            "DIALECT_PROVIDER_PYTHON_SHA256": "5" * 64,
            "PREPARE_ONLY": "1",
            **override,
        },
    )

    assert result.returncode == 70
    assert message in result.stderr


def test_pipeline_rejects_symlinked_dialect_import_before_outputs(
    tmp_path: Path,
) -> None:
    bash = shutil.which("bash")
    assert bash is not None
    fake_repo = tmp_path / "fake-repo"
    scripts = fake_repo / "scripts"
    scripts.mkdir(parents=True)
    pipeline = scripts / "run_cohort_pipeline.sh"
    pipeline.write_bytes(
        (REPO_ROOT / "scripts" / "run_cohort_pipeline.sh").read_bytes(),
    )
    outside_package = tmp_path / "outside" / "dialect"
    outside_package.mkdir(parents=True)
    (outside_package / "__init__.py").write_text("", encoding="utf-8")
    (fake_repo / "src").mkdir()
    (fake_repo / "src" / "dialect").symlink_to(
        outside_package,
        target_is_directory=True,
    )
    output = tmp_path / "provider-output"

    result = subprocess.run(  # noqa: S603
        [bash, str(pipeline), "SYN"],
        check=False,
        capture_output=True,
        text=True,
        cwd=fake_repo,
        env={**os.environ, "ROOT": str(output)},
    )

    assert result.returncode == 70
    assert "DIALECT source root traverses a symlink" in result.stderr
    assert not output.exists()


def test_pipeline_rejects_stale_dialect_tree_authority_before_outputs(
    tmp_path: Path,
) -> None:
    bash = Path(shutil.which("bash") or "").resolve()
    assert bash.is_file()
    fake_repo = tmp_path / "fake-repo"
    scripts = fake_repo / "scripts"
    scripts.mkdir(parents=True)
    pipeline = scripts / "run_cohort_pipeline.sh"
    pipeline.write_bytes(
        (REPO_ROOT / "scripts" / "run_cohort_pipeline.sh").read_bytes(),
    )
    dialect = fake_repo / "src" / "dialect"
    dialect.mkdir(parents=True)
    (dialect / "__init__.py").write_text("", encoding="utf-8")
    cbase = fake_repo / "external" / "CBaSE"
    cbase.mkdir(parents=True)
    (cbase / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    maf_dir = tmp_path / "mafs"
    maf_dir.mkdir()
    (maf_dir / "SYN.maf").write_text("synthetic input\n", encoding="utf-8")
    axis = tmp_path / "axis.txt"
    axis.write_text("sample\n", encoding="utf-8")
    output = tmp_path / "provider-output"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        f"#!/bin/sh\nprintf 'src/dialect/__init__.py\\n%s\\n' {'0' * 64}\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o700)
    runtime_authority = fake_repo / "runtime" / "authority.json"
    runtime_authority.parent.mkdir()
    runtime_authority.write_text("{}\n", encoding="utf-8")
    nice = Path(shutil.which("nice") or "").resolve()
    assert nice.is_file()

    result = subprocess.run(  # noqa: S603
        [bash.as_posix(), pipeline.as_posix(), "SYN"],
        check=False,
        capture_output=True,
        text=True,
        cwd=fake_repo,
        env={
            **os.environ,
            "DIALECT_PROVIDER_BASH": bash.as_posix(),
            "DIALECT_PROVIDER_BASH_SHA256": hashlib.sha256(
                bash.read_bytes(),
            ).hexdigest(),
            "DIALECT_PROVIDER_DIALECT_TREE_SHA256": "0" * 64,
            "DIALECT_PROVIDER_CBASE_INPUTS_TREE_SHA256": "2" * 64,
            "DIALECT_PROVIDER_NICE": nice.as_posix(),
            "DIALECT_PROVIDER_NICE_SHA256": hashlib.sha256(
                nice.read_bytes(),
            ).hexdigest(),
            "DIALECT_PROVIDER_RUNTIME_AUTHORITY_FILE": (runtime_authority.as_posix()),
            "DIALECT_PROVIDER_RUNTIME_AUTHORITY_SHA256": hashlib.sha256(
                runtime_authority.read_bytes(),
            ).hexdigest(),
            "DIALECT_PROVIDER_PYTHON": fake_python.as_posix(),
            "DIALECT_PROVIDER_PYTHON_RUNTIME_SHA256": "1" * 64,
            "DIALECT_PROVIDER_PYTHON_SHA256": hashlib.sha256(
                fake_python.read_bytes(),
            ).hexdigest(),
            "MAF_DIR": maf_dir.as_posix(),
            "MUTSIG_SAMPLE_AXIS_FILE": axis.as_posix(),
            "PREPARE_ONLY": "1",
            "ROOT": output.as_posix(),
        },
    )

    assert result.returncode == 70
    assert "DIALECT source tree differs from provider authority" in result.stderr
    assert not output.exists()


def test_shell_runner_receipt_rejects_partial_and_stale_absolute_paths(
    tmp_path: Path,
) -> None:
    _require_source_clone()
    bash = shutil.which("bash")
    assert bash is not None
    maf_dir = tmp_path / "maf's absolute"
    out_root = tmp_path / "out's absolute"
    axis = tmp_path / "axis's.txt"
    call_log = tmp_path / "fake-octave-calls.txt"
    fake_octave = tmp_path / "fake octave"
    maf_dir.mkdir()
    (maf_dir / "SYN.maf").write_text("synthetic result-blind input\n", encoding="utf-8")
    axis.write_text("s0\ns1\n", encoding="utf-8")
    fake_octave.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import os
            import sys
            from pathlib import Path

            required_flags = {"--no-init-all", "--no-history", "--no-gui"}
            if not required_flags.issubset(sys.argv):
                print("unsafe octave startup flags", file=sys.stderr)
                raise SystemExit(91)
            if "--version" in sys.argv:
                print("fake-octave 1.0")
                raise SystemExit(0)
            log = Path(os.environ["FAKE_CALL_LOG"])
            with log.open("a", encoding="utf-8") as handle:
                handle.write("run\\n")
            output = Path(os.environ["DIALECT_MUTSIG_OUT"])
            output.mkdir(parents=True, exist_ok=True)
            patients = Path(os.environ["DIALECT_MUTSIG_AXIS"]).read_text(
                encoding="utf-8",
            ).splitlines()
            (output / "persample_meta.txt").write_text(
                f"ng\\t1\\nnp\\t{len(patients)}\\nneff\\t2\\n",
                encoding="utf-8",
            )
            (output / "persample_genes.txt").write_text("G\\n", encoding="utf-8")
            (output / "persample_patients.txt").write_text(
                "".join(f"{patient}\\n" for patient in patients),
                encoding="utf-8",
            )
            (output / "persample_lambda.f32").write_bytes(
                bytes(1 * len(patients) * 2 * 4),
            )
            """,
        ),
        encoding="utf-8",
    )
    fake_octave.chmod(fake_octave.stat().st_mode | stat.S_IXUSR)
    env = {
        **os.environ,
        "OCTAVE_BIN": str(fake_octave),
        "FAKE_CALL_LOG": str(call_log),
    }
    command = [
        bash,
        str(REPO_ROOT / "scripts" / "run_mutsig_octave.sh"),
        "SYN",
        str(axis),
        str(maf_dir),
        str(out_root),
    ]

    first = subprocess.run(  # noqa: S603
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )
    assert first.returncode == 0, first.stderr
    assert call_log.read_text(encoding="utf-8").splitlines() == ["run"]

    second = subprocess.run(  # noqa: S603
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )
    assert second.returncode == 0, second.stderr
    assert call_log.read_text(encoding="utf-8").splitlines() == ["run"]

    bundle = out_root / "SYN"
    (bundle / "persample_meta.txt").unlink()
    repaired = subprocess.run(  # noqa: S603
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )
    assert repaired.returncode == 0, repaired.stderr
    assert call_log.read_text(encoding="utf-8").splitlines() == ["run", "run"]

    axis.write_text("s0\ns1\ns2\n", encoding="utf-8")
    refreshed = subprocess.run(  # noqa: S603
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )
    assert refreshed.returncode == 0, refreshed.stderr
    assert call_log.read_text(encoding="utf-8").splitlines() == [
        "run",
        "run",
        "run",
    ]
    assert (bundle / "persample_patients.txt").read_text(
        encoding="utf-8",
    ).splitlines() == ["s0", "s1", "s2"]
    assert (bundle / "persample_lambda.f32").stat().st_size == 24
    assert (bundle / "persample_receipt.tsv").is_file()


def test_revision_shells_parse_under_macos_system_bash() -> None:
    """The provider-pinned macOS Bash 3.2 must parse both revision wrappers."""
    system_bash = Path("/bin/bash")
    if not system_bash.is_file():
        pytest.skip("macOS system Bash is unavailable")
    for relative in (
        "scripts/run_cohort_pipeline.sh",
        "scripts/run_mutsig_octave.sh",
    ):
        completed = subprocess.run(  # noqa: S603
            [system_bash.as_posix(), "-n", relative],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        smoke = subprocess.run(  # noqa: S603
            [system_bash.as_posix(), relative],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert smoke.returncode == 64
        assert "usage:" in smoke.stderr.lower()
    assert "mapfile" not in (
        REPO_ROOT / "scripts" / "run_cohort_pipeline.sh"
    ).read_text(encoding="utf-8")
    assert "mapfile" not in (REPO_ROOT / "scripts" / "run_mutsig_octave.sh").read_text(
        encoding="utf-8",
    )
