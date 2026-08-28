import os
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = REPO_ROOT / "external" / "mutsig2cv_octave_dialect.patch"
SOURCE_DOC = REPO_ROOT / "external" / "mutsig2cv_source.md"
MUTSIG_ROOT = REPO_ROOT / "external" / "MutSig2CV_src"
MUTSIG_SRC = MUTSIG_ROOT / "src"
PINNED_COMMIT = "0109e27e70478181695f31ca8dd281bb44f0b3af"


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
    assert 'git -C "$MUTSIG_SOURCE" diff --cached --binary' in shell
    assert "getenv('DIALECT_MUTSIG_MAF')" in shell
    assert "mktemp -d" in shell
    assert "persample_receipt.tsv" in shell
    assert "runner_sha256" in shell
    assert "runtime_sha256" in shell
    assert shell.index("persample_patients.txt") < shell.rindex(
        'mv -f -- "${STAGING_DIR}/persample_receipt.tsv"',
    )
    assert '--cbase-sample-axis "$MUTSIG_SAMPLE_AXIS_FILE"' in pipeline
    assert "count_axis_matches" in pipeline
    assert "set -euo pipefail" in pipeline
    assert "directory_files_sha256" in pipeline
    assert "CBASE_INPUTS_SHA256" in pipeline
    assert 'files+=("$CB_Q")' in pipeline
    assert "cbase_outputs_sha256" in pipeline
    assert '|| log "STAGE-FAIL' not in pipeline
    assert "cbase_stage_receipt.tsv" in pipeline
    assert "dig_stage_receipt.tsv" in pipeline
    assert "identify_stage_receipt.tsv" in pipeline


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
