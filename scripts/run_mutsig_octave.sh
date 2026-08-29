#!/usr/bin/env bash
# Run the pinned DIALECT-patched MutSig2CV source under GNU Octave for one cohort.
# A validated receipt is published last, so incomplete or input/source-stale output
# bundles are never treated as reusable.
#
# Usage: run_mutsig_octave.sh <COHORT> <sample_axis_file> [maf_dir] [out_root]
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
  echo "usage: $0 <COHORT> <sample_axis_file> [maf_dir] [out_root]" >&2
  exit 64
fi
if [ "${PREPARE_ONLY+x}" = "x" ] && [ "$PREPARE_ONLY" != "1" ]; then
  echo "PREPARE_ONLY must be unset or exactly 1" >&2
  exit 64
fi

C="$1"
SAMPLE_AXIS_ARG="$2"
MAF_DIR_ARG="${3:-data/mafs_pancan}"
OUT_ROOT_ARG="${4:-output/mutsigsrc}"
UPSTREAM_COMMIT="0109e27e70478181695f31ca8dd281bb44f0b3af"

if [[ ! "$C" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "invalid cohort identifier: ${C}" >&2
  exit 65
fi

SCRIPT_PATH="${BASH_SOURCE[0]}"
case "$SCRIPT_PATH" in
  /*) ;;
  *) SCRIPT_PATH="${PWD}/${SCRIPT_PATH}" ;;
esac
SCRIPT_DIR="$(cd -- "${SCRIPT_PATH%/*}" && pwd -P)"
REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
MUTSIG_SOURCE="${REPO}/external/MutSig2CV_src"
PATCH_FILE="${REPO}/external/mutsig2cv_octave_dialect.patch"
DEFAULT_PY="/opt/anaconda3/envs/dialect/bin/python"
if [ "${PREPARE_ONLY:-}" = "1" ]; then
  : "${DIALECT_PROVIDER_PYTHON:?missing provider Python authority}"
  : "${DIALECT_PROVIDER_PYTHON_SHA256:?missing provider Python SHA-256}"
  : "${DIALECT_PROVIDER_GIT:?missing provider Git authority}"
  : "${DIALECT_PROVIDER_GIT_SHA256:?missing provider Git SHA-256}"
  : "${DIALECT_PROVIDER_OCTAVE:?missing provider Octave authority}"
  : "${DIALECT_PROVIDER_OCTAVE_SHA256:?missing provider Octave SHA-256}"
  : "${DIALECT_PROVIDER_OCTAVE_ID:?missing provider Octave identity}"
  : "${DIALECT_PROVIDER_JAVA_HOME:?missing provider Java home authority}"
  : "${DIALECT_PROVIDER_JAVA:?missing provider Java authority}"
  : "${DIALECT_PROVIDER_JAVA_SHA256:?missing provider Java SHA-256}"
  : "${DIALECT_PROVIDER_JAVA_ID:?missing provider Java identity}"
  : "${DIALECT_PROVIDER_MUTSIG_RUNTIME_SHA256:?missing provider MutSig runtime SHA-256}"
  if [ -n "${OCTAVE_BIN:-}" ] && [ "$OCTAVE_BIN" != "$DIALECT_PROVIDER_OCTAVE" ]; then
    echo "OCTAVE_BIN override differs from provider authority" >&2
    exit 70
  fi
  if [ -n "${JAVA_HOME:-}" ] && [ "$JAVA_HOME" != "$DIALECT_PROVIDER_JAVA_HOME" ]; then
    echo "JAVA_HOME override differs from provider authority" >&2
    exit 70
  fi
  PY="$DIALECT_PROVIDER_PYTHON"
else
  PY="$DEFAULT_PY"
fi
[ -x "$PY" ] || { echo "pinned Python is not executable: ${PY}" >&2; exit 69; }

run_python() {
  PYTHONPATH="${REPO}:${REPO}/src" PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
    "$PY" -P -s "$@"
}

resolve_from_repo() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$REPO" "$1" ;;
  esac
}

sha256_file() {
  run_python -c '
import hashlib
import os
import stat
import sys
from pathlib import Path

digest = hashlib.sha256()
descriptor = os.open(
    Path(sys.argv[1]),
    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
)
try:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise SystemExit("hash input must be a single-link regular file")
    observed_bytes = 0
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
        observed_bytes += len(chunk)
    after = os.fstat(descriptor)
finally:
    os.close(descriptor)
if (
    before.st_dev != after.st_dev
    or before.st_ino != after.st_ino
    or before.st_mode != after.st_mode
    or after.st_nlink != 1
    or before.st_size != after.st_size
    or before.st_mtime_ns != after.st_mtime_ns
    or before.st_ctime_ns != after.st_ctime_ns
    or observed_bytes != before.st_size
):
    raise SystemExit("hash input changed during stable descriptor read")
print(digest.hexdigest())
' "$1"
}

sha256_stream() {
  run_python -c '
import hashlib
import sys

digest = hashlib.sha256()
while chunk := sys.stdin.buffer.read(1024 * 1024):
    digest.update(chunk)
print(digest.hexdigest())
'
}

line_count_stream() {
  run_python -c '
import sys

print(sum(1 for _line in sys.stdin.buffer))
'
}

receipt_value() {
  run_python -c '
import sys
from pathlib import Path

rows = [line.split("\t") for line in Path(sys.argv[1]).read_text(encoding="ascii").splitlines()]
if any(len(row) != 2 for row in rows):
    raise SystemExit(1)
matches = [row[1] for row in rows if row[0] == sys.argv[2]]
if len(matches) != 1:
    raise SystemExit(1)
print(matches[0])
' "$1" "$2"
}

make_directory() {
  run_python -c '
import sys
from pathlib import Path

Path(sys.argv[1]).mkdir(parents=True, exist_ok=True)
' "$1"
}

remove_tree() {
  run_python -c '
import shutil
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.exists():
    shutil.rmtree(path)
' "$1"
}

remove_file() {
  run_python -c '
import sys
from pathlib import Path

Path(sys.argv[1]).unlink(missing_ok=True)
' "$1"
}

MAF_DIR="$(resolve_from_repo "$MAF_DIR_ARG")"
OUT_ROOT="$(resolve_from_repo "$OUT_ROOT_ARG")"
SAMPLE_AXIS_FILE="$(resolve_from_repo "$SAMPLE_AXIS_ARG")"
MAF="${MAF_DIR%/}/${C}.maf"
OUT_DIR="${OUT_ROOT%/}/${C}"
RECEIPT="${OUT_DIR}/persample_receipt.tsv"

[ -f "$MAF" ] || { echo "MAF not found: ${MAF}" >&2; exit 66; }
[ -f "$SAMPLE_AXIS_FILE" ] || {
  echo "sample axis file not found: ${SAMPLE_AXIS_FILE}" >&2
  exit 66
}
[ -f "$PATCH_FILE" ] || { echo "MutSig patch not found: ${PATCH_FILE}" >&2; exit 66; }
if [ "${PREPARE_ONLY:-}" != "1" ]; then
  [ -d "${MUTSIG_SOURCE}/.git" ] || {
    echo "patched MutSig source clone not found: ${MUTSIG_SOURCE}" >&2
    exit 69
  }
fi

if [ "${PREPARE_ONLY:-}" = "1" ]; then
  GIT_BIN="$DIALECT_PROVIDER_GIT"
  PY_EXECUTABLE_IDENTITY=()
  while IFS= read -r identity_line; do
    PY_EXECUTABLE_IDENTITY+=("$identity_line")
  done < <(
    run_python -c '
import hashlib
import os
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
try:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise SystemExit("executable must be a single-link regular file")
    hasher = hashlib.sha256()
    observed_bytes = 0
    while chunk := os.read(descriptor, 1024 * 1024):
        hasher.update(chunk)
        observed_bytes += len(chunk)
    after = os.fstat(descriptor)
finally:
    os.close(descriptor)
if (
    before.st_dev != after.st_dev
    or before.st_ino != after.st_ino
    or before.st_mode != after.st_mode
    or after.st_nlink != 1
    or before.st_size != after.st_size
    or before.st_mtime_ns != after.st_mtime_ns
    or before.st_ctime_ns != after.st_ctime_ns
    or observed_bytes != before.st_size
):
    raise SystemExit("executable changed during stable descriptor read")
digest = hasher.hexdigest()
print(path.as_posix())
print(digest)
' "$PY"
  )
  [ "${#PY_EXECUTABLE_IDENTITY[@]}" -eq 2 ] \
    && [ "${PY_EXECUTABLE_IDENTITY[0]}" = "$DIALECT_PROVIDER_PYTHON" ] \
    && [ "${PY_EXECUTABLE_IDENTITY[1]}" = "$DIALECT_PROVIDER_PYTHON_SHA256" ] || {
      echo "Python executable differs from provider authority" >&2
      exit 70
    }
  GIT_EXECUTABLE_IDENTITY=()
  while IFS= read -r identity_line; do
    GIT_EXECUTABLE_IDENTITY+=("$identity_line")
  done < <(
    run_python -c '
import hashlib
import os
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
try:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise SystemExit("executable must be a single-link regular file")
    hasher = hashlib.sha256()
    observed_bytes = 0
    while chunk := os.read(descriptor, 1024 * 1024):
        hasher.update(chunk)
        observed_bytes += len(chunk)
    after = os.fstat(descriptor)
finally:
    os.close(descriptor)
if (
    before.st_dev != after.st_dev
    or before.st_ino != after.st_ino
    or before.st_mode != after.st_mode
    or after.st_nlink != 1
    or before.st_size != after.st_size
    or before.st_mtime_ns != after.st_mtime_ns
    or before.st_ctime_ns != after.st_ctime_ns
    or observed_bytes != before.st_size
):
    raise SystemExit("executable changed during stable descriptor read")
digest = hasher.hexdigest()
print(path.as_posix())
print(digest)
' "$GIT_BIN"
  )
  [ "${#GIT_EXECUTABLE_IDENTITY[@]}" -eq 2 ] \
    && [ "${GIT_EXECUTABLE_IDENTITY[0]}" = "$DIALECT_PROVIDER_GIT" ] \
    && [ "${GIT_EXECUTABLE_IDENTITY[1]}" = "$DIALECT_PROVIDER_GIT_SHA256" ] || {
      echo "Git executable differs from provider authority" >&2
      exit 70
    }
else
  GIT_BIN="${GIT_BIN:-git}"
  command -v "$GIT_BIN" >/dev/null 2>&1 || {
    echo "Git executable not found" >&2
    exit 69
  }
fi
export GIT_CONFIG_COUNT=0
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_NOSYSTEM=1
export GIT_PAGER=
export GIT_TERMINAL_PROMPT=0

PATCH_SHA256="$(sha256_file "$PATCH_FILE")"
if [ "${PREPARE_ONLY:-}" = "1" ]; then
  : "${DIALECT_PROVIDER_MUTSIG_SOURCE_TREE_SHA256:?missing MutSig source-tree authority}"
  : "${DIALECT_PROVIDER_MUTSIG_SOURCE_FILE_COUNT:?missing MutSig source file-count authority}"
  SOURCE_TREE_SHA256="$DIALECT_PROVIDER_MUTSIG_SOURCE_TREE_SHA256"
  SOURCE_FILE_COUNT="$DIALECT_PROVIDER_MUTSIG_SOURCE_FILE_COUNT"
else
  SOURCE_HEAD="$("$GIT_BIN" --no-pager -C "$MUTSIG_SOURCE" rev-parse HEAD)"
  if [ "$SOURCE_HEAD" != "$UPSTREAM_COMMIT" ]; then
    echo "MutSig source HEAD ${SOURCE_HEAD} does not match pinned ${UPSTREAM_COMMIT}" >&2
    exit 70
  fi
  if ! "$GIT_BIN" --no-pager -C "$MUTSIG_SOURCE" diff \
    --quiet --no-ext-diff --no-textconv; then
    echo "MutSig source has unstaged drift; regenerate the tracked patch first" >&2
    exit 70
  fi
  if [ -n "$("$GIT_BIN" --no-pager -C "$MUTSIG_SOURCE" ls-files --others --exclude-standard)" ]; then
    echo "MutSig source has untracked drift; regenerate the tracked patch first" >&2
    exit 70
  fi
  SOURCE_DIFF_SHA256="$(
    "$GIT_BIN" --no-pager -C "$MUTSIG_SOURCE" diff --cached --binary \
      --no-ext-diff --no-textconv | sha256_stream
  )"
  if [ "$SOURCE_DIFF_SHA256" != "$PATCH_SHA256" ]; then
    echo "MutSig source index does not reconstruct the tracked patch" >&2
    exit 70
  fi
  SOURCE_TREE_SHA256="$SOURCE_DIFF_SHA256"
  SOURCE_FILE_COUNT="$("$GIT_BIN" --no-pager -C "$MUTSIG_SOURCE" ls-files | line_count_stream)"
fi

if [ "${PREPARE_ONLY:-}" = "1" ]; then
  OCTAVE_BIN="$DIALECT_PROVIDER_OCTAVE"
  JAVA_HOME="$DIALECT_PROVIDER_JAVA_HOME"
  JAVA_BIN="$DIALECT_PROVIDER_JAVA"
  export JAVA_HOME
MUTSIG_EXECUTABLE_IDENTITY=()
while IFS= read -r identity_line; do
  MUTSIG_EXECUTABLE_IDENTITY+=("$identity_line")
done < <(
    run_python -c '
import hashlib
import os
import stat
import sys
from pathlib import Path

def stable_digest(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise SystemExit("executable must be a single-link regular file")
        hasher = hashlib.sha256()
        observed_bytes = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            hasher.update(chunk)
            observed_bytes += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_mode != after.st_mode
        or after.st_nlink != 1
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
        or observed_bytes != before.st_size
    ):
        raise SystemExit("executable changed during stable descriptor read")
    return hasher.hexdigest()

for raw in sys.argv[1:]:
    path = Path(raw).resolve()
    print(path.as_posix())
    print(stable_digest(path))
' "$OCTAVE_BIN" "$JAVA_BIN"
  )
  [ "${#MUTSIG_EXECUTABLE_IDENTITY[@]}" -eq 4 ] \
    && [ "${MUTSIG_EXECUTABLE_IDENTITY[0]}" = "$DIALECT_PROVIDER_OCTAVE" ] \
    && [ "${MUTSIG_EXECUTABLE_IDENTITY[1]}" = "$DIALECT_PROVIDER_OCTAVE_SHA256" ] \
    && [ "${MUTSIG_EXECUTABLE_IDENTITY[2]}" = "$DIALECT_PROVIDER_JAVA" ] \
    && [ "${MUTSIG_EXECUTABLE_IDENTITY[3]}" = "$DIALECT_PROVIDER_JAVA_SHA256" ] || {
      echo "Octave or Java executable differs from provider authority" >&2
      exit 70
    }
else
  export PATH="/opt/homebrew/bin:${PATH}"
  if [ -z "${JAVA_HOME:-}" ]; then
    export JAVA_HOME="/Library/Java/JavaVirtualMachines/amazon-corretto-11.jdk/Contents/Home"
  fi
  OCTAVE_BIN="${OCTAVE_BIN:-octave}"
  JAVA_BIN="${JAVA_HOME}/bin/java"
fi
command -v "$OCTAVE_BIN" >/dev/null 2>&1 || { echo "GNU Octave not found" >&2; exit 69; }
OCTAVE_VERSION_OUTPUT="$("$OCTAVE_BIN" --no-init-all --no-history --no-gui --version 2>&1)"
OCTAVE_ID="${OCTAVE_VERSION_OUTPUT%%$'\n'*}"
if [ "${PREPARE_ONLY:-}" = "1" ]; then
  [ "$OCTAVE_ID" = "$DIALECT_PROVIDER_OCTAVE_ID" ] || {
    echo "Octave version differs from provider authority" >&2
    exit 70
  }
  JAVA_VERSION_OUTPUT="$("$JAVA_BIN" -version 2>&1)"
  JAVA_ID="${JAVA_VERSION_OUTPUT%%$'\n'*}"
  [ "$JAVA_ID" = "$DIALECT_PROVIDER_JAVA_ID" ] || {
    echo "Java version differs from provider authority" >&2
    exit 70
  }
  RUNTIME_SHA256="$DIALECT_PROVIDER_MUTSIG_RUNTIME_SHA256"
else
  RUNTIME_SHA256="$(printf '%s\n%s\n' "$OCTAVE_ID" "$JAVA_HOME" | sha256_stream)"
fi
RUNNER_SHA256="$(sha256_file "${SCRIPT_DIR}/run_mutsig_octave.sh")"
MAF_SHA256="$(sha256_file "$MAF")"
SAMPLE_AXIS_SHA256="$(sha256_file "$SAMPLE_AXIS_FILE")"
SAMPLE_AXIS_COUNT="$(
  run_python -c '
import os
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
try:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise SystemExit(1)
    chunks = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    after = os.fstat(descriptor)
finally:
    os.close(descriptor)
raw = b"".join(chunks)
if (
    before.st_dev != after.st_dev
    or before.st_ino != after.st_ino
    or before.st_mode != after.st_mode
    or after.st_nlink != 1
    or before.st_size != after.st_size
    or before.st_mtime_ns != after.st_mtime_ns
    or before.st_ctime_ns != after.st_ctime_ns
    or len(raw) != before.st_size
):
    raise SystemExit(1)
print(len(raw.splitlines()))
' "$SAMPLE_AXIS_FILE"
)"

validate_bundle() {
  run_python -c '
import re
import os
import stat
import sys
from pathlib import Path

bundle = Path(sys.argv[1])
axis = Path(sys.argv[2])

def stable_read(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise SystemExit(1)
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_mode != after.st_mode
        or after.st_nlink != 1
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
        or len(raw) != before.st_size
    ):
        raise SystemExit(1)
    return raw

def stable_size(path: Path) -> int:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise SystemExit(1)
        return opened.st_size
    finally:
        os.close(descriptor)

artifacts = (
    "persample_meta.txt",
    "persample_genes.txt",
    "persample_patients.txt",
)
artifact_bytes = {name: stable_read(bundle / name) for name in artifacts}
lambda_size = stable_size(bundle / "persample_lambda.f32")
if lambda_size <= 0 or any(not raw for raw in artifact_bytes.values()):
    raise SystemExit(1)
try:
    rows = [
        line.split("\t")
        for line in artifact_bytes["persample_meta.txt"].decode("ascii").splitlines()
    ]
except UnicodeDecodeError:
    raise SystemExit(1) from None
if any(len(row) != 2 for row in rows):
    raise SystemExit(1)
metadata = {key: value for key, value in rows}
if len(metadata) != len(rows) or set(metadata) != {"ng", "np", "neff"}:
    raise SystemExit(1)
if re.fullmatch(r"[1-9][0-9]*", metadata["ng"]) is None:
    raise SystemExit(1)
if re.fullmatch(r"[1-9][0-9]*", metadata["np"]) is None or metadata["neff"] != "2":
    raise SystemExit(1)
ng = int(metadata["ng"])
np = int(metadata["np"])
genes = artifact_bytes["persample_genes.txt"].splitlines()
patients = artifact_bytes["persample_patients.txt"].splitlines()
signed_axis = stable_read(axis).splitlines()
if (
    len(genes) != ng
    or len(patients) != np
    or len(patients) != int(sys.argv[3])
    or any(not value for value in genes)
    or any(not value for value in patients)
    or patients != signed_axis
    or lambda_size != ng * np * 2 * 4
):
    raise SystemExit(1)
' "$1" "$SAMPLE_AXIS_FILE" "$SAMPLE_AXIS_COUNT"
}

receipt_matches_current_inputs() {
  local artifact key
  [ -s "$RECEIPT" ] || return 1
  validate_bundle "$OUT_DIR" || return 1
  [ "$(receipt_value "$RECEIPT" schema_version)" = "1" ] || return 1
  [ "$(receipt_value "$RECEIPT" cohort)" = "$C" ] || return 1
  [ "$(receipt_value "$RECEIPT" upstream_commit)" = "$UPSTREAM_COMMIT" ] || return 1
  [ "$(receipt_value "$RECEIPT" source_tree_sha256)" = "$SOURCE_TREE_SHA256" ] || return 1
  [ "$(receipt_value "$RECEIPT" source_file_count)" = "$SOURCE_FILE_COUNT" ] || return 1
  [ "$(receipt_value "$RECEIPT" patch_sha256)" = "$PATCH_SHA256" ] || return 1
  [ "$(receipt_value "$RECEIPT" runner_sha256)" = "$RUNNER_SHA256" ] || return 1
  [ "$(receipt_value "$RECEIPT" runtime_sha256)" = "$RUNTIME_SHA256" ] || return 1
  [ "$(receipt_value "$RECEIPT" maf_sha256)" = "$MAF_SHA256" ] || return 1
  [ "$(receipt_value "$RECEIPT" sample_axis_sha256)" = "$SAMPLE_AXIS_SHA256" ] || return 1
  [ "$(receipt_value "$RECEIPT" sample_axis_count)" = "$SAMPLE_AXIS_COUNT" ] || return 1

  for artifact in lambda meta genes patients; do
    case "$artifact" in
      lambda) key="persample_lambda.f32" ;;
      meta) key="persample_meta.txt" ;;
      genes) key="persample_genes.txt" ;;
      patients) key="persample_patients.txt" ;;
    esac
    [ "$(receipt_value "$RECEIPT" "${artifact}_sha256")" = \
      "$(sha256_file "${OUT_DIR}/${key}")" ] || return 1
  done
}

if receipt_matches_current_inputs; then
  echo "DIALECT: validated current MutSig receipt at ${RECEIPT}"
  exit 0
fi

make_directory "$OUT_ROOT"
STAGING_DIR="$(
  run_python -c '
import sys
import tempfile

print(tempfile.mkdtemp(prefix=f".{sys.argv[2]}.mutsig.", dir=sys.argv[1]))
' "$OUT_ROOT" "$C"
)"
cleanup() {
  if [ -n "${STAGING_DIR:-}" ] && [ -d "$STAGING_DIR" ]; then
    remove_tree "$STAGING_DIR"
  fi
}
trap cleanup EXIT

DIALECT_MUTSIG_SOURCE="$MUTSIG_SOURCE" \
DIALECT_MUTSIG_MAF="$MAF" \
DIALECT_MUTSIG_OUT="$STAGING_DIR" \
DIALECT_MUTSIG_AXIS="$SAMPLE_AXIS_FILE" \
  "$OCTAVE_BIN" --no-init-all --no-history --no-gui --eval \
    "addpath(getenv('DIALECT_MUTSIG_SOURCE')); run_mutsig_persample(getenv('DIALECT_MUTSIG_MAF'),getenv('DIALECT_MUTSIG_OUT'),getenv('DIALECT_MUTSIG_AXIS'))"

validate_bundle "$STAGING_DIR" || {
  echo "MutSig did not produce a complete, axis-exact lambda bundle" >&2
  exit 74
}

{
  printf 'schema_version\t1\n'
  printf 'cohort\t%s\n' "$C"
  printf 'upstream_commit\t%s\n' "$UPSTREAM_COMMIT"
  printf 'source_tree_sha256\t%s\n' "$SOURCE_TREE_SHA256"
  printf 'source_file_count\t%s\n' "$SOURCE_FILE_COUNT"
  printf 'patch_sha256\t%s\n' "$PATCH_SHA256"
  printf 'runner_sha256\t%s\n' "$RUNNER_SHA256"
  printf 'runtime_sha256\t%s\n' "$RUNTIME_SHA256"
  printf 'maf_sha256\t%s\n' "$MAF_SHA256"
  printf 'sample_axis_sha256\t%s\n' "$SAMPLE_AXIS_SHA256"
  printf 'sample_axis_count\t%s\n' "$SAMPLE_AXIS_COUNT"
  printf 'lambda_sha256\t%s\n' "$(sha256_file "${STAGING_DIR}/persample_lambda.f32")"
  printf 'meta_sha256\t%s\n' "$(sha256_file "${STAGING_DIR}/persample_meta.txt")"
  printf 'genes_sha256\t%s\n' "$(sha256_file "${STAGING_DIR}/persample_genes.txt")"
  printf 'patients_sha256\t%s\n' "$(sha256_file "${STAGING_DIR}/persample_patients.txt")"
} > "${STAGING_DIR}/persample_receipt.tsv"

run_python -c '
import os
import stat
import sys
from pathlib import Path

staging = Path(sys.argv[1])
destination = Path(os.path.abspath(sys.argv[2]))
staging = Path(os.path.abspath(staging))
no_follow = getattr(os, "O_NOFOLLOW", 0)
directory_only = getattr(os, "O_DIRECTORY", 0)
if not no_follow or not directory_only:
    raise SystemExit("MutSig publication requires no-follow directory descriptors")

def open_directory(path: Path) -> int:
    flags = os.O_RDONLY | directory_only | no_follow
    descriptor = None
    try:
        descriptor = os.open(path.anchor, flags)
        for part in path.parts[1:]:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except (OSError, TypeError) as error:
        if descriptor is not None:
            os.close(descriptor)
        raise SystemExit(
            "unable to open MutSig publication directory without symlinks",
        ) from error
    opened = os.fstat(descriptor)
    if not stat.S_ISDIR(opened.st_mode):
        os.close(descriptor)
        raise SystemExit("MutSig publication directory is not a directory")
    return descriptor

def open_regular_at(parent: int, name: str, *, absent_ok: bool = False) -> int | None:
    try:
        descriptor = os.open(name, os.O_RDONLY | no_follow, dir_fd=parent)
    except FileNotFoundError:
        if absent_ok:
            return None
        raise
    except (OSError, TypeError) as error:
        raise SystemExit("MutSig publication encountered an unsafe file") from error
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
        os.close(descriptor)
        raise SystemExit("MutSig publication requires single-link regular files")
    return descriptor

def require_replaceable_at(parent: int, name: str) -> None:
    existing = open_regular_at(parent, name, absent_ok=True)
    if existing is not None:
        os.close(existing)

def remove_regular_at(parent: int, name: str) -> None:
    existing = open_regular_at(parent, name, absent_ok=True)
    if existing is None:
        return
    os.close(existing)
    try:
        os.unlink(name, dir_fd=parent)
    except (OSError, TypeError) as error:
        raise SystemExit("unable to unlink stale MutSig receipt through dirfd") from error

def publish_regular(source_parent: int, destination_parent: int, name: str) -> None:
    source = open_regular_at(source_parent, name)
    assert source is not None
    try:
        staged = os.fstat(source)
        os.fsync(source)
        require_replaceable_at(destination_parent, name)
        try:
            os.replace(
                name,
                name,
                src_dir_fd=source_parent,
                dst_dir_fd=destination_parent,
            )
        except (OSError, TypeError) as error:
            raise SystemExit(
                "unable to atomically publish MutSig artifact through dirfd",
            ) from error
        published = open_regular_at(destination_parent, name)
        assert published is not None
        try:
            observed = os.fstat(published)
            if observed.st_dev != staged.st_dev or observed.st_ino != staged.st_ino:
                raise SystemExit("published MutSig artifact differs from staged file")
            os.fsync(published)
        finally:
            os.close(published)
    finally:
        os.close(source)

def path_matches_directory(path: Path, descriptor: int) -> bool:
    try:
        observed = path.stat(follow_symlinks=False)
    except OSError:
        return False
    opened = os.fstat(descriptor)
    return (
        stat.S_ISDIR(observed.st_mode)
        and observed.st_dev == opened.st_dev
        and observed.st_ino == opened.st_ino
    )

staging_descriptor = open_directory(staging)
destination_parent = open_directory(destination.parent)
destination_descriptor = None
try:
    try:
        os.mkdir(destination.name, mode=0o700, dir_fd=destination_parent)
    except FileExistsError:
        pass
    except (OSError, TypeError) as error:
        raise SystemExit("unable to create MutSig destination through dirfd") from error
    try:
        destination_descriptor = os.open(
            destination.name,
            os.O_RDONLY | directory_only | no_follow,
            dir_fd=destination_parent,
        )
    except (OSError, TypeError) as error:
        raise SystemExit("unable to open MutSig destination through dirfd") from error
    if not stat.S_ISDIR(os.fstat(destination_descriptor).st_mode):
        raise SystemExit("MutSig destination must remain a directory")

    remove_regular_at(destination_descriptor, "persample_receipt.tsv")
    for name in (
        "persample_lambda.f32",
        "persample_meta.txt",
        "persample_genes.txt",
        "persample_patients.txt",
    ):
        publish_regular(staging_descriptor, destination_descriptor, name)
    os.fsync(destination_descriptor)
    publish_regular(
        staging_descriptor,
        destination_descriptor,
        "persample_receipt.tsv",
    )
    os.fsync(destination_descriptor)
    if not path_matches_directory(destination, destination_descriptor):
        raise SystemExit("MutSig destination pathname changed during publication")
finally:
    if destination_descriptor is not None:
        os.close(destination_descriptor)
    os.close(destination_parent)
    os.close(staging_descriptor)
' "$STAGING_DIR" "$OUT_DIR"

if ! receipt_matches_current_inputs; then
  remove_file "$RECEIPT"
  echo "published MutSig bundle failed receipt validation" >&2
  exit 74
fi

echo "DIALECT: published complete MutSig lambda bundle and receipt at ${OUT_DIR}"
