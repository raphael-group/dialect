#!/usr/bin/env bash
# Per-cohort DIALECT pipeline on the uniform cBioPortal PanCancer MAF.
#
# Stages (each receipt-validated before reuse):
#   1. CBaSE  generate   -> output/pancan/<C>/{bmr_pmfs.csv,count_matrix.csv,CBaSE_output}
#   2. DIG    generate   -> output/pancan/<C>/bmr_pmfs.dig.csv   (Pancan DIG model)
#   3. DIALECT identify  -> output/pancan/<C>/id_{cbase,dig}/pairwise...csv
#   4. MutSig2CV source + identify -> output/pancan/<C>/id_mutsig/pairwise...csv
#
# Set PREPARE_ONLY=1 for the signed revision input rebuild. That mode generates
# and receipt-validates CBaSE, DIG, and MutSig inputs but never invokes DIALECT
# identify or creates association outputs.
#
# Any requested stage failure aborts the cohort. Usage: run_cohort_pipeline.sh ACC
# This is the historical provider-specific workflow, not the frozen revision
# analysis. Use analysis.run_tcga_revision_k500 for the common-universe K=500 grid.
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "usage: $0 <COHORT>" >&2
  exit 64
fi
if [ "${PREPARE_ONLY+x}" = "x" ] && [ "$PREPARE_ONLY" != "1" ]; then
  echo "PREPARE_ONLY must be unset or exactly 1" >&2
  exit 64
fi
if [ "${PREPARE_ONLY:-}" = "1" ] && [ -n "${SKIP_MUTSIG:-}" ]; then
  echo "PREPARE_ONLY=1 forbids SKIP_MUTSIG" >&2
  exit 64
fi
C="$1"
SCRIPT_PATH="${BASH_SOURCE[0]}"
case "$SCRIPT_PATH" in
  /*) ;;
  *) SCRIPT_PATH="${PWD}/${SCRIPT_PATH}" ;;
esac
SCRIPT_DIR="$(cd -- "${SCRIPT_PATH%/*}" && pwd -P)"
REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "$REPO" || exit 70
# Parameterized (defaults = TCGA pancan); MSK runs set MAF_DIR + ROOT in the environment.
ROOT="${ROOT:-output/pancan}"
MAF_DIR="${MAF_DIR:-data/mafs_pancan}"
MAF="${MAF_DIR}/${C}.maf"
MUTSIG_ROOT="${MUTSIG_ROOT:-output/mutsigsrc}"
MUTSIG_SAMPLE_AXIS_FILE="${MUTSIG_SAMPLE_AXIS_FILE:-${ROOT}/${C}/sample_axis.txt}"
# Top-K genes for the identify stage. K=100 keeps the historical id_{cbase,dig,mutsig}
# directory names; any other K writes to id_<bmr>_k<K> so runs at different K coexist.
TOP_K="${TOP_K:-100}"
if [ "$TOP_K" = "100" ]; then IDSUF=""; else IDSUF="_k${TOP_K}"; fi
DIG_RESULTS="external/DIGDriver/run/Pancan.genes.results.txt"
DEFAULT_PY="/opt/anaconda3/envs/dialect/bin/python"
if [ "${PREPARE_ONLY:-}" = "1" ]; then
  : "${DIALECT_PROVIDER_PYTHON:?missing provider Python authority}"
  : "${DIALECT_PROVIDER_PYTHON_SHA256:?missing provider Python SHA-256}"
  : "${DIALECT_PROVIDER_PYTHON_RUNTIME_SHA256:?missing provider Python runtime SHA-256}"
  : "${DIALECT_PROVIDER_DIALECT_TREE_SHA256:?missing DIALECT tree SHA-256}"
  : "${DIALECT_PROVIDER_BASH:?missing provider Bash authority}"
  : "${DIALECT_PROVIDER_BASH_SHA256:?missing provider Bash SHA-256}"
  PY="$DIALECT_PROVIDER_PYTHON"
  BASH_RUNNER="$DIALECT_PROVIDER_BASH"
else
  PY="$DEFAULT_PY"
  BASH_RUNNER="bash"
fi
[ -x "$PY" ] || { echo "pinned Python is not executable: ${PY}" >&2; exit 69; }

run_python() {
  PYTHONPATH="${REPO}:${REPO}/src" PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
    "$PY" -P -s "$@"
}

run_dialect() {
  run_python -m dialect "$@"
}

log() { printf '[%s] %s\n' "$C" "$*"; }

sha256_file() {
  run_python -c '
import hashlib
import os
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
digest = hashlib.sha256()
descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
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

tree_sha256() {
  run_python -c '
import hashlib
import os
import stat
import struct
import sys
from pathlib import Path

root = Path(sys.argv[1])
python_only = sys.argv[2] == "python-only"
digest = hashlib.sha256()
files = []
for path in root.rglob("*"):
    relative = path.relative_to(root)
    if "__pycache__" in relative.parts:
        continue
    opened = path.lstat()
    if stat.S_ISLNK(opened.st_mode):
        raise SystemExit(f"source/input tree contains a symlink: {relative}")
    if stat.S_ISDIR(opened.st_mode):
        continue
    if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
        raise SystemExit(f"source/input tree contains a non-private file: {relative}")
    if not python_only or path.suffix == ".py":
        files.append((relative.as_posix(), path))
if not files:
    raise SystemExit(f"empty source/input directory: {root}")
for relative, path in sorted(files):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise SystemExit(f"source/input is no longer a private file: {relative}")
        encoded = relative.encode()
        mode = 0o500 if stat.S_IMODE(before.st_mode) & 0o111 else 0o400
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack(">Q", mode))
        digest.update(struct.pack(">Q", before.st_size))
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
        or before.st_nlink != 1
        or after.st_nlink != 1
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
        or observed_bytes != before.st_size
    ):
        raise SystemExit(f"source/input changed while hashing: {relative}")
print(digest.hexdigest())
' "$1" "$2"
}

source_tree_sha256() {
  tree_sha256 "$1" python-only
}

directory_files_sha256() {
  tree_sha256 "$1" all-files
}

files_sha256() {
  local path
  for path in "$@"; do
    [ -s "$path" ] || return 1
    printf '%s\t%s\n' "${path##*/}" "$(sha256_file "$path")"
  done | sha256_stream
}

cbase_outputs_sha256() {
  local files=("$CBASE_BMR" "$COUNT_MATRIX")
  [ ! -e "$CB_Q" ] || files+=("$CB_Q")
  files_sha256 "${files[@]}"
}

receipt_value() {
  run_python -c '
import sys
from pathlib import Path

lines = Path(sys.argv[1]).read_text(encoding="ascii").splitlines()
rows = [line.split("\t") for line in lines]
if any(len(row) != 2 for row in rows):
    raise SystemExit(1)
matches = [row[1] for row in rows if row[0] == sys.argv[2]]
if len(matches) != 1:
    raise SystemExit(1)
print(matches[0])
' "$1" "$2"
}

receipt_matches() {
  local receipt="$1"
  local input_sha="$2"
  local output_sha="$3"
  [ -s "$receipt" ] \
    && [ "$(receipt_value "$receipt" schema_version)" = "1" ] \
    && [ "$(receipt_value "$receipt" input_sha256)" = "$input_sha" ] \
    && [ "$(receipt_value "$receipt" output_sha256)" = "$output_sha" ]
}

publish_receipt() {
  local receipt="$1"
  local input_sha="$2"
  local output_sha="$3"
  run_python -c '
import os
import stat
import sys
from pathlib import Path

destination = Path(os.path.abspath(sys.argv[1]))
temporary_name = f"{destination.name}.tmp.{os.getpid()}"
payload = (
    "schema_version\t1\n"
    f"input_sha256\t{sys.argv[2]}\n"
    f"output_sha256\t{sys.argv[3]}\n"
).encode("ascii")

no_follow = getattr(os, "O_NOFOLLOW", 0)
directory_only = getattr(os, "O_DIRECTORY", 0)
if not no_follow or not directory_only:
    raise SystemExit("receipt publication requires no-follow directory descriptors")

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
            "unable to open receipt parent through no-follow descriptors",
        ) from error
    opened = os.fstat(descriptor)
    if not stat.S_ISDIR(opened.st_mode):
        os.close(descriptor)
        raise SystemExit("receipt parent must be a directory")
    return descriptor

def open_regular_at(parent: int, name: str, *, absent_ok: bool = False) -> int | None:
    try:
        descriptor = os.open(name, os.O_RDONLY | no_follow, dir_fd=parent)
    except FileNotFoundError:
        if absent_ok:
            return None
        raise
    except (OSError, TypeError) as error:
        raise SystemExit("receipt publication encountered an unsafe file") from error
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
        os.close(descriptor)
        raise SystemExit("receipt publication requires single-link regular files")
    return descriptor

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

parent_descriptor = open_directory(destination.parent)
temporary_descriptor = None
try:
    try:
        temporary_descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow,
            0o600,
            dir_fd=parent_descriptor,
        )
    except (OSError, TypeError) as error:
        raise SystemExit("unable to create private staged receipt") from error
    before = os.fstat(temporary_descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise SystemExit("staged receipt must be a single-link regular file")
    view = memoryview(payload)
    while view:
        written = os.write(temporary_descriptor, view)
        if written <= 0:
            raise SystemExit("short write while staging receipt")
        view = view[written:]
    os.fsync(temporary_descriptor)
    staged = os.fstat(temporary_descriptor)
    if (
        staged.st_dev != before.st_dev
        or staged.st_ino != before.st_ino
        or not stat.S_ISREG(staged.st_mode)
        or staged.st_nlink != 1
        or staged.st_size != len(payload)
    ):
        raise SystemExit("staged receipt identity changed before publication")

    existing = open_regular_at(
        parent_descriptor,
        destination.name,
        absent_ok=True,
    )
    if existing is not None:
        os.close(existing)
    try:
        os.replace(
            temporary_name,
            destination.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
    except (OSError, TypeError) as error:
        raise SystemExit("unable to atomically publish receipt through dirfd") from error
    published = open_regular_at(parent_descriptor, destination.name)
    assert published is not None
    try:
        observed = os.fstat(published)
        if observed.st_dev != staged.st_dev or observed.st_ino != staged.st_ino:
            raise SystemExit("published receipt is not the staged receipt")
        os.fsync(published)
    finally:
        os.close(published)
    os.fsync(parent_descriptor)
    if not path_matches_directory(destination.parent, parent_descriptor):
        raise SystemExit("receipt parent pathname changed during publication")
finally:
    if temporary_descriptor is not None:
        os.close(temporary_descriptor)
    try:
        os.unlink(temporary_name, dir_fd=parent_descriptor)
    except FileNotFoundError:
        pass
    except (OSError, TypeError) as error:
        raise SystemExit("unable to clean staged receipt through dirfd") from error
    os.close(parent_descriptor)
' "$receipt" "$input_sha" "$output_sha"
}

count_axis_matches() {
  run_python -c '
import csv
import os
import stat
import sys
from io import StringIO
from pathlib import Path

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

rows = list(csv.reader(StringIO(stable_read(Path(sys.argv[1])).decode("utf-8"))))
actual = [row[0] for row in rows[1:] if row]
expected = stable_read(Path(sys.argv[2])).decode("utf-8").splitlines()
raise SystemExit(actual != expected)
' "$1" "$2"
}

make_directory() {
  run_python -c '
import sys
from pathlib import Path

Path(sys.argv[1]).mkdir(parents=True, exist_ok=True)
' "$1"
}

remove_file() {
  run_python -c '
import sys
from pathlib import Path

Path(sys.argv[1]).unlink(missing_ok=True)
' "$1"
}

publish_axis_if_absent() {
  run_python -c '
import os
import stat
import sys
from pathlib import Path

def stable_read(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise SystemExit(74)
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
        raise SystemExit(74)
    return raw

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
source_bytes = stable_read(source)
if destination.exists():
    if stable_read(destination) != source_bytes:
        raise SystemExit(74)
    raise SystemExit(0)
temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(source_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, destination, follow_symlinks=False)
finally:
    temporary.unlink(missing_ok=True)
' "$1" "$2"
}

line_count() {
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
' "$1"
}

verify_runtime_authority() {
  run_python -c '
import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path, PurePosixPath

authority_path = Path(sys.argv[1]).absolute()
expected_path = (Path(sys.argv[2]).absolute() / "runtime" / "authority.json")
if authority_path != expected_path:
    raise SystemExit("runtime authority path differs from the execution snapshot")
descriptor = os.open(authority_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
try:
    before = os.fstat(descriptor)
    chunks = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    after = os.fstat(descriptor)
finally:
    os.close(descriptor)
raw = b"".join(chunks)
if (
    not stat.S_ISREG(before.st_mode)
    or before.st_nlink != 1
    or stat.S_IMODE(before.st_mode) != 0o400
    or before.st_dev != after.st_dev
    or before.st_ino != after.st_ino
    or before.st_mode != after.st_mode
    or after.st_nlink != 1
    or before.st_size != after.st_size
    or before.st_mtime_ns != after.st_mtime_ns
    or before.st_ctime_ns != after.st_ctime_ns
    or len(raw) != before.st_size
    or hashlib.sha256(raw).hexdigest()
    != os.environ["DIALECT_PROVIDER_RUNTIME_AUTHORITY_SHA256"]
):
    raise SystemExit("runtime authority file failed stable same-descriptor validation")

def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate runtime authority key: {key}")
        value[key] = item
    return value

def finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite runtime authority number")
    return parsed

try:
    payload = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant: {value}"),
        ),
        parse_float=finite_float,
    )
except (UnicodeDecodeError, ValueError) as error:
    raise SystemExit("runtime authority is not unambiguous canonical JSON") from error
canonical = json.dumps(
    payload,
    allow_nan=False,
    ensure_ascii=True,
    sort_keys=True,
    separators=(",", ":"),
).encode() + b"\n"
if raw != canonical:
    raise SystemExit("runtime authority is not canonical JSON with one LF")
if set(payload) != {
    "schema_version",
    "contract",
    "tools",
    "python_runtime",
    "mutsig_runtime",
} or payload["schema_version"] != "1.0.0" or payload["contract"] != "provider-child-runtime-authority-v1":
    raise SystemExit("runtime authority top-level schema is invalid")
tools = payload["tools"]
if not isinstance(tools, dict) or set(tools) != {"bash", "git", "java", "nice", "octave", "python"}:
    raise SystemExit("runtime authority tool closure is invalid")
environment_names = {
    "bash": ("DIALECT_PROVIDER_BASH", "DIALECT_PROVIDER_BASH_SHA256"),
    "git": ("DIALECT_PROVIDER_GIT", "DIALECT_PROVIDER_GIT_SHA256"),
    "java": ("DIALECT_PROVIDER_JAVA", "DIALECT_PROVIDER_JAVA_SHA256"),
    "nice": ("DIALECT_PROVIDER_NICE", "DIALECT_PROVIDER_NICE_SHA256"),
    "octave": ("DIALECT_PROVIDER_OCTAVE", "DIALECT_PROVIDER_OCTAVE_SHA256"),
    "python": ("DIALECT_PROVIDER_PYTHON", "DIALECT_PROVIDER_PYTHON_SHA256"),
}
for name, record in tools.items():
    if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
        raise SystemExit(f"runtime authority {name} record is invalid")
    path = Path(record["path"])
    path_name, hash_name = environment_names[name]
    if (
        not path.is_absolute()
        or path.as_posix() != record["path"]
        or record["path"] != os.environ[path_name]
        or record["sha256"] != os.environ[hash_name]
        or not isinstance(record["bytes"], int)
        or isinstance(record["bytes"], bool)
        or record["bytes"] <= 0
    ):
        raise SystemExit(f"runtime authority {name} environment binding is invalid")
    tool_fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        tool_before = os.fstat(tool_fd)
        digest = hashlib.sha256()
        byte_count = 0
        while chunk := os.read(tool_fd, 1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
        tool_after = os.fstat(tool_fd)
    finally:
        os.close(tool_fd)
    if (
        not stat.S_ISREG(tool_before.st_mode)
        or tool_before.st_nlink != 1
        or tool_before.st_dev != tool_after.st_dev
        or tool_before.st_ino != tool_after.st_ino
        or tool_before.st_mode != tool_after.st_mode
        or tool_after.st_nlink != 1
        or tool_before.st_size != tool_after.st_size
        or tool_before.st_mtime_ns != tool_after.st_mtime_ns
        or tool_before.st_ctime_ns != tool_after.st_ctime_ns
        or byte_count != tool_before.st_size
        or byte_count != record["bytes"]
        or digest.hexdigest() != record["sha256"]
    ):
        raise SystemExit(f"runtime authority {name} executable changed")

python_runtime = payload["python_runtime"]
expected_python_keys = {
    "launcher",
    "entrypoint_shebang",
    "python_executable",
    "dialect_entrypoint",
    "dialect_import",
    "dialect_tree_hash_contract",
    "dialect_tree_sha256",
    "imported_modules",
    "distributions",
    "versions",
    "runtime_sha256",
}
if not isinstance(python_runtime, dict) or set(python_runtime) != expected_python_keys:
    raise SystemExit("Python runtime authority schema is invalid")
python_digest_payload = dict(python_runtime)
python_digest = python_digest_payload.pop("runtime_sha256")
if (
    python_runtime["python_executable"] != tools["python"]
    or python_runtime["dialect_tree_sha256"]
    != os.environ["DIALECT_PROVIDER_DIALECT_TREE_SHA256"]
    or python_digest != os.environ["DIALECT_PROVIDER_PYTHON_RUNTIME_SHA256"]
    or hashlib.sha256(
        json.dumps(
            python_digest_payload,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
    ).hexdigest()
    != python_digest
):
    raise SystemExit("Python runtime authority digest is invalid")
dialect_import = python_runtime["dialect_import"]
if not isinstance(dialect_import, dict) or set(dialect_import) != {"path", "bytes", "sha256"}:
    raise SystemExit("DIALECT import authority is invalid")
parts = PurePosixPath(dialect_import["path"]).parts
matches = [index for index in range(len(parts) - 1) if parts[index:index + 2] == ("src", "dialect")]
if len(matches) != 1:
    raise SystemExit("DIALECT import authority path is not repository-relative")
dialect_relative = PurePosixPath(*parts[matches[0]:]).as_posix()

mutsig_runtime = payload["mutsig_runtime"]
if not isinstance(mutsig_runtime, dict) or set(mutsig_runtime) != {
    "octave",
    "octave_id",
    "java_home",
    "java_executable",
    "java_id",
    "runtime_sha256",
}:
    raise SystemExit("MutSig runtime authority schema is invalid")
mutsig_digest_payload = dict(mutsig_runtime)
mutsig_digest = mutsig_digest_payload.pop("runtime_sha256")
if (
    mutsig_runtime["octave"] != tools["octave"]
    or mutsig_runtime["java_executable"] != tools["java"]
    or mutsig_runtime["octave_id"] != os.environ["DIALECT_PROVIDER_OCTAVE_ID"]
    or mutsig_runtime["java_home"] != os.environ["DIALECT_PROVIDER_JAVA_HOME"]
    or mutsig_runtime["java_id"] != os.environ["DIALECT_PROVIDER_JAVA_ID"]
    or mutsig_digest != os.environ["DIALECT_PROVIDER_MUTSIG_RUNTIME_SHA256"]
    or hashlib.sha256(
        json.dumps(
            mutsig_digest_payload,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
    ).hexdigest()
    != mutsig_digest
):
    raise SystemExit("MutSig runtime authority digest is invalid")
print(dialect_relative)
print(dialect_import["sha256"])
' "$DIALECT_PROVIDER_RUNTIME_AUTHORITY_FILE" "$REPO"
}

if [ "${PREPARE_ONLY:-}" = "1" ]; then
  : "${DIALECT_PROVIDER_NICE:?missing provider nice authority}"
  : "${DIALECT_PROVIDER_NICE_SHA256:?missing provider nice SHA-256}"
  : "${DIALECT_PROVIDER_CBASE_INPUTS_TREE_SHA256:?missing CBaSE input-tree authority}"
  : "${DIALECT_PROVIDER_RUNTIME_AUTHORITY_FILE:?missing runtime authority file}"
  : "${DIALECT_PROVIDER_RUNTIME_AUTHORITY_SHA256:?missing runtime authority SHA-256}"
  RUNTIME_AUTHORITY_IDENTITY=()
  while IFS= read -r identity_line; do
    RUNTIME_AUTHORITY_IDENTITY+=("$identity_line")
  done < <(verify_runtime_authority)
  [ "${#RUNTIME_AUTHORITY_IDENTITY[@]}" -eq 2 ] || {
    echo "provider runtime authority failed closed validation" >&2
    exit 70
  }
  AUTHORIZED_DIALECT_RELATIVE="${RUNTIME_AUTHORITY_IDENTITY[0]}"
  AUTHORIZED_DIALECT_SHA256="${RUNTIME_AUTHORITY_IDENTITY[1]}"
fi

if ! DIALECT_SOURCE_SHA256="$(source_tree_sha256 "${REPO}/src/dialect")"; then
  echo "DIALECT source tree failed closed validation" >&2
  exit 70
fi
if [ "${PREPARE_ONLY:-}" = "1" ] \
  && [ "$DIALECT_SOURCE_SHA256" != "$DIALECT_PROVIDER_DIALECT_TREE_SHA256" ]; then
  echo "DIALECT source tree differs from provider authority" >&2
  exit 70
fi

PY_IDENTITY=()
while IFS= read -r identity_line; do
  PY_IDENTITY+=("$identity_line")
done < <(
  run_python -c '
import hashlib
import os
import platform
import stat
import sys
from pathlib import Path

python_path = Path(sys.executable).resolve()
requested_python = Path(sys.argv[1]).resolve()
source_root_unresolved = Path(sys.argv[2]).absolute()
source_root = source_root_unresolved.resolve()
if python_path != requested_python:
    raise SystemExit("Python executable resolved outside its pinned path")
if source_root != source_root_unresolved:
    raise SystemExit("DIALECT source root traverses a symlink")
import dialect
import numpy
import pandas
import scipy

module_path = Path(dialect.__file__).resolve()
if module_path == source_root or not module_path.is_relative_to(source_root):
    raise SystemExit("DIALECT import resolved outside the pinned repository source")

def digest(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise SystemExit("runtime file must be a single-link regular file")
        value = hashlib.sha256()
        observed_bytes = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            value.update(chunk)
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
        raise SystemExit("runtime file changed during stable descriptor read")
    return value.hexdigest()

versions = " ".join(
    (
        platform.python_version(),
        numpy.__version__,
        pandas.__version__,
        scipy.__version__,
    ),
)
print(python_path.as_posix())
print(digest(python_path))
print(module_path.as_posix())
print(digest(module_path))
print(versions)
print(hashlib.sha256((versions + "\n").encode()).hexdigest())
' "$PY" "${REPO}/src/dialect"
)
[ "${#PY_IDENTITY[@]}" -eq 6 ] || {
  echo "pinned Python/DIALECT identity probe returned an invalid record" >&2
  exit 70
}
PY_RESOLVED="${PY_IDENTITY[0]}"
PY_EXECUTABLE_SHA256="${PY_IDENTITY[1]}"
DIALECT_MODULE_PATH="${PY_IDENTITY[2]}"
DIALECT_MODULE_SHA256="${PY_IDENTITY[3]}"
PY_VERSION_ID="${PY_IDENTITY[4]}"
PY_CORE_RUNTIME_SHA256="${PY_IDENTITY[5]}"
if [ "${PREPARE_ONLY:-}" = "1" ]; then
  DIALECT_MODULE_RELATIVE="${DIALECT_MODULE_PATH#${REPO}/}"
  [ "$DIALECT_MODULE_RELATIVE" = "$AUTHORIZED_DIALECT_RELATIVE" ] \
    && [ "$DIALECT_MODULE_SHA256" = "$AUTHORIZED_DIALECT_SHA256" ] || {
      echo "DIALECT module differs from provider runtime authority" >&2
      exit 70
    }
fi

[ -f "$MAF" ] || { log "no MAF at ${MAF}"; exit 66; }
[ -f "$MUTSIG_SAMPLE_AXIS_FILE" ] || {
  log "no exact sample axis at ${MUTSIG_SAMPLE_AXIS_FILE}; refusing mutation-derived cohort"
  exit 66
}
if [ "${PREPARE_ONLY:-}" = "1" ]; then
  [ "$PY_RESOLVED" = "$DIALECT_PROVIDER_PYTHON" ] || {
    echo "resolved Python differs from provider authority" >&2
    exit 70
  }
  [ "$PY_EXECUTABLE_SHA256" = "$DIALECT_PROVIDER_PYTHON_SHA256" ] || {
    echo "Python executable differs from provider authority" >&2
    exit 70
  }
  BASH_IDENTITY=()
  while IFS= read -r identity_line; do
    BASH_IDENTITY+=("$identity_line")
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
        raise SystemExit("Bash executable must be a single-link regular file")
    digest = hashlib.sha256()
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
    raise SystemExit("Bash executable changed during stable descriptor read")
print(path.as_posix())
print(digest.hexdigest())
' "$BASH_RUNNER"
  )
  [ "${#BASH_IDENTITY[@]}" -eq 2 ] \
    && [ "${BASH_IDENTITY[0]}" = "$DIALECT_PROVIDER_BASH" ] \
    && [ "${BASH_IDENTITY[1]}" = "$DIALECT_PROVIDER_BASH_SHA256" ] || {
      echo "Bash executable differs from provider authority" >&2
      exit 70
    }
fi
PIPELINE_SHA256="$(sha256_file "${SCRIPT_DIR}/run_cohort_pipeline.sh")"
MAF_SHA256="$(sha256_file "$MAF")"
SAMPLE_AXIS_SHA256="$(sha256_file "$MUTSIG_SAMPLE_AXIS_FILE")"
if [ "${PREPARE_ONLY:-}" = "1" ]; then
  CBASE_INPUTS_SHA256="$DIALECT_PROVIDER_CBASE_INPUTS_TREE_SHA256"
else
  CBASE_INPUTS_SHA256="$(directory_files_sha256 "${REPO}/external/CBaSE")"
fi
if [ "${PREPARE_ONLY:-}" = "1" ]; then
  PY_RUNTIME_SHA256="$DIALECT_PROVIDER_PYTHON_RUNTIME_SHA256"
else
  PY_RUNTIME_SHA256="$PY_CORE_RUNTIME_SHA256"
fi
make_directory "${ROOT}/${C}"
AUTHORITATIVE_AXIS="${ROOT}/${C}/sample_axis.txt"
if [ "$MUTSIG_SAMPLE_AXIS_FILE" != "$AUTHORITATIVE_AXIS" ]; then
  publish_axis_if_absent "$MUTSIG_SAMPLE_AXIS_FILE" "$AUTHORITATIVE_AXIS" || {
    log "existing provider sample axis differs from the signed source axis"
    exit 74
  }
fi
MUTSIG_SAMPLE_AXIS_FILE="$AUTHORITATIVE_AXIS"
LOGF="${ROOT}/${C}/pipeline.log"; : > "$LOGF"  # per-cohort verbose log (keeps main log small)

# 1. CBaSE ------------------------------------------------------------------
CBASE_BMR="${ROOT}/${C}/bmr_pmfs.csv"
COUNT_MATRIX="${ROOT}/${C}/count_matrix.csv"
CB_Q="${ROOT}/${C}/CBaSE_output/q_values.txt"
CBASE_RECEIPT="${ROOT}/${C}/cbase_stage_receipt.tsv"
CBASE_INPUT_SHA256="$(
  printf '%s\n' "$PIPELINE_SHA256" "$MAF_SHA256" "$SAMPLE_AXIS_SHA256" \
    "$DIALECT_SOURCE_SHA256" "$CBASE_INPUTS_SHA256" "$PY_RUNTIME_SHA256" \
    "hg19" | sha256_stream
)"
CBASE_OUTPUT_SHA256="$(cbase_outputs_sha256 2>/dev/null || true)"
if receipt_matches "$CBASE_RECEIPT" "$CBASE_INPUT_SHA256" "$CBASE_OUTPUT_SHA256" \
   && count_axis_matches "$COUNT_MATRIX" "$MUTSIG_SAMPLE_AXIS_FILE"; then
  log "skip cbase (receipt current)"
else
  log "CBaSE generate"
  remove_file "$CBASE_RECEIPT"
  if ! run_dialect generate -m "$MAF" -o "${ROOT}/${C}" --bmr cbase -r hg19 \
       --cbase-sample-axis "$MUTSIG_SAMPLE_AXIS_FILE" >>"$LOGF" 2>&1; then
    log "STAGE-FAIL cbase"
    exit 74
  fi
  if ! count_axis_matches "$COUNT_MATRIX" "$MUTSIG_SAMPLE_AXIS_FILE"; then
    log "CBaSE count matrix does not preserve the exact sample axis"
    exit 74
  fi
  CBASE_OUTPUT_SHA256="$(cbase_outputs_sha256)" || exit 74
  publish_receipt "$CBASE_RECEIPT" "$CBASE_INPUT_SHA256" "$CBASE_OUTPUT_SHA256"
fi

N="$(line_count "$MUTSIG_SAMPLE_AXIS_FILE")"
[ "$N" -gt 0 ] || { log "exact sample axis is empty"; exit 74; }

# 2. DIG (writes bmr_pmfs.dig.csv directly; does not clobber CBaSE's BMR) ---
[ -f "$DIG_RESULTS" ] || { log "no DIG results at ${DIG_RESULTS}"; exit 66; }
DIG_BMR="${ROOT}/${C}/bmr_pmfs.dig.csv"
DIG_RECEIPT="${ROOT}/${C}/dig_stage_receipt.tsv"
DIG_RESULTS_SHA256="$(sha256_file "$DIG_RESULTS")"
COUNT_MATRIX_SHA256="$(sha256_file "$COUNT_MATRIX")"
DIG_INPUT_SHA256="$(
  printf '%s\n' "$PIPELINE_SHA256" "$MAF_SHA256" "$SAMPLE_AXIS_SHA256" \
    "$COUNT_MATRIX_SHA256" "$DIALECT_SOURCE_SHA256" "$DIG_RESULTS_SHA256" \
    "$PY_RUNTIME_SHA256" "$N" "hg19" | sha256_stream
)"
DIG_OUTPUT_SHA256="$(files_sha256 "$DIG_BMR" 2>/dev/null || true)"
if receipt_matches "$DIG_RECEIPT" "$DIG_INPUT_SHA256" "$DIG_OUTPUT_SHA256"; then
  log "skip dig (receipt current)"
else
  log "DIG generate (N=${N})"
  remove_file "$DIG_RECEIPT"
  if ! run_dialect generate -m "$MAF" -o "${ROOT}/${C}" --bmr dig \
       --dig-results "$DIG_RESULTS" --dig-samples "$N" -r hg19 >>"$LOGF" 2>&1; then
    log "STAGE-FAIL dig"
    exit 74
  fi
  DIG_OUTPUT_SHA256="$(files_sha256 "$DIG_BMR")" || exit 74
  publish_receipt "$DIG_RECEIPT" "$DIG_INPUT_SHA256" "$DIG_OUTPUT_SHA256"
fi

# 3. DIALECT identify -- CBaSE + DIG (fast; run before the slow MutSig) ------
if [ -z "${PREPARE_ONLY:-}" ]; then
CB_Q_SHA256="none"
[ ! -f "$CB_Q" ] || CB_Q_SHA256="$(sha256_file "$CB_Q")"
CBASE_ID_DIR="${ROOT}/${C}/id_cbase${IDSUF}"
CBASE_ID_RESULT="${CBASE_ID_DIR}/pairwise_interaction_results.csv"
CBASE_ID_RECEIPT="${CBASE_ID_DIR}/identify_stage_receipt.tsv"
CBASE_ID_INPUT_SHA256="$(
  printf '%s\n' "$PIPELINE_SHA256" "$DIALECT_SOURCE_SHA256" "$TOP_K" \
    "$PY_RUNTIME_SHA256" "$COUNT_MATRIX_SHA256" "$CBASE_OUTPUT_SHA256" \
    "$CB_Q_SHA256" | sha256_stream
)"
CBASE_ID_OUTPUT_SHA256="$(files_sha256 "$CBASE_ID_RESULT" 2>/dev/null || true)"
if receipt_matches "$CBASE_ID_RECEIPT" "$CBASE_ID_INPUT_SHA256" "$CBASE_ID_OUTPUT_SHA256"; then
  log "skip identify cbase (receipt current)"
else
  log "identify cbase"
  mkdir -p "$CBASE_ID_DIR"
  rm -f -- "$CBASE_ID_RECEIPT"
  cb_arg=(); [ -f "$CB_Q" ] && cb_arg=(-cb "$CB_Q")
  if ! run_dialect identify -c "$COUNT_MATRIX" -b "$CBASE_BMR" \
       -o "$CBASE_ID_DIR" -k "$TOP_K" "${cb_arg[@]+"${cb_arg[@]}"}" >>"$LOGF" 2>&1; then
    log "STAGE-FAIL id_cbase${IDSUF}"
    exit 74
  fi
  CBASE_ID_OUTPUT_SHA256="$(files_sha256 "$CBASE_ID_RESULT")" || exit 74
  publish_receipt "$CBASE_ID_RECEIPT" "$CBASE_ID_INPUT_SHA256" "$CBASE_ID_OUTPUT_SHA256"
fi

DIG_ID_DIR="${ROOT}/${C}/id_dig${IDSUF}"
DIG_ID_RESULT="${DIG_ID_DIR}/pairwise_interaction_results.csv"
DIG_ID_RECEIPT="${DIG_ID_DIR}/identify_stage_receipt.tsv"
DIG_ID_INPUT_SHA256="$(
  printf '%s\n' "$PIPELINE_SHA256" "$DIALECT_SOURCE_SHA256" "$TOP_K" \
    "$PY_RUNTIME_SHA256" "$COUNT_MATRIX_SHA256" "$DIG_OUTPUT_SHA256" | sha256_stream
)"
DIG_ID_OUTPUT_SHA256="$(files_sha256 "$DIG_ID_RESULT" 2>/dev/null || true)"
if receipt_matches "$DIG_ID_RECEIPT" "$DIG_ID_INPUT_SHA256" "$DIG_ID_OUTPUT_SHA256"; then
  log "skip identify dig (receipt current)"
else
  log "identify dig"
  mkdir -p "$DIG_ID_DIR"
  rm -f -- "$DIG_ID_RECEIPT"
  if ! run_dialect identify -c "$COUNT_MATRIX" -b "$DIG_BMR" \
       -o "$DIG_ID_DIR" -k "$TOP_K" >>"$LOGF" 2>&1; then
    log "STAGE-FAIL id_dig${IDSUF}"
    exit 74
  fi
  DIG_ID_OUTPUT_SHA256="$(files_sha256 "$DIG_ID_RESULT")" || exit 74
  publish_receipt "$DIG_ID_RECEIPT" "$DIG_ID_INPUT_SHA256" "$DIG_ID_OUTPUT_SHA256"
fi
else
  log "prepare-only: association identify stages remain sealed"
fi

# 4. MutSig2CV (Octave-patched source -> native per-sample lambda) + identify --
if [ -n "${SKIP_MUTSIG:-}" ]; then
  log "skip mutsig (SKIP_MUTSIG set)"
else
  log "MutSig2CV (validate receipt or rebuild atomically)"
  if ! "$BASH_RUNNER" scripts/run_mutsig_octave.sh "$C" "$MUTSIG_SAMPLE_AXIS_FILE" \
       "$MAF_DIR" "$MUTSIG_ROOT" >>"$LOGF" 2>&1; then
    log "STAGE-FAIL mutsig"
    exit 74
  fi

  MUTSIG_RECEIPT="${MUTSIG_ROOT}/${C}/persample_receipt.tsv"
  [ -s "$MUTSIG_RECEIPT" ] || { log "MutSig receipt is missing"; exit 74; }
  if [ -n "${PREPARE_ONLY:-}" ]; then
    log "prepare-only: provider inputs DONE; MutSig identify remains sealed"
    exit 0
  fi
  MUTSIG_RECEIPT_SHA256="$(sha256_file "$MUTSIG_RECEIPT")"
  MUTSIG_ANALYSIS_SHA256="$(sha256_file "${REPO}/analysis/mutsig_lambda_co.py")"
  MUTSIG_ID_DIR="${ROOT}/${C}/id_mutsig${IDSUF}"
  MUTSIG_ID_RESULT="${MUTSIG_ID_DIR}/pairwise_interaction_results.csv"
  MUTSIG_ID_RECEIPT="${MUTSIG_ID_DIR}/identify_stage_receipt.tsv"
  MUTSIG_ID_INPUT_SHA256="$(
    printf '%s\n' "$PIPELINE_SHA256" "$DIALECT_SOURCE_SHA256" "$MUTSIG_ANALYSIS_SHA256" \
      "$PY_RUNTIME_SHA256" "$TOP_K" "$COUNT_MATRIX_SHA256" "$CBASE_OUTPUT_SHA256" \
      "$MUTSIG_RECEIPT_SHA256" | sha256_stream
  )"
  MUTSIG_ID_OUTPUT_SHA256="$(files_sha256 "$MUTSIG_ID_RESULT" 2>/dev/null || true)"
  if receipt_matches "$MUTSIG_ID_RECEIPT" "$MUTSIG_ID_INPUT_SHA256" "$MUTSIG_ID_OUTPUT_SHA256"; then
    log "skip identify mutsig (receipt current)"
  else
    log "identify mutsig (native per-sample lambda)"
    mkdir -p "$MUTSIG_ID_DIR"
    rm -f -- "$MUTSIG_ID_RECEIPT"
    if ! run_python -m analysis.mutsig_lambda_co --cohort "$C" --results-root "$ROOT" \
         --mutsig-root "$MUTSIG_ROOT" --suffix "mutsig${IDSUF}" \
         -k "$TOP_K" >>"$LOGF" 2>&1; then
      log "STAGE-FAIL id_mutsig${IDSUF}"
      exit 74
    fi
    MUTSIG_ID_OUTPUT_SHA256="$(files_sha256 "$MUTSIG_ID_RESULT")" || exit 74
    publish_receipt "$MUTSIG_ID_RECEIPT" "$MUTSIG_ID_INPUT_SHA256" "$MUTSIG_ID_OUTPUT_SHA256"
  fi
fi
log "cohort pipeline DONE"
