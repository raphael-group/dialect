"""Focused tests for the role-generic native document derivation launcher."""

from __future__ import annotations

import contextlib
import hashlib
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from analysis import build_tcga_revision_rebuttal_native_producer as native
from analysis import build_tcga_revision_rendered_document_machine_closure as machine
from analysis import tcga_revision_document_roles as roles

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

ROLES = ("clean", "marked", "s1", "rebuttal")
SUBPROCESS_TIMEOUT_SECONDS = 15.0
SUBPROCESS_CLEANUP_TIMEOUT_SECONDS = 2.0
ROLE_MISMATCHES = tuple(
    (compiled_role, requested_role)
    for compiled_role in ROLES
    for requested_role in ROLES
    if compiled_role != requested_role
)
SOURCE = Path(native.__file__).parents[1] / roles.SHARED_LAUNCHER_SOURCE_MEMBER
LEGACY_SOURCE = Path(native.__file__).parent / "native/rebuttal_derivation_launcher.c"
REQUIRES_ARM64_DARWIN = pytest.mark.skipif(
    sys.platform != "darwin" or os.uname().machine != "arm64",
    reason="native launcher execution requires arm64 Darwin",
)

PROBE_RENDERER = b"""\
import fcntl
import os
import sys

if len(sys.argv) != 9 or __file__ != sys.argv[0]:
    raise SystemExit(90)
source_fd = int(sys.argv[6], 10)
open_fds = sorted(
    int(item)
    for item in os.listdir("/dev/fd")
    if item.isdecimal() and int(item) >= 3
)
for descriptor in open_fds:
    if descriptor == source_fd:
        continue
    try:
        fcntl.fcntl(descriptor, fcntl.F_GETFD)
    except OSError:
        continue
    raise SystemExit(91)
if source_fd not in open_fds:
    raise SystemExit(91)
if fcntl.fcntl(source_fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC:
    raise SystemExit(92)
if os.lseek(source_fd, 0, os.SEEK_CUR) != 0:
    raise SystemExit(93)
source_flags = fcntl.fcntl(source_fd, fcntl.F_GETFL)
if source_flags & (os.O_ACCMODE | os.O_NONBLOCK) != os.O_NONBLOCK:
    raise SystemExit(94)
if os.getcwd() != "/":
    raise SystemExit(95)
if os.environ != {"LANG": "C", "LC_ALL": "C", "TZ": "UTC"}:
    raise SystemExit(96)
print(__file__)
for argument in sys.argv[1:]:
    print(argument)
"""

REPLACEMENT_RUNTIME_SOURCE = r"""
#include <fcntl.h>
#include <unistd.h>

int main(void) {
  const int marker = open(DIALECT_MARKER_PATH, O_WRONLY | O_CREAT | O_EXCL, 0600);
  if (marker >= 0) {
    (void)close(marker);
  }
  return 77;
}
"""

EARLY_TERMINAL_RUNTIME_SOURCE = "int main(void) { return 77; }\n"
HANGING_PROCESS_GROUP_SOURCE = r"""
#include <unistd.h>

int main(void) {
  const pid_t child = fork();
  if (child < 0) {
    return 80;
  }
  for (;;) {
    (void)pause();
  }
}
"""

NONZERO_RENDERER = b"raise SystemExit(73)\n"
SIGNAL_RENDERER = b"import os,signal\nos.kill(os.getpid(),signal.SIGTERM)\n"
SIGKILL_RENDERER = b"import os,signal\nos.kill(os.getpid(),signal.SIGKILL)\n"
PROCESS_GROUP_RENDERER = (
    b"import os\n"
    b"if os.getpgrp()!=os.getppid():raise SystemExit(74)\n"
    b"os.write(2,b'fd2-inherited\\n')\n"
)

RACE_HARNESS_SUFFIX = r"""
static const char *race_action = NULL;
static const char *race_backup_path = NULL;
static const char *race_original_path = NULL;
static const char *race_replacement_path = NULL;
static pid_t race_spawned_pid = 0;
static int race_path_swap_completed = 0;
static int race_spawn_interposed = 0;
static int race_waitid_calls = 0;
static int race_waitid_options_valid = 1;

int dialect_test_posix_spawn(
    pid_t *process_id, const char *path,
    const posix_spawn_file_actions_t *actions,
    const posix_spawnattr_t *attributes, char *const arguments[],
    char *const environment[]) {
  const char *spawn_path = path;
  const posix_spawnattr_t *spawn_attributes = attributes;
  int result;

  if (race_spawn_interposed != 0) {
    return EALREADY;
  }
  race_spawn_interposed = 1;
  if (strcmp(race_action, "runtime") == 0 ||
      strcmp(race_action, "renderer") == 0) {
    if (rename(race_original_path, race_backup_path) != 0 ||
        rename(race_replacement_path, race_original_path) != 0) {
      return EIO;
    }
    race_path_swap_completed = 1;
  } else if (strcmp(race_action, "early-terminal") == 0) {
    spawn_path = race_replacement_path;
    spawn_attributes = NULL;
  }
  result = posix_spawn(process_id, spawn_path, actions, spawn_attributes,
                       arguments, environment);
  if (result == 0) {
    race_spawned_pid = *process_id;
  }
  return result;
}

int dialect_test_waitid(idtype_t id_type, id_t identifier,
                         siginfo_t *information, int options) {
  const int expected_options = WSTOPPED | WEXITED | WNOHANG | WNOWAIT;

  ++race_waitid_calls;
  if (options != expected_options) {
    race_waitid_options_valid = 0;
  }
  return waitid(id_type, identifier, information, options);
}

int dialect_test_csops(pid_t process_id, unsigned int operation, void *buffer,
                        size_t buffer_bytes) {
  uint32_t status;
  int result;

  result = csops(process_id, operation, buffer, buffer_bytes);
  if (result != 0) {
    return result;
  }
  if (((strcmp(race_action, "csops-status-failure") == 0 &&
        operation == DIALECT_CS_OPS_STATUS) ||
       (strcmp(race_action, "csops-cdhash-failure") == 0 &&
        operation == DIALECT_CS_OPS_CDHASH))) {
    errno = EPERM;
    return -1;
  }
  if (operation != DIALECT_CS_OPS_STATUS) {
    return 0;
  }
  if (buffer_bytes != sizeof(status)) {
    errno = EINVAL;
    return -1;
  }
  memcpy(&status, buffer, sizeof(status));
  if (strcmp(race_action, "status-missing-valid") == 0) {
    status &= ~DIALECT_CS_VALID;
  } else if (strcmp(race_action, "status-missing-kill") == 0) {
    status &= ~DIALECT_CS_KILL;
  } else if (strcmp(race_action, "status-missing-signed") == 0) {
    status &= ~DIALECT_CS_SIGNED;
  } else if (strcmp(race_action, "status-invalid-allowed") == 0) {
    status |= DIALECT_CS_INVALID_ALLOWED;
  } else if (strcmp(race_action, "status-killed") == 0) {
    status |= DIALECT_CS_KILLED;
  } else if (strcmp(race_action, "status-debugged") == 0) {
    status |= DIALECT_CS_DEBUGGED;
  }
  memcpy(buffer, &status, sizeof(status));
  return 0;
}

static int race_child_was_reaped(void) {
  int child_status;
  pid_t waited;

  if (race_spawned_pid <= 0) {
    return 0;
  }
  do {
    waited = waitpid(race_spawned_pid, &child_status, WNOHANG);
  } while (waited < 0 && errno == EINTR);
  return waited < 0 && errno == ECHILD;
}

int main(int argc, char *argv[]) {
  char *launch_argv[10];
  int renderer_descriptor = -1;
  int result;
  int runtime_descriptor = -1;
  int source_descriptor;
  struct dialect_handoff handoff;
  struct stat renderer_identity;
  struct stat runtime_identity;

  if (argc != 5 || !parse_source_descriptor(argv[1], &source_descriptor) ||
      !verify_source_descriptor(source_descriptor) || chdir("/") != 0) {
    return 80;
  }
  if (!pin_dependency(DIALECT_RUNTIME_PATH, DIALECT_RUNTIME_SHA256,
                      (off_t)DIALECT_RUNTIME_BYTES,
                      (mode_t)DIALECT_RUNTIME_MODE, 1, &runtime_descriptor,
                      &runtime_identity) ||
      !pin_dependency(DIALECT_RENDERER_PATH, DIALECT_RENDERER_SHA256,
                      (off_t)DIALECT_RENDERER_BYTES,
                      (mode_t)DIALECT_RENDERER_MODE, 0, &renderer_descriptor,
                      &renderer_identity)) {
    return 81;
  }
  launch_argv[0] = (char *)"race-harness";
  launch_argv[1] = (char *)"--dialect-derivation-protocol";
  launch_argv[2] = (char *)DIALECT_PROTOCOL;
  launch_argv[3] = (char *)"--pdf-id";
  launch_argv[4] = (char *)DIALECT_ROLE_TEXT;
  launch_argv[5] = (char *)"--source-fd";
  launch_argv[6] = argv[1];
  launch_argv[7] = (char *)"--pdf-output";
  launch_argv[8] = (char *)"stdout";
  launch_argv[9] = NULL;
  result = prepare_verified_handoff(
      source_descriptor, runtime_descriptor, &runtime_identity,
      renderer_descriptor, &renderer_identity, launch_argv, &handoff);
  if (result != 0) {
    return 85;
  }
  race_action = argv[2];
  race_backup_path = argv[3];
  race_replacement_path = argv[4];
  if (strcmp(race_action, "runtime") == 0) {
    race_original_path = DIALECT_RUNTIME_PATH;
  } else if (strcmp(race_action, "renderer") == 0) {
    race_original_path = DIALECT_RENDERER_PATH;
  } else if (strcmp(race_action, "status-missing-valid") != 0 &&
             strcmp(race_action, "status-missing-kill") != 0 &&
             strcmp(race_action, "status-missing-signed") != 0 &&
             strcmp(race_action, "status-invalid-allowed") != 0 &&
             strcmp(race_action, "status-killed") != 0 &&
             strcmp(race_action, "status-debugged") != 0 &&
             strcmp(race_action, "csops-status-failure") != 0 &&
             strcmp(race_action, "csops-cdhash-failure") != 0 &&
             strcmp(race_action, "early-terminal") != 0) {
    return 84;
  }
  result = spawn_attested_runtime(
      source_descriptor, runtime_descriptor, &runtime_identity,
      renderer_descriptor, &renderer_identity, &handoff);
  (void)close(renderer_descriptor);
  (void)close(runtime_descriptor);
  if (race_spawn_interposed == 0 ||
      ((strcmp(race_action, "runtime") == 0 ||
        strcmp(race_action, "renderer") == 0) &&
       race_path_swap_completed == 0)) {
    return 86;
  }
  if (race_waitid_calls == 0 || race_waitid_options_valid == 0) {
    return 88;
  }
  if (!race_child_was_reaped()) {
    return 87;
  }
  return result;
}
"""


def _run(
    arguments: Sequence[str],
    *,
    check: bool = False,
    timeout_seconds: float = SUBPROCESS_TIMEOUT_SECONDS,
    **kwargs: object,
) -> subprocess.CompletedProcess[bytes]:
    requested_new_session = kwargs.pop("start_new_session", True)
    assert requested_new_session is True
    process = subprocess.Popen(  # noqa: S603 - exact local executable paths.
        list(arguments),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        **kwargs,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        try:
            process.communicate(timeout=SUBPROCESS_CLEANUP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=SUBPROCESS_CLEANUP_TIMEOUT_SECONDS)
        raise
    result = subprocess.CompletedProcess(
        list(arguments),
        process.returncode,
        stdout,
        stderr,
    )
    if check:
        result.check_returncode()
    return result


def _c_string_macro(name: str, value: str) -> str:
    assert all(character not in value for character in ('"', "\\", "\n", "\0"))
    return f'-D{name}="{value}"'


def _race_harness_source() -> str:
    source_path = str(SOURCE)
    assert all(character not in source_path for character in ('"', "\\", "\n", "\0"))
    return (
        "#include <spawn.h>\n"
        "#include <stddef.h>\n"
        "#include <stdint.h>\n"
        "#include <stdio.h>\n"
        "#include <sys/types.h>\n"
        "#include <sys/wait.h>\n"
        "extern int csops(pid_t, unsigned int, void *, size_t);\n"
        "int dialect_test_posix_spawn(\n"
        "    pid_t *, const char *, const posix_spawn_file_actions_t *,\n"
        "    const posix_spawnattr_t *, char *const [], char *const []);\n"
        "int dialect_test_waitid(idtype_t, id_t, siginfo_t *, int);\n"
        "int dialect_test_csops(pid_t, unsigned int, void *, size_t);\n"
        "int dialect_embedded_launcher_main(int argc, char *argv[]);\n"
        "#define posix_spawn dialect_test_posix_spawn\n"
        "#define waitid dialect_test_waitid\n"
        "#define csops dialect_test_csops\n"
        "#define main dialect_embedded_launcher_main\n"
        f'#include "{source_path}"\n'
        "#undef main\n"
        "#undef csops\n"
        "#undef waitid\n"
        "#undef posix_spawn\n"
        f"{RACE_HARNESS_SUFFIX}"
    )


def _definition_set(
    role: str,
    *,
    runtime: Path | None = None,
    renderer: Path | None = None,
) -> tuple[str, ...]:
    if runtime is None or renderer is None:
        return (
            f"-DDIALECT_ROLE={role}",
            "-DDIALECT_MAX_BUNDLE_BYTES=8388608LL",
            _c_string_macro("DIALECT_RUNTIME_PATH", "/synthetic/runtime"),
            _c_string_macro("DIALECT_RUNTIME_SHA256", "0" * 64),
            _c_string_macro("DIALECT_RUNTIME_CDHASH", "0" * 40),
            "-DDIALECT_RUNTIME_BYTES=1",
            "-DDIALECT_RUNTIME_MODE=0500",
            _c_string_macro("DIALECT_RENDERER_PATH", "/synthetic/renderer"),
            _c_string_macro("DIALECT_RENDERER_SHA256", "0" * 64),
            "-DDIALECT_RENDERER_BYTES=1",
            "-DDIALECT_RENDERER_MODE=0400",
        )
    runtime_raw = runtime.read_bytes()
    renderer_raw = renderer.read_bytes()
    runtime_pin = machine._pin_file(  # noqa: SLF001
        runtime,
        maximum=128 * 1024 * 1024,
        context="synthetic runtime",
    )
    try:
        runtime_cdhash = str(
            machine._parse_arm64_code_directory(runtime_pin)["cdhash"],  # noqa: SLF001
        )
    finally:
        runtime_pin.close()
    return (
        f"-DDIALECT_ROLE={role}",
        "-DDIALECT_MAX_BUNDLE_BYTES=8388608LL",
        _c_string_macro("DIALECT_RUNTIME_PATH", str(runtime)),
        _c_string_macro(
            "DIALECT_RUNTIME_SHA256",
            hashlib.sha256(runtime_raw).hexdigest(),
        ),
        _c_string_macro("DIALECT_RUNTIME_CDHASH", runtime_cdhash),
        f"-DDIALECT_RUNTIME_BYTES={len(runtime_raw)}",
        f"-DDIALECT_RUNTIME_MODE=0{stat.S_IMODE(runtime.stat().st_mode):o}",
        _c_string_macro("DIALECT_RENDERER_PATH", str(renderer)),
        _c_string_macro(
            "DIALECT_RENDERER_SHA256",
            hashlib.sha256(renderer_raw).hexdigest(),
        ),
        f"-DDIALECT_RENDERER_BYTES={len(renderer_raw)}",
        f"-DDIALECT_RENDERER_MODE=0{stat.S_IMODE(renderer.stat().st_mode):o}",
    )


def _clang_arguments(
    source: Path,
    definitions: Sequence[str],
) -> list[str]:
    return [
        str(native.EXPECTED_CLANG),
        "-arch",
        "arm64",
        "-target",
        "arm64-apple-macos13.0",
        "--no-default-config",
        "-std=c11",
        "-Os",
        "-Weverything",
        "-Werror",
        "-Wno-poison-system-directories",
        "-isysroot",
        str(native.EXPECTED_SDK_ROOT),
        "-resource-dir",
        str(native.EXPECTED_COMPILER_RESOURCE_ROOT),
        *definitions,
        str(source),
    ]


def _compile_executable(
    source: Path,
    output: Path,
    *,
    definitions: Sequence[str] = (),
    identifier: str,
) -> None:
    object_path = output.with_suffix(".o")
    _run(
        [*_clang_arguments(source, definitions), "-c", "-o", str(object_path)],
        check=True,
    )
    sdk_version = native._sdk_version(native.EXPECTED_SDK_ROOT)  # noqa: SLF001
    _run(
        [
            str(native.EXPECTED_LD),
            "-arch",
            "arm64",
            "-syslibroot",
            str(native.EXPECTED_SDK_ROOT),
            "-platform_version",
            "macos",
            native.MACOS_MINIMUM,
            sdk_version,
            "-lSystem",
            "-dead_strip",
            "-no_adhoc_codesign",
            "-o",
            str(output),
            str(object_path),
        ],
        check=True,
    )
    _run(
        [
            str(native.EXPECTED_CODESIGN),
            "--force",
            "--sign",
            "-",
            "--options",
            "kill",
            "--timestamp=none",
            "--identifier",
            identifier,
            str(output),
        ],
        check=True,
    )
    output.chmod(0o500)


def _compile_custom_renderer_launcher(
    tmp_path: Path,
    runtime: Path,
    renderer_raw: bytes,
    *,
    label: str,
) -> Path:
    renderer = tmp_path / f"{label}-renderer.py"
    renderer.write_bytes(renderer_raw)
    renderer.chmod(0o400)
    launcher = tmp_path / f"derive-clean-{label}"
    _compile_executable(
        SOURCE,
        launcher,
        definitions=_definition_set("clean", runtime=runtime, renderer=renderer),
        identifier=f"org.raphaelgroup.dialect.synthetic-clean-{label}-launcher",
    )
    return launcher


def _run_clean_launcher(
    tmp_path: Path,
    launcher: Path,
    *,
    start_new_session: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    source = tmp_path / f"{launcher.name}.bundle"
    source.write_bytes(b"synthetic canonical source bundle\n")
    source.chmod(0o400)
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    try:
        return _run(
            [
                str(launcher),
                "--dialect-derivation-protocol",
                "dialect-pdf-derivation-fd-protocol-v1",
                "--pdf-id",
                "clean",
                "--source-fd",
                str(descriptor),
                "--pdf-output",
                "stdout",
            ],
            pass_fds=(descriptor,),
            start_new_session=start_new_session,
        )
    finally:
        os.close(descriptor)


def _prepare_race_paths(
    tmp_path: Path,
    runtime: Path,
    renderer: Path,
    *,
    race_action: str,
    replacement_kind: str,
) -> tuple[Path, Path, Path, Path | None, bytes | None, bytes | None]:
    replacement = tmp_path / f"replacement-{race_action}"
    backup = tmp_path / f"verified-{race_action}"
    execution_marker = tmp_path / "replacement-runtime-executed"
    replaced_path: Path | None = None
    expected_backup: bytes | None = None
    expected_replacement: bytes | None = None
    if race_action == "runtime":
        if replacement_kind == "same-bytes":
            replacement.write_bytes(runtime.read_bytes())
            replacement.chmod(0o500)
        else:
            replacement_source = tmp_path / "replacement-runtime.c"
            replacement_source.write_text(REPLACEMENT_RUNTIME_SOURCE, encoding="ascii")
            _compile_executable(
                replacement_source,
                replacement,
                definitions=(
                    _c_string_macro("DIALECT_MARKER_PATH", str(execution_marker)),
                ),
                identifier="org.raphaelgroup.dialect.synthetic-replacement-runtime",
            )
        replaced_path = runtime
        expected_backup = runtime.read_bytes()
        expected_replacement = replacement.read_bytes()
    elif race_action == "renderer":
        replacement.write_bytes(b"raise SystemExit(77)\n")
        replacement.chmod(0o400)
        replaced_path = renderer
        expected_backup = renderer.read_bytes()
        expected_replacement = replacement.read_bytes()
    elif race_action == "early-terminal":
        replacement_source = tmp_path / "early-terminal-runtime.c"
        replacement_source.write_text(EARLY_TERMINAL_RUNTIME_SOURCE, encoding="ascii")
        _compile_executable(
            replacement_source,
            replacement,
            identifier="org.raphaelgroup.dialect.synthetic-early-terminal-runtime",
        )
    return (
        replacement,
        backup,
        execution_marker,
        replaced_path,
        expected_backup,
        expected_replacement,
    )


def _compile_race_harness(
    tmp_path: Path,
    runtime: Path,
    renderer: Path,
    *,
    race_action: str,
) -> Path:
    harness_source = tmp_path / f"{race_action}-race-harness.c"
    harness = tmp_path / f"{race_action}-race-harness"
    harness_source.write_text(_race_harness_source(), encoding="ascii")
    _compile_executable(
        harness_source,
        harness,
        definitions=_definition_set("clean", runtime=runtime, renderer=renderer),
        identifier=f"org.raphaelgroup.dialect.synthetic-{race_action}-race-harness",
    )
    return harness


def _run_race_harness(
    tmp_path: Path,
    harness: Path,
    backup: Path,
    replacement: Path,
    *,
    race_action: str,
) -> tuple[subprocess.CompletedProcess[bytes], list[str]]:
    source = tmp_path / "source.bundle"
    source.write_bytes(b"synthetic canonical source bundle\n")
    source.chmod(0o400)
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    arguments = [
        "--dialect-derivation-protocol",
        "dialect-pdf-derivation-fd-protocol-v1",
        "--pdf-id",
        "clean",
        "--source-fd",
        str(descriptor),
        "--pdf-output",
        "stdout",
    ]
    try:
        result = _run(
            [
                str(harness),
                str(descriptor),
                race_action,
                str(backup),
                str(replacement),
            ],
            pass_fds=(descriptor,),
            cwd=tmp_path,
            env={"UNEXPECTED": "discarded"},
        )
    finally:
        os.close(descriptor)
    return result, arguments


@pytest.fixture(scope="session")
def role_launchers(
    tmp_path_factory: pytest.TempPathFactory,
) -> Mapping[str, Path]:
    if sys.platform != "darwin" or os.uname().machine != "arm64":
        pytest.skip("native launcher execution requires arm64 Darwin")
    root = tmp_path_factory.mktemp("document-derivation-launchers")
    runtime = root / "python3.12"
    renderer = root / "synthetic-renderer.py"
    runtime.write_bytes(Path(sys.executable).resolve().read_bytes())
    runtime.chmod(0o500)
    renderer.write_bytes(PROBE_RENDERER)
    renderer.chmod(0o400)
    launchers: dict[str, Path] = {"runtime": runtime, "renderer": renderer}
    for role in ROLES:
        launcher = root / f"derive-{role}"
        _compile_executable(
            SOURCE,
            launcher,
            definitions=_definition_set(role, runtime=runtime, renderer=renderer),
            identifier=f"org.raphaelgroup.dialect.synthetic-{role}-launcher",
        )
        launchers[role] = launcher
    return launchers


def test_shared_source_preserves_contract_and_uses_pinned_fd_handoff() -> None:
    shared = SOURCE.read_text(encoding="ascii")
    assert LEGACY_SOURCE.is_file()
    assert '#define DIALECT_ROLE "rebuttal"' not in shared
    for role in ROLES:
        assert f"#define DIALECT_ROLE_SUPPORTED_{role} 1" in shared
    assert '#define DIALECT_PROTOCOL "dialect-pdf-derivation-fd-protocol-v1"' in shared
    assert "POSIX_SPAWN_START_SUSPENDED" in shared
    assert "POSIX_SPAWN_CLOEXEC_DEFAULT" in shared
    assert "POSIX_SPAWN_SETSID" not in shared
    assert "kill(process_id, SIGCONT)" in shared
    assert "killpg(" not in shared
    assert "PROC_PIDREGIONPATHINFO" in shared
    assert "DIALECT_CS_OPS_CDHASH" in shared
    assert "waitid(P_PID" in shared
    assert "WSTOPPED | WEXITED | WNOHANG | WNOWAIT" in shared
    assert "posix_spawn_file_actions_addinherit_np(&actions, 0)" in shared
    assert "DIALECT_RENDERER_BOOTSTRAP" in shared
    assert "'__file__':p" in shared
    assert "execve(DIALECT_RUNTIME_PATH" not in shared
    assert "active same-UID peer sending SIGCONT" in shared
    assert "Dylib and Python" in shared
    assert "same-vnode non-code" in shared


@REQUIRES_ARM64_DARWIN
@pytest.mark.parametrize(
    ("missing", "expected"),
    [
        ("DIALECT_ROLE", "DIALECT_ROLE must be supplied"),
        (
            "DIALECT_MAX_BUNDLE_BYTES",
            "DIALECT_MAX_BUNDLE_BYTES must be supplied",
        ),
        ("DIALECT_RUNTIME_PATH", "DIALECT_RUNTIME_PATH must be supplied"),
        ("DIALECT_RUNTIME_CDHASH", "DIALECT_RUNTIME_CDHASH must be supplied"),
        ("DIALECT_RENDERER_PATH", "DIALECT_RENDERER_PATH must be supplied"),
    ],
)
def test_required_compile_time_pins_fail_closed(
    missing: str,
    expected: str,
) -> None:
    definitions = tuple(
        value
        for value in _definition_set("clean")
        if not value.startswith(f"-D{missing}=")
    )
    result = _run([*_clang_arguments(SOURCE, definitions), "-fsyntax-only"])
    assert result.returncode != 0
    assert expected.encode() in result.stderr
    assert result.stdout == b""


@REQUIRES_ARM64_DARWIN
@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("", b"DIALECT_ROLE must be exactly clean, marked, s1, or rebuttal"),
        ('"clean"', b"pasting formed"),
        ("other", b"DIALECT_ROLE must be exactly clean, marked, s1, or rebuttal"),
        ("Clean", b"DIALECT_ROLE must be exactly clean, marked, s1, or rebuttal"),
        (
            "clean_marked",
            b"DIALECT_ROLE must be exactly clean, marked, s1, or rebuttal",
        ),
    ],
)
def test_invalid_compile_time_role_fails_closed(
    role: str,
    expected: bytes,
) -> None:
    result = _run(
        [*_clang_arguments(SOURCE, _definition_set(role)), "-fsyntax-only"],
    )
    assert result.returncode != 0
    assert expected in result.stderr
    assert result.stdout == b""


@REQUIRES_ARM64_DARWIN
@pytest.mark.parametrize("value", ["0", "-1"])
def test_bundle_byte_limit_must_be_positive(value: str) -> None:
    definitions = tuple(
        f"-DDIALECT_MAX_BUNDLE_BYTES={value}"
        if item.startswith("-DDIALECT_MAX_BUNDLE_BYTES=")
        else item
        for item in _definition_set("clean")
    )
    result = _run([*_clang_arguments(SOURCE, definitions), "-fsyntax-only"])
    assert result.returncode != 0
    assert b"DIALECT_MAX_BUNDLE_BYTES must be positive" in result.stderr
    assert result.stdout == b""


@REQUIRES_ARM64_DARWIN
def test_compiled_bundle_byte_limit_controls_source_validation(
    tmp_path: Path,
    role_launchers: Mapping[str, Path],
) -> None:
    launcher = tmp_path / "derive-clean-limit-16"
    definitions = tuple(
        "-DDIALECT_MAX_BUNDLE_BYTES=16LL"
        if item.startswith("-DDIALECT_MAX_BUNDLE_BYTES=")
        else item
        for item in _definition_set(
            "clean",
            runtime=role_launchers["runtime"],
            renderer=role_launchers["renderer"],
        )
    )
    _compile_executable(
        SOURCE,
        launcher,
        definitions=definitions,
        identifier="org.raphaelgroup.dialect.synthetic-clean-limit-launcher",
    )
    source = tmp_path / "seventeen-byte.bundle"
    source.write_bytes(b"x" * 17)
    source.chmod(0o400)
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    try:
        result = _run(
            [
                str(launcher),
                "--dialect-derivation-protocol",
                "dialect-pdf-derivation-fd-protocol-v1",
                "--pdf-id",
                "clean",
                "--source-fd",
                str(descriptor),
                "--pdf-output",
                "stdout",
            ],
            pass_fds=(descriptor,),
        )
    finally:
        os.close(descriptor)
    assert result.returncode == 65
    assert result.stdout == b""
    assert result.stderr == b""


@REQUIRES_ARM64_DARWIN
@pytest.mark.parametrize("runtime_cdhash", ["0" * 40, "A" * 40])
def test_runtime_cdhash_pin_fails_closed(
    tmp_path: Path,
    role_launchers: Mapping[str, Path],
    runtime_cdhash: str,
) -> None:
    launcher = tmp_path / f"derive-clean-cdhash-{len(runtime_cdhash)}"
    definitions = tuple(
        _c_string_macro("DIALECT_RUNTIME_CDHASH", runtime_cdhash)
        if item.startswith("-DDIALECT_RUNTIME_CDHASH=")
        else item
        for item in _definition_set(
            "clean",
            runtime=role_launchers["runtime"],
            renderer=role_launchers["renderer"],
        )
    )
    _compile_executable(
        SOURCE,
        launcher,
        definitions=definitions,
        identifier="org.raphaelgroup.dialect.synthetic-bad-cdhash-launcher",
    )
    source = tmp_path / "source.bundle"
    source.write_bytes(b"synthetic canonical source bundle\n")
    source.chmod(0o400)
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    try:
        result = _run(
            [
                str(launcher),
                "--dialect-derivation-protocol",
                "dialect-pdf-derivation-fd-protocol-v1",
                "--pdf-id",
                "clean",
                "--source-fd",
                str(descriptor),
                "--pdf-output",
                "stdout",
            ],
            pass_fds=(descriptor,),
        )
    finally:
        os.close(descriptor)
    assert result.returncode == 67
    assert result.stdout == b""
    assert result.stderr == b""


@REQUIRES_ARM64_DARWIN
@pytest.mark.parametrize("runtime_cdhash", ["0" * 39, "0" * 41])
def test_runtime_cdhash_pin_requires_exact_length(runtime_cdhash: str) -> None:
    definitions = tuple(
        _c_string_macro("DIALECT_RUNTIME_CDHASH", runtime_cdhash)
        if item.startswith("-DDIALECT_RUNTIME_CDHASH=")
        else item
        for item in _definition_set("clean")
    )
    result = _run([*_clang_arguments(SOURCE, definitions), "-fsyntax-only"])
    assert result.returncode != 0
    assert result.stdout == b""


@REQUIRES_ARM64_DARWIN
def test_role_allowlist_cannot_be_extended_by_a_compile_definition() -> None:
    result = _run(
        [
            *_clang_arguments(
                SOURCE,
                (
                    *_definition_set("other"),
                    "-DDIALECT_ROLE_SUPPORTED_other=1",
                ),
            ),
            "-fsyntax-only",
        ],
    )
    assert result.returncode != 0
    assert result.stdout == b""


@REQUIRES_ARM64_DARWIN
@pytest.mark.parametrize("role", ROLES)
def test_every_role_compiles_with_weverything_werror(
    role_launchers: Mapping[str, Path],
    role: str,
) -> None:
    launcher = role_launchers[role]
    assert launcher.is_file()
    assert stat.S_IMODE(launcher.stat().st_mode) == 0o500


@REQUIRES_ARM64_DARWIN
def test_role_builds_are_byte_distinct(
    role_launchers: Mapping[str, Path],
) -> None:
    digests = {
        hashlib.sha256(role_launchers[role].read_bytes()).hexdigest() for role in ROLES
    }
    assert len(digests) == len(ROLES)


@REQUIRES_ARM64_DARWIN
@pytest.mark.parametrize("role", ROLES)
def test_each_role_preserves_exact_fd_argv_env_and_cwd_handoff(
    tmp_path: Path,
    role_launchers: Mapping[str, Path],
    role: str,
) -> None:
    source = tmp_path / f"{role}.bundle"
    source.write_bytes(b"synthetic canonical source bundle\n")
    source.chmod(0o400)
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    extra_path = tmp_path / f"{role}.extra"
    extra_path.write_bytes(b"extra")
    extra = os.open(extra_path, os.O_RDONLY)
    arguments = [
        "--dialect-derivation-protocol",
        "dialect-pdf-derivation-fd-protocol-v1",
        "--pdf-id",
        role,
        "--source-fd",
        str(descriptor),
        "--pdf-output",
        "stdout",
    ]
    try:
        result = _run(
            [str(role_launchers[role]), *arguments],
            pass_fds=(descriptor, extra),
            cwd=tmp_path,
            env={"UNEXPECTED": "discarded"},
        )
    finally:
        os.close(extra)
        os.close(descriptor)
    assert result.returncode == 0
    assert result.stderr == b""
    assert result.stdout.decode("ascii").splitlines() == [
        str(role_launchers["renderer"]),
        *arguments,
    ]


@REQUIRES_ARM64_DARWIN
@pytest.mark.parametrize(
    ("label", "renderer_raw", "expected_return_code"),
    [
        ("nonzero", NONZERO_RENDERER, 73),
        ("signal", SIGNAL_RENDERER, -signal.SIGTERM),
        ("sigkill", SIGKILL_RENDERER, -signal.SIGKILL),
    ],
)
def test_attested_runtime_terminal_status_is_propagated_exactly(
    tmp_path: Path,
    role_launchers: Mapping[str, Path],
    label: str,
    renderer_raw: bytes,
    expected_return_code: int,
) -> None:
    launcher = _compile_custom_renderer_launcher(
        tmp_path,
        role_launchers["runtime"],
        renderer_raw,
        label=label,
    )
    result = _run_clean_launcher(tmp_path, launcher)
    assert result.returncode == expected_return_code
    assert result.stdout == result.stderr == b""


@REQUIRES_ARM64_DARWIN
def test_attested_runtime_stays_in_outer_launcher_process_group(
    tmp_path: Path,
    role_launchers: Mapping[str, Path],
) -> None:
    launcher = _compile_custom_renderer_launcher(
        tmp_path,
        role_launchers["runtime"],
        PROCESS_GROUP_RENDERER,
        label="process-group",
    )
    result = _run_clean_launcher(tmp_path, launcher, start_new_session=True)
    assert result.returncode == 0
    assert result.stdout == b""
    assert result.stderr == b"fd2-inherited\n"


@REQUIRES_ARM64_DARWIN
def test_subprocess_timeout_terminates_owned_process_group(tmp_path: Path) -> None:
    source = tmp_path / "hanging-process-group.c"
    executable = tmp_path / "hanging-process-group"
    source.write_text(HANGING_PROCESS_GROUP_SOURCE, encoding="ascii")
    _compile_executable(
        source,
        executable,
        identifier="org.raphaelgroup.dialect.synthetic-hanging-process-group",
    )
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        _run([str(executable)], timeout_seconds=0.1)
    assert time.monotonic() - started < SUBPROCESS_CLEANUP_TIMEOUT_SECONDS


@REQUIRES_ARM64_DARWIN
@pytest.mark.parametrize(
    ("race_action", "replacement_kind"),
    [
        ("runtime", "different-bytes"),
        ("runtime", "same-bytes"),
        ("renderer", "different-bytes"),
        ("status-missing-valid", "none"),
        ("status-missing-kill", "none"),
        ("status-missing-signed", "none"),
        ("status-invalid-allowed", "none"),
        ("status-killed", "none"),
        ("status-debugged", "none"),
        ("csops-status-failure", "none"),
        ("csops-cdhash-failure", "none"),
        ("early-terminal", "different-bytes"),
    ],
)
def test_suspended_attestation_rejects_runtime_and_signing_identity_races(
    tmp_path: Path,
    race_action: str,
    replacement_kind: str,
) -> None:
    runtime = tmp_path / "python3.12"
    renderer = tmp_path / "synthetic-renderer.py"
    runtime.write_bytes(Path(sys.executable).resolve().read_bytes())
    runtime.chmod(0o500)
    renderer.write_bytes(PROBE_RENDERER)
    renderer.chmod(0o400)
    (
        replacement,
        backup,
        execution_marker,
        replaced_path,
        expected_backup,
        expected_replacement,
    ) = _prepare_race_paths(
        tmp_path,
        runtime,
        renderer,
        race_action=race_action,
        replacement_kind=replacement_kind,
    )
    harness = _compile_race_harness(
        tmp_path,
        runtime,
        renderer,
        race_action=race_action,
    )
    result, arguments = _run_race_harness(
        tmp_path,
        harness,
        backup,
        replacement,
        race_action=race_action,
    )
    logical_renderer = renderer
    assert result.stderr == b""
    if race_action == "renderer":
        assert result.returncode == 0
        assert result.stdout.decode("ascii").splitlines() == [
            str(logical_renderer),
            *arguments,
        ]
    else:
        assert result.returncode == 67
        assert result.stdout == b""
    assert not execution_marker.exists()
    if replaced_path is not None:
        assert expected_backup is not None
        assert expected_replacement is not None
        assert backup.read_bytes() == expected_backup
        assert replaced_path.read_bytes() == expected_replacement


@REQUIRES_ARM64_DARWIN
@pytest.mark.parametrize(("compiled_role", "requested_role"), ROLE_MISMATCHES)
def test_role_binary_rejects_every_other_role(
    tmp_path: Path,
    role_launchers: Mapping[str, Path],
    compiled_role: str,
    requested_role: str,
) -> None:
    source = tmp_path / f"{compiled_role}-rejects-{requested_role}.bundle"
    source.write_bytes(b"synthetic source\n")
    source.chmod(0o400)
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    try:
        result = _run(
            [
                str(role_launchers[compiled_role]),
                "--dialect-derivation-protocol",
                "dialect-pdf-derivation-fd-protocol-v1",
                "--pdf-id",
                requested_role,
                "--source-fd",
                str(descriptor),
                "--pdf-output",
                "stdout",
            ],
            pass_fds=(descriptor,),
        )
    finally:
        os.close(descriptor)
    assert result.returncode == 64
    assert result.stdout == result.stderr == b""
