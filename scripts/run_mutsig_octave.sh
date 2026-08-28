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

C="$1"
SAMPLE_AXIS_ARG="$2"
MAF_DIR_ARG="${3:-data/mafs_pancan}"
OUT_ROOT_ARG="${4:-output/mutsigsrc}"
UPSTREAM_COMMIT="0109e27e70478181695f31ca8dd281bb44f0b3af"

if [[ ! "$C" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "invalid cohort identifier: ${C}" >&2
  exit 65
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
MUTSIG_SOURCE="${REPO}/external/MutSig2CV_src"
PATCH_FILE="${REPO}/external/mutsig2cv_octave_dialect.patch"

resolve_from_repo() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$REPO" "$1" ;;
  esac
}

sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    echo "no SHA-256 utility found (need shasum or sha256sum)" >&2
    return 69
  fi
}

sha256_stream() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum | awk '{print $1}'
  else
    echo "no SHA-256 utility found (need shasum or sha256sum)" >&2
    return 69
  fi
}

file_size() {
  if stat -f '%z' "$1" >/dev/null 2>&1; then
    stat -f '%z' "$1"
  else
    stat -c '%s' "$1"
  fi
}

receipt_value() {
  awk -F '\t' -v wanted="$2" '
    $1 == wanted { print $2; found = 1; exit }
    END { if (!found) exit 1 }
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
[ -d "${MUTSIG_SOURCE}/.git" ] || {
  echo "patched MutSig source clone not found: ${MUTSIG_SOURCE}" >&2
  exit 69
}

SOURCE_HEAD="$(git -C "$MUTSIG_SOURCE" rev-parse HEAD)"
if [ "$SOURCE_HEAD" != "$UPSTREAM_COMMIT" ]; then
  echo "MutSig source HEAD ${SOURCE_HEAD} does not match pinned ${UPSTREAM_COMMIT}" >&2
  exit 70
fi
if ! git -C "$MUTSIG_SOURCE" diff --quiet; then
  echo "MutSig source has unstaged drift; regenerate the tracked patch first" >&2
  exit 70
fi
if [ -n "$(git -C "$MUTSIG_SOURCE" ls-files --others --exclude-standard)" ]; then
  echo "MutSig source has untracked drift; regenerate the tracked patch first" >&2
  exit 70
fi

PATCH_SHA256="$(sha256_file "$PATCH_FILE")"
SOURCE_DIFF_SHA256="$(git -C "$MUTSIG_SOURCE" diff --cached --binary | sha256_stream)"
if [ "$SOURCE_DIFF_SHA256" != "$PATCH_SHA256" ]; then
  echo "MutSig source index does not reconstruct the tracked patch" >&2
  exit 70
fi

export PATH="/opt/homebrew/bin:${PATH}"
if [ -z "${JAVA_HOME:-}" ]; then
  export JAVA_HOME="/Library/Java/JavaVirtualMachines/amazon-corretto-11.jdk/Contents/Home"
fi
OCTAVE_BIN="${OCTAVE_BIN:-octave}"
command -v "$OCTAVE_BIN" >/dev/null 2>&1 || { echo "GNU Octave not found" >&2; exit 69; }
OCTAVE_VERSION_OUTPUT="$("$OCTAVE_BIN" --version 2>&1)"
OCTAVE_ID="${OCTAVE_VERSION_OUTPUT%%$'\n'*}"
RUNTIME_SHA256="$(printf '%s\n%s\n' "$OCTAVE_ID" "$JAVA_HOME" | sha256_stream)"
RUNNER_SHA256="$(sha256_file "${SCRIPT_DIR}/run_mutsig_octave.sh")"
MAF_SHA256="$(sha256_file "$MAF")"
SAMPLE_AXIS_SHA256="$(sha256_file "$SAMPLE_AXIS_FILE")"
SAMPLE_AXIS_COUNT="$(awk 'END { print NR + 0 }' "$SAMPLE_AXIS_FILE")"

validate_bundle() {
  local bundle="$1"
  local ng np neff expected_bytes actual_bytes genes_count patients_count
  local artifact

  for artifact in persample_lambda.f32 persample_meta.txt persample_genes.txt persample_patients.txt; do
    [ -s "${bundle}/${artifact}" ] || return 1
  done

  ng="$(awk -F '\t' '$1 == "ng" { print $2 }' "${bundle}/persample_meta.txt")"
  np="$(awk -F '\t' '$1 == "np" { print $2 }' "${bundle}/persample_meta.txt")"
  neff="$(awk -F '\t' '$1 == "neff" { print $2 }' "${bundle}/persample_meta.txt")"
  [[ "$ng" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ "$np" =~ ^[1-9][0-9]*$ ]] || return 1
  [ "$neff" = "2" ] || return 1

  genes_count="$(awk 'END { print NR + 0 }' "${bundle}/persample_genes.txt")"
  patients_count="$(awk 'END { print NR + 0 }' "${bundle}/persample_patients.txt")"
  [ "$genes_count" = "$ng" ] || return 1
  [ "$patients_count" = "$np" ] || return 1
  [ "$patients_count" = "$SAMPLE_AXIS_COUNT" ] || return 1
  awk 'NF == 0 { exit 1 }' "${bundle}/persample_genes.txt" || return 1
  awk 'NF == 0 { exit 1 }' "${bundle}/persample_patients.txt" || return 1
  cmp -s <(sed 's/\r$//' "$SAMPLE_AXIS_FILE") \
    "${bundle}/persample_patients.txt" || return 1

  expected_bytes=$((ng * np * neff * 4))
  actual_bytes="$(file_size "${bundle}/persample_lambda.f32")"
  [ "$actual_bytes" = "$expected_bytes" ] || return 1
}

receipt_matches_current_inputs() {
  local artifact key
  [ -s "$RECEIPT" ] || return 1
  validate_bundle "$OUT_DIR" || return 1
  [ "$(receipt_value "$RECEIPT" schema_version)" = "1" ] || return 1
  [ "$(receipt_value "$RECEIPT" cohort)" = "$C" ] || return 1
  [ "$(receipt_value "$RECEIPT" upstream_commit)" = "$UPSTREAM_COMMIT" ] || return 1
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

mkdir -p "$OUT_ROOT"
STAGING_DIR="$(mktemp -d "${OUT_ROOT%/}/.${C}.mutsig.XXXXXX")"
cleanup() {
  if [ -n "${STAGING_DIR:-}" ] && [ -d "$STAGING_DIR" ]; then
    rm -rf -- "$STAGING_DIR"
  fi
}
trap cleanup EXIT

DIALECT_MUTSIG_SOURCE="$MUTSIG_SOURCE" \
DIALECT_MUTSIG_MAF="$MAF" \
DIALECT_MUTSIG_OUT="$STAGING_DIR" \
DIALECT_MUTSIG_AXIS="$SAMPLE_AXIS_FILE" \
  "$OCTAVE_BIN" --no-gui --eval \
    "addpath(getenv('DIALECT_MUTSIG_SOURCE')); run_mutsig_persample(getenv('DIALECT_MUTSIG_MAF'),getenv('DIALECT_MUTSIG_OUT'),getenv('DIALECT_MUTSIG_AXIS'))"

validate_bundle "$STAGING_DIR" || {
  echo "MutSig did not produce a complete, axis-exact lambda bundle" >&2
  exit 74
}

{
  printf 'schema_version\t1\n'
  printf 'cohort\t%s\n' "$C"
  printf 'upstream_commit\t%s\n' "$UPSTREAM_COMMIT"
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

mkdir -p "$OUT_DIR"
rm -f -- "$RECEIPT"
for artifact in persample_lambda.f32 persample_meta.txt persample_genes.txt persample_patients.txt; do
  mv -f -- "${STAGING_DIR}/${artifact}" "${OUT_DIR}/${artifact}"
done
mv -f -- "${STAGING_DIR}/persample_receipt.tsv" "$RECEIPT"

if ! receipt_matches_current_inputs; then
  rm -f -- "$RECEIPT"
  echo "published MutSig bundle failed receipt validation" >&2
  exit 74
fi

echo "DIALECT: published complete MutSig lambda bundle and receipt at ${OUT_DIR}"
