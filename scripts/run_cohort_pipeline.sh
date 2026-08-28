#!/usr/bin/env bash
# Per-cohort DIALECT pipeline on the uniform cBioPortal PanCancer MAF.
#
# Stages (each receipt-validated before reuse):
#   1. CBaSE  generate   -> output/pancan/<C>/{bmr_pmfs.csv,count_matrix.csv,CBaSE_output}
#   2. DIG    generate   -> output/pancan/<C>/bmr_pmfs.dig.csv   (Pancan DIG model)
#   3. DIALECT identify  -> output/pancan/<C>/id_{cbase,dig}/pairwise...csv
#   4. MutSig2CV source + identify -> output/pancan/<C>/id_mutsig/pairwise...csv
#
# Any requested stage failure aborts the cohort. Usage: run_cohort_pipeline.sh ACC
# This is the historical provider-specific workflow, not the frozen revision
# analysis. Use analysis.run_tcga_revision_k500 for the common-universe K=500 grid.
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "usage: $0 <COHORT>" >&2
  exit 64
fi
C="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
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
DIALECT="/opt/anaconda3/envs/dialect/bin/dialect"
PY="/opt/anaconda3/envs/dialect/bin/python"
log() { echo "[$(date +%H:%M:%S)] ${C}: $*"; }

sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    sha256sum "$1" | awk '{print $1}'
  fi
}

sha256_stream() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 | awk '{print $1}'
  else
    sha256sum | awk '{print $1}'
  fi
}

source_tree_sha256() {
  "$PY" -c '
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
digest = hashlib.sha256()
for path in sorted(candidate for candidate in root.rglob("*.py") if candidate.is_file()):
    digest.update(path.relative_to(root).as_posix().encode())
    digest.update(b"\\0")
    digest.update(path.read_bytes())
    digest.update(b"\\0")
print(digest.hexdigest())
' "$1"
}

directory_files_sha256() {
  "$PY" -c '
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
if not root.is_dir():
    raise SystemExit(f"missing source/input directory: {root}")
digest = hashlib.sha256()
files = sorted(
    candidate
    for candidate in root.rglob("*")
    if candidate.is_file() and "__pycache__" not in candidate.parts
)
if not files:
    raise SystemExit(f"empty source/input directory: {root}")
for path in files:
    digest.update(path.relative_to(root).as_posix().encode())
    digest.update(b"\\0")
    digest.update(path.read_bytes())
    digest.update(b"\\0")
print(digest.hexdigest())
' "$1"
}

files_sha256() {
  local path
  for path in "$@"; do
    [ -s "$path" ] || return 1
    printf '%s\t%s\n' "$(basename -- "$path")" "$(sha256_file "$path")"
  done | sha256_stream
}

cbase_outputs_sha256() {
  local files=("$CBASE_BMR" "$COUNT_MATRIX")
  [ ! -e "$CB_Q" ] || files+=("$CB_Q")
  files_sha256 "${files[@]}"
}

receipt_value() {
  awk -F '\t' -v wanted="$2" '
    $1 == wanted { print $2; found = 1; exit }
    END { if (!found) exit 1 }
  ' "$1"
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
  {
    printf 'schema_version\t1\n'
    printf 'input_sha256\t%s\n' "$input_sha"
    printf 'output_sha256\t%s\n' "$output_sha"
  } > "${receipt}.tmp.$$"
  mv -f -- "${receipt}.tmp.$$" "$receipt"
}

count_axis_matches() {
  "$PY" -c '
import csv
import sys
from pathlib import Path

with Path(sys.argv[1]).open(encoding="utf-8", newline="") as handle:
    rows = list(csv.reader(handle))
actual = [row[0] for row in rows[1:] if row]
expected = Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()
raise SystemExit(actual != expected)
' "$1" "$2"
}

[ -f "$MAF" ] || { log "no MAF at ${MAF}"; exit 66; }
[ -f "$MUTSIG_SAMPLE_AXIS_FILE" ] || {
  log "no exact sample axis at ${MUTSIG_SAMPLE_AXIS_FILE}; refusing mutation-derived cohort"
  exit 66
}
PIPELINE_SHA256="$(sha256_file "${SCRIPT_DIR}/run_cohort_pipeline.sh")"
MAF_SHA256="$(sha256_file "$MAF")"
SAMPLE_AXIS_SHA256="$(sha256_file "$MUTSIG_SAMPLE_AXIS_FILE")"
DIALECT_SOURCE_SHA256="$(source_tree_sha256 "${REPO}/src/dialect")"
CBASE_INPUTS_SHA256="$(directory_files_sha256 "${REPO}/external/CBaSE")"
PY_RUNTIME_SHA256="$(
  "$PY" -c 'import platform; import numpy; import pandas; import scipy; print(platform.python_version(), numpy.__version__, pandas.__version__, scipy.__version__)' \
    | sha256_stream
)"
mkdir -p "${ROOT}/${C}"
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
  rm -f -- "$CBASE_RECEIPT"
  if ! "$DIALECT" generate -m "$MAF" -o "${ROOT}/${C}" --bmr cbase -r hg19 \
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

N="$(awk 'END { print NR + 0 }' "$MUTSIG_SAMPLE_AXIS_FILE")"
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
  rm -f -- "$DIG_RECEIPT"
  if ! "$DIALECT" generate -m "$MAF" -o "${ROOT}/${C}" --bmr dig \
       --dig-results "$DIG_RESULTS" --dig-samples "$N" -r hg19 >>"$LOGF" 2>&1; then
    log "STAGE-FAIL dig"
    exit 74
  fi
  DIG_OUTPUT_SHA256="$(files_sha256 "$DIG_BMR")" || exit 74
  publish_receipt "$DIG_RECEIPT" "$DIG_INPUT_SHA256" "$DIG_OUTPUT_SHA256"
fi

# 3. DIALECT identify -- CBaSE + DIG (fast; run before the slow MutSig) ------
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
  if ! "$DIALECT" identify -c "$COUNT_MATRIX" -b "$CBASE_BMR" \
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
  if ! "$DIALECT" identify -c "$COUNT_MATRIX" -b "$DIG_BMR" \
       -o "$DIG_ID_DIR" -k "$TOP_K" >>"$LOGF" 2>&1; then
    log "STAGE-FAIL id_dig${IDSUF}"
    exit 74
  fi
  DIG_ID_OUTPUT_SHA256="$(files_sha256 "$DIG_ID_RESULT")" || exit 74
  publish_receipt "$DIG_ID_RECEIPT" "$DIG_ID_INPUT_SHA256" "$DIG_ID_OUTPUT_SHA256"
fi

# 4. MutSig2CV (Octave-patched source -> native per-sample lambda) + identify --
if [ -n "${SKIP_MUTSIG:-}" ]; then
  log "skip mutsig (SKIP_MUTSIG set)"
else
  log "MutSig2CV (validate receipt or rebuild atomically)"
  if ! bash scripts/run_mutsig_octave.sh "$C" "$MUTSIG_SAMPLE_AXIS_FILE" \
       "$MAF_DIR" "$MUTSIG_ROOT" >>"$LOGF" 2>&1; then
    log "STAGE-FAIL mutsig"
    exit 74
  fi

  MUTSIG_RECEIPT="${MUTSIG_ROOT}/${C}/persample_receipt.tsv"
  [ -s "$MUTSIG_RECEIPT" ] || { log "MutSig receipt is missing"; exit 74; }
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
    if ! "$PY" -m analysis.mutsig_lambda_co --cohort "$C" --results-root "$ROOT" \
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
