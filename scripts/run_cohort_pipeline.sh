#!/usr/bin/env bash
# Per-cohort DIALECT pipeline on the uniform cBioPortal PanCancer MAF.
#
# Stages (each idempotent -- skips if its output already exists):
#   1. CBaSE  generate   -> output/pancan/<C>/{bmr_pmfs.csv,count_matrix.csv,CBaSE_output}
#   2. DIG    generate   -> output/pancan/<C>/bmr_pmfs.dig.csv   (Pancan DIG model)
#   3. MutSig2CV (Docker)-> output/pancan/<C>_mutsig/results.mat  (amd64 MCR, emulated)
#   4. DIALECT identify  -> output/pancan/<C>/id_{cbase,dig,mutsig}/pairwise...csv
#
# A failed stage is logged and the rest continue. Usage: run_cohort_pipeline.sh ACC
# This is the historical provider-specific workflow, not the frozen revision
# analysis. Use analysis.run_tcga_revision_k500 for the common-universe K=500 grid.
set -u
C="$1"
# Parameterized (defaults = TCGA pancan); MSK runs set MAF_DIR + ROOT in the environment.
ROOT="${ROOT:-output/pancan}"
MAF_DIR="${MAF_DIR:-data/mafs_pancan}"
MAF="${MAF_DIR}/${C}.maf"
# Top-K genes for the identify stage. K=100 keeps the historical id_{cbase,dig,mutsig}
# directory names; any other K writes to id_<bmr>_k<K> so runs at different K coexist.
TOP_K="${TOP_K:-100}"
if [ "$TOP_K" = "100" ]; then IDSUF=""; else IDSUF="_k${TOP_K}"; fi
DIG_RESULTS="external/DIGDriver/run/Pancan.genes.results.txt"
DIALECT="/opt/anaconda3/envs/dialect/bin/dialect"
PY="/opt/anaconda3/envs/dialect/bin/python"
MCR="/opt/mcr/v81"
LDP="${MCR}/runtime/glnxa64:${MCR}/bin/glnxa64:${MCR}/sys/os/glnxa64:${MCR}/sys/java/jre/glnxa64/jre/lib/amd64/server"

log() { echo "[$(date +%H:%M:%S)] ${C}: $*"; }

[ -f "$MAF" ] || { log "no MAF at ${MAF}; skipping cohort"; exit 0; }
mkdir -p "${ROOT}/${C}"
LOGF="${ROOT}/${C}/pipeline.log"; : > "$LOGF"  # per-cohort verbose log (keeps main log small)

# 1. CBaSE ------------------------------------------------------------------
if [ ! -f "${ROOT}/${C}/bmr_pmfs.csv" ]; then
  log "CBaSE generate"
  "$DIALECT" generate -m "$MAF" -o "${ROOT}/${C}" --bmr cbase -r hg19 >>"$LOGF" 2>&1 \
    || log "STAGE-FAIL cbase"
else log "skip cbase"; fi

N=0
[ -f "${ROOT}/${C}/count_matrix.csv" ] && N=$(( $(wc -l < "${ROOT}/${C}/count_matrix.csv") - 1 ))

# 2. DIG (writes bmr_pmfs.dig.csv directly; does not clobber CBaSE's bmr_pmfs.csv) ---
if [ ! -f "${ROOT}/${C}/bmr_pmfs.dig.csv" ] && [ "$N" -gt 0 ]; then
  log "DIG generate (N=${N})"
  "$DIALECT" generate -m "$MAF" -o "${ROOT}/${C}" --bmr dig \
    --dig-results "$DIG_RESULTS" --dig-samples "$N" -r hg19 >>"$LOGF" 2>&1 \
    || log "STAGE-FAIL dig"
else log "skip dig"; fi

# 3. DIALECT identify -- CBaSE + DIG (fast; run before the slow MutSig) ------
CB_Q="${ROOT}/${C}/CBaSE_output/q_values.txt"
if [ ! -f "${ROOT}/${C}/id_cbase${IDSUF}/pairwise_interaction_results.csv" ] \
   && [ -f "${ROOT}/${C}/bmr_pmfs.csv" ]; then
  log "identify cbase"
  mkdir -p "${ROOT}/${C}/id_cbase${IDSUF}"
  cb_arg=(); [ -f "$CB_Q" ] && cb_arg=(-cb "$CB_Q")
  "$DIALECT" identify -c "${ROOT}/${C}/count_matrix.csv" -b "${ROOT}/${C}/bmr_pmfs.csv" \
    -o "${ROOT}/${C}/id_cbase${IDSUF}" -k "$TOP_K" "${cb_arg[@]+"${cb_arg[@]}"}" >>"$LOGF" 2>&1 \
    || log "STAGE-FAIL id_cbase${IDSUF}"
fi
if [ ! -f "${ROOT}/${C}/id_dig${IDSUF}/pairwise_interaction_results.csv" ] \
   && [ -f "${ROOT}/${C}/bmr_pmfs.dig.csv" ]; then
  log "identify dig"
  mkdir -p "${ROOT}/${C}/id_dig${IDSUF}"
  "$DIALECT" identify -c "${ROOT}/${C}/count_matrix.csv" -b "${ROOT}/${C}/bmr_pmfs.dig.csv" \
    -o "${ROOT}/${C}/id_dig${IDSUF}" -k "$TOP_K" >>"$LOGF" 2>&1 || log "STAGE-FAIL id_dig${IDSUF}"
fi

# 4. MutSig2CV (Octave-patched source -> PROPER per-sample lambda) + identify --
#    Replaces the old Docker compiled binary + scalar-f_p reconstruction. Skipped
#    when SKIP_MUTSIG is set. MUTSIG_ROOT holds the per-cohort lambda dumps.
MUTSIG_ROOT="${MUTSIG_ROOT:-output/mutsigsrc}"
if [ -n "${SKIP_MUTSIG:-}" ]; then
  log "skip mutsig (SKIP_MUTSIG set)"
else
  if [ ! -f "${MUTSIG_ROOT}/${C}/persample_lambda.f32" ]; then
    log "MutSig2CV (Octave-patched, per-sample lambda)"
    bash scripts/run_mutsig_octave.sh "$C" "$MAF_DIR" "$MUTSIG_ROOT" >>"$LOGF" 2>&1 \
      || log "STAGE-FAIL mutsig"
  else log "skip mutsig (lambda done)"; fi
  if [ ! -f "${ROOT}/${C}/id_mutsig${IDSUF}/pairwise_interaction_results.csv" ] \
     && [ -f "${MUTSIG_ROOT}/${C}/persample_lambda.f32" ]; then
    log "identify mutsig (proper per-sample lambda)"
    "$PY" -m analysis.mutsig_lambda_co --cohort "$C" --results-root "$ROOT" \
      --mutsig-root "$MUTSIG_ROOT" --suffix "mutsig${IDSUF}" -k "$TOP_K" >>"$LOGF" 2>&1 \
      || log "STAGE-FAIL id_mutsig${IDSUF}"
  fi
fi
log "cohort pipeline DONE"
