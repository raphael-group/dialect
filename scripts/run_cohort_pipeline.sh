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
set -u
C="$1"
ROOT="output/pancan"
MAF="data/mafs_pancan/${C}.maf"
DIG_RESULTS="external/DIGDriver/run/Pancan.genes.results.txt"
DIALECT="/opt/anaconda3/envs/dialect/bin/dialect"
PY="/opt/anaconda3/envs/dialect/bin/python"
MCR="/opt/mcr/v81"
LDP="${MCR}/runtime/glnxa64:${MCR}/bin/glnxa64:${MCR}/sys/os/glnxa64:${MCR}/sys/java/jre/glnxa64/jre/lib/amd64/server"

log() { echo "[$(date +%H:%M:%S)] ${C}: $*"; }

[ -f "$MAF" ] || { log "no MAF at ${MAF}; skipping cohort"; exit 0; }
mkdir -p "${ROOT}/${C}"

# 1. CBaSE ------------------------------------------------------------------
if [ ! -f "${ROOT}/${C}/bmr_pmfs.csv" ]; then
  log "CBaSE generate"
  "$DIALECT" generate -m "$MAF" -o "${ROOT}/${C}" --bmr cbase -r hg19 || log "STAGE-FAIL cbase"
else log "skip cbase"; fi

N=0
[ -f "${ROOT}/${C}/count_matrix.csv" ] && N=$(( $(wc -l < "${ROOT}/${C}/count_matrix.csv") - 1 ))

# 2. DIG --------------------------------------------------------------------
if [ ! -f "${ROOT}/${C}/bmr_pmfs.dig.csv" ] && [ "$N" -gt 0 ]; then
  log "DIG generate (N=${N})"
  rm -rf "/tmp/dig_${C}"
  if "$DIALECT" generate -m "$MAF" -o "/tmp/dig_${C}" --bmr dig \
       --dig-results "$DIG_RESULTS" --dig-samples "$N" -r hg19; then
    cp "/tmp/dig_${C}/bmr_pmfs.csv" "${ROOT}/${C}/bmr_pmfs.dig.csv"
  else log "STAGE-FAIL dig"; fi
else log "skip dig"; fi

# 3. MutSig2CV (Docker) -----------------------------------------------------
if [ ! -f "${ROOT}/${C}_mutsig/results.mat" ]; then
  log "MutSig2CV (Docker, emulated)"
  mkdir -p "${ROOT}/${C}_mutsig"
  docker run --rm -v "${PWD}:/work" -w /work/external/MutSig2CV/mutsig2cv \
    -e LD_LIBRARY_PATH="$LDP" -e MCR_CACHE_ROOT="/tmp/mcr_${C}" \
    flywheel/matlab-mcr:v81 \
    ./MutSig2CV "/work/${MAF}" "/work/${ROOT}/${C}_mutsig" || log "STAGE-FAIL mutsig"
else log "skip mutsig"; fi

# 4. DIALECT identify (cbase, dig, mutsig) ----------------------------------
CB_Q="${ROOT}/${C}/CBaSE_output/q_values.txt"
if [ ! -f "${ROOT}/${C}/id_cbase/pairwise_interaction_results.csv" ] \
   && [ -f "${ROOT}/${C}/bmr_pmfs.csv" ]; then
  log "identify cbase"
  mkdir -p "${ROOT}/${C}/id_cbase"
  cb_arg=(); [ -f "$CB_Q" ] && cb_arg=(-cb "$CB_Q")
  "$DIALECT" identify -c "${ROOT}/${C}/count_matrix.csv" -b "${ROOT}/${C}/bmr_pmfs.csv" \
    -o "${ROOT}/${C}/id_cbase" -k 100 "${cb_arg[@]}" || log "STAGE-FAIL id_cbase"
fi
if [ ! -f "${ROOT}/${C}/id_dig/pairwise_interaction_results.csv" ] \
   && [ -f "${ROOT}/${C}/bmr_pmfs.dig.csv" ]; then
  log "identify dig"
  mkdir -p "${ROOT}/${C}/id_dig"
  "$DIALECT" identify -c "${ROOT}/${C}/count_matrix.csv" -b "${ROOT}/${C}/bmr_pmfs.dig.csv" \
    -o "${ROOT}/${C}/id_dig" -k 100 || log "STAGE-FAIL id_dig"
fi
if [ ! -f "${ROOT}/${C}/id_mutsig/pairwise_interaction_results.csv" ] \
   && [ -f "${ROOT}/${C}_mutsig/results.mat" ]; then
  log "identify mutsig (per-sample extractor)"
  "$PY" -m analysis.mutsig_persample_co --cohort "$C" --results-root "$ROOT" -k 100 \
    || log "STAGE-FAIL id_mutsig"
fi
log "cohort pipeline DONE"
