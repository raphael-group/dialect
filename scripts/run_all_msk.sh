#!/usr/bin/env bash
# Drive the per-cohort pipeline across the MSK per-cancer-type sub-cohorts.
#
# Reuses run_cohort_pipeline.sh via MAF_DIR + ROOT env vars. All three BMRs are run
# (per Ahmed's choice) -- note CBaSE/MutSig are calibrated for exome, not a ~500-gene
# panel, so for MSK they also document how those BMRs behave on panel data; DIG is the
# methodologically appropriate provider here. Two passes (fast CBaSE+DIG, then MutSig),
# fully idempotent. Run AFTER the TCGA sweep to avoid CPU contention.
set -u

run_study() {
  local split_dir="$1" out_root="$2"
  local mafs; mafs=$(ls "${split_dir}"/*.maf 2>/dev/null)
  echo "##### MSK STUDY ${out_root} ($(echo "$mafs" | wc -w) cohorts) $(date) #####"
  for maf in $mafs; do
    C=$(basename "$maf" .maf)
    echo "----- passA ${out_root} ${C} $(date +%H:%M:%S) -----"
    MAF_DIR="$split_dir" ROOT="$out_root" SKIP_MUTSIG=1 \
      bash scripts/run_cohort_pipeline.sh "$C"
  done
  for maf in $mafs; do
    C=$(basename "$maf" .maf)
    echo "----- passB ${out_root} ${C} $(date +%H:%M:%S) -----"
    MAF_DIR="$split_dir" ROOT="$out_root" \
      bash scripts/run_cohort_pipeline.sh "$C"
  done
}

run_study data/mafs_msk_split/IMPACT2026 output/msk/IMPACT2026
run_study data/mafs_msk_split/CHORD2024  output/msk/CHORD2024
echo "##### ALL MSK COHORTS DONE $(date) #####"
