#!/usr/bin/env bash
# Drive the per-cohort pipeline across every cohort with a pancan MAF.
#
# Two passes so the fast results land first:
#   Pass A: CBaSE + DIG + identify for ALL cohorts (SKIP_MUTSIG=1)  -> ~hours
#   Pass B: MutSig2CV (Docker) + identify-mutsig for ALL cohorts    -> ~a day (emulated)
# Everything is idempotent, so re-running resumes where it left off.
set -u
ROOT="output/pancan"
mafs=$(ls data/mafs_pancan/*.maf 2>/dev/null)

echo "##### PASS A (CBaSE + DIG + identify) $(date) #####"
for maf in $mafs; do
  C=$(basename "$maf" .maf)
  echo "----- passA ${C} $(date +%H:%M:%S) -----"
  SKIP_MUTSIG=1 bash scripts/run_cohort_pipeline.sh "$C"
done

echo "##### PASS B (MutSig2CV + identify-mutsig) $(date) #####"
for maf in $mafs; do
  C=$(basename "$maf" .maf)
  echo "----- passB ${C} $(date +%H:%M:%S) -----"
  bash scripts/run_cohort_pipeline.sh "$C"
done
echo "##### ALL PANCAN COHORTS DONE $(date) #####"
