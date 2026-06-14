#!/usr/bin/env bash
# Master driver: run the full DIALECT pipeline (CBaSE + DIG + proper Octave-MutSig
# per-sample lambda + DIALECT x3) across EVERY cohort -- all TCGA PanCancer cohorts
# then all MSK per-cancer-type sub-cohorts. Fully idempotent: re-running resumes
# where it left off (completed stages are skipped). Sequential to avoid CPU thrash.
set -u
echo "########## RUN EVERYTHING -- start $(date) ##########"

echo "===== TCGA PanCancer ($(ls data/mafs_pancan/*.maf 2>/dev/null | wc -l) cohorts) ====="
bash scripts/run_all_pancan.sh

echo "===== MSK (IMPACT 2026 + CHORD 2024, per cancer type) ====="
bash scripts/run_all_msk.sh

echo "########## ALL COHORTS DONE $(date) ##########"
