#!/usr/bin/env bash
# Run the DIALECT-patched MutSig2CV v3.11 source under GNU Octave for one cohort,
# dumping <out_root>/<C>/persample_bmr.mat (per-(gene,patient,effect) background).
# Octave's Java needs a working JVM; corretto-11 reads the FixedWidthBinary jar fine.
#
# Usage: run_mutsig_octave.sh <COHORT> [maf_dir] [out_root]
set -u
C="$1"
MAF_DIR="${2:-data/mafs_pancan}"
OUT_ROOT="${3:-output/mutsigsrc}"
export PATH="/opt/homebrew/bin:$PATH"
export JAVA_HOME="/Library/Java/JavaVirtualMachines/amazon-corretto-11.jdk/Contents/Home"
REPO="$(pwd)"
mkdir -p "${OUT_ROOT}/${C}"
octave --no-gui --eval \
  "addpath('${REPO}/external/MutSig2CV_src'); run_mutsig_persample('${REPO}/${MAF_DIR}/${C}.maf','${REPO}/${OUT_ROOT}/${C}')"
