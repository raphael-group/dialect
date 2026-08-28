# MutSig2CV source receipt

The DIALECT Octave source patch is based on the upstream repository
`https://github.com/getzlab/MutSig2CV.git` at the immutable commit
`0109e27e70478181695f31ca8dd281bb44f0b3af`.

Reconstruct the ignored working source from the repository root:

```bash
git clone https://github.com/getzlab/MutSig2CV.git external/MutSig2CV_src
git -C external/MutSig2CV_src checkout --detach 0109e27e70478181695f31ca8dd281bb44f0b3af
git -C external/MutSig2CV_src apply --check ../mutsig2cv_octave_dialect.patch
git -C external/MutSig2CV_src apply --index ../mutsig2cv_octave_dialect.patch
git -C external/MutSig2CV_src diff --cached --binary > /tmp/mutsig2cv_reconstructed.patch
cmp external/mutsig2cv_octave_dialect.patch /tmp/mutsig2cv_reconstructed.patch
```

`scripts/run_mutsig_octave.sh` enforces this commit and exact indexed diff
before either accepting an existing receipt or running Octave. The ignored
source clone is never the receipt of record; the tracked patch and this pinned
upstream commit are.
