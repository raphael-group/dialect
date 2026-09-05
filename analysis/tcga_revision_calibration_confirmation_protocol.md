# Focused calibration confirmation protocol

This protocol is fixed before opening any observed DIALECT pairwise result.
It preserves the completed v4 calibration tree byte-for-byte and uses it only
as fitted-null simulation evidence plus production-integrity metadata.

## Stage 1

The implementation fully validates and recomputes all 2,048 primary MutSig
cohort--sentinel-pair--alpha endpoints from the existing 10,000-replicate
calibration tasks. For every endpoint, it constructs a one-sided exact
Clopper--Pearson upper confidence bound using endpoint error
`0.025 / 2,048`. Every endpoint whose upper bound exceeds its predeclared
acceptance bound (0.02 at alpha 0.01; 0.07 at alpha 0.05) is selected. The
complete selection is frozen in a write-once run manifest before confirmation
simulation begins.

## Stage 2

If stage 1 selects `M` endpoints, each receives 100,000 new fitted-null
replicates generated with a new root seed and endpoint- and shard-specific
SHA-256 domain separation. Its one-sided exact Clopper--Pearson upper bound uses
endpoint error `0.025 / M`. Simulation, the one-degree-of-freedom profile LRT,
and the policy assigning p=1 to nonreportable fits are unchanged from v4.

Stage-1 and stage-2 counts are not pooled. There is no outcome-dependent change
to the thresholds and no third stage. The composite gate passes only when every
unselected endpoint passed stage 1 and every selected endpoint passes stage 2.
The union-bound familywise-error budget is at most 0.05 across both stages.

## Execution and evidence

Only one selected endpoint can run at a time. It uses five single-threaded
spawned workers on the frozen 14-logical-CPU host, below the half-machine limit
of seven. Run, task, final-table, and summary artifacts are atomic and
write-once. Validation recomputes both stages and every digest before returning
the composite gate decision. Observed association files are never parsed or
inspected; their hashes are used only to preserve the production-run integrity
binding.
