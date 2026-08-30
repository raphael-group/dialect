Estimating Background Mutation Rates (BMR)
==========================================

DIALECT provides an interface with external methods for estimating BMR distributions.

DIG finite-support contract
---------------------------

The DIG adapter converts each gene- and consequence-specific background to its
native per-sample negative-binomial distribution. All emitted distributions use
one shared inclusive support from zero through ``K``. ``K`` is the larger of the
largest observed count in an existing ``count_matrix.csv`` and the largest
effect-specific negative-binomial quantile whose omitted upper-tail mass is at
most ``tail_eps`` (``1e-7`` by default). The retained native probabilities are
renormalized only after this support is selected.

There is no fixed count cap and no probability floor. ``tail_eps`` controls only
finite-support truncation; it is not an epsilon added to unsupported observations.
When a count matrix is present, the DIG provider also requires its row count to
equal the ``n_samples`` used to convert cohort-level DIG parameters. Gene and
sample axes must be unique, counts must be finite nonnegative integers, ``ALPHA``
and ``THETA`` must be finite and positive, and each consequence fraction must be
finite and lie in ``[0, 1]``. Any invalid row or effect fails the conversion; it
is never silently omitted.
