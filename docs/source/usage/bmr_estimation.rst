Estimating Background Mutation Rates (BMR)
==========================================

DIALECT provides an interface with external methods for estimating BMR distributions.

CBaSE auxiliary-data requirement
--------------------------------

The CBaSE provider is the default for ``dialect generate``. It requires a Git
checkout installed in editable mode because the runtime resolves DIALECT's
tracked CBaSE fork under ``external/CBaSE/``. Neither the configured wheel nor
source distribution contains those runtime scripts or the auxiliary data.

The checkout does not track the large ``external/CBaSE/auxiliary/`` directory.
Provision a compatible auxiliary data set there separately; DIALECT does not
automate its acquisition, and the notice records the upstream landing page and
archive identity. For the default hg19 trinucleotide workflow, the fork reads
these paths:

* ``triplets_user.txt``
* ``COSMIC_genes_v80.txt``
* ``used_genes_new_CBaSE.txt``
* ``abundances_trinucleotides_tx.txt``
* ``context_alt_effect_by_gene_new_encoding_hg19_trinucleotides.txt.gz``
* ``gene_annotations_hg19/chr*.txt.gz``

The hg38 reference choice requires its corresponding context and annotation
files. Review ``external/CBaSE/NOTICE`` before use or redistribution. The current
official v1.2 archive recorded there is a comparison and provenance reference,
not a claimed byte-identical parent of DIALECT's historical fork; keep the
tracked DIALECT scripts rather than replacing them with the archive's scripts.

This requirement does not apply to ``dialect identify`` when the count matrix
and BMR PMFs already exist.

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
