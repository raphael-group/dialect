Installation
============

Core package
------------

The wheel configured in ``pyproject.toml`` contains the installable ``dialect``
package, CLI metadata, and license. The configured source distribution contains
the source package, tests, and selected README, license, and build metadata.
Neither distribution contains the ``external/CBaSE`` runtime scripts or
auxiliary data. A wheel is therefore sufficient for ``dialect identify`` with
existing ``count_matrix.csv`` and ``bmr_pmfs.csv`` inputs, but not for CBaSE
generation.

Default CBaSE workflow
----------------------

Run the default ``dialect generate --bmr cbase`` workflow from a Git checkout
installed in editable mode:

.. code-block:: bash

   git clone https://github.com/raphael-group/dialect.git
   cd dialect
   python -m pip install -e .

Use ``python -m pip install -e ".[dev]"`` for the test and development tools.

CBaSE auxiliary-data boundary
-----------------------------

The checkout supplies DIALECT's tracked CBaSE fork and
``external/CBaSE/NOTICE``. It intentionally does not track the large
``external/CBaSE/auxiliary/`` directory. Provision a compatible CBaSE auxiliary
data set at that exact path before generation; DIALECT does not automate its
acquisition. The notice records the upstream landing page and archive identity.
The provider resolves both the tracked fork and the auxiliary data relative to
the editable checkout, so a wheel, non-editable install, or source distribution
is not sufficient for this workflow.

Review ``external/CBaSE/NOTICE`` before use or redistribution. Its current
official CBaSE v1.2 archive is a comparison and provenance reference, not a
claimed byte-identical parent of DIALECT's historical two-script fork. Keep the
tracked DIALECT fork scripts; do not replace them with the current archive's
scripts.

This source-checkout and external-data requirement applies to CBaSE BMR
generation, not to interaction identification from existing inputs. None of the
configured Python distributions is a self-contained CBaSE data release.

.. note:: See the :doc:`usage/index` section for provider-specific contracts and
   data preparation.
