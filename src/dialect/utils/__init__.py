"""Legacy glue retained behind the re-layout.

Only three modules still live here: ``identify`` (EM orchestration, called by
``api``), ``merge`` (result merging, called by ``api``), and ``argument_parser``
(the argparse builder for the ``analysis/`` research scripts). Everything else moved
to ``data`` / ``bmr`` / ``models`` / ``baselines`` / ``stats`` / ``viz`` /
``experiments`` / ``config``; the old re-export shims have been removed.
"""
