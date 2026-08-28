"""Explicit cohort-size support for the vendored CBaSE command-line scripts.

CBaSE historically inferred the cohort size from samples with at least one retained
mutation.  A cohort may also contain mutation-free samples, so revision analyses can
pass the complete cohort size explicitly while legacy invocations retain the original
inference behavior.
"""

from __future__ import annotations

from collections.abc import Sequence

N_SAMPLES_FLAG = "--n-samples"

_OUTPUT_HEADER_PREFIX = (
    "gene\tl_m\tl_k\tl_s\tm_obs\tk_obs\ts_obs\tL_gene\tlambda_s"
    "\ts_max_per_sample"
)
_OUTPUT_N_SAMPLES_PREFIX = "N_samples="


def parse_explicit_n_samples(arguments: Sequence[str]) -> int | None:
    """Parse the optional named cohort-size argument following CBaSE's legacy args."""
    if not arguments:
        return None
    if len(arguments) != 2 or arguments[0] != N_SAMPLES_FLAG:
        msg = (
            "Optional CBaSE arguments must be exactly "
            f"'{N_SAMPLES_FLAG} POSITIVE_INTEGER'."
        )
        raise ValueError(msg)
    try:
        n_samples = int(arguments[1])
    except ValueError as err:
        msg = f"{N_SAMPLES_FLAG} must be a positive integer."
        raise ValueError(msg) from err
    if n_samples <= 0:
        msg = f"{N_SAMPLES_FLAG} must be a positive integer."
        raise ValueError(msg)
    return n_samples


def resolve_n_samples(
    mutation_bearing_n_samples: int,
    explicit_n_samples: int | None,
) -> int:
    """Resolve cohort size and reject an explicit size below observed membership."""
    if mutation_bearing_n_samples < 0:
        msg = "Mutation-bearing sample count cannot be negative."
        raise ValueError(msg)
    if explicit_n_samples is None:
        if mutation_bearing_n_samples == 0:
            msg = "Cannot infer CBaSE cohort size from zero mutation-bearing samples."
            raise ValueError(msg)
        return mutation_bearing_n_samples
    if explicit_n_samples <= 0:
        msg = "Explicit cohort size must be a positive integer."
        raise ValueError(msg)
    if explicit_n_samples < mutation_bearing_n_samples:
        msg = (
            f"Explicit cohort size ({explicit_n_samples}) is smaller than the "
            "number of mutation-bearing samples retained by CBaSE "
            f"({mutation_bearing_n_samples})."
        )
        raise ValueError(msg)
    return explicit_n_samples


def output_data_preparation_header(n_samples: int) -> str:
    """Build the header consumed by ``CBaSE_qvals_v1.2.py``."""
    if n_samples <= 0:
        msg = "Cohort size written to the CBaSE header must be positive."
        raise ValueError(msg)
    return f"{_OUTPUT_HEADER_PREFIX}\t{_OUTPUT_N_SAMPLES_PREFIX}{n_samples}\n"


def parse_output_data_preparation_n_samples(header: str) -> int:
    """Read the positive cohort size persisted by the params entrypoint."""
    field = header.rstrip("\r\n").split("\t")[-1]
    if not field.startswith(_OUTPUT_N_SAMPLES_PREFIX):
        msg = "CBaSE data-preparation header is missing its N_samples field."
        raise ValueError(msg)
    raw_n_samples = field.removeprefix(_OUTPUT_N_SAMPLES_PREFIX)
    try:
        n_samples = int(raw_n_samples)
    except ValueError as err:
        msg = "CBaSE data-preparation N_samples field must be a positive integer."
        raise ValueError(msg) from err
    if n_samples <= 0:
        msg = "CBaSE data-preparation N_samples field must be a positive integer."
        raise ValueError(msg)
    return n_samples


def per_sample_rate(cohort_rate: float, n_samples: int) -> float:
    """Convert a cohort-wide CBaSE rate to its homogeneous per-sample rate."""
    if n_samples <= 0:
        msg = "CBaSE cohort size must be positive before per-sample division."
        raise ValueError(msg)
    return cohort_rate / n_samples
