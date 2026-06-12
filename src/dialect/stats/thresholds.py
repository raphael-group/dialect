"""Significance thresholds for DIALECT's marginal driver-rate (epsilon) calls."""

from math import sqrt

from scipy.optimize import brentq
from scipy.stats import norm


def compute_epsilon_threshold(num_samples: int, alpha: float = 0.001) -> float:
    r"""Compute the epsilon threshold for the one-sided normal approximation CI.

    We solve the equation:

    .. math::

        \\max\\Bigl(\\epsilon - z_{(1 - \\alpha)}
        \\sqrt{\\frac{\\epsilon (1 - \\epsilon)}{n}}, 0\\Bigr) = 0

    for \\(\\epsilon \\in [0, 1]\\), where:
    - \\(n\\) = ``num_samples``,
    - \\(\\alpha\\) is the one-sided significance level,
    - \\(z_{(1 - \\alpha)}\\) is critical value from the standard normal distribution
      at the \\((1 - \\alpha)\\) quantile.

    **Parameters**:
      :param num_samples: (int) Number of samples (\\(n\\))
      :param alpha: (float) Significance level for the one-sided interval
                    (default 0.001 for a 99.9% one-sided CI)

    **Returns**:
      :return: (float) The marginal driver mutation rate threshold \\(\\epsilon\\) s.t.
               the lower bound of the one-sided normal CI is exactly 0.
    """
    z = norm.ppf(1 - alpha)

    def f(eps: float) -> float:
        return eps - z * sqrt(eps * (1 - eps) / num_samples)

    return brentq(f, 1e-6, 1.0)
