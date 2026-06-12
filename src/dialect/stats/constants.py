"""Per-method ranking criteria and significance thresholds for ME/CO post-processing.

Each ``*_RANKING_CRITERIA`` maps a method name to ``(metric_column, sort_order)``;
each ``*_SIGNIFICANCE_THRESHOLDS`` maps a method to ``(q/p-value column, cutoff)``.
"""

ME_METHOD_RANKING_CRITERIA = {
    "DIALECT (Rho)": ("Rho", "ascending"),
    "DIALECT (LRT)": ("Likelihood Ratio", "descending"),
    "DIALECT (LOR)": ("Log Odds Ratio", "descending"),
    "DIALECT (Wald)": ("Wald Statistic", "descending"),
    "DISCOVER": ("Discover ME P-Val", "ascending"),
    "Fisher's Exact Test": ("Fisher's ME P-Val", "ascending"),
    "MEGSA": ("MEGSA S-Score (LRT)", "descending"),
    "WeSME": ("WeSME P-Val", "ascending"),
}

CO_METHOD_RANKING_CRITERIA = {
    "DIALECT (Rho)": ("Rho", "descending"),
    "DIALECT (LRT)": ("Likelihood Ratio", "descending"),
    "DIALECT (LOR)": ("Log Odds Ratio", "ascending"),
    "DIALECT (Wald)": ("Wald Statistic", "ascending"),
    "DISCOVER": ("Discover CO P-Val", "ascending"),
    "Fisher's Exact Test": ("Fisher's CO P-Val", "ascending"),
    "WeSCO": ("WeSCO P-Val", "ascending"),
}

ME_METHOD_SIGNIFICANCE_THRESHOLDS = {
    "DISCOVER": ("Discover ME Q-Val", 0.01),
    "Fisher's Exact Test": ("Fisher's ME Q-Val", 0.01),
    "MEGSA": ("MEGSA P-Val", 1e-3),
    "WeSME": ("WeSME Q-Val", 0.01),
}

CO_METHOD_SIGNIFICANCE_THRESHOLDS = {
    "DISCOVER": ("Discover CO Q-Val", 0.01),
    "Fisher's Exact Test": ("Fisher's CO Q-Val", 0.01),
    "WeSCO": ("WeSCO Q-Val", 0.01),
}
