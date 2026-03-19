"""Population Stability Index (PSI) calculation.

Measures distribution shift between a baseline period and a current period.
Computed per feature and for the score distribution using histogram binning.
"""

from typing import Any

import duckdb
import numpy as np


# Número de bins para el histograma PSI
PSI_BINS: int = 10
# Valor mínimo para evitar log(0) en el cálculo de PSI
EPSILON: float = 1e-4


def compute_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    bins: int = PSI_BINS,
) -> float:
    """Compute PSI between two distributions.

    Uses shared bin edges derived from the baseline distribution to ensure
    consistent comparison.

    Args:
        baseline: Array of values from the baseline period.
        current: Array of values from the current period.
        bins: Number of histogram bins.

    Returns:
        PSI value (float). Higher values indicate greater distributional shift.
    """
    # Bins definidos por el baseline para comparación consistente
    bin_edges = np.histogram_bin_edges(baseline, bins=bins)

    baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
    current_counts, _ = np.histogram(current, bins=bin_edges)

    # Proporciones con suavizado epsilon
    baseline_pct = (baseline_counts + EPSILON) / (baseline_counts.sum() + EPSILON * bins)
    current_pct = (current_counts + EPSILON) / (current_counts.sum() + EPSILON * bins)

    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(psi)


def compute_psi_for_week(
    con: duckdb.DuckDBPyConnection,
    table: str,
    baseline_start: str,
    baseline_end: str,
    current_start: str,
    current_end: str,
    features: list[str],
) -> dict[str, float]:
    """Compute PSI for multiple features comparing baseline vs current period.

    Args:
        con: DuckDB connection.
        table: Name of the inference logs table.
        baseline_start: Start date of baseline period (inclusive).
        baseline_end: End date of baseline period (inclusive).
        current_start: Start date of current period (inclusive).
        current_end: End date of current period (inclusive).
        features: List of column names to compute PSI for.

    Returns:
        Dict mapping feature name to its PSI value.
    """
    results: dict[str, float] = {}

    for feature in features:
        # Extraer valores del baseline
        baseline_vals = con.execute(
            f"SELECT {feature} FROM {table} "
            f"WHERE fecha >= ? AND fecha <= ?",
            [baseline_start, baseline_end],
        ).fetchnumpy()[feature]

        # Extraer valores del periodo actual
        current_vals = con.execute(
            f"SELECT {feature} FROM {table} "
            f"WHERE fecha >= ? AND fecha <= ?",
            [current_start, current_end],
        ).fetchnumpy()[feature]

        if len(baseline_vals) == 0 or len(current_vals) == 0:
            results[feature] = 0.0
            continue

        results[feature] = compute_psi(
            np.asarray(baseline_vals, dtype=float),
            np.asarray(current_vals, dtype=float),
        )

    return results
