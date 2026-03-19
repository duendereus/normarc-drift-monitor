"""Expected Calibration Error (ECE) calculation.

Measures how well predicted probabilities match observed default rates.
Uses equal-width bins (n=10) as specified in the governance framework.
"""

import duckdb
import numpy as np


ECE_BINS: int = 10


def compute_ece(
    prob_predicted: np.ndarray,
    labels: np.ndarray,
    bins: int = ECE_BINS,
) -> float:
    """Compute Expected Calibration Error.

    Args:
        prob_predicted: Predicted probabilities of default (0-1).
        labels: Binary outcomes (1=default, 0=no default).
        bins: Number of equal-width bins.

    Returns:
        ECE value (float). Lower is better.
    """
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n_total = len(labels)

    if n_total == 0:
        return 0.0

    for i in range(bins):
        # Registros en este bin
        mask = (prob_predicted >= bin_edges[i]) & (prob_predicted < bin_edges[i + 1])
        # Incluir el borde superior en el último bin
        if i == bins - 1:
            mask = (prob_predicted >= bin_edges[i]) & (prob_predicted <= bin_edges[i + 1])

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        # Confianza promedio vs frecuencia observada
        avg_confidence = prob_predicted[mask].mean()
        avg_accuracy = labels[mask].mean()

        ece += (n_bin / n_total) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def compute_ece_for_week(
    con: duckdb.DuckDBPyConnection,
    table: str,
    week_start: str,
    week_end: str,
) -> float:
    """Compute ECE for a specific week's predictions.

    Args:
        con: DuckDB connection.
        table: Name of the inference logs table.
        week_start: Start date of the week (inclusive).
        week_end: End date of the week (inclusive).

    Returns:
        ECE value for the week.
    """
    result = con.execute(
        f"SELECT prob_default, default_real FROM {table} "
        f"WHERE fecha >= ? AND fecha <= ?",
        [week_start, week_end],
    ).fetchnumpy()

    prob_predicted = np.asarray(result["prob_default"], dtype=float)
    labels = np.asarray(result["default_real"], dtype=float)

    if len(prob_predicted) == 0:
        return 0.0

    return compute_ece(prob_predicted, labels)


def get_calibration_curve(
    con: duckdb.DuckDBPyConnection,
    table: str,
    week_start: str,
    week_end: str,
    bins: int = ECE_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get calibration curve data for plotting the reliability diagram.

    Args:
        con: DuckDB connection.
        table: Name of the inference logs table.
        week_start: Start date (inclusive).
        week_end: End date (inclusive).
        bins: Number of equal-width bins.

    Returns:
        Tuple of (mean_predicted, fraction_positive, bin_counts).
    """
    result = con.execute(
        f"SELECT prob_default, default_real FROM {table} "
        f"WHERE fecha >= ? AND fecha <= ?",
        [week_start, week_end],
    ).fetchnumpy()

    prob_predicted = np.asarray(result["prob_default"], dtype=float)
    labels = np.asarray(result["default_real"], dtype=float)

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    mean_predicted = np.zeros(bins)
    fraction_positive = np.zeros(bins)
    bin_counts = np.zeros(bins, dtype=int)

    for i in range(bins):
        if i == bins - 1:
            mask = (prob_predicted >= bin_edges[i]) & (prob_predicted <= bin_edges[i + 1])
        else:
            mask = (prob_predicted >= bin_edges[i]) & (prob_predicted < bin_edges[i + 1])

        n_bin = mask.sum()
        bin_counts[i] = n_bin

        if n_bin > 0:
            mean_predicted[i] = prob_predicted[mask].mean()
            fraction_positive[i] = labels[mask].mean()

    return mean_predicted, fraction_positive, bin_counts
