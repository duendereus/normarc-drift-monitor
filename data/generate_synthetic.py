"""Generate synthetic credit scoring inference logs with injected drift.

Produces a 24-week dataset (~200 records per week) where drift is injected
at week 18 by shifting the ingreso_mensual distribution upward, simulating
a fintech that starts receiving higher-income applicants without revalidating
the model.
"""

import csv
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np


# --- Configuración del generador ---
TOTAL_WEEKS: int = 24
RECORDS_PER_WEEK: int = 500
DRIFT_START_WEEK: int = 18
SEED: int = 42

# Distribuciones base (pre-drift)
INGRESO_MEAN_BASE: float = 15_000.0
INGRESO_STD_BASE: float = 4_000.0

# Distribuciones post-drift
INGRESO_MEAN_DRIFT: float = 22_000.0
INGRESO_STD_DRIFT: float = 5_000.0

# Otras features
EDAD_MEAN: float = 35.0
EDAD_STD: float = 10.0
MESES_HIST_MEAN: float = 48.0
MESES_HIST_STD: float = 24.0

OUTPUT_DIR: Path = Path(__file__).resolve().parent
OUTPUT_FILE: Path = OUTPUT_DIR / "inference_logs.csv"

START_DATE: date = date(2024, 1, 1)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Standard sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


def generate_week(
    rng: np.random.Generator,
    week_num: int,
    week_start: date,
) -> list[dict]:
    """Generate inference records for a single week.

    Args:
        rng: NumPy random generator instance.
        week_num: Week number (0-indexed).
        week_start: First date of the week.

    Returns:
        List of record dicts ready for CSV writing.
    """
    n = RECORDS_PER_WEEK
    is_drift = week_num >= DRIFT_START_WEEK

    # --- Features ---
    if is_drift:
        ingreso = rng.normal(INGRESO_MEAN_DRIFT, INGRESO_STD_DRIFT, n)
    else:
        ingreso = rng.normal(INGRESO_MEAN_BASE, INGRESO_STD_BASE, n)

    ingreso = np.clip(ingreso, 2_000.0, 80_000.0)

    edad = rng.normal(EDAD_MEAN, EDAD_STD, n)
    edad = np.clip(edad, 18, 75).astype(int)

    meses_historial = rng.normal(MESES_HIST_MEAN, MESES_HIST_STD, n)
    meses_historial = np.clip(meses_historial, 0, 240).astype(int)

    # --- Modelo latente de default ---
    # Coeficientes diseñados para que el modelo esté bien calibrado pre-drift
    # y se descalibre post-drift (ingreso alto reduce prob_default, pero
    # el segmento nuevo tiene tasas de default distintas)
    z = (
        -0.8
        + 0.00003 * (ingreso - INGRESO_MEAN_BASE)
        - 0.008 * (edad - 30)
        - 0.003 * (meses_historial - 48)
        + rng.normal(0, 0.15, n)
    )
    prob_default = _sigmoid(z)

    # Score del modelo (0-1000, inversamente proporcional a prob_default)
    score = np.clip((1 - prob_default) * 1000, 0, 1000).round(1)
    prob_default = prob_default.round(4)

    # --- Resultado real ---
    # Pre-drift: bien calibrado. Post-drift: tasa real de default más alta
    # de lo que el modelo predice para ingresos altos.
    if is_drift:
        # El modelo subestima el riesgo del segmento nuevo
        real_prob = np.clip(prob_default + 0.12, 0, 1)
    else:
        real_prob = prob_default

    default_real = (rng.random(n) < real_prob).astype(int)

    # --- Fechas distribuidas en la semana ---
    fechas = [week_start + timedelta(days=int(d)) for d in rng.integers(0, 7, n)]

    records = []
    for i in range(n):
        records.append({
            "fecha": fechas[i].isoformat(),
            "cliente_id": f"CLI-{week_num:02d}-{i:04d}",
            "score": score[i],
            "prob_default": prob_default[i],
            "ingreso_mensual": round(ingreso[i], 2),
            "edad": int(edad[i]),
            "meses_historial": int(meses_historial[i]),
            "default_real": int(default_real[i]),
        })

    return records


def main() -> None:
    """Generate the full synthetic dataset and write to CSV."""
    rng = np.random.default_rng(SEED)

    all_records: list[dict] = []
    for week in range(TOTAL_WEEKS):
        week_start = START_DATE + timedelta(weeks=week)
        records = generate_week(rng, week, week_start)
        all_records.extend(records)

    # Escribir CSV
    fieldnames = [
        "fecha", "cliente_id", "score", "prob_default",
        "ingreso_mensual", "edad", "meses_historial", "default_real",
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"✓ Generated {len(all_records)} records across {TOTAL_WEEKS} weeks")
    print(f"  Drift injected from week {DRIFT_START_WEEK} onward")
    print(f"  Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
