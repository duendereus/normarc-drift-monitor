"""Monitoring agent — orchestrates the full governance pipeline.

Loads inference logs into DuckDB, iterates week by week computing PSI and ECE,
generates PDF reports, and sends Telegram alerts when drift is detected.
"""

import os
import sys
from datetime import timedelta
from pathlib import Path

import duckdb
import numpy as np
from dotenv import load_dotenv

# Agregar root del proyecto al path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.psi import compute_psi_for_week
from metrics.ece import compute_ece_for_week, get_calibration_curve
from reports.report_generator import generate_report

load_dotenv(PROJECT_ROOT / ".env")

# --- Configuración desde .env ---
PSI_THRESHOLD: float = float(os.getenv("PSI_THRESHOLD", "0.20"))
ECE_THRESHOLD: float = float(os.getenv("ECE_THRESHOLD", "0.05"))
BASELINE_WEEKS: int = int(os.getenv("BASELINE_WEEKS", "8"))
TELEGRAM_BOT_TOKEN: str | None = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID: str | None = os.getenv("TELEGRAM_CHAT_ID")

# Features a monitorear con PSI
MONITORED_FEATURES: list[str] = ["ingreso_mensual", "edad", "meses_historial", "score"]

CSV_PATH: Path = PROJECT_ROOT / "data" / "inference_logs.csv"
DB_PATH: str = ":memory:"
TABLE_NAME: str = "inference_logs"


def load_data(con: duckdb.DuckDBPyConnection) -> None:
    """Load CSV into DuckDB table.

    Args:
        con: DuckDB connection.
    """
    con.execute(f"""
        CREATE TABLE {TABLE_NAME} AS
        SELECT * FROM read_csv_auto('{CSV_PATH.as_posix()}',
            types={{
                'fecha': 'DATE',
                'cliente_id': 'VARCHAR',
                'score': 'DOUBLE',
                'prob_default': 'DOUBLE',
                'ingreso_mensual': 'DOUBLE',
                'edad': 'INTEGER',
                'meses_historial': 'INTEGER',
                'default_real': 'INTEGER'
            }}
        )
    """)
    count = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    print(f"✓ Loaded {count} records into DuckDB")


def get_week_boundaries(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str]]:
    """Get start and end dates for each week in the dataset.

    Args:
        con: DuckDB connection.

    Returns:
        List of (week_start, week_end) date string tuples.
    """
    rows = con.execute(f"""
        SELECT
            MIN(fecha) as week_start,
            MAX(fecha) as week_end
        FROM (
            SELECT fecha,
                   FLOOR(DATEDIFF('day', (SELECT MIN(fecha) FROM {TABLE_NAME}), fecha) / 7) as week_num
            FROM {TABLE_NAME}
        )
        GROUP BY week_num
        ORDER BY week_num
    """).fetchall()

    return [(str(r[0]), str(r[1])) for r in rows]


def send_telegram_alert(message: str, report_path: Path | None = None) -> None:
    """Send alert via Telegram bot.

    Gracefully skips if credentials are not configured.

    Args:
        message: Alert text.
        report_path: Optional path to PDF report to attach.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("  ⚠ Telegram not configured — skipping alert")
        return

    if TELEGRAM_BOT_TOKEN == "your_token":
        print("  ⚠ Telegram token is placeholder — skipping alert")
        return

    try:
        import asyncio
        from telegram import Bot

        bot = Bot(token=TELEGRAM_BOT_TOKEN)

        async def _send() -> None:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
            if report_path and report_path.exists() and report_path.suffix == ".pdf":
                with open(report_path, "rb") as f:
                    await bot.send_document(chat_id=TELEGRAM_CHAT_ID, document=f)

        asyncio.run(_send())
        print("  ✓ Telegram alert sent")
    except Exception as e:
        print(f"  ⚠ Telegram alert failed: {e}")


def run_monitor() -> None:
    """Execute the full monitoring pipeline."""
    if not CSV_PATH.exists():
        print(f"✗ Data file not found: {CSV_PATH}")
        print("  Run: python data/generate_synthetic.py")
        sys.exit(1)

    con = duckdb.connect(DB_PATH)
    load_data(con)

    weeks = get_week_boundaries(con)
    total_weeks = len(weeks)
    print(f"\n{'='*60}")
    print(f"  Normarc Drift Monitor — {total_weeks} weeks detected")
    print(f"  Baseline: weeks 0-{BASELINE_WEEKS - 1} | Monitoring: weeks {BASELINE_WEEKS}-{total_weeks - 1}")
    print(f"  PSI threshold: {PSI_THRESHOLD} | ECE threshold: {ECE_THRESHOLD}")
    print(f"{'='*60}\n")

    # Periodo baseline
    baseline_start = weeks[0][0]
    baseline_end = weeks[BASELINE_WEEKS - 1][1]

    drift_weeks: list[int] = []
    stable_weeks: list[int] = []

    # Iterar semanas post-baseline
    for week_idx in range(BASELINE_WEEKS, total_weeks):
        week_start, week_end = weeks[week_idx]
        print(f"Week {week_idx} ({week_start} → {week_end})")

        # --- Calcular PSI ---
        psi_results = compute_psi_for_week(
            con, TABLE_NAME,
            baseline_start, baseline_end,
            week_start, week_end,
            MONITORED_FEATURES,
        )

        max_psi_feature = max(psi_results, key=psi_results.get)
        max_psi_value = psi_results[max_psi_feature]

        # --- Calcular ECE ---
        ece_value = compute_ece_for_week(con, TABLE_NAME, week_start, week_end)

        # --- Evaluar drift ---
        psi_drift = max_psi_value > PSI_THRESHOLD
        ece_drift = ece_value > ECE_THRESHOLD
        has_drift = psi_drift or ece_drift

        # --- Curva de calibración para el reporte ---
        mean_pred, frac_pos, bin_counts = get_calibration_curve(
            con, TABLE_NAME, week_start, week_end
        )

        # --- Generar reporte PDF ---
        report_path = generate_report(
            week_num=week_idx,
            week_start=week_start,
            week_end=week_end,
            psi_results=psi_results,
            psi_threshold=PSI_THRESHOLD,
            ece_value=ece_value,
            ece_threshold=ECE_THRESHOLD,
            mean_predicted=mean_pred,
            fraction_positive=frac_pos,
            bin_counts=bin_counts,
            has_drift=has_drift,
        )

        # --- Output y alertas ---
        if has_drift:
            drift_weeks.append(week_idx)
            alert_parts = []
            if psi_drift:
                alert_parts.append(f"PSI {max_psi_feature}: {max_psi_value:.4f}")
            if ece_drift:
                alert_parts.append(f"ECE: {ece_value:.4f}")

            detail = " | ".join(alert_parts)
            print(f"  ⚠ DRIFT DETECTED — {detail}")
            print(f"  Report: {report_path}")

            # Alerta Telegram
            telegram_msg = (
                f"⚠️ Drift detected — Week {week_idx}\n"
                f"Period: {week_start} to {week_end}\n"
                f"{detail}\n"
                f"Action: Model revalidation recommended."
            )
            send_telegram_alert(telegram_msg, report_path)
        else:
            stable_weeks.append(week_idx)
            print(f"  ✓ Stable — ECE: {ece_value:.4f} | Max PSI ({max_psi_feature}): {max_psi_value:.4f}")
            print(f"  Report: {report_path}")

        print()

    # --- Resumen final ---
    print(f"{'='*60}")
    print(f"  SUMMARY")
    print(f"  Stable weeks: {len(stable_weeks)} | Drift weeks: {len(drift_weeks)}")
    if drift_weeks:
        print(f"  First drift at week: {drift_weeks[0]}")
    print(f"{'='*60}")

    con.close()


if __name__ == "__main__":
    run_monitor()
