"""PDF report generator with reliability diagrams and metric summaries.

Generates weekly audit-ready PDF reports using matplotlib for charts
and WeasyPrint for PDF rendering. Styled to match the Normarc brand identity.
"""

import io
import sys
import base64
import subprocess
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def _check_weasyprint() -> bool:
    """Check if WeasyPrint can be imported safely (avoids segfault on misconfigured fontconfig)."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "from weasyprint import HTML; print('ok')"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except Exception:
        return False


OUTPUT_DIR: Path = Path(__file__).resolve().parent / "output"

# --- Paleta Normarc ---
_GOLD = "#d4a853"
_GOLD_LIGHT = "#e8c675"
_GOLD_DIM = "#a78338"
_GOLD_DEEP = "#7a5f28"
_CREAM = "#f5e6c4"
_BG = "#0c0a06"
_BG2 = "#111008"
_BG3 = "#1a170e"
_TEXT_DIM = "#8a7a55"
_GREEN = "#7bc77e"
_RED = "#d47a6a"
_BLUE = "#7aa8d4"

# Normarc logo SVG inline (arc + dots)
_LOGO_SVG = (
    '<svg width="28" height="28" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M8 36 C8 18, 24 4, 40 4" stroke="url(#gR)" stroke-width="3" stroke-linecap="round" fill="none"/>'
    '<line x1="14" y1="28" x2="36" y2="10" stroke="url(#gR)" stroke-width="1.5" stroke-linecap="round" opacity="0.4"/>'
    '<circle cx="40" cy="4" r="3" fill="#d4a853"/>'
    '<circle cx="8" cy="36" r="2.5" fill="#a78338"/>'
    '<defs><linearGradient id="gR" x1="8" y1="36" x2="40" y2="4">'
    '<stop stop-color="#a78338"/><stop offset="1" stop-color="#d4a853"/>'
    '</linearGradient></defs></svg>'
)


def _setup_chart_style() -> None:
    """Configure matplotlib for Normarc dark theme."""
    plt.rcParams.update({
        "figure.facecolor": _BG2,
        "axes.facecolor": _BG3,
        "axes.edgecolor": _GOLD_DEEP,
        "axes.labelcolor": _CREAM,
        "axes.titlepad": 14,
        "text.color": _CREAM,
        "xtick.color": _TEXT_DIM,
        "ytick.color": _TEXT_DIM,
        "grid.color": _GOLD_DEEP,
        "grid.alpha": 0.15,
        "legend.facecolor": _BG3,
        "legend.edgecolor": _GOLD_DEEP,
        "legend.labelcolor": _CREAM,
        "font.size": 10,
    })


def _reliability_diagram_base64(
    mean_predicted: np.ndarray,
    fraction_positive: np.ndarray,
    bin_counts: np.ndarray,
    ece: float,
    week_label: str,
) -> str:
    """Generate reliability diagram and return as base64-encoded PNG.

    Args:
        mean_predicted: Mean predicted probability per bin.
        fraction_positive: Observed fraction of positives per bin.
        bin_counts: Number of samples per bin.
        ece: ECE value for the week.
        week_label: Label for the week (e.g., "Week 12").

    Returns:
        Base64-encoded PNG string.
    """
    _setup_chart_style()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 5.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    mask = bin_counts > 0

    # Línea de calibración perfecta
    ax1.plot([0, 1], [0, 1], color=_GOLD_DEEP, linestyle="--", alpha=0.5,
             label="Perfect calibration", linewidth=1)

    # Barras de calibración del modelo
    ax1.bar(
        mean_predicted[mask],
        fraction_positive[mask],
        width=0.075,
        color=_GOLD,
        alpha=0.85,
        edgecolor=_GOLD_DIM,
        linewidth=0.5,
        label="Model",
        zorder=3,
    )

    ax1.set_ylabel("Fraction of positives", fontsize=10, color=_TEXT_DIM)
    ax1.set_title(f"Reliability Diagram — {week_label}",
                  fontsize=13, fontweight="bold", color=_CREAM, pad=12)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.8)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.1)

    # Histograma de predicciones
    ax2.bar(
        mean_predicted[mask],
        bin_counts[mask],
        width=0.075,
        color=_GOLD_DIM,
        alpha=0.7,
        edgecolor=_GOLD_DEEP,
        linewidth=0.5,
        zorder=3,
    )
    ax2.set_xlabel("Mean predicted probability", fontsize=10, color=_TEXT_DIM)
    ax2.set_ylabel("Count", fontsize=10, color=_TEXT_DIM)
    ax2.grid(True, alpha=0.1)

    plt.tight_layout(pad=1.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _psi_bar_chart_base64(
    psi_results: dict[str, float],
    threshold: float,
    week_label: str,
) -> str:
    """Generate PSI bar chart and return as base64-encoded PNG.

    Args:
        psi_results: Dict mapping feature name to PSI value.
        threshold: PSI alert threshold.
        week_label: Label for the week.

    Returns:
        Base64-encoded PNG string.
    """
    _setup_chart_style()
    fig, ax = plt.subplots(figsize=(7, 3.5))

    features = list(psi_results.keys())
    values = list(psi_results.values())
    colors = [_RED if v > threshold else (_GOLD if v > 0.10 else _GREEN) for v in values]

    bars = ax.barh(features, values, color=colors, alpha=0.85,
                   edgecolor=[c + "80" for c in colors], linewidth=0.5, height=0.55, zorder=3)
    ax.axvline(x=threshold, color=_RED, linestyle="--", alpha=0.6,
               label=f"Threshold ({threshold})", linewidth=1)
    ax.set_xlabel("PSI", fontsize=10, color=_TEXT_DIM)
    ax.set_title(f"Population Stability Index — {week_label}",
                 fontsize=13, fontweight="bold", color=_CREAM, pad=12)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, axis="x", alpha=0.1)

    # Valores al lado de las barras
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max(max(values) * 0.03, 0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=9,
            color=_CREAM,
            fontfamily="monospace",
        )

    # Padding derecho para las etiquetas
    ax.set_xlim(right=max(max(values) * 1.25, threshold * 1.5))
    ax.tick_params(axis="y", labelsize=10)

    plt.tight_layout(pad=1.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_report(
    week_num: int,
    week_start: str,
    week_end: str,
    psi_results: dict[str, float],
    psi_threshold: float,
    ece_value: float,
    ece_threshold: float,
    mean_predicted: np.ndarray,
    fraction_positive: np.ndarray,
    bin_counts: np.ndarray,
    has_drift: bool,
) -> Path:
    """Generate a weekly PDF audit report.

    Args:
        week_num: Week number in the monitoring period.
        week_start: Start date of the week.
        week_end: End date of the week.
        psi_results: PSI values per feature.
        psi_threshold: Configured PSI threshold.
        ece_value: ECE for the week.
        ece_threshold: Configured ECE threshold.
        mean_predicted: Calibration curve - mean predicted per bin.
        fraction_positive: Calibration curve - observed fraction per bin.
        bin_counts: Calibration curve - sample count per bin.
        has_drift: Whether drift was detected this week.

    Returns:
        Path to the generated PDF file.
    """
    week_label = f"Week {week_num}"
    status = "DRIFT DETECTED" if has_drift else "STABLE"
    status_color = _RED if has_drift else _GREEN

    # Generar gráficos con paleta Normarc
    reliability_img = _reliability_diagram_base64(
        mean_predicted, fraction_positive, bin_counts, ece_value, week_label
    )
    psi_img = _psi_bar_chart_base64(psi_results, psi_threshold, week_label)

    # Max PSI para el resumen
    max_psi_feature = max(psi_results, key=psi_results.get)
    max_psi_value = psi_results[max_psi_feature]

    # Tabla de PSI
    psi_rows = ""
    for feature, value in psi_results.items():
        if value > psi_threshold:
            row_color = f"rgba(212, 122, 106, 0.08)"
            badge = f'<span class="badge badge-alert">ALERT</span>'
            val_color = _RED
        elif value > 0.10:
            row_color = f"rgba(212, 168, 83, 0.06)"
            badge = f'<span class="badge badge-warn">MONITOR</span>'
            val_color = _GOLD
        else:
            row_color = "transparent"
            badge = f'<span class="badge badge-ok">OK</span>'
            val_color = _GREEN

        psi_rows += (
            f'<tr style="background:{row_color};">'
            f'<td class="td-feature">{feature}</td>'
            f'<td class="td-value" style="color:{val_color};">{value:.4f}</td>'
            f'<td class="td-badge">{badge}</td>'
            f'</tr>\n'
        )

    # ECE badge
    if ece_value > ece_threshold:
        ece_badge = f'<span class="badge badge-alert">ALERT</span>'
        ece_color = _RED
    else:
        ece_badge = f'<span class="badge badge-ok">OK</span>'
        ece_color = _GREEN

    # Status icon SVG
    if has_drift:
        status_icon = (
            '<svg width="20" height="20" viewBox="0 0 20 20" fill="none">'
            f'<path d="M10 2L18 17H2L10 2Z" stroke="{_RED}" stroke-width="1.5" fill="{_RED}20"/>'
            f'<text x="10" y="14" text-anchor="middle" fill="{_CREAM}" font-size="10" font-weight="bold">!</text>'
            '</svg>'
        )
    else:
        status_icon = (
            '<svg width="20" height="20" viewBox="0 0 20 20" fill="none">'
            f'<circle cx="10" cy="10" r="8" stroke="{_GREEN}" stroke-width="1.5" fill="{_GREEN}20"/>'
            f'<path d="M6 10L9 13L14 7" stroke="{_GREEN}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
            '</svg>'
        )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=Instrument+Serif:ital@0;1&family=Space+Mono:wght@400;700&display=swap');

    :root {{
        --gold: {_GOLD};
        --gold-dim: {_GOLD_DIM};
        --gold-deep: {_GOLD_DEEP};
        --cream: {_CREAM};
        --bg: {_BG};
        --bg2: {_BG2};
        --bg3: {_BG3};
        --text-dim: {_TEXT_DIM};
        --green: {_GREEN};
        --red: {_RED};
        --blue: {_BLUE};
        --border: rgba(212, 168, 83, 0.08);
        --border-med: rgba(212, 168, 83, 0.15);
    }}

    * {{ margin: 0; padding: 0; box-sizing: border-box; }}

    body {{
        background: var(--bg);
        color: var(--cream);
        font-family: 'Sora', 'Segoe UI', sans-serif;
        padding: 0;
        -webkit-font-smoothing: antialiased;
    }}

    .page {{
        max-width: 900px;
        margin: 0 auto;
        padding: 40px 48px;
    }}

    /* --- Header --- */
    .header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 20px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 28px;
    }}
    .header-left {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .header-brand {{
        font-size: 20px;
        font-weight: 700;
        color: var(--cream);
        letter-spacing: 0.01em;
    }}
    .header-sub {{
        font-family: 'Instrument Serif', serif;
        font-style: italic;
        font-size: 12px;
        color: var(--text-dim);
    }}
    .header-right {{
        text-align: right;
    }}
    .header-date {{
        font-family: 'Space Mono', monospace;
        font-size: 10px;
        color: var(--text-dim);
        letter-spacing: 0.05em;
    }}

    /* --- Status banner --- */
    .status-banner {{
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 16px 22px;
        border-radius: 8px;
        border: 1px solid;
        margin-bottom: 28px;
    }}
    .status-banner.stable {{
        background: rgba(123, 199, 126, 0.06);
        border-color: rgba(123, 199, 126, 0.2);
    }}
    .status-banner.drift {{
        background: rgba(212, 122, 106, 0.06);
        border-color: rgba(212, 122, 106, 0.2);
    }}
    .status-label {{
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 0.04em;
    }}
    .status-detail {{
        font-size: 11px;
        color: var(--text-dim);
        font-family: 'Space Mono', monospace;
    }}

    /* --- Metric cards row --- */
    .metrics-row {{
        display: flex;
        gap: 16px;
        margin-bottom: 28px;
    }}
    .metric-card {{
        flex: 1;
        padding: 18px 20px;
        background: var(--bg3);
        border-radius: 8px;
        border: 1px solid var(--border);
        border-left: 3px solid var(--gold);
    }}
    .metric-card.alert {{
        border-left-color: var(--red);
    }}
    .metric-card.ok {{
        border-left-color: var(--green);
    }}
    .metric-value {{
        font-size: 28px;
        font-weight: 700;
        font-family: 'Sora', sans-serif;
    }}
    .metric-label {{
        font-size: 11px;
        color: var(--cream);
        font-weight: 600;
        margin-top: 4px;
    }}
    .metric-sub {{
        font-size: 9px;
        color: var(--text-dim);
        margin-top: 2px;
        font-family: 'Space Mono', monospace;
    }}

    /* --- Section titles --- */
    .section-title {{
        font-family: 'Space Mono', monospace;
        font-size: 9px;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--text-dim);
        margin-bottom: 14px;
        margin-top: 32px;
    }}

    /* --- Chart container --- */
    .chart-container {{
        background: var(--bg2);
        border-radius: 8px;
        border: 1px solid var(--border);
        padding: 4px;
        margin-bottom: 24px;
        overflow: hidden;
    }}
    .chart-container img {{
        width: 100%;
        display: block;
        border-radius: 6px;
    }}

    /* --- PSI table --- */
    .psi-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        background: var(--bg3);
        border-radius: 8px;
        border: 1px solid var(--border);
        overflow: hidden;
        margin-bottom: 24px;
    }}
    .psi-table thead th {{
        font-family: 'Space Mono', monospace;
        font-size: 8px;
        font-weight: 700;
        color: var(--text-dim);
        letter-spacing: 0.12em;
        text-transform: uppercase;
        padding: 12px 18px;
        text-align: left;
        background: var(--bg2);
        border-bottom: 1px solid var(--border-med);
    }}
    .psi-table tbody tr {{
        border-bottom: 1px solid var(--border);
    }}
    .psi-table tbody tr:last-child {{
        border-bottom: none;
    }}
    .td-feature {{
        padding: 11px 18px;
        font-size: 12px;
        color: var(--cream);
        font-weight: 500;
    }}
    .td-value {{
        padding: 11px 18px;
        font-size: 13px;
        font-weight: 700;
        font-family: 'Space Mono', monospace;
    }}
    .td-badge {{
        padding: 11px 18px;
        text-align: center;
    }}

    /* --- Badges --- */
    .badge {{
        font-family: 'Space Mono', monospace;
        font-size: 9px;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
        letter-spacing: 0.05em;
    }}
    .badge-ok {{
        color: var(--green);
        background: rgba(123, 199, 126, 0.1);
        border: 1px solid rgba(123, 199, 126, 0.25);
    }}
    .badge-warn {{
        color: var(--gold);
        background: rgba(212, 168, 83, 0.1);
        border: 1px solid rgba(212, 168, 83, 0.25);
    }}
    .badge-alert {{
        color: var(--red);
        background: rgba(212, 122, 106, 0.1);
        border: 1px solid rgba(212, 122, 106, 0.25);
    }}

    /* --- Footer --- */
    .footer {{
        margin-top: 40px;
        padding-top: 16px;
        border-top: 1px solid var(--border);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .footer-left {{
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .footer-text {{
        font-family: 'Instrument Serif', serif;
        font-style: italic;
        font-size: 11px;
        color: var(--gold-deep);
    }}
    .footer-right {{
        font-family: 'Space Mono', monospace;
        font-size: 9px;
        color: var(--text-dim);
        letter-spacing: 0.03em;
    }}

    @media print {{
        body {{ background: var(--bg); -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    }}
</style>
</head>
<body>
<div class="page">

    <!-- Header -->
    <div class="header">
        <div class="header-left">
            {_LOGO_SVG}
            <div>
                <div class="header-brand">normarc</div>
                <div class="header-sub">Model Governance Report</div>
            </div>
        </div>
        <div class="header-right">
            <div class="header-date">{week_label.upper()} &middot; {week_start} &mdash; {week_end}</div>
        </div>
    </div>

    <!-- Status banner -->
    <div class="status-banner {"drift" if has_drift else "stable"}">
        {status_icon}
        <div>
            <div class="status-label" style="color:{status_color};">{status}</div>
            <div class="status-detail">
                {"Model revalidation recommended. " if has_drift else "All metrics within defined thresholds. "}
                ECE: {ece_value:.4f} &middot; Max PSI: {max_psi_value:.4f} ({max_psi_feature})
            </div>
        </div>
    </div>

    <!-- Metric cards -->
    <div class="metrics-row">
        <div class="metric-card {"alert" if ece_value > ece_threshold else "ok"}">
            <div class="metric-value" style="color:{"var(--red)" if ece_value > ece_threshold else "var(--green)"};">{ece_value:.4f}</div>
            <div class="metric-label">Expected Calibration Error</div>
            <div class="metric-sub">threshold: {ece_threshold} {ece_badge}</div>
        </div>
        <div class="metric-card {"alert" if max_psi_value > psi_threshold else "ok"}">
            <div class="metric-value" style="color:{"var(--red)" if max_psi_value > psi_threshold else "var(--green)"};">{max_psi_value:.4f}</div>
            <div class="metric-label">Max PSI ({max_psi_feature})</div>
            <div class="metric-sub">threshold: {psi_threshold} {"<span class='badge badge-alert'>ALERT</span>" if max_psi_value > psi_threshold else "<span class='badge badge-ok'>OK</span>"}</div>
        </div>
        <div class="metric-card" style="border-left-color:var(--gold);">
            <div class="metric-value" style="color:var(--gold);">{int(sum(bin_counts))}</div>
            <div class="metric-label">Inferences this week</div>
            <div class="metric-sub">scored &amp; monitored</div>
        </div>
    </div>

    <!-- Calibration section -->
    <div class="section-title">Calibration &mdash; Reliability Diagram</div>
    <div class="chart-container">
        <img src="data:image/png;base64,{reliability_img}" alt="Reliability Diagram">
    </div>

    <!-- PSI section -->
    <div class="section-title">Population Stability Index</div>
    <table class="psi-table">
        <thead>
            <tr><th>Feature</th><th>PSI Value</th><th>Status</th></tr>
        </thead>
        <tbody>
            {psi_rows}
        </tbody>
    </table>

    <div class="chart-container">
        <img src="data:image/png;base64,{psi_img}" alt="PSI Chart">
    </div>

    <!-- Footer -->
    <div class="footer">
        <div class="footer-left">
            <svg width="16" height="16" viewBox="0 0 48 48" fill="none">
                <path d="M8 36 C8 18, 24 4, 40 4" stroke="{_GOLD_DEEP}" stroke-width="3.5" stroke-linecap="round" fill="none"/>
                <circle cx="40" cy="4" r="3.5" fill="{_GOLD_DIM}"/>
                <circle cx="8" cy="36" r="2.5" fill="{_GOLD_DEEP}"/>
            </svg>
            <span class="footer-text">Governance that bends with your architecture, not against it.</span>
        </div>
        <div class="footer-right">{date.today().isoformat()}</div>
    </div>

</div>
</body>
</html>"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"report_week_{week_num:02d}.pdf"

    if _check_weasyprint():
        from weasyprint import HTML as WeasyprintHTML
        WeasyprintHTML(string=html_content).write_pdf(str(output_path))
    else:
        # Fallback: guardar como HTML si WeasyPrint no está instalado
        output_path = output_path.with_suffix(".html")
        output_path.write_text(html_content, encoding="utf-8")
        print(f"  > WeasyPrint not available, saved as HTML: {output_path}")

    return output_path
