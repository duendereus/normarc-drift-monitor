# normarc-drift-monitor

Automated model governance for regulated fintechs in LATAM.

A demo of what continuous AI governance looks like in practice: a monitoring agent that tracks drift and calibration on a credit scoring model, generates audit-ready reports, and alerts compliance teams when something goes wrong — before a regulator asks.

Built as a reference implementation for [Normarc] — governance that bends with your architecture, not against it.

---

## The problem

Your scoring model approved 800 credits this month. Do you know if it made those decisions with the same logic it had when it was validated?

Most fintechs can't answer that systematically. Governance today is:

- Manual — someone runs a script when something already went wrong
- Periodic — monthly or quarterly reviews instead of continuous monitoring
- Reactive — built case by case when an audit request arrives

This repo shows what the alternative looks like.

---

## What this demo does

Simulates 6 months of credit scoring inferences with intentional drift injected at week 18. The monitoring agent:

1. **Reads inference logs** — scores, input features, and realized outcomes
2. **Computes governance metrics** on a weekly cadence:
   - **PSI** (Population Stability Index) — detects when the model is seeing a different population than it was trained on
   - **ECE** (Expected Calibration Error) — verifies the model's probability estimates are still trustworthy
3. **Generates audit-ready reports** in PDF — weekly snapshots with reliability diagrams and trend analysis
4. **Alerts the compliance team** via Telegram when metrics breach defined thresholds

Two scenarios:

- ✅ **Stable week** — metrics within bounds, report archived silently: `"Model stable. ECE: 0.03. Max PSI: 0.08"`
- ⚠️ **Drift detected** — immediate alert: `"Drift detected. PSI ingreso_mensual: 0.23 (threshold: 0.20). Report attached."`

---

## Project structure

```
normarc-drift-monitor/
├── data/
│   └── generate_synthetic.py     # Synthetic inference log generator with drift
├── metrics/
│   ├── psi.py                    # Population Stability Index
│   └── ece.py                    # Expected Calibration Error
├── agent/
│   └── monitor.py                # Orchestrates the full monitoring pipeline
├── reports/
│   └── report_generator.py       # PDF report with reliability diagram
├── notebooks/
│   └── exploration.ipynb         # EDA and metric visualization
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/duendereus/normarc-drift-monitor
cd normarc-drift-monitor
pip install -r requirements.txt

# Generate synthetic dataset (6 months, drift at week 18)
python data/generate_synthetic.py

# Run the monitoring agent on the full dataset
python agent/monitor.py

# Reports are written to /reports/output/
# Alerts are sent to Telegram (configure .env first)
```

---

## Configuration

Copy `.env.example` to `.env` and fill in:

```
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
PSI_THRESHOLD=0.20
ECE_THRESHOLD=0.05
BASELINE_WEEKS=8        # weeks used to establish baseline distribution
```

---

## Metrics reference

### PSI — Population Stability Index

Measures how much the distribution of a feature or score has shifted relative to a baseline period.

| PSI value   | Interpretation                               |
| ----------- | -------------------------------------------- |
| < 0.10      | No significant change                        |
| 0.10 – 0.20 | Minor shift, monitor closely                 |
| > 0.20      | Major shift — model revalidation recommended |

### ECE — Expected Calibration Error

Measures the gap between predicted probabilities and observed frequencies. A well-calibrated model that says "70% probability of default" should observe ~70% defaults in that bucket.

Lower is better. Threshold for alert: **ECE > 0.05**.

---

## From demo to production

This repo is intentionally simple. In a production deployment:

- **Inference logs** come from your data warehouse (Snowflake, BigQuery, DuckDB) instead of a CSV
- **The agent** runs as a FastAPI microservice scheduled via cron or triggered post-batch
- **Reports** are stored in S3 or GCS with versioning for audit trail
- **Alerts** extend to Slack, email, or your incident management system

That production layer is what Normarc provides. [normarc.ai]

---

## Tech stack

- Python 3.11+
- DuckDB — analytical queries over inference logs
- scikit-learn — calibration utilities
- matplotlib — reliability diagrams
- WeasyPrint — PDF report generation
- python-telegram-bot — alerting

---

## License

MIT — use it, fork it, build on it.

---

_Built by [Fernando Céspedes](https://linkedin.com/in/fernandocespedes) · [Normarc]_
