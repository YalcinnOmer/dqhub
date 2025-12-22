# DQ Hub (Data Quality Ops Hub)

End-to-end data quality pipeline that generates synthetic input, runs data-quality checks (completeness/validity/uniqueness/timeliness), and produces an interactive Plotly dashboard + Excel report.

## Outputs
- `reports/DQ_Report.html` (interactive dashboard)
- `reports/DQ_Report.xlsx`
- `reports/dq_summary.csv`
- `reports/dq_issues.csv`

## Quickstart
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt

python -m dqhub pipeline
python -m dqhub verify
