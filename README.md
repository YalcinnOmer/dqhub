# DQHub

DQHub is a lightweight, end-to-end **data quality CLI pipeline** that can:
1) generate synthetic input data,
2) run cleaning rules,
3) verify constraints and formats,
4) produce reports (Excel + HTML),
5) and package outputs as CI artifacts via GitHub Actions.

## Features

- **CLI-first workflow** with a single entrypoint: `dqhub pipeline`
- **Rules-driven** validation via YAML (`rules/rules.yaml`)
- Generates:
  - cleaned dataset: `data/output/clean.csv`
  - Excel report: `reports/DQ_Report.xlsx`
  - HTML report: `reports/DQ_Report.html`
  - CI pipeline via GitHub Actions, uploading key outputs as artifacts

## Project structure

```text
dqhub/
  src/dqhub/            # Python package + CLI
  rules/                # YAML rules (rules.yaml)
  data/                 # generated input/output
  reports/              # generated reports
  .github/workflows/    # CI workflow (pipeline.yml)
