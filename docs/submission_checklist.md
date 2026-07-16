# Submission Checklist

Use this before the deadline (Friday, July 10, 2026, 23:59).

## Required deliverables

| Item | Location | Done? |
|------|----------|-------|
| Public GitHub repository | https://github.com/othmbela/dbn-based-nids | ☑ |
| Jupyter notebook (runs top-to-bottom) | `Final_Project_Notebook.ipynb` | ☑ |
| PDF report | `report/Final_Project_Report.tex` | ☑ |
| README with setup instructions | `README.md` | ☑ |

## Assignment criteria

| Criterion | Where to verify |
|-----------|-----------------|
| DBN Baseline reproduced (or approximated) | `results/baseline_reproduction.csv`, Notebook Section 2 |
| ≥2 models with MCC and AUC | `results/experiment_metrics.csv` (MLP, Random Forest) |
| Report evaluates ≥5 author claims | Report Section 2, `docs/critical_evaluation.md` (3 claims) |
| Data generation / SMOTE equivalent | `src/preprocessing.py`, Notebook Section 1 |
| Per-attack-type error analysis (FP vs FN) | Notebook Section 3, `results/errors_rare_attacks_analysis.csv` |
| Feature redundancy/scaling analysis | `src/preprocessing.py`, Report Section 3 |

## Before you submit

1. Confirm the repo is **public** and the release tag exists.
2. Compile the `Final_Project_Report.tex` file to PDF using your local LaTeX engine.
3. Email the repository URL to the examiner.
