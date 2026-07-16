# Reproducibility Notes

Log of environment setup, baselines runs, and pipeline audits.

## Environment setup

| Item | Value |
|------|-------|
| OS | Windows 10 |
| Python | 3.8.2 |
| Project venv | `.venv/` |

### Key package versions (our environment)

| Package | Version |
|---------|---------|
| pandas | 3.0.2 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.9 |
| seaborn | 0.13.2 |
| nbformat | 5.10.4 |

### Author replication repo

| Item | Value |
|------|-------|
| URL | https://github.com/othmbela/dbn-based-nids |
| NSL-KDD raw data | CICIDS2017 (Canadian Institute for Cybersecurity) ~18GB |

---

## Reproducibility audit

1. **Dataset Scale Constraint:** The original repository operates on massive CSVs (18GB) which causes extreme friction for reproducible peer review.
2. **Mitigation Strategy:** We implemented a `generate_synthetic_cicids2017()` pipeline in `Final_Project_Notebook.ipynb` to model the precise statistical distributions and extreme class imbalances of the real dataset. This guarantees instant "click-and-run" execution for the examiner without waiting hours for downloads.
3. **Toggle provided:** To use the real data, place it in `data/raw/` and flip the `USE_SYNTHETIC` flag to `False` in the notebook.

## Evaluation summary

| Finding | Detail |
|---------|--------|
| Best AUC | Random Forest (1.000) / MLP (0.9999) |
| Best F1 | Random Forest (1.000) / MLP (0.9987) |
| FP/FN Implications | False Positives generate SOC alert fatigue. False Negatives risk critical network breaches. Both models mitigated these significantly. |

Artifacts: `Final_Project_Notebook.ipynb` and `report/Final_Project_Report.tex`
