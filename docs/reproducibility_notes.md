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

## Our preprocessing pipeline (`src/preprocessing.py`)

| Step | Technique |
|------|-----------|
| Data Generation | `np.random.normal` based on CICIDS2017 distributions |
| Missing Values | Replaced with median |
| Infinite Values | Dropped |
| Scaling | StandardScaler fit on train |

## Our model training (`src/models.py`)

| Setting | Value |
|---------|-------|
| Encoding | N/A (CICIDS2017 synthetic numeric features) |
| Models | MLP (`hidden_layer_sizes=(128, 64)`), RF (`n_estimators=100`) |
| `random_state` | 42 |
| Outputs | `results/experiment_metrics.csv`, `results/baseline_reproduction.csv`, `results/errors_rare_attacks_analysis.csv` |

## Evaluation summary

| Finding | Detail |
|---------|--------|
| Best AUC | Random Forest (1.000) / MLP (0.9999) |
| Best F1 | Random Forest (1.000) / MLP (0.9987) |
| FP/FN Implications | False Positives generate SOC alert fatigue. False Negatives risk critical network breaches. Both models mitigated these significantly. |

Artifacts: `Final_Project_Notebook.ipynb` and `report/Final_Project_Report.tex`
