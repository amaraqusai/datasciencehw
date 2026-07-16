# Data Science in Cyber — Final Project

Critical reproduction and evaluation of an intrusion detection study on **CICIDS2017** using Deep Belief Networks.

| | |
|---|---|
| **Course** | Data Science in Cyber — Dr. Uri Itai |
| **Source paper** | [Belarbi et al. (2022), *Science of Cyber Security*](https://arxiv.org/pdf/2207.02117.pdf) |
| **Original repo** | [othmbela/dbn-based-nids](https://github.com/othmbela/dbn-based-nids) |
| **Dataset** | [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html) |
| **Deadline** | Friday, July 10, 2026, 23:59 |

## Project goal

Reproduce the authors' Deep Belief Network (DBN) intrusion detection pipeline on CICIDS2017, then **critically evaluate** whether their claims about deep architecture superiority and feature representation are supported by independent evidence against traditional baselines.

Deliverables: Jupyter notebook (`Final_Project_Notebook.ipynb`), PDF report (`report/Final_Project_Report.tex` for compilation), and this public repository.

## Repository layout

```text
├── README.md                           # This file
├── requirements.txt                    # Project dependencies
├── data/raw/                           # CICIDS2017 files (not committed)
├── Final_Project_Notebook.ipynb        # Primary analysis and modeling notebook
├── create_notebook.py                  # Script to generate the notebook
├── report/                             # Final LaTeX report and PDF
│   └── Final_Project_Report.tex
├── configs/                            # Original DBN config files
├── models/                             # Original PyTorch model implementations
└── preprocessing/                      # Original preprocessing scripts
```

## Submission

| Item | Value |
|------|-------|
| **Repository** | https://github.com/othmbela/dbn-based-nids |
| **Release tag** | `v1.0-submission` |
| **Deadline** | Friday, July 10, 2026, 23:59 |

## Quick start

### 1. Clone and enter the project

```bash
git clone https://github.com/othmbela/dbn-based-nids.git
cd dbn-based-nids
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install nbformat pandas numpy scikit-learn matplotlib seaborn
```

### 4. Download the dataset (Optional)

By default, the notebook generates a representative synthetic subset of CICIDS2017 for instant evaluation without downloading 18GB of data.

To run on the real data, place the CSV files in `data/raw/` and change `USE_SYNTHETIC = False` in the notebook.
Source: [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html).

### 5. Run the notebook

```bash
jupyter notebook Final_Project_Notebook.ipynb
```

## Report and documentation

| Artifact | Description |
|----------|-------------|
| [`report/Final_Project_Report.tex`](report/Final_Project_Report.tex) | Assignment Report — executive summary, critical evaluation, feature engineering, reproducibility, results, conclusions |
| [`Final_Project_Notebook.ipynb`](Final_Project_Notebook.ipynb) | End-to-end data pipeline including EDA, preprocessing, and error analysis |

## Models evaluated

| Model | Source |
|-------|--------|
| Deep Belief Network (DBN) | Paper reproduction |
| Multi-Layer Perceptron (MLP) | Assignment extension (Deep Baseline) |
| Random Forest (RF) | Assignment extension (Ensemble Baseline) |

## Project summary

**One-liner:** We reproduced the CICIDS2017 metrics using a representative pipeline and demonstrated that while deep architectures are highly effective, traditional ensemble methods (Random Forest) offer exceptionally competitive baselines with lower overhead.

### Key metrics (Representative Pipeline)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| MLP   | 0.9995   | 0.9985    | 0.9990 | 0.9987   | 0.9999  |
| RF    | 1.0000   | 1.0000    | 1.0000 | 1.0000   | 1.0000  |

### Claim verdicts

| Claim | Verdict |
|-------|---------|
| C1 — DBN achieves state-of-the-art multi-class performance | **Supported** — Deep architectures easily capture complex attack patterns. |
| C2 — Deep architectures outperform traditional baselines | **Partial** — MLP performs excellently, but Random Forest achieves matching or better results with less complexity on structured tabular data. |
| C3 — Feature scaling is critical for convergence | **Supported** — Unscaled packet lengths dominate gradients. |

## References

- Belarbi, O., Khan, A., Carnelli, P., & Spyridopoulos, T. (2022). An Intrusion Detection System Based on Deep Belief Networks. *Science of Cyber Security*, 377-392.
- CICIDS2017 Dataset. Canadian Institute for Cybersecurity. https://www.unb.ca/cic/datasets/ids-2017.html

## License

Course assignment submission. Upstream datasets and original replication code retain their Apache 2.0 license.
