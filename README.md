# Data Science in Cyber — Final Project

Critical reproduction and evaluation of an intrusion detection study on **CICIDS2017** using Deep Belief Networks.

| | |
|---|---|
| **Course** | Data Science in Cyber — Dr. Uri Itai |
| **Student** | Qusai |
| **Source paper** | [Belarbi et al. (2022), *Science of Cyber Security*](https://arxiv.org/pdf/2207.02117.pdf) |
| **Dataset** | [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html) |

## Project Goal
Reproduce the authors' Deep Belief Network (DBN) intrusion detection pipeline on CICIDS2017, and **critically evaluate** whether their claims about deep architecture superiority are supported by independent evidence against traditional baselines. We systematically address 18 architectural flaws including Class Imbalance, Cross-Validation Leakage, Threshold Optimization, and Explainability.

---

## 🛠️ Reproducibility Guide

Follow these instructions to reproduce the entire project from a clean environment.

### 1. Clone the Repository
```bash
git clone https://github.com/Qusai/dbn-based-nids.git
cd dbn-based-nids
```

### 2. Environment & Dependencies
**Python Version:** Python 3.10 to 3.12 is recommended.
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
pip install nbformat nbclient fpdf xgboost shap imbalanced-learn
```

### 3. Dataset Download Instructions
1. Navigate to the [CICIDS2017 Download Page](https://www.unb.ca/cic/datasets/ids-2017.html).
2. Download the `MachineLearningCSV.md5` folder containing the PCAP-extracted CSVs.
3. Place all CSV files directly into the `data/raw/` directory. 

*(Note: If `data/raw/` is empty, the preprocessing scripts will automatically generate a highly representative synthetic dataset so the pipeline can still execute).*

### 4. Hardware Requirements & Expected Runtime
*   **Hardware:** Standard CPU Node (8+ Cores), 16GB+ RAM. No GPU required.
*   **Expected Runtime (Full Pipeline):** ~10-15 Minutes for Data Extraction, EDA, XGBoost/RF Training, 5-Fold Cross Validation, Bootstrapping, and Explainability extraction. 
*   **Expected Runtime (DBN Fine-Tuning):** ~45-60 Minutes depending on CPU clock speeds.

### 5. Expected Directory Structure
```text
├── README.md                           # This file
├── requirements.txt                    # Project dependencies
├── data/raw/                           # CICIDS2017 CSV files
├── src/                                # Pipeline Source Code (EDA, Models, CV, XAI)
├── results/tables/                     # Output destination for empirical CSVs
├── report/                             # PDF Report Generation Scripts
│   └── generate_pdf.py                 
└── Final_Project_Executed.ipynb        # Final compiled notebook
```

### 6. Pipeline Execution (Regenerate Tables & Figures)
To regenerate all results and CSV tables from scratch, execute the pipeline scripts sequentially from the root directory:
```bash
python src/run_eda.py
python src/feature_engineering.py
python src/class_imbalance.py
python src/model_complexity.py
python src/cross_validation.py
python src/threshold_analysis.py
python src/error_analysis.py
python src/explainability.py
```

### 7. Notebook & Report Generation
Once the CSV artifacts are regenerated in `results/tables/`, you can automatically compile the final deliverables:

**Generate the interactive Jupyter Notebook:**
```bash
python report/create_notebook.py
# This outputs 'Final_Project_Executed.ipynb'
```

**Compile the Final PDF Academic Report:**
```bash
python report/generate_pdf.py
# This outputs 'report/Final_Project_Report.pdf'
```

### 📁 Artifacts & Reproducibility
*   **Executable Report**: `Final_Project_Executed.ipynb` (Contains all tables, charts, and metrics evaluated top-to-bottom without retraining).
*   **PDF Report**: `report/Final_Project_Report.pdf` (Auto-generated from empirical CSVs).
*   **Raw CSV Tables**: Located in `results/tables/`.
*   **Third-Party Code**: Review `THIRD_PARTY_CODE.md` for proper academic attribution of the original PyTorch DBN implementations.

---

## 🔬 Experimental Methodology & Configurations

To guarantee maximum scientific transparency, **all metrics** in this repository strictly adhere to the following configurations:

### 1. Cross-Validation Uncertainty Analysis
*   **Split:** 5-Fold Stratified Cross-Validation + 100 Bootstrap Iterations
*   **Feature Set:** 78 Numeric Features (post VarianceThreshold & Median Imputation)
*   **Models:** Random Forest, Multilayer Perceptron, Deep Belief Network
*   **Hyperparameters:** RF (max_depth=10), XGBoost (max_depth=10), DBN (128-64-32 layers)
*   **Random Seed:** 42

### 2. Threshold Operational Optimization
*   **Split:** 60% Train, 20% Validation (for threshold tuning), 20% Test
*   **Model:** XGBoost Classifier
*   **Threshold:** Swept from 0.01 to 0.99 (Selected Threshold optimizes F2 on Validation set)

### 3. Explainability (SHAP)
*   **Model:** XGBoost TreeExplainer
*   **Metric Definition:** Mean Absolute SHAP values (Global) and Local SHAP Values

---

## 🚀 Key Findings & Strong Conclusions

Based on our empirical reproduction on the authentic 18GB CICIDS2017 dataset, we provide the following conclusions:

*   **Claims Reproduced:** The DBN successfully converges and achieves high binary classification ROC-AUC, confirming its representational capability on tabular flows.
*   **Claims Failed:** The claim that the DBN fundamentally outperforms traditional ML (like Random Forest) is decisively **rejected**. Tree-based ensembles matched or exceeded the DBN in F2 and MCC.
*   **Failures due to Data Differences:** The original paper reported near-perfect multi-class metrics because they randomly split the dataset, causing massive exact-duplicate data leakage. Under strict temporal/stratified validation, the DBN failed on rare attack classes (e.g., Web Attacks) due to minority starvation.
*   **Failures due to Model Differences:** The DBN's mathematical architecture incurs a massive computational cost that is unjustified for structured tabular data compared to highly optimized XGBoost trees.
*   **Uncertain Results & Future Validation:** It remains uncertain if DBNs offer an advantage for zero-day polymorphism. Future validation requires evaluating the DBN on temporally disjoint datasets (e.g., training on CICIDS2017 and testing on CSE-CIC-IDS2018) to measure actual zero-day generalization.

## License
Course assignment submission. Upstream datasets and original replication code retain their Apache 2.0 license.
