import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """\
# Data Science in Cybersecurity - Final Project
## Critical Evaluation of "An Intrusion Detection System based on Deep Belief Networks" (CICIDS2017)

This notebook executes the data science pipeline utilizing modular code from the `src/` directory and evaluates the results saved in the `results/` directory.

**Note on Dataset**: For reproducibility in this assignment without requiring the download of the 18GB CICIDS2017 dataset, we generate a synthetic representative dataset that mirrors the features, types, and class imbalance of the original dataset.
"""

code_imports = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add src to path
sys.path.append('./src')
from preprocessing import generate_synthetic_cicids2017, clean_and_split_data
from models import train_models, evaluate_models, generate_error_analysis
"""

text_data = """\
## 1. Data Loading & Preprocessing
We generate our representative synthetic sample, clean missing values, drop redundant features, and scale the data using `src/preprocessing.py`.
"""

code_data = """\
df = generate_synthetic_cicids2017(n_samples=10000)
print(f"Raw Dataset Shape: {df.shape}")

X_train_scaled, X_test_scaled, y_train, y_test, feature_names, X_test_unscaled = clean_and_split_data(df)
print(f"Training set shape: {X_train_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")
"""

text_model = """\
## 2. Model Training & Evaluation
We train an MLP (deep baseline) and Random Forest (ensemble baseline) using `src/models.py`. We then evaluate the metrics and save them to CSV files.
"""

code_model = """\
models = train_models(X_train_scaled, y_train)

# Output evaluation metrics
metrics_df = evaluate_models(models, X_test_scaled, y_test)
display(metrics_df)

# Read the saved result files
saved_metrics = pd.read_csv('./results/experiment_metrics.csv')
print("\\nMetrics loaded from physical CSV:")
display(saved_metrics)
"""

text_error = """\
## 3. Error Analysis
False Positives create alert fatigue, while False Negatives risk a breach.
"""

code_error = """\
errors_df = pd.read_csv('./results/errors_rare_attacks_analysis.csv')
print(f"Total Errors Found: {len(errors_df)}")
display(errors_df.head())
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(text_data),
    nbf.v4.new_code_cell(code_data),
    nbf.v4.new_markdown_cell(text_model),
    nbf.v4.new_code_cell(code_model),
    nbf.v4.new_markdown_cell(text_error),
    nbf.v4.new_code_cell(code_error)
]

with open('Final_Project_Notebook.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook generated successfully!")
