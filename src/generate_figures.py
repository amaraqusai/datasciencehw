"""
Generate all result figures for the report and notebook.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from preprocessing import load_real_cicids2017, run_eda, run_duplicate_audit, clean_data, split_and_scale
from model_utils import train_all_models, evaluate_all_models, get_multiclass_error_analysis

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'figure.figsize': (12, 6), 'figure.dpi': 150, 'font.size': 11})

os.makedirs('../results/figures', exist_ok=True)

print("[1/9] Generating dataset...")
df = load_real_cicids2017(sample_frac=0.10)
eda = run_eda(df)

# ============================================================
# DUPLICATE AUDIT
# ============================================================
print("[2/9] Running Duplicate Audit...")
audit = run_duplicate_audit(df)

# Save duplicate audit results
import json
with open('../results/figures/duplicate_audit.json', 'w') as f:
    json.dump(audit, f, indent=4)

# ============================================================
# FIGURE 1: Class Distribution
# ============================================================
print("[3/9] Plotting class distribution...")
class_dist = eda['class_distribution']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
class_dist.plot(kind='barh', ax=axes[0], color=sns.color_palette('viridis', len(class_dist)))
axes[0].set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Number of Samples')

binary_counts = pd.Series({'Benign': class_dist.get('BENIGN', 0),
                           'Attack': class_dist.sum() - class_dist.get('BENIGN', 0)})
binary_counts.plot(kind='pie', ax=axes[1], autopct='%1.1f%%',
                   colors=['#2ecc71', '#e74c3c'], startangle=90, fontsize=12)
axes[1].set_title('Binary Class Split', fontsize=14, fontweight='bold')
axes[1].set_ylabel('')
plt.tight_layout()
plt.savefig('../results/figures/class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# CLEAN DATA
# ============================================================
print("[4/9] Cleaning Data...")
df_clean, _, _, label_encoder = clean_data(df)

# Helper for plotting tables
def plot_results_table(metrics_df, filename, title, header_color='#1e4682', row_color='#f0f5ff'):
    cols = ['Model', 'Accuracy', 'Balanced Acc', 'Macro Prec', 'Macro Rec', 'Macro F1', 'Weighted F1', 'MCC', 'ROC-AUC (OVR)']
    fig, ax = plt.subplots(figsize=(16, max(4, len(metrics_df)*0.8)))
    ax.axis('off')
    table_data = []
    
    for _, row in metrics_df.iterrows():
        table_data.append([
            row['Model'], f"{row['Accuracy']:.4f}", f"{row['Balanced Acc']:.4f}",
            f"{row['Macro Precision']:.4f}", f"{row['Macro Recall']:.4f}", 
            f"{row['Macro F1']:.4f}", f"{row['Weighted F1']:.4f}",
            f"{row['MCC']:.4f}", f"{row['ROC-AUC (OVR)']:.4f}"
        ])
            
    table = ax.table(cellText=table_data, colLabels=cols, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for j in range(len(cols)):
        table[0, j].set_facecolor(header_color)
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(table_data) + 1):
        for j in range(len(cols)):
            if i % 2 == 0:
                table[i, j].set_facecolor(row_color)
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'../results/figures/{filename}', dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# EXPERIMENTAL SPLIT CONFIGURATIONS
# ============================================================
print("[5/9] Evaluating 4 Split Strategies for Leakage Analysis...")
strategies = [
    ('random_original', 'Baseline (With Duplicates)', '#1e4682'),
    ('random_deduplicated', 'Exact Deduplicated', '#d35400'),
    ('grouped', 'Grouped by Feature Hash', '#27ae60'),
    ('chronological', 'Chronological (Day-based)', '#8e44ad')
]

leakage_results = []
metrics_results = {}

for strategy, desc, color in strategies:
    print(f"\\n--- Running Strategy: {strategy} ---")
    X_train, X_val, X_test, y_train, y_val, y_test, _, leakage_pct = split_and_scale(
        df_clean, task_type='multiclass', split_strategy=strategy
    )
    leakage_results.append({'Strategy': strategy, 'Leakage_Pct': leakage_pct})
    
    # Train models
    # To save time, we only run DBN on 'random_original' and 'chronological'
    models, _ = train_all_models(X_train, y_train, X_val=X_val, y_val=y_val)
    if strategy in ['random_deduplicated', 'grouped']:
        if 'Authentic DBN' in models:
            del models['Authentic DBN']
            
    metrics = evaluate_all_models(models, X_test, y_test, task_type='multiclass')
    metrics_results[strategy] = metrics
    
    # Save table
    plot_results_table(metrics, f'results_table_{strategy}.png', f'Evaluation: {desc}', header_color=color)
    
    # Save confusion matrix for best models (just RF for consistency, plus DBN if available)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot RF
    if 'Random Forest' in models:
        y_pred = models['Random Forest'].predict(X_test)
        unique_test_classes = np.unique(np.concatenate((y_test, y_pred)))
        test_class_names = label_encoder.inverse_transform(unique_test_classes)
        cm = confusion_matrix(y_test, y_pred, labels=unique_test_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_class_names)
        disp.plot(cmap='Blues', ax=axes[0], values_format='d', xticks_rotation='vertical')
        axes[0].set_title('Random Forest', fontsize=14, fontweight='bold')
        axes[0].grid(False)
        
    # Plot DBN if available, else XGBoost
    second_model = 'Authentic DBN' if 'Authentic DBN' in models else 'XGBoost'
    if second_model in models:
        y_pred = models[second_model].predict(X_test)
        unique_test_classes = np.unique(np.concatenate((y_test, y_pred)))
        test_class_names = label_encoder.inverse_transform(unique_test_classes)
        cm = confusion_matrix(y_test, y_pred, labels=unique_test_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_class_names)
        disp.plot(cmap='Purples', ax=axes[1], values_format='d', xticks_rotation='vertical')
        axes[1].set_title(second_model, fontsize=14, fontweight='bold')
        axes[1].grid(False)

    plt.suptitle(f'Confusion Matrices: {desc}', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'../results/figures/confusion_{strategy}.png', dpi=150, bbox_inches='tight')
    plt.close()

# Save Leakage JSON
with open('../results/figures/leakage_analysis.json', 'w') as f:
    json.dump(leakage_results, f, indent=4)

print("[6/9] All experimental configurations completed.")
print("[7/9] Skipped Error Analysis CSV (Optional for now).")
print("[8/9] Skipped Feature Importance.")
print("[9/9] Success.")
