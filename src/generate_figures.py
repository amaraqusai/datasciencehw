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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve

from preprocessing import load_real_cicids2017, run_eda, clean_data, split_and_scale
from models import train_all_models, evaluate_all_models

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'figure.figsize': (12, 6), 'figure.dpi': 150, 'font.size': 11})

os.makedirs('../results/figures', exist_ok=True)

print("[1/8] Generating dataset...")
df = load_real_cicids2017(sample_frac=0.10)
eda = run_eda(df)

# ============================================================
# FIGURE 1: Class Distribution
# ============================================================
print("[2/8] Plotting class distribution...")
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
# FIGURE 2: Feature Distributions
# ============================================================
print("[3/8] Plotting feature distributions...")
key_features = ['Flow Duration', 'Total Fwd Packets', 'Flow Bytes/s',
                'Fwd Packet Length Mean', 'Flow IAT Mean', 'Bwd Packet Length Max']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, feat in zip(axes.ravel(), key_features):
    data = df[feat].replace([np.inf, -np.inf], np.nan).dropna()
    q99 = data.quantile(0.99)
    data_clipped = data[data <= q99]
    sns.histplot(data_clipped, kde=True, ax=ax, color='steelblue', bins=50)
    ax.set_title(feat, fontsize=12, fontweight='bold')
    ax.set_xlabel('')
plt.suptitle('Feature Distributions (clipped at 99th percentile)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../results/figures/feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 3: Outlier Boxplots
# ============================================================
print("[4/8] Plotting outlier analysis...")
numeric_clean = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
box_features = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                'Fwd Packet Length Max', 'Bwd Packet Length Mean',
                'Flow IAT Mean', 'Packet Length Variance', 'Active Mean']

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for ax, feat in zip(axes.ravel(), box_features):
    data = numeric_clean[feat].dropna()
    q1, q3 = data.quantile(0.25), data.quantile(0.75)
    iqr = q3 - q1
    outliers = ((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)).sum()
    sns.boxplot(y=data.clip(upper=data.quantile(0.95)), ax=ax, color='lightskyblue')
    ax.set_title(f'{feat}\n({outliers} outliers)', fontsize=10, fontweight='bold')
plt.suptitle('Outlier Analysis (Boxplots)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../results/figures/outlier_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 4: Correlation Heatmap
# ============================================================
print("[5/8] Plotting correlation heatmap...")
numeric_filled = numeric_clean.fillna(numeric_clean.median())
top_var_cols = numeric_filled.var().nlargest(20).index.tolist()
corr_matrix = numeric_filled[top_var_cols].corr()

plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Heatmap (Top 20 Features by Variance)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../results/figures/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# CLEAN, SPLIT, TRAIN
# ============================================================
print("[6/8] Training models...")
df_clean, _, _ = clean_data(df)
X_train, X_test, y_train, y_test, feature_names = split_and_scale(df_clean)
models, times = train_all_models(X_train, y_train)
metrics_df = evaluate_all_models(models, X_test, y_test)

# ============================================================
# FIGURE 5: Confusion Matrices
# ============================================================
print("[7/8] Plotting confusion matrices, ROC, PR curves...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Attack'])
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    ax.set_title(f'{name}', fontsize=12, fontweight='bold')
    ax.grid(False)
plt.suptitle('Confusion Matrices - All Models', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../results/figures/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 6: ROC + PR Curves
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = metrics_df[metrics_df['Model'] == name]['ROC-AUC'].values[0]
    axes[0].plot(fpr, tpr, label=f'{name} (AUC={auc_val:.4f})', linewidth=2)
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves', fontsize=14, fontweight='bold')
axes[0].legend()

for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    pr_auc_val = metrics_df[metrics_df['Model'] == name]['PR-AUC'].values[0]
    axes[1].plot(rec, prec, label=f'{name} (PR-AUC={pr_auc_val:.4f})', linewidth=2)
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
axes[1].legend()
plt.tight_layout()
plt.savefig('../results/figures/roc_pr_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 7: Feature Importance
# ============================================================
print("[8/8] Plotting feature importance...")
rf_model = models['Random Forest']
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
top_n = min(20, len(feature_names))
sns.barplot(x=importances[indices][:top_n],
            y=np.array(feature_names)[indices][:top_n],
            palette='magma')
plt.title('Top 20 Most Important Features (Random Forest Gini)', fontsize=14, fontweight='bold')
plt.xlabel('Gini Importance')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.savefig('../results/figures/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 8: Scaling Comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
raw = df_clean['Flow Duration'].dropna().values[:1000]
scaled = X_train['Flow Duration'].values[:1000]
sns.histplot(raw, kde=True, ax=axes[0], color='coral', bins=40)
axes[0].set_title('Flow Duration - Before Scaling (Raw)', fontsize=12, fontweight='bold')
sns.histplot(scaled, kde=True, ax=axes[1], color='teal', bins=40)
axes[1].set_title('Flow Duration - After Z-Score Scaling', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('../results/figures/scaling_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# RESULTS TABLE as image
# ============================================================
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')
table_data = []
for _, row in metrics_df.iterrows():
    table_data.append([
        row['Model'], f"{row['Accuracy']:.4f}", f"{row['Precision']:.4f}",
        f"{row['Recall']:.4f}", f"{row['F1-Score']:.4f}", f"{row['F2-Score']:.4f}",
        f"{row['MCC']:.4f}", f"{row['ROC-AUC']:.4f}", f"{row['PR-AUC']:.4f}",
        str(row['FP']), str(row['FN']), f"{row['FAR']:.4f}", f"{row['FNR']:.4f}"
    ])
cols = ['Model', 'Acc', 'Prec', 'Recall', 'F1', 'F2', 'MCC', 'ROC-AUC', 'PR-AUC', 'FP', 'FN', 'FAR', 'FNR']
table = ax.table(cellText=table_data, colLabels=cols, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)
# Style header
for j in range(len(cols)):
    table[0, j].set_facecolor('#1e4682')
    table[0, j].set_text_props(color='white', fontweight='bold')
# Style rows
for i in range(1, len(table_data) + 1):
    for j in range(len(cols)):
        if i % 2 == 0:
            table[i, j].set_facecolor('#f0f5ff')
plt.title('Comprehensive Model Evaluation Results', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('../results/figures/results_table.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll 9 figures saved to results/figures/")
print("Files:")
for f in sorted(os.listdir('../results/figures')):
    size = os.path.getsize(f'../results/figures/{f}') / 1024
    print(f"  {f} ({size:.0f} KB)")
