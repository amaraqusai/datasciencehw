import sys
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

from preprocessing import load_real_cicids2017

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'figure.figsize': (12, 6), 'figure.dpi': 150, 'font.size': 11})

os.makedirs('../results/figures', exist_ok=True)

print("=== [1/7] Ingesting Dataset ===")
# Use 10% sample for tractable analysis
df = load_real_cicids2017(sample_frac=0.10)
findings = {}

print("=== [2/7] Dataset Structure & Quality ===")
findings['num_rows'] = int(df.shape[0])
findings['num_cols'] = int(df.shape[1])
findings['data_types'] = df.dtypes.astype(str).value_counts().to_dict()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

findings['num_numeric_features'] = len(numeric_cols)
findings['num_categorical_features'] = len(categorical_cols)
findings['unique_labels'] = df['Label'].nunique()

# Extreme and invalid values
# e.g., negative flow duration
negative_flow_duration = int((df['Flow Duration'] < 0).sum()) if 'Flow Duration' in df.columns else 0
findings['negative_flow_duration'] = negative_flow_duration

# Extremely large values from division by zero (Infs)
inf_counts = {}
for col in numeric_cols:
    n_inf = int(np.isinf(df[col]).sum())
    if n_inf > 0:
        inf_counts[col] = n_inf
findings['infinite_values'] = inf_counts

# Constant columns
constant_cols = [col for col in numeric_cols if df[col].nunique() <= 1]
findings['constant_columns'] = constant_cols

# Near-constant
near_const = []
for col in numeric_cols:
    if col not in constant_cols and len(df) > 0:
        if df[col].value_counts(normalize=True).iloc[0] > 0.99:
            near_const.append(col)
findings['near_constant_columns'] = near_const

print("=== [3/7] Class Analysis ===")
# Multi-class and binary class imbalance
class_counts = df['Label'].value_counts()
findings['records_per_class'] = class_counts.to_dict()

binary_label = df['Label'].apply(lambda x: 'Benign' if x == 'BENIGN' else 'Attack')
findings['binary_class_imbalance'] = binary_label.value_counts().to_dict()

# Attack prevalence by day
if 'Day' in df.columns:
    attack_by_day = df[df['Label'] != 'BENIGN'].groupby('Day').size().to_dict()
    findings['attack_prevalence_by_day'] = attack_by_day
    findings['records_per_day'] = df['Day'].value_counts().to_dict()

print("=== [4/7] Distribution Analysis (Benign vs Attack) ===")
# Plot specific features
features_to_plot = [
    'Flow Duration', 'Total Fwd Packets', 'Total Length of Fwd Packets', 
    'Fwd IAT Total', 'Flow Bytes/s', 'PSH Flag Count', 'SYN Flag Count', 'Init_Win_bytes_forward'
]

# We must replace Inf/NaN to plot
plot_df = df.copy()
plot_df['Binary'] = binary_label
for col in features_to_plot:
    if col in plot_df.columns:
        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
        plot_df[col] = plot_df[col].replace([np.inf, -np.inf], np.nan)
        plot_df[col] = plot_df[col].fillna(plot_df[col].median())
        
        plt.figure(figsize=(10, 5))
        # Use log scale if values are large
        is_large = plot_df[col].max() > 1000
        
        sns.kdeplot(data=plot_df, x=col, hue='Binary', common_norm=False, fill=True, alpha=0.5)
        if is_large:
            plt.xscale('symlog')
        plt.title(f'Distribution of {col}: Benign vs Attack')
        plt.savefig(f'../results/figures/dist_{col.replace("/", "_").replace(" ", "_")}.png', bbox_inches='tight')
        plt.close()

print("=== [5/7] Outlier Analysis (Clipping Experiment) ===")
# Test if clipping outliers hurts recall for rare classes
# We will use a fast Random Forest on a small subset
clip_cols = [c for c in numeric_cols if c not in ['Label', 'Day']]
df_clip = plot_df.copy()
df_clip[clip_cols] = df_clip[clip_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

le = LabelEncoder()
df_clip['Label_Enc'] = le.fit_transform(df_clip['Label'])

# Filter out classes with < 2 members to allow stratify
class_counts_enc = df_clip['Label_Enc'].value_counts()
valid_classes = class_counts_enc[class_counts_enc >= 2].index
df_clip_filtered = df_clip[df_clip['Label_Enc'].isin(valid_classes)]

y_enc = df_clip_filtered['Label_Enc'].values
X_all = df_clip_filtered[clip_cols].values

X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_enc, test_size=0.3, random_state=42, stratify=y_enc)

print("  Training baseline RF...")
rf_base = RandomForestClassifier(n_estimators=20, n_jobs=-1, random_state=42)
rf_base.fit(X_tr, y_tr)
preds_base = rf_base.predict(X_te)
recall_base = recall_score(y_te, preds_base, average=None)

# Clip at 99.9th percentile
print("  Training clipped RF...")
X_tr_clipped = np.clip(X_tr, a_min=None, a_max=np.percentile(X_tr, 99.9, axis=0))
X_te_clipped = np.clip(X_te, a_min=None, a_max=np.percentile(X_tr, 99.9, axis=0)) # clip using train limits
rf_clip = RandomForestClassifier(n_estimators=20, n_jobs=-1, random_state=42)
rf_clip.fit(X_tr_clipped, y_tr)
preds_clip = rf_clip.predict(X_te_clipped)
recall_clip = recall_score(y_te, preds_clip, average=None)

rare_classes = [cls for cls, count in class_counts.items() if count < 100]
rare_class_impact = {}
unique_y_te = np.unique(y_te)

for cls in rare_classes:
    enc_val = list(le.classes_).index(cls)
    if enc_val in unique_y_te:
        idx = list(unique_y_te).index(enc_val)
        rare_class_impact[cls] = {
            'recall_base': float(recall_base[idx]),
            'recall_clipped': float(recall_clip[idx]),
            'impact': float(recall_clip[idx] - recall_base[idx])
        }
findings['outlier_clipping_impact'] = rare_class_impact

print("=== [6/7] Correlation and Redundancy ===")
# Compute Pearson and Spearman on 10k sample to avoid taking forever
corr_sample = df_clip[clip_cols].sample(n=min(10000, len(df_clip)), random_state=42)

pearson_corr = corr_sample.corr(method='pearson')
spearman_corr = corr_sample.corr(method='spearman')

# Find highly correlated pairs (Pearson > 0.95)
high_corr_pairs = []
for i in range(len(pearson_corr.columns)):
    for j in range(i):
        if abs(pearson_corr.iloc[i, j]) > 0.95:
            high_corr_pairs.append((pearson_corr.columns[i], pearson_corr.columns[j], float(pearson_corr.iloc[i, j])))
findings['high_correlation_pairs'] = high_corr_pairs

# Duplicate columns
dup_cols = []
cols_list = corr_sample.columns
for i in range(len(cols_list)):
    for j in range(i+1, len(cols_list)):
        if corr_sample[cols_list[i]].equals(corr_sample[cols_list[j]]):
            dup_cols.append((cols_list[i], cols_list[j]))
findings['exact_duplicate_columns'] = dup_cols

# Variance Inflation Factor (VIF)
# To prevent matrix singularity, we drop constant columns and take a small subset of features
vif_cols = [c for c in clip_cols if c not in constant_cols][:20] # Take first 20 valid features
vif_data = corr_sample[vif_cols]
# Add jitter to prevent singular matrix
vif_data = vif_data + np.random.normal(0, 1e-6, vif_data.shape)
vif_results = {}
for i, col in enumerate(vif_cols):
    try:
        vif_val = variance_inflation_factor(vif_data.values, i)
        vif_results[col] = float(vif_val)
    except Exception:
        vif_results[col] = float('inf')
findings['vif_sample'] = vif_results

# Redundancy Reduction Experiment
# Drop highly correlated features
to_drop = set([pair[1] for pair in high_corr_pairs]) # drop one from each pair
reduced_cols = [c for c in clip_cols if c not in to_drop]

print(f"  Training RF on Reduced Feature Set ({len(reduced_cols)} features)...")
X_tr_red = df_clip.iloc[X_tr_clipped[:1000].shape[0]] # Just dummy to get indices
X_tr_red, X_te_red = X_tr[:, [clip_cols.index(c) for c in reduced_cols]], X_te[:, [clip_cols.index(c) for c in reduced_cols]]
rf_red = RandomForestClassifier(n_estimators=20, n_jobs=-1, random_state=42)
t0 = time.time()
rf_red.fit(X_tr_red, y_tr)
red_train_time = time.time() - t0
preds_red = rf_red.predict(X_te_red)

# baseline time
t0 = time.time()
rf_base_time = RandomForestClassifier(n_estimators=20, n_jobs=-1, random_state=42)
rf_base_time.fit(X_tr, y_tr)
base_train_time = time.time() - t0

from sklearn.metrics import f1_score
f1_base = f1_score(y_te, rf_base_time.predict(X_te), average='macro')
f1_red = f1_score(y_te, preds_red, average='macro')

findings['redundancy_reduction_experiment'] = {
    'features_before': len(clip_cols),
    'features_after': len(reduced_cols),
    'f1_macro_before': float(f1_base),
    'f1_macro_after': float(f1_red),
    'train_time_before_sec': float(base_train_time),
    'train_time_after_sec': float(red_train_time)
}

print("=== [7/7] Saving Results ===")
with open('../results/figures/eda_findings.json', 'w') as f:
    json.dump(findings, f, indent=4)
print("Complete. Saved to eda_findings.json")
