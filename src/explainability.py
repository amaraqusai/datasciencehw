import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import xgboost as xgb
import shap
import warnings

warnings.filterwarnings('ignore')

from preprocessing import load_real_cicids2017

os.makedirs('../results/tables', exist_ok=True)

def replace_infs(X):
    df = pd.DataFrame(X)
    return df.replace([np.inf, -np.inf], np.nan).values

print("=== [1/4] Loading Data for XAI ===")
df = load_real_cicids2017(sample_frac=0.05)

original_labels = np.array(df['Label'].tolist())
y_binary = (df['Label'] != 'BENIGN').astype(int).values

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Day' in numeric_cols:
    numeric_cols.remove('Day')
X = df[numeric_cols].values

# We do NOT use StandardScaler here so SHAP values are in original interpretable units!
X_train, X_test, y_train, y_test, ol_train, ol_test = train_test_split(
    X, y_binary, original_labels, test_size=0.2, random_state=42, stratify=y_binary
)

print("=== [2/4] Training XGBoost Baseline ===")
# Use pipeline for missing values only
prep_pipe = Pipeline([
    ('inf_replacer', FunctionTransformer(replace_infs)),
    ('var_thresh', VarianceThreshold(threshold=0.0)),
    ('imputer', SimpleImputer(strategy='median'))
])

X_train_proc = prep_pipe.fit_transform(X_train)
X_test_proc = prep_pipe.transform(X_test)

# To keep feature names aligned, we get the mask of retained features
mask = prep_pipe.named_steps['var_thresh'].get_support()
active_features = np.array(numeric_cols)[mask]

model = xgb.XGBClassifier(n_estimators=100, max_depth=10, random_state=42, eval_metric='logloss', use_label_encoder=False, n_jobs=-1)
model.fit(X_train_proc, y_train)

print("=== [3/4] Global Explainability ===")
# 1. Permutation Importance
print("  Calculating Permutation Importance...")
perm_importance = permutation_importance(model, X_test_proc, y_test, n_repeats=5, random_state=42, scoring='f1', n_jobs=-1)
global_df = pd.DataFrame({
    'Feature': active_features,
    'Permutation Importance (Mean)': perm_importance.importances_mean,
    'XGB Feature Importance': model.feature_importances_
})

# 2. Global SHAP
print("  Calculating Global SHAP...")
explainer = shap.TreeExplainer(model)
# Sample 1000 for fast global SHAP
sample_indices = np.random.choice(X_test_proc.shape[0], min(1000, X_test_proc.shape[0]), replace=False)
X_test_sample = X_test_proc[sample_indices]
shap_values = explainer.shap_values(X_test_sample)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
global_df['Mean Absolute SHAP'] = mean_abs_shap
global_df = global_df.sort_values(by='Mean Absolute SHAP', ascending=False).head(20) # Top 20

global_df.to_csv('../results/tables/global_shap.csv', index=False)

print("=== [4/4] Local Cohort Explainability ===")
preds = model.predict(X_test_proc)

tp_indices = np.where((y_test == 1) & (preds == 1))[0]
tn_indices = np.where((y_test == 0) & (preds == 0))[0]
fp_indices = np.where((y_test == 0) & (preds == 1))[0]
fn_indices = np.where((y_test == 1) & (preds == 0))[0]

local_results = []

def extract_top_reasons(idx, cohort_name):
    if len(idx) == 0: return
    # Select first representative
    i = idx[0]
    sv = explainer.shap_values(X_test_proc[i:i+1])[0]
    
    # Sort by absolute impact
    top_indices = np.argsort(np.abs(sv))[::-1][:3]
    
    local_results.append({
        'Cohort': cohort_name,
        'Original Label': ol_test[i],
        'Top 1 Feature': active_features[top_indices[0]],
        'Top 1 Value': X_test_proc[i][top_indices[0]],
        'Top 1 SHAP Impact': round(sv[top_indices[0]], 4),
        'Top 2 Feature': active_features[top_indices[1]],
        'Top 2 Value': X_test_proc[i][top_indices[1]],
        'Top 2 SHAP Impact': round(sv[top_indices[1]], 4),
        'Top 3 Feature': active_features[top_indices[2]],
        'Top 3 Value': X_test_proc[i][top_indices[2]],
        'Top 3 SHAP Impact': round(sv[top_indices[2]], 4)
    })

extract_top_reasons(tp_indices, 'Representative True Positive (Detected Attack)')
extract_top_reasons(tn_indices, 'Representative True Negative (Normal Traffic)')
extract_top_reasons(fp_indices, 'Representative False Positive (False Alarm)')
extract_top_reasons(fn_indices, 'Representative False Negative (Missed Attack)')

if local_results:
    local_df = pd.DataFrame(local_results)
    local_df.to_csv('../results/tables/local_shap.csv', index=False)
    print("\nLocal SHAP Explanations extracted successfully.")

print("\nXAI Module Complete!")
