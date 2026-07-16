import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, matthews_corrcoef, confusion_matrix
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

from preprocessing import load_real_cicids2017

os.makedirs('../results/tables', exist_ok=True)

def replace_infs(X):
    df = pd.DataFrame(X)
    return df.replace([np.inf, -np.inf], np.nan).values

print("=== [1/3] Loading Data for Threshold Analysis ===")
df = load_real_cicids2017(sample_frac=0.05)

# Convert to Binary Classification (0: Benign, 1: Attack)
y = (df['Label'] != 'BENIGN').astype(int).values

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Day' in numeric_cols:
    numeric_cols.remove('Day')

X = df[numeric_cols].values

# Split: 60% Train, 20% Validation, 20% Test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp) # 0.25 of 0.8 = 0.2

print("=== [2/3] Training XGBoost Baseline ===")
pipe = Pipeline([
    ('inf_replacer', FunctionTransformer(replace_infs)),
    ('var_thresh', VarianceThreshold(threshold=0.0)),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=-1
    ))
])

pipe.fit(X_train, y_train)

print("=== [3/3] Evaluating Thresholds on Validation Set ===")
val_probs = pipe.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.01, 1.00, 0.01)
val_results = []

for t in thresholds:
    preds = (val_probs >= t).astype(int)
    
    prec = precision_score(y_val, preds, zero_division=0)
    rec = recall_score(y_val, preds, zero_division=0)
    f1 = f1_score(y_val, preds, zero_division=0)
    f2 = fbeta_score(y_val, preds, beta=2, zero_division=0)
    mcc = matthews_corrcoef(y_val, preds)
    
    tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
    alerts_per_10k = (tp + fp) / len(y_val) * 10000
    
    val_results.append({
        'Threshold': t,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'F2': f2,
        'MCC': mcc,
        'FP': fp,
        'FN': fn,
        'Alerts/10k': alerts_per_10k
    })

val_df = pd.DataFrame(val_results)
val_df.to_csv('../results/tables/threshold_val_table.csv', index=False)

# Select Optimal Thresholds
best_f2_t = val_df.loc[val_df['F2'].idxmax()]['Threshold']
best_mcc_t = val_df.loc[val_df['MCC'].idxmax()]['Threshold']

candidates = {
    'Default (0.5)': 0.5,
    'Max F2': best_f2_t,
    'Max MCC': best_mcc_t
}

print(f"  Selected Max F2 Threshold: {best_f2_t:.2f}")
print(f"  Selected Max MCC Threshold: {best_mcc_t:.2f}")

print("\n=== Evaluating Selected Thresholds on Unseen Test Set ===")
test_probs = pipe.predict_proba(X_test)[:, 1]
test_results = []

for name, t in candidates.items():
    preds = (test_probs >= t).astype(int)
    
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    f2 = fbeta_score(y_test, preds, beta=2, zero_division=0)
    mcc = matthews_corrcoef(y_test, preds)
    
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    alerts_per_10k = (tp + fp) / len(y_test) * 10000
    
    test_results.append({
        'Strategy': name,
        'Threshold': t,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'F2': f2,
        'MCC': mcc,
        'FP': fp,
        'FN': fn,
        'Alerts/10k': alerts_per_10k
    })

test_df = pd.DataFrame(test_results)
test_df.to_csv('../results/tables/threshold_test_table.csv', index=False)
print("\nTest Set Threshold Results:")
print(test_df.to_string())
