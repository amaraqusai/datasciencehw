import os
import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, label_binarize
from sklearn.metrics import f1_score, fbeta_score, matthews_corrcoef, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

from preprocessing import load_real_cicids2017

os.makedirs('../results/tables', exist_ok=True)

class DropCorrelatedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.to_drop = []
        
    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return self
        
    def transform(self, X):
        df = pd.DataFrame(X)
        if len(self.to_drop) > 0:
            df = df.drop(columns=self.to_drop)
        return df.values

# Define safe, stateless function transformers
def replace_infs(X):
    df = pd.DataFrame(X)
    return df.replace([np.inf, -np.inf], np.nan).values

def apply_log1p(X):
    # Apply log1p to heavily skewed columns (max > 1000)
    df = pd.DataFrame(X)
    for col in df.columns:
        if df[col].max() > 1000 and df[col].min() >= 0:
            df[col] = np.log1p(df[col])
    return df.values

print("=== [1/2] Loading Data & Engineering Row-Wise Features ===")
df = load_real_cicids2017(sample_frac=0.05)

le = LabelEncoder()
df['Label_Enc'] = le.fit_transform(df['Label'])
class_counts = df['Label_Enc'].value_counts()
valid_classes = class_counts[class_counts >= 2].index
df = df[df['Label_Enc'].isin(valid_classes)].copy()

le2 = LabelEncoder()
df['Label_Enc'] = le2.fit_transform(df['Label'])
y = df['Label_Enc'].values
classes = np.unique(y)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('Label_Enc')
if 'Day' in numeric_cols:
    numeric_cols.remove('Day')

# Note: Row-wise mathematical engineering does not leak data across rows.
X_raw = df[numeric_cols].copy()
X_eng = X_raw.copy()

if 'Total Fwd Packets' in df.columns and 'Total Backward Packets' in df.columns:
    X_eng['Eng_Total_Packets'] = df['Total Fwd Packets'] + df['Total Backward Packets']
    X_eng['Eng_Fwd_Bwd_Pkt_Ratio'] = df['Total Fwd Packets'] / (df['Total Backward Packets'] + 1e-6)
    
if 'Total Length of Fwd Packets' in df.columns and 'Total Length of Bwd Packets' in df.columns:
    X_eng['Eng_Total_Bytes'] = df['Total Length of Fwd Packets'] + df['Total Length of Bwd Packets']
    X_eng['Eng_Fwd_Bwd_Byte_Ratio'] = df['Total Length of Fwd Packets'] / (df['Total Length of Bwd Packets'] + 1e-6)
    X_eng['Eng_Bytes_Per_Packet'] = X_eng['Eng_Total_Bytes'] / (X_eng['Eng_Total_Packets'] + 1e-6)

if 'FIN Flag Count' in df.columns and 'SYN Flag Count' in df.columns:
    X_eng['Eng_Combined_Flags'] = df['FIN Flag Count'] + df['SYN Flag Count'] + df['RST Flag Count'] + df['PSH Flag Count'] + df['ACK Flag Count'] + df['URG Flag Count']

# One split to rule them all
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw.values, y, test_size=0.3, random_state=42, stratify=y)
X_train_eng, X_test_eng, _, _ = train_test_split(X_eng.values, y, test_size=0.3, random_state=42, stratify=y)

print("=== [2/2] Running Strict Pipeline Ablation ===")

# Base Model
xgb = XGBClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1, eval_metric='mlogloss')

pipelines = {
    '1. Original Features': Pipeline([
        ('inf_replacer', FunctionTransformer(replace_infs)),
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)), # Raw original has 0 fill to just make it run
        ('clf', xgb)
    ]),
    
    '2. Cleaned Features': Pipeline([
        ('inf_replacer', FunctionTransformer(replace_infs)),
        ('var_thresh', VarianceThreshold(threshold=0.0)), # Drops constants
        ('imputer', SimpleImputer(strategy='median')), # Learns median on X_train ONLY
        ('clf', xgb)
    ]),
    
    '3. Reduced Non-Redundant Features': Pipeline([
        ('inf_replacer', FunctionTransformer(replace_infs)),
        ('var_thresh', VarianceThreshold(threshold=0.0)),
        ('imputer', SimpleImputer(strategy='median')),
        ('corr_drop', DropCorrelatedFeatures(threshold=0.95)), # Learns correlations on X_train ONLY
        ('clf', xgb)
    ]),
    
    '4. Log-Transformed Features': Pipeline([
        ('inf_replacer', FunctionTransformer(replace_infs)),
        ('var_thresh', VarianceThreshold(threshold=0.0)),
        ('imputer', SimpleImputer(strategy='median')),
        ('corr_drop', DropCorrelatedFeatures(threshold=0.95)),
        ('log1p', FunctionTransformer(apply_log1p)),
        ('clf', xgb)
    ])
}

# The remaining pipelines use the Engineered dataset
pipelines_eng = {
    '5. Engineered Cybersecurity Features': Pipeline([
        ('inf_replacer', FunctionTransformer(replace_infs)),
        ('var_thresh', VarianceThreshold(threshold=0.0)),
        ('imputer', SimpleImputer(strategy='median')),
        ('corr_drop', DropCorrelatedFeatures(threshold=0.95)),
        ('log1p', FunctionTransformer(apply_log1p)),
        ('clf', xgb)
    ]),
    
    '6. Selected Top-K Features': Pipeline([
        ('inf_replacer', FunctionTransformer(replace_infs)),
        ('var_thresh', VarianceThreshold(threshold=0.0)),
        ('imputer', SimpleImputer(strategy='median')),
        ('corr_drop', DropCorrelatedFeatures(threshold=0.95)),
        ('log1p', FunctionTransformer(apply_log1p)),
        ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1), max_features=25)), # Learns importances on X_train ONLY
        ('clf', xgb)
    ])
}

results = []
y_test_binarized = label_binarize(y_test, classes=classes)

def evaluate_pipeline(name, pipe, X_tr, X_te, y_tr, y_te):
    print(f"  Evaluating: {name}...")
    
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    train_time = time.time() - t0
    
    # Calculate Number of Features processed by the classifier
    if hasattr(pipe.named_steps['clf'], 'n_features_in_'):
        n_features = pipe.named_steps['clf'].n_features_in_
    else:
        n_features = "Unknown"
        
    t0 = time.time()
    preds = pipe.predict(X_te)
    probas = pipe.predict_proba(X_te)
    infer_time = time.time() - t0
    
    f1 = f1_score(y_te, preds, average='macro')
    f2 = fbeta_score(y_te, preds, beta=2, average='macro')
    mcc = matthews_corrcoef(y_te, preds)
    pr_auc = average_precision_score(y_test_binarized, probas, average='macro')
    
    results.append({
        'Feature Set': name,
        'Num Features': n_features,
        'F1-Macro': round(f1, 4),
        'F2-Macro': round(f2, 4),
        'MCC': round(mcc, 4),
        'PR-AUC': round(pr_auc, 4),
        'Train Time (s)': round(train_time, 2),
        'Inference Time (s)': round(infer_time, 4)
    })

# Evaluate first 4 on X_train_raw
for name, pipe in pipelines.items():
    evaluate_pipeline(name, pipe, X_train_raw, X_test_raw, y_train, y_test)

# Evaluate last 2 on X_train_eng
for name, pipe in pipelines_eng.items():
    evaluate_pipeline(name, pipe, X_train_eng, X_test_eng, y_train, y_test)

results_df = pd.DataFrame(results)
results_df.to_csv('../results/tables/ablation_table.csv', index=False)
print("\nAblation Experiment Complete:")
print(results_df.to_string())
