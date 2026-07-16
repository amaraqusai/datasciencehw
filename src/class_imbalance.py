import os
import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, label_binarize
from sklearn.metrics import f1_score, fbeta_score, matthews_corrcoef, average_precision_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
import warnings

# Use imblearn pipeline to ensure zero-leakage resampling
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

# Suppress warnings
warnings.filterwarnings('ignore')

from preprocessing import load_real_cicids2017

os.makedirs('../results/tables', exist_ok=True)

def replace_infs(X):
    df = pd.DataFrame(X)
    return df.replace([np.inf, -np.inf], np.nan).values

print("=== [1/2] Loading Data ===")
df = load_real_cicids2017(sample_frac=0.05)

# Drop classes with <5 members so stratify works and train split has >=2 for SMOTE
le = LabelEncoder()
df['Label_Enc'] = le.fit_transform(df['Label'])
class_counts = df['Label_Enc'].value_counts()
valid_classes = class_counts[class_counts >= 5].index
df = df[df['Label_Enc'].isin(valid_classes)].copy()

# Re-encode to ensure contiguous
le2 = LabelEncoder()
df['Label_Enc'] = le2.fit_transform(df['Label'])
y = df['Label_Enc'].values
classes = np.unique(y)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('Label_Enc')
if 'Day' in numeric_cols:
    numeric_cols.remove('Day')

X_raw = df[numeric_cols].values

# Split strictly before ANY processing
X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y, test_size=0.3, random_state=42, stratify=y)

print("=== [2/2] Running Rigorous Imbalance Strategies ===")

xgb = XGBClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1, eval_metric='mlogloss')

# Identify minority class and benign class in train set
class_counts_train = pd.Series(y_tr).value_counts()
minority_class = class_counts_train.idxmin()
minority_class_name = le2.inverse_transform([minority_class])[0]
minority_class_name = str(minority_class_name).replace('\ufffd', '-').encode('ascii', 'ignore').decode('ascii')
print(f"  Rarest Class in Train: {minority_class_name} (Count: {class_counts_train[minority_class]})")

# Benign class index
benign_idx = -1
for i, name in enumerate(le2.classes_):
    if name == 'BENIGN':
        benign_idx = i
        break

# Dynamic SMOTE neighbors
k_neighbors = min(5, class_counts_train.min() - 1)
if k_neighbors < 1:
    k_neighbors = 1

strategies = {
    '1. No Imbalance Correction': Pipeline([
        ('inf_replacer', FunctionTransformer(replace_infs)),
        ('var_thresh', VarianceThreshold(threshold=0.0)),
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', xgb)
    ]),
    
    '2. Class Weights': Pipeline([
        ('inf_replacer', FunctionTransformer(replace_infs)),
        ('var_thresh', VarianceThreshold(threshold=0.0)),
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', xgb)
    ]),
    
    '3. Random Undersampling': Pipeline([
        ('inf_replacer', FunctionTransformer(replace_infs)),
        ('var_thresh', VarianceThreshold(threshold=0.0)),
        ('imputer', SimpleImputer(strategy='median')),
        ('sampler', RandomUnderSampler(random_state=42)),
        ('clf', xgb)
    ]),
    
    '4. Random Oversampling': Pipeline([
        ('inf_replacer', FunctionTransformer(replace_infs)),
        ('var_thresh', VarianceThreshold(threshold=0.0)),
        ('imputer', SimpleImputer(strategy='median')),
        ('sampler', RandomOverSampler(random_state=42)),
        ('clf', xgb)
    ]),
    
    '5. SMOTE': Pipeline([
        ('inf_replacer', FunctionTransformer(replace_infs)),
        ('var_thresh', VarianceThreshold(threshold=0.0)),
        ('imputer', SimpleImputer(strategy='median')),
        ('sampler', SMOTE(random_state=42, k_neighbors=k_neighbors)),
        ('clf', xgb)
    ])
}

results = []
y_test_binarized = label_binarize(y_te, classes=classes)

for name, pipe in strategies.items():
    print(f"  Evaluating: {name}...")
    
    t0 = time.time()
    if name == '2. Class Weights':
        weights = compute_sample_weight('balanced', y_tr)
        pipe.fit(X_tr, y_tr, clf__sample_weight=weights)
    else:
        pipe.fit(X_tr, y_tr)
    
    preds = pipe.predict(X_te)
    probas = pipe.predict_proba(X_te)
    
    f1 = f1_score(y_te, preds, average='macro')
    f2 = fbeta_score(y_te, preds, beta=2, average='macro')
    mcc = matthews_corrcoef(y_te, preds)
    pr_auc = average_precision_score(y_test_binarized, probas, average='macro')
    
    # Calculate Minority Recall
    recalls = recall_score(y_te, preds, average=None)
    min_recall = recalls[list(classes).index(minority_class)]
    
    # Calculate Benign FPR (False Positive Rate)
    # FPR = FP / (FP + TN)
    # For multiclass, FP for Benign = sum of all attack traffic predicted as Benign
    # TN for Benign = sum of all attack traffic predicted as NOT Benign
    cm = confusion_matrix(y_te, preds, labels=classes)
    # The row is actual, col is predicted
    # FP for benign = sum of column `benign_idx` excluding row `benign_idx`
    FP_benign = cm[:, benign_idx].sum() - cm[benign_idx, benign_idx]
    # TN for benign = sum of entire matrix excluding row `benign_idx` and col `benign_idx`
    TN_benign = cm.sum() - cm[benign_idx, :].sum() - cm[:, benign_idx].sum() + cm[benign_idx, benign_idx]
    
    benign_fpr = FP_benign / (FP_benign + TN_benign + 1e-9)
    
    results.append({
        'Imbalance Strategy': name,
        'F1-Macro': round(f1, 4),
        'F2-Macro': round(f2, 4),
        'MCC': round(mcc, 4),
        'PR-AUC': round(pr_auc, 4),
        'Minority Recall': round(min_recall, 4),
        'Benign FPR': round(benign_fpr, 4)
    })

results_df = pd.DataFrame(results)
results_df.to_csv('../results/tables/imbalance_table.csv', index=False)
print("\nImbalance Experiment Complete:")
print(results_df.to_string())
