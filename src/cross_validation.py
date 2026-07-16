import os
import time
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, StandardScaler
from sklearn.metrics import make_scorer, f1_score, fbeta_score, matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import resample
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

from model_utils import PyTorchDBNWrapper
from preprocessing import load_real_cicids2017
import os

os.makedirs('../results/tables', exist_ok=True)

def replace_infs(X):
    df = pd.DataFrame(X)
    return df.replace([np.inf, -np.inf], np.nan).values

def safe_roc_auc_score(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
    except:
        return 0.0

def safe_pr_auc_score(y_true, y_score):
    try:
        # Require one-hot encoding for PR-AUC macro
        from sklearn.preprocessing import label_binarize
        classes = np.unique(y_true)
        if len(classes) == 1:
            return 0.0
        y_bin = label_binarize(y_true, classes=classes)
        if y_bin.shape[1] == 1:
            y_bin = np.hstack([1 - y_bin, y_bin])
        return average_precision_score(y_bin, y_score, average='macro')
    except:
        return 0.0

print("=== [1/3] Loading Data ===")
df = load_real_cicids2017(sample_frac=0.05)

le = LabelEncoder()
df['Label_Enc'] = le.fit_transform(df['Label'])
class_counts = df['Label_Enc'].value_counts()
valid_classes = class_counts[class_counts >= 5].index # Require at least 5 for 5-fold CV
df = df[df['Label_Enc'].isin(valid_classes)].copy()

le2 = LabelEncoder()
df['Label_Enc'] = le2.fit_transform(df['Label'])
y = df['Label_Enc'].values
classes = np.unique(y)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('Label_Enc')
if 'Day' in numeric_cols:
    numeric_cols.remove('Day')

X_raw = df[numeric_cols].values

# Hold out a final test set for bootstrapping
X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y, test_size=0.3, random_state=42, stratify=y)

print("=== [2/3] Running 5-Fold Stratified CV ===")

# To ensure the DBN matches the input size after variance thresholding, we'll dummy-run it
inf_rep = FunctionTransformer(replace_infs)
var_thr = VarianceThreshold(threshold=0.0)
imp = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_dummy = imp.fit_transform(var_thr.fit_transform(inf_rep.fit_transform(X_tr)))
n_features = X_dummy.shape[1]

models = {
    'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
    'Multilayer Perceptron': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=20, random_state=42),
    'Deep Belief Network': PyTorchDBNWrapper(n_hidden=(64, 32), num_epochs=(2, 2), fine_tune_epochs=3, num_classes=len(classes))
}

scorers = {
    'f1_macro': make_scorer(f1_score, average='macro'),
    'f1_weighted': make_scorer(f1_score, average='weighted'),
    'f2_macro': make_scorer(fbeta_score, beta=2, average='macro'),
    'mcc': make_scorer(matthews_corrcoef),
    'roc_auc': make_scorer(safe_roc_auc_score, response_method='predict_proba'),
    'pr_auc': make_scorer(safe_pr_auc_score, response_method='predict_proba')
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results_list = []
fitted_pipelines = {}

for name, model in models.items():
    print(f"  Cross-Validating {name}...")
    # Scale data for neural networks (and also RF for consistency in this test)
    pipe = Pipeline([
        ('inf_replacer', FunctionTransformer(replace_infs)),
        ('var_thresh', VarianceThreshold(threshold=0.0)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', model)
    ])
    
    cv_res = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=scorers, n_jobs=1, return_estimator=True)
    
    # Save the first estimator to run Bootstrap on the test set
    fitted_pipelines[name] = cv_res['estimator'][0]
    
    cv_results_list.append({
        'Model': name,
        'Macro-F1 (Mean ± SD)': f"{cv_res['test_f1_macro'].mean():.4f} ± {cv_res['test_f1_macro'].std():.4f}",
        'Weighted-F1': f"{cv_res['test_f1_weighted'].mean():.4f} ± {cv_res['test_f1_weighted'].std():.4f}",
        'F2-Macro': f"{cv_res['test_f2_macro'].mean():.4f} ± {cv_res['test_f2_macro'].std():.4f}",
        'MCC': f"{cv_res['test_mcc'].mean():.4f} ± {cv_res['test_mcc'].std():.4f}",
        'ROC-AUC': f"{cv_res['test_roc_auc'].mean():.4f} ± {cv_res['test_roc_auc'].std():.4f}",
        'PR-AUC': f"{cv_res['test_pr_auc'].mean():.4f} ± {cv_res['test_pr_auc'].std():.4f}",
        'Fit Time (s)': f"{cv_res['fit_time'].mean():.2f}"
    })

cv_df = pd.DataFrame(cv_results_list)
cv_df.to_csv('../results/tables/cv_results.csv', index=False)
print("\nCV Results:")
print(cv_df.to_string())

print("\n=== [3/3] Uncertainty Analysis (100 Bootstrap Iterations) ===")
# We will use the trained pipeline to predict on X_test, then bootstrap the predictions
n_iterations = 100
alpha = 0.95

bootstrap_results = []

for name, pipe in fitted_pipelines.items():
    print(f"  Bootstrapping {name}...")
    preds = pipe.predict(X_te)
    
    boot_f1 = []
    boot_mcc = []
    
    # Bootstrap the predictions and true labels together
    for i in range(n_iterations):
        # resample indices
        indices = resample(np.arange(len(y_te)), replace=True, n_samples=len(y_te), random_state=i)
        y_te_boot = y_te[indices]
        preds_boot = preds[indices]
        
        # Only compute if we have enough classes
        if len(np.unique(y_te_boot)) > 1:
            boot_f1.append(f1_score(y_te_boot, preds_boot, average='macro'))
            boot_mcc.append(matthews_corrcoef(y_te_boot, preds_boot))
            
    # Calculate 95% Confidence Intervals
    p_lower = ((1.0 - alpha) / 2.0) * 100
    p_upper = (alpha + ((1.0 - alpha) / 2.0)) * 100
    
    f1_lower = np.percentile(boot_f1, p_lower)
    f1_upper = np.percentile(boot_f1, p_upper)
    f1_mean = np.mean(boot_f1)
    
    mcc_lower = np.percentile(boot_mcc, p_lower)
    mcc_upper = np.percentile(boot_mcc, p_upper)
    mcc_mean = np.mean(boot_mcc)
    
    bootstrap_results.append({
        'Model': name,
        'Macro-F1 (95% CI)': f"{f1_mean:.4f} ({f1_lower:.4f} - {f1_upper:.4f})",
        'MCC (95% CI)': f"{mcc_mean:.4f} ({mcc_lower:.4f} - {mcc_upper:.4f})"
    })

boot_df = pd.DataFrame(bootstrap_results)
boot_df.to_csv('../results/tables/bootstrap_ci.csv', index=False)
print("\nBootstrap 95% Confidence Intervals:")
print(boot_df.to_string())
print("\nSuccess.")
