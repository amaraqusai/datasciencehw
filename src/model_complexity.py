import os
import time
import tracemalloc
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import warnings

warnings.filterwarnings('ignore')

from preprocessing import load_real_cicids2017
from model_utils import PyTorchDBNWrapper

os.makedirs('../results/tables', exist_ok=True)
os.makedirs('../results/models', exist_ok=True)

def replace_infs(X):
    df = pd.DataFrame(X)
    return df.replace([np.inf, -np.inf], np.nan).values

print("=== [1/2] Loading Data for Complexity Audit ===")
# Use a 5% sample to keep hardware metrics realistic but feasible
df = load_real_cicids2017(sample_frac=0.05)

le = LabelEncoder()
df['Label_Enc'] = le.fit_transform(df['Label'])
class_counts = df['Label_Enc'].value_counts()
valid_classes = class_counts[class_counts >= 5].index
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

# Hold out exactly 10,000 for standardizing inference time
X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y, test_size=10000, random_state=42, stratify=y)

inf_rep = FunctionTransformer(replace_infs)
var_thr = VarianceThreshold(threshold=0.0)
imp = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_tr_proc = scaler.fit_transform(imp.fit_transform(var_thr.fit_transform(inf_rep.fit_transform(X_tr))))
X_te_proc = scaler.transform(imp.transform(var_thr.transform(inf_rep.transform(X_te))))
n_features = X_tr_proc.shape[1]

models = {
    'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1),
    'Multilayer Perceptron': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=20, random_state=42),
    'Deep Belief Network': PyTorchDBNWrapper(n_hidden=(64, 32), num_epochs=(2, 2), fine_tune_epochs=3)
}

print("=== [2/2] Profiling Models ===")
results = []

for name, model in models.items():
    print(f"  Profiling {name}...")
    
    # Measure RAM and Train Time
    tracemalloc.start()
    t0 = time.time()
    
    if name == 'Deep Belief Network':
        # Provide validation set for DBN early stopping
        model.fit(X_tr_proc, y_tr, X_val=X_te_proc, y_val=y_te)
    else:
        model.fit(X_tr_proc, y_tr)
        
    train_time = time.time() - t0
    current, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak_memory / (1024 * 1024)
    
    # Measure Inference Time
    t0 = time.time()
    _ = model.predict(X_te_proc)
    infer_time = time.time() - t0
    
    # Calculate Parameters and Disk Size
    n_params = "Unknown"
    disk_size_mb = 0.0
    hw_context = "CPU"
    
    if name == 'Random Forest':
        n_params = sum([tree.tree_.node_count for tree in model.estimators_])
        joblib.dump(model, '../results/models/rf.pkl')
        disk_size_mb = os.path.getsize('../results/models/rf.pkl') / (1024 * 1024)
        
    elif name == 'Multilayer Perceptron':
        n_params = sum([coef.size for coef in model.coefs_]) + sum([intercept.size for intercept in model.intercepts_])
        joblib.dump(model, '../results/models/mlp.pkl')
        disk_size_mb = os.path.getsize('../results/models/mlp.pkl') / (1024 * 1024)
        
    elif name == 'Deep Belief Network':
        # The underlying PyTorch model parameters
        total_params = 0
        if model.model is not None:
            total_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            torch.save(model.model.state_dict(), '../results/models/dbn.pt')
            disk_size_mb = os.path.getsize('../results/models/dbn.pt') / (1024 * 1024)
            hw_context = "GPU" if torch.cuda.is_available() else "CPU"
        n_params = total_params
        
    results.append({
        'Model': name,
        'Params/Nodes': n_params,
        'Train Latency (s)': round(train_time, 2),
        'Inference Latency (10k flows) (s)': round(infer_time, 4),
        'Peak RAM (MB)': round(peak_memory_mb, 2),
        'Disk Size (MB)': round(disk_size_mb, 2),
        'Hardware': hw_context
    })

complexity_df = pd.DataFrame(results)
complexity_df.to_csv('../results/tables/complexity_table.csv', index=False)
print("\nComplexity Audit Complete:")
print(complexity_df.to_string())
