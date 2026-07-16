import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

from preprocessing import load_real_cicids2017

os.makedirs('../results/tables', exist_ok=True)

def replace_infs(X):
    df = pd.DataFrame(X)
    return df.replace([np.inf, -np.inf], np.nan).values

print("=== [1/4] Loading Data for Error Forensics ===")
# Use a 10% sample to get enough test errors
df = load_real_cicids2017(sample_frac=0.1)

# Preserve Original Labels and Days
original_labels = df['Label'].astype(str).values
capture_days = df['Day'].astype(str).values if 'Day' in df.columns else np.array(['Unknown'] * len(df))

# Convert to Binary for Training
y_binary = (df['Label'] != 'BENIGN').astype(int).values

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Day' in numeric_cols:
    numeric_cols.remove('Day')
X = df[numeric_cols].values

# Hold out Test Set
X_train, X_test, y_train, y_test, ol_train, ol_test, day_train, day_test = train_test_split(
    X, y_binary, original_labels, capture_days, test_size=0.2, random_state=42, stratify=y_binary
)

print("=== [2/4] Training Forensic Model ===")
pipe = Pipeline([
    ('inf_replacer', FunctionTransformer(replace_infs)),
    ('var_thresh', VarianceThreshold(threshold=0.0)),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', use_label_encoder=False, n_jobs=-1))
])

pipe.fit(X_train, y_train)

# Transform the test set features to get scaled vectors for Euclidean distance
processor = Pipeline(pipe.steps[:-1])
X_test_scaled = processor.transform(X_test)

# Calculate centroids of the True Negative (Benign) and True Positive (Attack) classes on Train
X_train_scaled = processor.transform(X_train)
benign_centroid = X_train_scaled[y_train == 0].mean(axis=0).reshape(1, -1)

# Centroids for each specific attack type in train
attack_centroids = {}
for attack_type in np.unique(ol_train):
    if attack_type == 'BENIGN': continue
    mask = ol_train == attack_type
    if mask.sum() > 0:
        attack_centroids[attack_type] = X_train_scaled[mask].mean(axis=0).reshape(1, -1)

print("=== [3/4] Running Inference & Isolating Errors ===")
probs = pipe.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(int)

# Identify cohorts
fp_indices = np.where((y_test == 0) & (preds == 1))[0]
fn_indices = np.where((y_test == 1) & (preds == 0))[0]

print(f"  Found {len(fp_indices)} False Positives")
print(f"  Found {len(fn_indices)} False Negatives")

print("=== [4/4] Generating Forensic Reports ===")

# --- False Negative Analysis ---
fn_results = []
for idx in fn_indices:
    original_type = ol_test[idx]
    
    # Calculate Similarity to Benign (Euclidean Distance)
    vec = X_test_scaled[idx].reshape(1, -1)
    dist_to_benign = cdist(vec, benign_centroid, metric='euclidean')[0][0]
    
    # Check if the attack was rare in training
    train_count = (ol_train == original_type).sum()
    rare = "Yes" if train_count < 100 else "No"
    
    fn_results.append({
        'Original Attack Type': original_type,
        'Capture Day': day_test[idx],
        'Predicted Probability (Attack)': round(probs[idx], 4),
        'Distance to Benign Centroid': round(dist_to_benign, 2),
        'Rare in Training? (<100 samples)': rare,
        'Raw Feature: Total Fwd Packets': X_test[idx][numeric_cols.index('Total Fwd Packets')] if 'Total Fwd Packets' in numeric_cols else 0,
        'Raw Feature: Flow Duration': X_test[idx][numeric_cols.index('Flow Duration')] if 'Flow Duration' in numeric_cols else 0,
    })

if fn_results:
    fn_df = pd.DataFrame(fn_results)
    fn_df.to_csv('../results/tables/fn_analysis.csv', index=False)
    print("\nSample of False Negatives:")
    print(fn_df.head(10).to_string())

# --- False Positive Analysis ---
fp_results = []
for idx in fp_indices:
    # Calculate Similarity to Attack Classes
    vec = X_test_scaled[idx].reshape(1, -1)
    min_dist = float('inf')
    closest_attack = "Unknown"
    
    for atk, centroid in attack_centroids.items():
        dist = cdist(vec, centroid, metric='euclidean')[0][0]
        if dist < min_dist:
            min_dist = dist
            closest_attack = atk
            
    fp_results.append({
        'Benign Traffic Type': ol_test[idx],
        'Capture Day': day_test[idx],
        'Predicted Probability (Attack)': round(probs[idx], 4),
        'Closest Attack Centroid': closest_attack,
        'Distance to Closest Attack': round(min_dist, 2),
        'Raw Feature: Total Fwd Packets': X_test[idx][numeric_cols.index('Total Fwd Packets')] if 'Total Fwd Packets' in numeric_cols else 0,
        'Raw Feature: Flow Duration': X_test[idx][numeric_cols.index('Flow Duration')] if 'Flow Duration' in numeric_cols else 0,
    })

if fp_results:
    fp_df = pd.DataFrame(fp_results)
    fp_df.to_csv('../results/tables/fp_analysis.csv', index=False)
    print("\nSample of False Positives:")
    print(fp_df.head(10).to_string())

print("\nError Analysis Module Complete!")
