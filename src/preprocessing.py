import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# CICIDS2017 Feature Names (Official CICFlowMeter columns)
# ============================================================
CICIDS_FEATURES = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min',
    'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd Header Length', 'Bwd Header Length',
    'Fwd Packets/s', 'Bwd Packets/s',
    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
    'Packet Length Std', 'Packet Length Variance',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
    'CWE Flag Count', 'ECE Flag Count',
    'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size',
    'Avg Bwd Segment Size', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
    'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
    'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes',
    'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
]

ATTACK_TYPES = ['BENIGN', 'DoS Hulk', 'PortScan', 'DDoS',
                'DoS GoldenEye', 'FTP-Patator', 'SSH-Patator',
                'DoS slowloris', 'DoS Slowhttptest', 'Bot',
                'Web Attack - Brute Force', 'Web Attack - XSS',
                'Infiltration', 'Web Attack - Sql Injection', 'Heartbleed']


import glob
import os

def load_real_cicids2017(data_dir=None, sample_frac=0.10, random_state=42):
    """
    Load the authentic CICIDS2017 dataset from CSV files.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'MachineLearningCVE')
        
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}. Please run src/download_dataset.py first.")
    
    df_list = []
    for f in csv_files:
        print(f"Loading {os.path.basename(f)}...")
        # Handle mixed types or bad lines if any
        df_part = pd.read_csv(f, skipinitialspace=True, encoding='utf-8', low_memory=False)
        # Strip whitespace from column names
        df_part.columns = [c.strip() for c in df_part.columns]
        df_list.append(df_part)
        
    df = pd.concat(df_list, ignore_index=True)
    
    # Strip whitespace from labels just in case
    if 'Label' in df.columns:
        df['Label'] = df['Label'].astype(str).str.strip()
        # Fix known labeling typos in CICIDS2017 (e.g. Web Attack  Brute Force)
        df['Label'] = df['Label'].replace({
            'Web Attack  Brute Force': 'Web Attack - Brute Force',
            'Web Attack  XSS': 'Web Attack - XSS',
            'Web Attack  Sql Injection': 'Web Attack - Sql Injection'
        })
        
    # Sample to reduce memory footprint
    if sample_frac < 1.0:
        print(f"Sampling {sample_frac*100}% of {len(df)} rows stratified by Label...")
        from sklearn.model_selection import train_test_split
        # Identify classes with extremely few samples to avoid ValueError in train_test_split
        counts = df['Label'].value_counts()
        valid_classes = counts[counts > 5].index
        df_filtered = df[df['Label'].isin(valid_classes)]
        
        _, sampled_df = train_test_split(df_filtered, test_size=sample_frac, 
                                         stratify=df_filtered['Label'], random_state=random_state)
        df = sampled_df.reset_index(drop=True)
        
    return df


def run_eda(df):
    """Run comprehensive EDA checks and return a dictionary of findings."""
    results = {}

    # Shape
    results['shape'] = df.shape

    # Missing values
    missing = df.isnull().sum()
    results['missing_values'] = missing[missing > 0]
    results['total_missing'] = df.isnull().sum().sum()

    # Infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    for col in numeric_cols:
        n_inf = np.isinf(df[col]).sum()
        if n_inf > 0:
            inf_counts[col] = n_inf
    results['infinite_values'] = inf_counts

    # Constant features (zero variance)
    constant_cols = [col for col in numeric_cols if df[col].nunique() <= 1]
    results['constant_features'] = constant_cols

    # Near-constant features (>99% same value)
    near_constant = []
    for col in numeric_cols:
        if col not in constant_cols:
            top_freq = df[col].value_counts(normalize=True).iloc[0]
            if top_freq > 0.99:
                near_constant.append(col)
    results['near_constant_features'] = near_constant

    # Duplicate rows
    results['duplicate_rows'] = df.duplicated().sum()

    # Class distribution
    results['class_distribution'] = df['Label'].value_counts()
    results['class_percentages'] = df['Label'].value_counts(normalize=True) * 100

    # Duplicate feature columns
    dup_cols = []
    cols = list(numeric_cols)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if df[cols[i]].equals(df[cols[j]]):
                dup_cols.append((cols[i], cols[j]))
    results['duplicate_columns'] = dup_cols

    return results


def clean_data(df):
    """Full cleaning pipeline with leakage-safe processing."""
    df_clean = df.copy()

    # 1. Replace infinities with NaN
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # 2. Median imputation
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

    # 3. Drop constant features
    constant_cols = [col for col in numeric_cols if df_clean[col].nunique() <= 1]
    df_clean.drop(columns=constant_cols, inplace=True)

    # 4. Drop duplicate columns
    cols_to_drop = []
    remaining_numeric = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    for i in range(len(remaining_numeric)):
        for j in range(i + 1, len(remaining_numeric)):
            if df_clean[remaining_numeric[i]].equals(df_clean[remaining_numeric[j]]):
                if remaining_numeric[j] not in cols_to_drop:
                    cols_to_drop.append(remaining_numeric[j])
    df_clean.drop(columns=cols_to_drop, inplace=True)

    # 5. Binary label encoding
    df_clean['Binary_Label'] = (df_clean['Label'] != 'BENIGN').astype(int)

    return df_clean, constant_cols, cols_to_drop


def split_and_scale(df_clean, test_size=0.2, random_state=42):
    """Leakage-safe train/test split and scaling."""
    feature_cols = [c for c in df_clean.columns if c not in ['Label', 'Binary_Label']]
    X = df_clean[feature_cols]
    y = df_clean['Binary_Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit scaler ONLY on training data (leakage-safe)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
