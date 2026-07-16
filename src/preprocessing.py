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
        
        # Extract Day from filename
        day_str = os.path.basename(f).split('-')[0].capitalize()
        df_part['Day'] = day_str
        
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


def run_duplicate_audit(df):
    """Run a comprehensive duplicate and leakage audit on the dataframe."""
    print("Running Duplicate Audit...")
    
    feature_cols = [c for c in df.columns if c not in ['Label', 'Binary_Label', 'Multi_Label', 'Day']]
    
    exact_dups = df.duplicated().sum()
    feature_dups = df.duplicated(subset=feature_cols).sum()
    
    # Identical features with conflicting labels
    # We find groups of identical feature vectors that have >1 unique label
    grouped = df.groupby(feature_cols, dropna=False)['Label'].nunique()
    conflicting_vectors = (grouped > 1).sum()
    
    dup_per_day = {}
    if 'Day' in df.columns:
        day_dups = df[df.duplicated(keep=False)]
        dup_per_day = day_dups.groupby('Day').size().to_dict()
        
    dup_per_class = {}
    if 'Label' in df.columns:
        class_dups = df[df.duplicated(keep=False)]
        dup_per_class = class_dups.groupby('Label').size().to_dict()
        
    audit_results = {
        'exact_duplicates': int(exact_dups),
        'feature_duplicates_ignoring_label': int(feature_dups),
        'conflicting_label_vectors': int(conflicting_vectors),
        'duplicates_per_day': dup_per_day,
        'duplicates_per_class': dup_per_class
    }
    
    print(f"  Exact duplicate rows: {exact_dups}")
    print(f"  Duplicate feature vectors (ignoring label): {feature_dups}")
    print(f"  Identical feature vectors with conflicting labels: {conflicting_vectors}")
    
    return audit_results

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

    # 5. Label encoding (Multi-class and Binary)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df_clean['Multi_Label'] = le.fit_transform(df_clean['Label'])
    df_clean['Binary_Label'] = (df_clean['Label'] != 'BENIGN').astype(int)

    return df_clean, constant_cols, cols_to_drop, le


def split_and_scale(df_clean, task_type='multiclass', split_strategy='random_original', test_size=0.2, val_size=0.2, random_state=42):
    """Leakage-safe train/val/test split and scaling with support for duplicate handling strategies."""
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler
    import hashlib
    
    feature_cols = [c for c in df_clean.columns if c not in ['Label', 'Binary_Label', 'Multi_Label', 'Day']]
    
    # 1. Apply Deduplication BEFORE splitting if requested
    if split_strategy == 'random_deduplicated':
        print("Deduplicating entire dataset before split...")
        df_clean = df_clean.drop_duplicates(subset=feature_cols)
        print(f"Remaining records: {len(df_clean)}")

    X = df_clean[feature_cols]
    if task_type == 'multiclass':
        y = df_clean['Multi_Label']
    else:
        y = df_clean['Binary_Label']

    day_col = df_clean['Day'] if 'Day' in df_clean.columns else pd.Series(['Monday'] * len(df_clean), index=df_clean.index)

    print(f"Using split strategy: {split_strategy}")

    if split_strategy == 'chronological':
        train_mask = day_col.isin(['Monday', 'Tuesday', 'Wednesday'])
        val_mask = day_col == 'Thursday'
        test_mask = day_col == 'Friday'
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
    elif split_strategy == 'grouped':
        # Hash features to create a group ID for identical flows
        print("Hashing feature vectors for Grouped Split...")
        # A fast way to hash rows:
        groups = X.apply(lambda row: hash(tuple(row)), axis=1)
        
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_val_idx, test_idx = next(gss_test.split(X, y, groups))
        
        X_temp, y_temp, groups_temp = X.iloc[train_val_idx], y.iloc[train_val_idx], groups.iloc[train_val_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        val_fraction = val_size / (1.0 - test_size)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
        train_idx, val_idx = next(gss_val.split(X_temp, y_temp, groups_temp))
        
        X_train, y_train = X_temp.iloc[train_idx], y_temp.iloc[train_idx]
        X_val, y_val = X_temp.iloc[val_idx], y_temp.iloc[val_idx]
        
    else: # random_original or random_deduplicated
        # Ensure no classes have < 3 samples to do Train/Val/Test
        class_counts = y.value_counts()
        valid_classes = class_counts[class_counts >= 3].index
        if len(valid_classes) < len(class_counts):
            print(f"Dropping classes with < 3 samples for stratification: {set(class_counts.index) - set(valid_classes)}")
            valid_mask = y.isin(valid_classes)
            X = X[valid_mask]
            y = y[valid_mask]

        # First split off the test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        val_fraction = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_fraction, random_state=random_state, stratify=y_temp
        )

    # 2. Fit scaler ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=feature_cols, index=X_val.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    # 3. Calculate Train-Test Leakage
    print("Calculating Train-Test Leakage...")
    # Convert training rows to a set of hashes for fast lookup
    train_hashes = set(X_train.apply(lambda row: hash(tuple(row)), axis=1))
    test_hashes = X_test.apply(lambda row: hash(tuple(row)), axis=1)
    leaked_count = test_hashes.isin(train_hashes).sum()
    leakage_pct = (leaked_count / len(test_hashes)) * 100 if len(test_hashes) > 0 else 0
    
    print(f"  Test Leakage: {leaked_count} / {len(test_hashes)} ({leakage_pct:.2f}%) records in Test are also in Train.")

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, feature_cols, leakage_pct
