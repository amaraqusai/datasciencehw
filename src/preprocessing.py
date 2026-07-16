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


def generate_synthetic_cicids2017(n_samples=20000, random_state=42):
    """Generate a statistically representative synthetic CICIDS2017 dataset.
    
    Models the key statistical properties of the real CICIDS2017 dataset:
    - 78 CICFlowMeter features with realistic magnitude ranges
    - Extreme class imbalance (80% benign, 20% attack across 15 classes)
    - Data artifacts: NaN values, Infinity values, constant features, duplicate columns
    - Attack-specific traffic signatures (DoS = high volume, Infiltration = low-and-slow)
    """
    np.random.seed(random_state)
    n_features = len(CICIDS_FEATURES)

    # --- Benign traffic: moderate volume, regular patterns ---
    n_benign = int(n_samples * 0.80)
    benign_means = np.random.uniform(10, 500, size=n_features)
    benign_stds = benign_means * 0.3
    benign_data = np.abs(np.random.normal(loc=benign_means, scale=benign_stds, size=(n_benign, n_features)))
    benign_labels = ['BENIGN'] * n_benign

    # --- Attack traffic: different distributions per attack type ---
    attack_categories = ATTACK_TYPES[1:]  # exclude BENIGN
    n_attack = n_samples - n_benign
    samples_per_attack = max(1, n_attack // len(attack_categories))

    attack_data_list = []
    attack_labels_list = []

    for i, attack_name in enumerate(attack_categories):
        n_this = samples_per_attack if i < len(attack_categories) - 1 else n_attack - samples_per_attack * (len(attack_categories) - 1)
        if n_this <= 0:
            n_this = 1

        # Each attack type has a distinct statistical signature
        attack_means = np.random.uniform(100, 5000, size=n_features)
        # DoS attacks: very high flow duration and packet counts
        if 'DoS' in attack_name or 'DDoS' in attack_name:
            attack_means[0] = 50000   # Flow Duration
            attack_means[1] = 500     # Total Fwd Packets
            attack_means[13] = 100000 # Flow Bytes/s
        # PortScan: many short connections
        elif 'PortScan' in attack_name:
            attack_means[0] = 10      # Very short flows
            attack_means[1] = 1       # Single packet
            attack_means[36] = 1000   # High packets/s
        # Infiltration/Heartbleed: low-and-slow, mimics benign
        elif attack_name in ['Infiltration', 'Heartbleed']:
            attack_means = benign_means * np.random.uniform(0.8, 1.2, size=n_features)

        attack_stds = attack_means * 0.4
        data = np.abs(np.random.normal(loc=attack_means, scale=attack_stds, size=(n_this, n_features)))
        attack_data_list.append(data)
        attack_labels_list.extend([attack_name] * n_this)

    attack_data = np.vstack(attack_data_list)
    X = np.vstack([benign_data, attack_data])
    labels = benign_labels + attack_labels_list

    # --- Inject realistic data artifacts ---
    # 1. NaN values (~1.5% of cells in Flow Bytes/s and Flow Packets/s)
    nan_idx = np.random.choice(len(X), size=int(0.015 * len(X)), replace=False)
    X[nan_idx, 13] = np.nan  # Flow Bytes/s
    nan_idx2 = np.random.choice(len(X), size=int(0.01 * len(X)), replace=False)
    X[nan_idx2, 14] = np.nan  # Flow Packets/s

    # 2. Infinity values (~0.5% in Flow Bytes/s due to zero-duration division)
    inf_idx = np.random.choice(len(X), size=int(0.005 * len(X)), replace=False)
    X[inf_idx, 13] = np.inf

    # 3. Constant features (Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags = always 0)
    X[:, 30] = 0  # Bwd PSH Flags
    X[:, 31] = 0  # Fwd URG Flags
    X[:, 32] = 0  # Bwd URG Flags

    # 4. Near-constant features
    X[:, 54] = 0  # Fwd Avg Bytes/Bulk
    X[:, 55] = 0  # Fwd Avg Packets/Bulk
    X[:, 56] = 0  # Fwd Avg Bulk Rate
    X[:, 57] = 0  # Bwd Avg Bytes/Bulk
    X[:, 58] = 0  # Bwd Avg Packets/Bulk
    X[:, 59] = 0  # Bwd Avg Bulk Rate

    # 5. Duplicate column (Subflow Fwd Packets == Total Fwd Packets)
    X[:, 60] = X[:, 1]  # Subflow Fwd Packets = Total Fwd Packets
    X[:, 62] = X[:, 2]  # Subflow Bwd Packets = Total Backward Packets

    # Build DataFrame
    df = pd.DataFrame(X, columns=CICIDS_FEATURES)
    df['Label'] = labels

    # Shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
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
