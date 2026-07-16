import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_synthetic_cicids2017(n_samples=10000):
    np.random.seed(42)
    features = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 
                'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 
                'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 
                'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
                'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
                'Constant_Feature', 'Duplicate_Feature']
    
    n_normal = int(n_samples * 0.8)
    normal_data = np.random.normal(loc=[100, 5, 5, 500, 1000, 150, 40, 50, 300, 40, 80, 1000, 100, 50, 10, 1, 1], 
                                   scale=[20, 2, 2, 100, 200, 30, 10, 15, 50, 10, 20, 200, 20, 10, 2, 0, 0], 
                                   size=(n_normal, len(features)))
    
    n_attack = n_samples - n_normal
    attack_data = np.random.normal(loc=[1000, 20, 20, 2000, 5000, 500, 0, 100, 1500, 0, 200, 5000, 500, 5, 1, 1, 1], 
                                   scale=[200, 5, 5, 500, 1000, 100, 0, 30, 300, 0, 50, 1000, 100, 2, 0.5, 0, 0], 
                                   size=(n_attack, len(features)))
    
    X = np.vstack([normal_data, attack_data])
    y = np.array([0]*n_normal + [1]*n_attack)
    
    idx_missing = np.random.choice(n_samples, size=int(0.01*n_samples), replace=False)
    X[idx_missing, 0] = np.nan
    idx_inf = np.random.choice(n_samples, size=int(0.01*n_samples), replace=False)
    X[idx_inf, 11] = np.inf
    
    df = pd.DataFrame(X, columns=features)
    df['Duplicate_Feature'] = df['Flow Duration'] 
    df['Label'] = y
    
    return df

def clean_and_split_data(df):
    # 1. Handle Missing and Infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)

    # 2. Remove Constant and Duplicate Features
    constant_features = [col for col in df.columns if df[col].nunique() == 1]
    df.drop(columns=constant_features, inplace=True)

    duplicate_features = []
    for i in range(len(df.columns)):
        for j in range(i + 1, len(df.columns)):
            if df.iloc[:, i].equals(df.iloc[:, j]):
                duplicate_features.append(df.columns[j])
    df.drop(columns=duplicate_features, inplace=True)

    X = df.drop(columns=['Label'])
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Store features for reference later
    feature_names = X.columns.tolist()

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, X_test
