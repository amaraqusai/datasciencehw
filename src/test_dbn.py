import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from preprocessing import load_real_cicids2017
from model_utils import PyTorchDBNWrapper
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, f1_score

df = load_real_cicids2017(sample_frac=0.01)
le = LabelEncoder()
df['Label_Enc'] = le.fit_transform(df['Label'])
y = df['Label_Enc'].values
X = df.select_dtypes(include=[np.number]).drop(columns=['Label_Enc']).values

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import FunctionTransformer, StandardScaler

def replace_infs(X):
    df = pd.DataFrame(X)
    return df.replace([np.inf, -np.inf], np.nan).values

pipe = Pipeline([
    ('inf_replacer', FunctionTransformer(replace_infs)),
    ('var_thresh', VarianceThreshold(threshold=0.0)),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', PyTorchDBNWrapper(n_hidden=(64, 32), num_epochs=(1, 1), fine_tune_epochs=1, num_classes=len(np.unique(y))))
])

cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
try:
    res = cross_validate(pipe, X, y, cv=cv, scoring={'f1': make_scorer(f1_score, average='macro')}, error_score='raise', return_estimator=True)
    print("SUCCESS")
except Exception as e:
    print("ERROR:", e)
