import time
import pandas as pd
import numpy as np
import sys
import os

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, balanced_accuracy_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.DBN import DBN
from utils.train import train

class MockDataset:
    def __init__(self, X):
        self.features = X

class MockLoader:
    def __init__(self, loader, X):
        self.loader = loader
        self.dataset = MockDataset(X)
    def __iter__(self):
        return iter(self.loader)
    def __len__(self):
        return len(self.loader)

class PyTorchDBNWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_hidden=(128, 64, 32), k=(1, 1, 1), batch_size=(64, 64, 64), learning_rate=(0.1, 0.1, 0.1), num_epochs=(3, 3, 3), fine_tune_epochs=5, fine_tune_lr=0.001, num_classes=15):
        self.n_hidden = n_hidden
        self.k = k
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.fine_tune_lr = fine_tune_lr
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):
        X_df = X
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X)
            
        n_visible = X.shape[1]
        n_classes = self.num_classes
        if y.max() >= n_classes:
            n_classes = int(y.max()) + 1
            
        self.classes_ = np.unique(y)
        self.fitted_ = True
        
        self.model = DBN(
            n_visible=n_visible,
            n_hidden=self.n_hidden,
            n_classes=n_classes,
            learning_rate=self.learning_rate,
            momentum=tuple([0.9]*len(self.n_hidden)),
            decay=tuple([0]*len(self.n_hidden)),
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            k=self.k,
            device=self.device
        )
        self.model.to(self.device)
        
        tensor_x = torch.tensor(X_df.values, dtype=torch.float32)
        tensor_y = torch.tensor(y.values if isinstance(y, pd.Series) else y, dtype=torch.long)
        
        dataset = TensorDataset(tensor_x, tensor_y)
        train_loader = DataLoader(dataset, batch_size=self.batch_size[0], shuffle=True)
        mock_loader = MockLoader(train_loader, X_df)
        
        print(f"Pretraining DBN with {len(self.n_hidden)} layers...")
        self.model.fit(mock_loader)
        
        print("Fine-tuning DBN...")
        criterion = nn.CrossEntropyLoss()
        optimizer = [optim.Adam(self.model.parameters(), lr=self.fine_tune_lr)]
        
        if X_val is not None and y_val is not None:
            tensor_x_val = torch.tensor(X_val.values if isinstance(X_val, pd.DataFrame) else X_val, dtype=torch.float32)
            tensor_y_val = torch.tensor(y_val.values if isinstance(y_val, pd.Series) else y_val, dtype=torch.long)
            val_dataset = TensorDataset(tensor_x_val, tensor_y_val)
            valid_loader = DataLoader(val_dataset, batch_size=self.batch_size[0], shuffle=False)
        else:
            valid_loader = train_loader

        train(
            model=self.model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            valid_loader=valid_loader, 
            num_epochs=self.fine_tune_epochs,
            device=self.device
        )
        return self

    def predict(self, X):
        self.model.eval()
        tensor_x = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor_x)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()

    def predict_proba(self, X):
        import torch.nn.functional as F
        self.model.eval()
        tensor_x = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor_x)
            probs = F.softmax(outputs, dim=1)
        return probs.cpu().numpy()


class MappedXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.xgb = XGBClassifier(**kwargs)
        self.classes_ = None

    def fit(self, X, y, eval_set=None, **kwargs):
        self.classes_ = np.unique(y)
        y_mapped = np.searchsorted(self.classes_, y)
        
        if eval_set is not None:
            X_val, y_val = eval_set[0]
            # Map val labels; mask out unseen classes
            y_val_mapped = np.searchsorted(self.classes_, y_val)
            # For unseen classes, searchsorted might return an index out of bounds or an incorrect index.
            # We can just filter out rows in validation that have unseen classes for early stopping.
            valid_mask = np.isin(y_val, self.classes_)
            if not np.all(valid_mask):
                X_val = X_val[valid_mask]
                y_val_mapped = y_val_mapped[valid_mask]
            
            if len(y_val_mapped) > 0:
                self.xgb.fit(X, y_mapped, eval_set=[(X_val, y_val_mapped)], **kwargs)
            else:
                self.xgb.fit(X, y_mapped, **kwargs)
        else:
            self.xgb.fit(X, y_mapped, **kwargs)
        return self

    def predict(self, X):
        preds_mapped = self.xgb.predict(X)
        return self.classes_[preds_mapped]
        
    def predict_proba(self, X):
        # Return probabilities aligned to the global 15 classes? 
        # Actually predict_proba isn't strictly required for our F1 metric, but we can pad it.
        probs_mapped = self.xgb.predict_proba(X)
        probs_full = np.zeros((len(X), 15)) # Assuming max 15 classes
        for i, cls in enumerate(self.classes_):
            if cls < 15:
                probs_full[:, cls] = probs_mapped[:, i]
        return probs_full

def train_all_models(X_train, y_train, X_val=None, y_val=None):
    models = {}
    times = {}

    t0 = time.time()
    lr = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
    lr.fit(X_train, y_train)
    times['Logistic Regression'] = time.time() - t0
    models['Logistic Regression'] = lr

    t0 = time.time()
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    times['Decision Tree'] = time.time() - t0
    models['Decision Tree'] = dt

    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    times['Random Forest'] = time.time() - t0
    models['Random Forest'] = rf

    t0 = time.time()
    xgb = MappedXGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1, early_stopping_rounds=10)
    if X_val is not None and y_val is not None:
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        xgb.fit(X_train, y_train)
    times['XGBoost'] = time.time() - t0
    models['XGBoost'] = xgb

    t0 = time.time()
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=50,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp.fit(X_train, y_train)
    times['Standard MLP'] = time.time() - t0
    models['Standard MLP'] = mlp

    # Authentic DBN Reproduction (using User-Approved 3-layer optimization)
    t0 = time.time()
    dbn = PyTorchDBNWrapper(
        n_hidden=(128, 64, 32),
        k=(1, 1, 1),
        batch_size=(128, 128, 128),
        learning_rate=(0.1, 0.1, 0.1),
        num_epochs=(3, 3, 3), # 3 epochs pretraining per layer
        fine_tune_epochs=5,   # 5 epochs supervised finetuning
        fine_tune_lr=0.001
    )
    dbn.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    times['Authentic DBN'] = time.time() - t0
    models['Authentic DBN'] = dbn

    return models, times


def evaluate_all_models(models, X_test, y_test, task_type='multiclass'):
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        if task_type == 'binary':
            y_prob = model.predict_proba(X_test)[:, 1]
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            results.append({
                'Model': name,
                'Accuracy': round(accuracy_score(y_test, y_pred), 6),
                'Balanced Acc': round(balanced_accuracy_score(y_test, y_pred), 6),
                'Precision': round(precision_score(y_test, y_pred, zero_division=0), 6),
                'Recall': round(recall_score(y_test, y_pred, zero_division=0), 6),
                'F1-Score': round(f1_score(y_test, y_pred, zero_division=0), 6),
                'F2-Score': round(fbeta_score(y_test, y_pred, beta=2, zero_division=0), 6),
                'MCC': round(matthews_corrcoef(y_test, y_pred), 6),
                'ROC-AUC': round(roc_auc_score(y_test, y_prob), 6),
                'PR-AUC': round(average_precision_score(y_test, y_prob), 6),
                'FAR': round(fp / (fp + tn), 6) if (fp + tn) > 0 else 0,
                'FNR': round(fn / (fn + tp), 6) if (fn + tp) > 0 else 0,
            })
        else:
            y_prob = model.predict_proba(X_test)
            try:
                roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
            except ValueError:
                roc_auc = np.nan # If a class is missing in y_test
                
            results.append({
                'Model': name,
                'Accuracy': round(accuracy_score(y_test, y_pred), 6),
                'Balanced Acc': round(balanced_accuracy_score(y_test, y_pred), 6),
                'Macro Precision': round(precision_score(y_test, y_pred, average='macro', zero_division=0), 6),
                'Macro Recall': round(recall_score(y_test, y_pred, average='macro', zero_division=0), 6),
                'Macro F1': round(f1_score(y_test, y_pred, average='macro', zero_division=0), 6),
                'Weighted F1': round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 6),
                'MCC': round(matthews_corrcoef(y_test, y_pred), 6),
                'ROC-AUC (OVR)': round(roc_auc, 6),
            })
    return pd.DataFrame(results)
def get_error_analysis(model, model_name, X_test, y_test):
    y_pred = model.predict(X_test)
    test_df = X_test.copy()
    test_df['True_Label'] = y_test.values if hasattr(y_test, 'values') else y_test
    test_df['Predicted_Label'] = y_pred

    fp_mask = (test_df['True_Label'] == 0) & (test_df['Predicted_Label'] == 1)
    fn_mask = (test_df['True_Label'] == 1) & (test_df['Predicted_Label'] == 0)

    fp_cases = test_df[fp_mask].copy()
    fn_cases = test_df[fn_mask].copy()
    fp_cases['Error_Type'] = 'False Positive'
    fn_cases['Error_Type'] = 'False Negative'
    fp_cases['Model'] = model_name
    fn_cases['Model'] = model_name

    return pd.concat([fp_cases, fn_cases]), len(fp_cases), len(fn_cases)

def get_multiclass_error_analysis(model, model_name, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    
    true_labels = label_encoder.inverse_transform(y_test.values if hasattr(y_test, 'values') else y_test)
    pred_labels = label_encoder.inverse_transform(y_pred)
    
    test_df = X_test.copy()
    test_df['True_Label'] = true_labels
    test_df['Predicted_Label'] = pred_labels
    
    # A False Negative for a specific attack type occurs when the true label is that attack, 
    # but the predicted label is anything else.
    # To keep it simple: we just find all misclassifications.
    errors = test_df[test_df['True_Label'] != test_df['Predicted_Label']].copy()
    errors['Model'] = model_name
    
    # Calculate false negatives per true attack type (predicting something else)
    fn_counts = {}
    for cls in label_encoder.classes_:
        if cls == 'BENIGN': continue
        fn = ((test_df['True_Label'] == cls) & (test_df['Predicted_Label'] != cls)).sum()
        total = (test_df['True_Label'] == cls).sum()
        fn_counts[cls] = {'FN': int(fn), 'Total': int(total)}
        
    return errors, pd.DataFrame.from_dict(fn_counts, orient='index')
