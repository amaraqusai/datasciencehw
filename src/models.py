import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)


def train_all_models(X_train, y_train):
    """Train three models and return them with training times."""
    models = {}
    times = {}

    # Model 1: Logistic Regression (simple linear baseline)
    t0 = time.time()
    lr = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
    lr.fit(X_train, y_train)
    times['Logistic Regression'] = time.time() - t0
    models['Logistic Regression'] = lr

    # Model 2: Random Forest (ensemble baseline)
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    times['Random Forest'] = time.time() - t0
    models['Random Forest'] = rf

    # Model 3: MLP (deep learning proxy for DBN)
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
    times['MLP (Deep Baseline)'] = time.time() - t0
    models['MLP (Deep Baseline)'] = mlp

    return models, times


def evaluate_all_models(models, X_test, y_test):
    """Evaluate all models with comprehensive cybersecurity metrics."""
    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        results.append({
            'Model': name,
            'Accuracy': round(accuracy_score(y_test, y_pred), 6),
            'Precision': round(precision_score(y_test, y_pred, zero_division=0), 6),
            'Recall': round(recall_score(y_test, y_pred, zero_division=0), 6),
            'F1-Score': round(f1_score(y_test, y_pred, zero_division=0), 6),
            'F2-Score': round(fbeta_score(y_test, y_pred, beta=2, zero_division=0), 6),
            'MCC': round(matthews_corrcoef(y_test, y_pred), 6),
            'ROC-AUC': round(roc_auc_score(y_test, y_prob), 6),
            'PR-AUC': round(average_precision_score(y_test, y_prob), 6),
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'FAR': round(fp / (fp + tn), 6) if (fp + tn) > 0 else 0,
            'FNR': round(fn / (fn + tp), 6) if (fn + tp) > 0 else 0,
        })

    return pd.DataFrame(results)


def get_error_analysis(model, model_name, X_test, y_test):
    """Extract and analyze False Positive and False Negative cases."""
    y_pred = model.predict(X_test)

    test_df = X_test.copy()
    test_df['True_Label'] = y_test.values
    test_df['Predicted_Label'] = y_pred

    fp_mask = (test_df['True_Label'] == 0) & (test_df['Predicted_Label'] == 1)
    fn_mask = (test_df['True_Label'] == 1) & (test_df['Predicted_Label'] == 0)

    fp_cases = test_df[fp_mask].copy()
    fn_cases = test_df[fn_mask].copy()
    fp_cases['Error_Type'] = 'False Positive'
    fn_cases['Error_Type'] = 'False Negative'
    fp_cases['Model'] = model_name
    fn_cases['Model'] = model_name

    errors = pd.concat([fp_cases, fn_cases])
    return errors, len(fp_cases), len(fn_cases)
