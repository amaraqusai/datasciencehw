from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             matthews_corrcoef, roc_auc_score, average_precision_score)
import pandas as pd

def train_models(X_train, y_train):
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=20, random_state=42)
    mlp.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    return {'MLP': mlp, 'Random Forest': rf}

def evaluate_models(models, X_test, y_test):
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'MCC': mcc,
            'ROC-AUC': roc_auc,
            'PR-AUC': pr_auc
        })
        
    return pd.DataFrame(results)

def generate_error_analysis(model, X_test, y_test, X_test_unscaled):
    y_pred = model.predict(X_test)
    
    test_df = X_test_unscaled.copy()
    test_df['True_Label'] = y_test.values
    test_df['Pred_Label'] = y_pred
    
    fp_cases = test_df[(test_df['True_Label'] == 0) & (test_df['Pred_Label'] == 1)]
    fn_cases = test_df[(test_df['True_Label'] == 1) & (test_df['Pred_Label'] == 0)]
    
    fp_cases['Error_Type'] = 'False Positive'
    fn_cases['Error_Type'] = 'False Negative'
    
    return pd.concat([fp_cases, fn_cases])
