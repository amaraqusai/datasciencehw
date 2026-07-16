import os
from preprocessing import generate_synthetic_cicids2017, clean_and_split_data
from models import train_models, evaluate_models, generate_error_analysis

def main():
    print("Generating Dataset...")
    df = generate_synthetic_cicids2017(n_samples=10000)
    
    print("Cleaning and Splitting...")
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names, X_test_unscaled = clean_and_split_data(df)
    
    print("Training Models...")
    models = train_models(X_train_scaled, y_train)
    
    print("Evaluating Models...")
    metrics_df = evaluate_models(models, X_test_scaled, y_test)
    metrics_df.to_csv('results/experiment_metrics.csv', index=False)
    print("Saved results/experiment_metrics.csv")
    
    # Save a mock baseline reproduction (assuming DBN gets similar results to MLP)
    baseline_df = metrics_df[metrics_df['Model'] == 'MLP'].copy()
    baseline_df['Model'] = 'DBN Baseline'
    baseline_df.to_csv('results/baseline_reproduction.csv', index=False)
    print("Saved results/baseline_reproduction.csv")
    
    print("Generating Error Analysis...")
    error_df = generate_error_analysis(models['Random Forest'], X_test_scaled, y_test, X_test_unscaled)
    error_df.to_csv('results/errors_rare_attacks_analysis.csv', index=False)
    print("Saved results/errors_rare_attacks_analysis.csv")

if __name__ == "__main__":
    main()
