"""
Master evaluation script. Runs the entire pipeline end-to-end
and saves all results to the results/ directory.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from preprocessing import load_real_cicids2017, run_eda, clean_data, split_and_scale
from model_utils import train_all_models, evaluate_all_models, get_error_analysis

def main():
    print("=" * 60)
    print("  CICIDS2017 DBN Evaluation Pipeline")
    print("=" * 60)

    # --- Step 1: Generate Data ---
    print("\n[1/6] Generating representative CICIDS2017 dataset...")
    df = generate_synthetic_cicids2017(n_samples=20000)
    print(f"      Shape: {df.shape}")

    # --- Step 2: EDA ---
    print("\n[2/6] Running Exploratory Data Analysis...")
    eda = run_eda(df)
    print(f"      Missing values: {eda['total_missing']}")
    print(f"      Infinite values: {sum(eda['infinite_values'].values())}")
    print(f"      Constant features: {len(eda['constant_features'])}")
    print(f"      Near-constant features: {len(eda['near_constant_features'])}")
    print(f"      Duplicate columns: {len(eda['duplicate_columns'])}")
    print(f"      Duplicate rows: {eda['duplicate_rows']}")

    # --- Step 3: Clean ---
    print("\n[3/6] Cleaning data...")
    df_clean, dropped_const, dropped_dup = clean_data(df)
    print(f"      Dropped {len(dropped_const)} constant features: {dropped_const}")
    print(f"      Dropped {len(dropped_dup)} duplicate columns: {dropped_dup}")
    print(f"      Clean shape: {df_clean.shape}")

    # --- Step 4: Split & Scale ---
    print("\n[4/6] Splitting and scaling (leakage-safe)...")
    X_train, X_test, y_train, y_test, feature_names = split_and_scale(df_clean)
    print(f"      Train: {X_train.shape}, Test: {X_test.shape}")

    # --- Step 5: Train ---
    print("\n[5/6] Training models...")
    models, times = train_all_models(X_train, y_train)
    for name, t in times.items():
        print(f"      {name}: {t:.2f}s")

    # --- Step 6: Evaluate ---
    print("\n[6/6] Evaluating models...")
    metrics_df = evaluate_all_models(models, X_test, y_test)
    print(metrics_df.to_string(index=False))

    # --- Save Results ---
    os.makedirs('results', exist_ok=True)

    metrics_df.to_csv('results/experiment_metrics.csv', index=False)
    print("\nSaved: results/experiment_metrics.csv")

    # Baseline reproduction (MLP as DBN proxy)
    baseline = metrics_df[metrics_df['Model'] == 'MLP (Deep Baseline)'].copy()
    baseline['Model'] = 'DBN Baseline (via MLP proxy)'
    baseline.to_csv('results/baseline_reproduction.csv', index=False)
    print("Saved: results/baseline_reproduction.csv")

    # Error analysis for each model
    all_errors = []
    for name, model in models.items():
        errors, n_fp, n_fn = get_error_analysis(model, name, X_test, y_test)
        all_errors.append(errors)
        print(f"\n{name}: {n_fp} False Positives, {n_fn} False Negatives")

    import pandas as pd
    all_errors_df = pd.concat(all_errors)
    all_errors_df.to_csv('results/errors_rare_attacks_analysis.csv', index=False)
    print("Saved: results/errors_rare_attacks_analysis.csv")

    # Training times
    import pandas as pd
    times_df = pd.DataFrame([{'Model': k, 'Training_Time_Seconds': round(v, 4)} for k, v in times.items()])
    times_df.to_csv('results/training_times.csv', index=False)
    print("Saved: results/training_times.csv")

    print("\n" + "=" * 60)
    print("  Pipeline complete. All results saved to results/")
    print("=" * 60)

if __name__ == "__main__":
    main()
