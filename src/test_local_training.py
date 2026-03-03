import pandas as pd
import numpy as np
import tensorflow as tf

from .model_tf import build_model
from .train_tf import train_local, evaluate, predict_risk, set_weights

# Set seeds for reproducibility (same results every run)
tf.random.set_seed(42)
np.random.seed(42)

TARGET_COL = "Outcome"

# All hospital datasets
HOSPITAL_PATHS = [
    "dataset/hospital_1.csv",
    "dataset/hospital_2.csv",
    "dataset/hospital_3.csv",
]

# Common test set
TEST_PATH = "dataset/test_set.csv"


def load_xy(csv_path: str):
    """
    Load CSV file and split into features (X) and target (y).
    """
    df = pd.read_csv(csv_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {csv_path}")

    X = df.drop(TARGET_COL, axis=1).values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int32)
    cols = df.drop(TARGET_COL, axis=1).columns.tolist()

    return X, y, cols


def main():
    # Load test set once
    X_test, y_test, test_cols = load_xy(TEST_PATH)

    input_dim = X_test.shape[1]
    print(f"Input features = {input_dim}")
    print("Features:", test_cols)

    # Build base model and capture initial weights
    base_model = build_model(input_dim)
    initial_weights = base_model.get_weights()

    print("\n=== FL-Style Local Training (Same Initial Weights) ===")

    # Loop through hospitals
    for hospital_path in HOSPITAL_PATHS:

        hospital_name = hospital_path.split("/")[-1].replace(".csv", "").replace("_", " ").title()

        # Load hospital training data
        X_train, y_train, train_cols = load_xy(hospital_path)

        # Ensure columns match test set
        if train_cols != test_cols:
            print(f"\n❌ Column mismatch for {hospital_name}")
            continue

        # Build fresh model and reset to same initial weights
        model = build_model(input_dim)
        set_weights(model, initial_weights)

        print(f"\n--- Training local model on {hospital_name} ---")
        train_local(model, X_train, y_train, epochs=15, batch_size=32)
        print("Training completed!")

        # Evaluate on common test set
        metrics = evaluate(model, X_test, y_test)

        print(f"Evaluation on test_set.csv for {hospital_name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # Predict risk for first test sample
        risk_pct, level = predict_risk(model, X_test[0])
        print("Sample Risk Prediction:")
        print(f"  Risk: {risk_pct:.2f}% | Category: {level}")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()