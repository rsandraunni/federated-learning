import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from config import (
    TARGET_COLUMN,
    DATASET_DIR,
    TEST_FILE,
    HOSPITAL_FILE_PATTERN,
    NUM_HOSPITALS,
    BATCH_SIZE,
)

from .model_tf import build_model
from .train_tf import evaluate

tf.random.set_seed(42)
np.random.seed(42)


def load_xy(csv_path: str):
    df = pd.read_csv(csv_path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in: {csv_path}")

    X = df.drop(TARGET_COLUMN, axis=1).values.astype(np.float32)
    y = df[TARGET_COLUMN].values.astype(np.int32)
    cols = df.drop(TARGET_COLUMN, axis=1).columns.tolist()
    return X, y, cols


def main():
    # 1) Load common test set
    test_csv = os.path.join(DATASET_DIR, TEST_FILE)
    X_test, y_test, test_cols = load_xy(test_csv)

    # 2) Load + combine hospital datasets
    X_parts, y_parts = [], []
    ref_cols = None

    for hid in range(1, NUM_HOSPITALS + 1):
        hospital_csv = os.path.join(DATASET_DIR, HOSPITAL_FILE_PATTERN.format(hid))
        X_train, y_train, train_cols = load_xy(hospital_csv)

        if ref_cols is None:
            ref_cols = train_cols

        if train_cols != ref_cols or train_cols != test_cols:
            raise ValueError(
                f"Column mismatch for hospital {hid}!\n"
                f"Hospital cols: {train_cols}\n"
                f"Reference cols: {ref_cols}\n"
                f"Test cols:      {test_cols}"
            )

        X_parts.append(X_train)
        y_parts.append(y_train)

    X_train_all = np.vstack(X_parts)
    y_train_all = np.concatenate(y_parts)

    print(f"[Centralized] Train samples: {len(X_train_all)}")
    print(f"[Centralized] Test samples : {len(X_test)}")
    print(f"[Centralized] Features     : {X_train_all.shape[1]}")

    # Label counts
    u, c = np.unique(y_train_all, return_counts=True)
    print("Train label counts:", dict(zip(u, c)))
    u, c = np.unique(y_test, return_counts=True)
    print("Test label counts:", dict(zip(u, c)))

    # 3) Build model (same architecture as FL)
    model = build_model(input_dim=X_train_all.shape[1])

    # 4) Train centrally (NO class weights, NO early stopping)
    CENTRAL_EPOCHS = 45  

    model.fit(
        X_train_all,
        y_train_all,
        epochs=CENTRAL_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    # 5) Evaluate on test set
    metrics = evaluate(model, X_test, y_test)
    print("\n========== FINAL CENTRALIZED METRICS ==========")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print("================================================\n")

    # Save model + metrics
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    model_path = os.path.join("models", "centralized_model.keras")
    metrics_path = os.path.join("logs", "centralized_metrics.json")

    model.save(model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[Centralized] Saved model  : {model_path}")
    print(f"[Centralized] Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()