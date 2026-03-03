import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import flwr as fl

from config import (
    TARGET_COLUMN, DATASET_DIR, TEST_FILE, HOSPITAL_FILE_PATTERN,
    SERVER_ADDRESS
)
from .model_tf import build_model
from .train_tf import train_local, evaluate, get_weights, set_weights

tf.random.set_seed(42)
np.random.seed(42)

def load_xy(csv_path: str):
    df = pd.read_csv(csv_path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in {csv_path}")

    X = df.drop(TARGET_COLUMN, axis=1).values.astype(np.float32)
    y = df[TARGET_COLUMN].values.astype(np.int32)
    cols = df.drop(TARGET_COLUMN, axis=1).columns.tolist()
    return X, y, cols

class HospitalClient(fl.client.NumPyClient):
    def __init__(self, hospital_csv: str, test_csv: str):
        # Load common test set
        self.X_test, self.y_test, self.test_cols = load_xy(test_csv)

        # Load local hospital dataset
        self.X_train, self.y_train, self.train_cols = load_xy(hospital_csv)

        # Ensure same feature columns/order
        if self.train_cols != self.test_cols:
            raise ValueError(
                f"Column mismatch!\nTrain: {self.train_cols}\nTest:  {self.test_cols}"
            )

        self.model = build_model(input_dim=self.X_train.shape[1])

    def get_parameters(self, config):
        return get_weights(self.model)

    def set_parameters(self, parameters):
        set_weights(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        local_epochs = int(config.get("local_epochs", 2))
        batch_size = int(config.get("batch_size", 32))

        train_local(self.model, self.X_train, self.y_train, epochs=local_epochs, batch_size=batch_size)

        # Optional metrics after local training (helps server logs)
        m = evaluate(self.model, self.X_test, self.y_test)
        return self.get_parameters(config), len(self.X_train), {
            "loss": m["loss"],
            "accuracy": m["accuracy"],
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        m = evaluate(self.model, self.X_test, self.y_test)

        print(
            f"| Accuracy: {m['accuracy']:.4f} "
            f"| Loss: {m['loss']:.4f} "
            f"| Recall: {m['recall']:.4f} "
            f"| F1: {m['f1']:.4f}"
        )

        return m["loss"], len(self.X_test), {
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hid", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--server", type=str, default=SERVER_ADDRESS)
    args = parser.parse_args()

    hospital_csv = os.path.join(DATASET_DIR, HOSPITAL_FILE_PATTERN.format(args.hid))
    test_csv = os.path.join(DATASET_DIR, TEST_FILE)

    client = HospitalClient(hospital_csv, test_csv)
    fl.client.start_numpy_client(server_address=args.server, client=client)

if __name__ == "__main__":
    main()