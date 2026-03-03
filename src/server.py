import os
import flwr as fl
import numpy as np
import pandas as pd

from config import (
    NUM_ROUNDS, LOCAL_EPOCHS, BATCH_SIZE, SERVER_ADDRESS, NUM_HOSPITALS,
    TARGET_COLUMN, DATASET_DIR, TEST_FILE
)

from .model_tf import build_model
from .train_tf import set_weights, evaluate


def fit_config(server_round: int):
    return {
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
    }


def load_testset():
    test_csv = os.path.join(DATASET_DIR, TEST_FILE)
    df = pd.read_csv(test_csv)
    X = df.drop(TARGET_COLUMN, axis=1).values.astype(np.float32)
    y = df[TARGET_COLUMN].values.astype(np.int32)
    return X, y


class FinalMetricsFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            return None

        params, metrics = aggregated

        # Convert Flower Parameters → numpy arrays
        ndarrays = fl.common.parameters_to_ndarrays(params)

        # If last round, evaluate final model
        if server_round == NUM_ROUNDS:
            print("\n========== TRAINING COMPLETE ==========")

            X_test, y_test = load_testset()
            input_dim = X_test.shape[1]

            model = build_model(input_dim=input_dim)
            set_weights(model, ndarrays)

            final_metrics = evaluate(model, X_test, y_test)

            print("\n========== FINAL FEDERATED METRICS ==========")
            print(f"Accuracy : {final_metrics['accuracy']:.4f}")
            print(f"Precision: {final_metrics['precision']:.4f}")
            print(f"Recall   : {final_metrics['recall']:.4f}")
            print(f"F1 Score : {final_metrics['f1']:.4f}")
            print("=============================================\n")

            os.makedirs("models", exist_ok=True)
            model.save("models/federated_model.keras")
            print("Saved final federated model → models/federated_model.keras")

        return params, metrics


def main():
    strategy = FinalMetricsFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_HOSPITALS,
        min_evaluate_clients=NUM_HOSPITALS,
        min_available_clients=NUM_HOSPITALS,
        on_fit_config_fn=fit_config,
    )

    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()