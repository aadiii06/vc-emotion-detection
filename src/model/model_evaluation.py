import numpy as np
import pandas as pd
import pickle
import json
import os
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# --------------------------
# Logging Configuration
# --------------------------
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_evaluation.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# --------------------------
# Load Trained Model
# --------------------------
def load_model(model_path: str):
    try:
        logger.info("Loading trained model...")

        if not os.path.exists(model_path):
            logger.error("Model file not found.")
            raise FileNotFoundError("Model file not found at provided path.")

        model = pickle.load(open(model_path, "rb"))
        logger.info("Model loaded successfully.")
        return model

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

# --------------------------
# Load Test Data
# --------------------------
def load_test_data(test_path: str):
    try:
        logger.info("Loading test feature data...")

        if not os.path.exists(test_path):
            logger.error("test_bow.csv not found.")
            raise FileNotFoundError("test_bow.csv not found in data/features/")

        df = pd.read_csv(test_path)
        logger.debug(f"Test data shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        raise RuntimeError(f"Error loading test data: {e}")

# --------------------------
# Evaluate the Model
# --------------------------
def evaluate_model(model, test_df: pd.DataFrame):
    try:
        logger.info("Evaluating model performance...")

        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc": auc
        }

        logger.info("Model evaluation completed.")
        logger.debug(f"Metrics: {metrics}")

        return metrics

    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise RuntimeError(f"Evaluation error: {e}")

# --------------------------
# Save Metrics
# --------------------------
def save_metrics(metrics: dict):
    try:
        logger.info("Saving evaluation metrics...")

        os.makedirs("./reports", exist_ok=True)
        with open("./reports/metric.json", "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info("Metrics saved successfully.")

    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")
        raise RuntimeError(f"Error saving metrics: {e}")

# --------------------------
# Main Execution
# --------------------------
def main():
    try:
        logger.info("Starting model evaluation pipeline...")

        model = load_model("./models/model.pkl")
        test_df = load_test_data("./data/processed/test_bow.csv")
        metrics = evaluate_model(model, test_df)
        save_metrics(metrics)

        logger.info("Model evaluation pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"❌ Pipeline failed: {e}")

if __name__ == "__main__":
    main()
