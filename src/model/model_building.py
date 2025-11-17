import numpy as np
import pandas as pd
import os
import yaml
import logging
import pickle
from sklearn.ensemble import GradientBoostingClassifier

# --------------------------
# Logging Configuration
# --------------------------
logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_building.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# --------------------------
# Load Model Parameters
# --------------------------
def load_params(params_path: str):
    try:
        logger.info("Loading model parameters...")

        if not os.path.exists(params_path):
            logger.error("params.yaml not found.")
            raise FileNotFoundError("params.yaml not found.")

        params = yaml.safe_load(open(params_path, 'r'))
        if params is None:
            logger.error("params.yaml is empty or invalid.")
            raise ValueError("params.yaml is empty or incorrectly formatted.")

        model_params = params["model_building"]
        logger.debug(f"Model parameters loaded: {model_params}")

        return model_params

    except KeyError as e:
        logger.error(f"Missing key in params.yaml: {e}")
        raise KeyError(f"Missing key in params.yaml: {e}")

    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        raise RuntimeError(f"Error loading parameters: {e}")

# --------------------------
# Load Training Features
# --------------------------
def load_train_features():
    try:
        logger.info("Loading training feature data...")

        train_path = "./data/processed/train_bow.csv"
        if not os.path.exists(train_path):
            logger.error("train_bow.csv not found in data/features/")
            raise FileNotFoundError("train_bow.csv not found in data/features/")

        train_df = pd.read_csv(train_path)
        logger.debug(f"Train features shape: {train_df.shape}")

        return train_df

    except Exception as e:
        logger.error(f"Failed to load training feature data: {e}")
        raise RuntimeError(f"Error loading training features: {e}")

# --------------------------
# Train the Model
# --------------------------
def build_model(train_df: pd.DataFrame, params: dict):
    try:
        logger.info("Training GradientBoostingClassifier model...")

        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values

        clf = GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"]
        )

        clf.fit(X_train, y_train)

        logger.info("Model training completed.")
        return clf

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise RuntimeError(f"Model training error: {e}")

# --------------------------
# Save the trained model
# --------------------------
def save_model(model):
    try:
        logger.info("Saving trained model...")

        os.makedirs("./models", exist_ok=True)
        pickle.dump(model, open("./models/model.pkl", "wb"))

        logger.info("Model saved successfully.")

    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise RuntimeError(f"Error saving model: {e}")

# --------------------------
# Main Execution
# --------------------------
def main():
    try:
        logger.info("Starting model building pipeline...")

        model_params = load_params("params.yaml")
        train_df = load_train_features()
        model = build_model(train_df, model_params)
        save_model(model)

        logger.info("Model building pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"❌ Pipeline failed: {e}")

if __name__ == "__main__":
    main()
