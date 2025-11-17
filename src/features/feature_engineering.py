import numpy as np
import pandas as pd
import os
import yaml
import logging
from sklearn.feature_extraction.text import CountVectorizer

# --------------------------
# Logging Configuration
# --------------------------
logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('feature_engineering.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# --------------------------
# Load params.yaml
# --------------------------
def load_params(params_path: str) -> int:
    try:
        logger.info("Loading feature engineering parameters...")

        if not os.path.exists(params_path):
            logger.error("params.yaml not found.")
            raise FileNotFoundError("params.yaml not found.")

        params = yaml.safe_load(open(params_path, "r"))
        if params is None:
            logger.error("params.yaml is empty or invalid.")
            raise ValueError("params.yaml is empty or incorrectly formatted.")

        max_features = params["feature_engineering"]["max_features"]
        logger.debug(f"max_features = {max_features}")

        return max_features

    except KeyError as e:
        logger.error(f"Missing key in params.yaml: {e}")
        raise KeyError(f"Missing key in params.yaml: {e}")

    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        raise RuntimeError(f"Error loading parameters: {e}")

# --------------------------
# Load processed train/test data
# --------------------------
def load_processed_data():
    try:
        logger.info("Loading processed train and test data...")

        train_path = "./data/interim/train_processed.csv"
        test_path = "./data/interim/test_processed.csv"

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            logger.error("Processed data files missing in data/processed/")
            raise FileNotFoundError("Processed data files not found in data/processed/")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        train_df = train_df.dropna(subset=["content"])
        test_df = test_df.dropna(subset=["content"])

        logger.info("Processed data loaded successfully.")
        logger.debug(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

        return train_df, test_df

    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        raise RuntimeError(f"Failed to load processed data: {e}")

# --------------------------
# Apply Bag-of-Words
# --------------------------
def build_bow_features(train_df: pd.DataFrame, test_df: pd.DataFrame, max_features: int):
    try:
        logger.info("Building Bag-of-Words features...")

        X_train = train_df["content"].values
        X_test = test_df["content"].values

        y_train = train_df["sentiment"].values
        y_test = test_df["sentiment"].values

        vectorizer = CountVectorizer(max_features=max_features)

        logger.debug("Fitting CountVectorizer on training data...")
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_features = pd.DataFrame(X_train_bow.toarray())
        test_features = pd.DataFrame(X_test_bow.toarray())

        train_features["sentiment"] = y_train
        test_features["sentiment"] = y_test

        logger.info("Bag-of-Words feature engineering completed.")
        logger.debug(f"Train features shape: {train_features.shape}")
        logger.debug(f"Test features shape: {test_features.shape}")

        return train_features, test_features

    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        raise RuntimeError(f"Feature engineering failed: {e}")

# --------------------------
# Save features
# --------------------------
def save_features(train_features: pd.DataFrame, test_features: pd.DataFrame):
    try:
        logger.info("Saving BOW features...")

        data_path = os.path.join("data", "processed")
        os.makedirs(data_path, exist_ok=True)

        train_features.to_csv(os.path.join(data_path, "train_bow.csv"), index=False)
        test_features.to_csv(os.path.join(data_path, "test_bow.csv"), index=False)

        logger.info("Feature files saved successfully.")

    except Exception as e:
        logger.error(f"Failed to save features: {e}")
        raise RuntimeError(f"Failed to save features: {e}")

# --------------------------
# Main Execution
# --------------------------
def main():
    try:
        logger.info("Starting feature engineering pipeline...")

        max_features = load_params("params.yaml")
        train_df, test_df = load_processed_data()
        train_features, test_features = build_bow_features(train_df, test_df, max_features)

        save_features(train_features, test_features)

        logger.info("Feature engineering pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"❌ Pipeline failed: {e}")

if __name__ == "__main__":
    main()
