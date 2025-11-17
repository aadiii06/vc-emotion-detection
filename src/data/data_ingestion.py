import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import yaml
import os
import logging

# --------------------------
# Logging Configure
# --------------------------
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('data_ingestion.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)
# --------------------------
# Load parameters safely
# --------------------------
def load_params(params_path: str) -> float:
    try:
        logger.info("Loading parameters...")

        if not os.path.exists(params_path):
            logger.error(f"params.yaml not found at path: {params_path}")
            raise FileNotFoundError(f"params.yaml not found at: {params_path}")

        params = yaml.safe_load(open(params_path, 'r'))
        
        if params is None:
            logger.error("params.yaml is empty or invalid.")
            raise ValueError("params.yaml is empty or incorrectly formatted.")

        test_size = params['data_ingestion']['test_size']
        logger.debug(f"Loaded test_size = {test_size}")

        return test_size

    except KeyError as e:
        logger.error(f"Missing key in params.yaml: {e}")
        raise KeyError(f"Missing key in params.yaml: {e}")

    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        raise RuntimeError(f"Error loading parameters: {e}")


# --------------------------
# Read data safely
# --------------------------
def read_data(url: str) -> pd.DataFrame:
    try:
        logger.info(f"Reading data from URL: {url}")
        df = pd.read_csv(url)
        logger.info("Data loaded successfully.")
        logger.debug(f"Data shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Failed to read data: {e}")
        raise RuntimeError(f"Failed to read data from URL: {e}")


# --------------------------
# Process data safely
# --------------------------
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Processing data...")

        if 'tweet_id' not in df.columns:
            logger.error("Column 'tweet_id' not found.")
            raise KeyError("Column 'tweet_id' not found in dataset.")

        if 'sentiment' not in df.columns:
            logger.error("Column 'sentiment' not found.")
            raise KeyError("Column 'sentiment' not found in dataset.")

        df = df.copy()
        df.drop(columns=['tweet_id'], inplace=True)

        logger.debug("Encoding sentiments...")
        le = LabelEncoder()
        df['sentiment'] = le.fit_transform(df['sentiment'])

        logger.info("Data processed successfully.")
        logger.debug(f"Processed data shape: {df.shape}")

        return df

    except Exception as e:
        logger.error(f"Error while processing data: {e}")
        raise RuntimeError(f"Error while processing data: {e}")


# --------------------------
# Save data safely
# --------------------------
def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        logger.info(f"Saving data to path: {data_path}")

        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)

        logger.info("Data saved successfully.")

    except Exception as e:
        logger.error(f"Failed to save data: {e}")
        raise RuntimeError(f"Failed to save processed data: {e}")


# --------------------------
# Main Execution
# --------------------------
def main():
    try:
        logger.info("Starting data ingestion pipeline...")

        test_size = load_params('params.yaml')
        df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        df = process_data(df)

        logger.info("Splitting data into train-test sets...")

        train_data, test_data = train_test_split(
            df,
            test_size=test_size,
            random_state=42,
            stratify=df['sentiment']
        )

        logger.debug(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")

        data_path = os.path.join('data', 'raw')
        save_data(data_path, train_data, test_data)

        logger.info("Data ingestion completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"❌ Pipeline failed: {e}")


if __name__ == '__main__':
    main()
