import numpy as np
import pandas as pd
import os
import re
import string
import nltk
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# --------------------------
# Logging Configuration
# --------------------------
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('data_preprocessing.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# --------------------------
# NLTK Setup
# --------------------------
try:
    nltk.download('stopwords')
    nltk.download('wordnet')
    logger.info("NLTK resources downloaded successfully.")
except Exception as e:
    logger.error(f"Failed to download NLTK data: {e}")
    raise RuntimeError(f"NLTK setup error: {e}")

# --------------------------
# Load Raw Data
# --------------------------
def load_data():
    try:
        logger.info("Loading raw train and test data...")

        train_path = "./data/raw/train.csv"
        test_path = "./data/raw/test.csv"

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            logger.error("Raw dataset files missing in data/raw/")
            raise FileNotFoundError("Raw data files not found in data/raw/")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logger.info("Raw data loaded successfully.")
        logger.debug(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

        return train_df, test_df

    except Exception as e:
        logger.error(f"Error loading raw data: {e}")
        raise RuntimeError(f"Failed to load raw data: {e}")

# --------------------------
# Cleaning Utilities
# --------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def lemmatization(text: str) -> str:
    try:
        words = text.split()
        return " ".join(lemmatizer.lemmatize(w) for w in words)
    except Exception as e:
        logger.error(f"Lemmatization error: {e}")
        return text

def remove_stopwords(text: str) -> str:
    try:
        words = text.split()
        filtered = [w for w in words if w.lower() not in stop_words]
        return " ".join(filtered)
    except Exception as e:
        logger.error(f"Stopword removal error: {e}")
        return text

def remove_numbers(text: str) -> str:
    return re.sub(r'\d+', '', text)

def to_lowercase(text: str) -> str:
    return text.lower()

def remove_punctuations(text: str) -> str:
    try:
        pattern = "[" + re.escape(string.punctuation) + "]"
        text = re.sub(pattern, " ", text)
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        logger.error(f"Punctuation removal error: {e}")
        return text

def remove_urls(text: str) -> str:
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def expand_contractions(text: str) -> str:
    try:
        return contractions.fix(text)
    except Exception:
        return text

def remove_mentions(text: str) -> str:
    return re.sub(r'@\w+', '', text)

def remove_hashtags(text: str) -> str:
    return re.sub(r'#(\w+)', r'\1', text)

def remove_html(text: str) -> str:
    return re.sub(r'<.*?>', '', text)

# --------------------------
# Main Clean Function
# --------------------------
def clean(text: str) -> str:
    try:
        if pd.isna(text):
            return ""

        text = to_lowercase(text)
        text = remove_urls(text)
        text = remove_mentions(text)
        text = remove_hashtags(text)
        text = remove_html(text)
        text = expand_contractions(text)
        text = remove_punctuations(text)
        text = remove_numbers(text)
        text = remove_stopwords(text)
        text = lemmatization(text)

        return text

    except Exception as e:
        logger.error(f"Text cleaning error: {e}")
        return text

# --------------------------
# Normalize DataFrame
# --------------------------
def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Normalizing text content...")

        if "content" not in df.columns:
            logger.error("Column 'content' not found in dataset.")
            raise KeyError("Missing 'content' column in dataframe.")

        df["content"] = df["content"].apply(clean)

        logger.info("Text normalization completed.")
        logger.debug(f"Processed DataFrame shape: {df.shape}")

        return df

    except Exception as e:
        logger.error(f"Error normalizing text: {e}")
        raise RuntimeError(f"Text normalization failed: {e}")

def normalize_sentence(sentence: str) -> str:
    return clean(sentence)

# --------------------------
# Save Processed Data
# --------------------------
def save_processed(train_df: pd.DataFrame, test_df: pd.DataFrame):
    try:
        data_path = os.path.join("data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_df.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_df.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        logger.info("Processed data saved successfully.")

    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise RuntimeError(f"Failed to save processed data: {e}")

# --------------------------
# Main Execution
# --------------------------
def main():
    try:
        logger.info("Starting data preprocessing pipeline...")

        train_df, test_df = load_data()

        train_clean = normalize_text(train_df)
        test_clean = normalize_text(test_df)

        save_processed(train_clean, test_clean)

        logger.info("Data preprocessing completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"❌ Pipeline failed: {e}")

if __name__ == "__main__":
    main()
