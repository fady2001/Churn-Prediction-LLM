from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATASET_NAME, RAW_DATA_DIR


def split_data(data, train_size: float = 0.7, test_size: float = 0.15):
    """Splits the data into training, validation and testing sets."""
    train_data, temp_data = train_test_split(data, train_size=train_size, random_state=42)
    val_data, test_data = train_test_split(
        temp_data, test_size=test_size / (1 - train_size), random_state=42
    )
    logger.info(
        f"Data split into train ({len(train_data)} samples), "
        f"validation ({len(val_data)} samples), "
        f"and test ({len(test_data)} samples) sets."
    )
    logger.info(
        f"Train size: {len(train_data) / len(data):.2%}, "
        f"Validation size: {len(val_data) / len(data):.2%}, "
        f"Test size: {len(test_data) / len(data):.2%}"
    )
    return train_data, val_data, test_data


if __name__ == "__main__":
    df = pd.read_csv(RAW_DATA_DIR / DATASET_NAME)
    train_data, val_data, test_data = split_data(df, train_size=0.7, test_size=0.15)
    train_data.to_csv(RAW_DATA_DIR / "train.csv", index=False)
    val_data.to_csv(RAW_DATA_DIR / "val.csv", index=False)
    test_data.to_csv(RAW_DATA_DIR / "test.csv", index=False)
