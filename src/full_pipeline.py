from loguru import logger
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import INTERIM_DATA_DIR, PIPELINE_CONFIG, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.features import ChurnFeatureEngineer
from src.preprocessor import Preprocessor


def full_pipeline():
    df = pd.read_csv(RAW_DATA_DIR / "telco_customer_churn.csv")
    feature_engineer = ChurnFeatureEngineer()
    preprocessor = Preprocessor(PIPELINE_CONFIG)
    df_train, df_test = df.iloc[:4000, :], df.iloc[4000:, :]

    df_train = feature_engineer.fit_transform(df_train)
    df_train = preprocessor.fit_transform(df_train)

    full_pipline = Pipeline(
        steps=[
            ("feature_engineering", feature_engineer),
            ("preprocessing", preprocessor),
        ]
    )

    df_test = full_pipline.transform(df_test)
    logger.info(f"Transformed test data shape: {df_test.shape}")

    df_train.to_csv(PROCESSED_DATA_DIR / "telco_customer_churn_train.csv", index=False)
    df_test.to_csv(PROCESSED_DATA_DIR / "telco_customer_churn_test.csv", index=False)


if __name__ == "__main__":
    full_pipeline()
