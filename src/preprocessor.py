from typing import Dict, List, Tuple

from loguru import logger
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)


class Preprocessor:
    def __init__(
        self,
        pipeline_config: Dict[str, str] = None,
    ):
        self.pipeline_config = pipeline_config or {}
        self.pipeline = None

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        transformers = []

        # if self.pipeline_config.get("drop"):
        #     columns_to_drop = self.pipeline_config["drop"]
        #     transformers.append(('drop', 'drop', columns_to_drop))

        # X.drop(columns=self.pipeline_config.get("drop", []), inplace=True, errors="ignore")
        logger.info(f"Columns to drop: {self.pipeline_config.get('drop', [])}")
        transformers.append(("drop_columns", "drop", self.pipeline_config.get("drop", [])))
        transformers.extend(self.__create_encode_steps())
        transformers.extend(self.__create_scaling_steps())
        transformers.extend(self.__create_imputing_steps())

        preprocessor = ColumnTransformer(transformers=transformers, remainder="passthrough")

        self.pipeline = preprocessor.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit() before transform().")

        # Transform the data
        transformed_data = self.pipeline.transform(X)

        # Get feature names and create DataFrame
        feature_names = self.get_feature_names_from_preprocessor()

        return pd.DataFrame(transformed_data, columns=feature_names, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        # Transform the data
        transformed_data = self.pipeline.fit_transform(X)

        # Get feature names and create DataFrame
        feature_names = self.get_feature_names_from_preprocessor()

        return pd.DataFrame(transformed_data, columns=feature_names, index=X.index)

    def get_pipeline(self) -> ColumnTransformer:
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit() before get_pipeline().")
        return self.pipeline

    def __create_encode_steps(self) -> List[Tuple[str, BaseEstimator, List[str]]]:
        encode_steps = []
        if self.pipeline_config.get("encoding"):
            for strategy in self.pipeline_config["encoding"].keys():
                if strategy == "onehot":
                    cols = self.pipeline_config["encoding"]["onehot"]
                    encode_steps.append(
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                            ),
                            cols,
                        )
                    )
                elif strategy == "ordinal":
                    cols = self.pipeline_config["encoding"]["ordinal"]
                    encode_steps.append(
                        (
                            "ordinal",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                            cols,
                        )
                    )
        return encode_steps

    def __create_scaling_steps(self) -> List[Tuple[str, BaseEstimator, List[str]]]:
        scaling_steps = []
        if self.pipeline_config.get("scaling"):
            for strategy in self.pipeline_config["scaling"].keys():
                if strategy == "standard":
                    cols = self.pipeline_config["scaling"]["standard"]
                    scaling_steps.append(("standard", StandardScaler(), cols))
                elif strategy == "minmax":
                    cols = self.pipeline_config["scaling"]["minmax"]
                    scaling_steps.append(("minmax", MinMaxScaler(), cols))
                elif strategy == "robust":
                    cols = self.pipeline_config["scaling"]["robust"]
                    scaling_steps.append(("robust", RobustScaler(), cols))
        return scaling_steps

    def __create_imputing_steps(self) -> List[Tuple[str, BaseEstimator, List[str]]]:
        imputing_steps = []
        if self.pipeline_config.get("imputation"):
            for strategy in self.pipeline_config["imputation"].keys():
                if strategy == "mean":
                    cols = self.pipeline_config["imputation"]["mean"]
                    imputing_steps.append(
                        ("mean", SimpleImputer(strategy="mean", add_indicator=True), cols)
                    )
                elif strategy == "median":
                    cols = self.pipeline_config["imputation"]["median"]
                    imputing_steps.append(
                        ("median", SimpleImputer(strategy="median", add_indicator=True), cols)
                    )
                elif strategy == "most_frequent":
                    cols = self.pipeline_config["imputation"]["most_frequent"]
                    imputing_steps.append(
                        (
                            "most_frequent",
                            SimpleImputer(strategy="most_frequent", add_indicator=True),
                            cols,
                        )
                    )
                elif strategy == "constant":
                    cols_fill: Dict = self.pipeline_config["imputation"]["constant"]
                    for col, fill_value in cols_fill.items():
                        imputing_steps.append(
                            (
                                "constant",
                                SimpleImputer(
                                    strategy="constant", fill_value=fill_value, add_indicator=True
                                ),
                                col,
                            )
                        )
        return imputing_steps

    def get_feature_names_from_preprocessor(self) -> List[str]:
        """
        Extract feature names from a ColumnTransformer after encoding.

        Parameters:
        - preprocessor: The fitted ColumnTransformer object.

        Returns:
        - A list of feature names.
        """
        feature_names = []
        for name, transformer, columns in self.pipeline.transformers_:
            if transformer == "drop" or transformer is None:
                continue  # Skip dropped columns
            elif hasattr(transformer, "get_feature_names_out"):
                # For transformers like OneHotEncoder
                feature_names.extend(transformer.get_feature_names_out(columns))
            else:
                # For other transformers, use the column names directly
                feature_names.extend(columns)
        return feature_names


if __name__ == "__main__":
    # Example usage
    from src.config import INTERIM_DATA_DIR

    df = pd.read_csv(f"{INTERIM_DATA_DIR}/telco_customer_churn_engineered.csv")
    df = df.head()

    PIPELINE_CONFIG = {
        "drop": ["customerID", "gender"],
        "scaling": {"standard": ["tenure", "Monthly_Charges", "Total_Charges"]},
        "encoding": {
            "onehot": [
                "Is_Married",
                "Dependents",
                "Phone_Service",
                "Dual",
                "Internet_Service",
                "Online_Security",
                "Online_Backup",
                "Device_Protection",
                "Tech_Support",
                "Streaming_TV",
                "Streaming_Movies",
                "Paperless_Billing",
            ],
            "ordinal": ["Contract", "Payment_Method", "Payment_Method_Grouped", "tenure_group"],
        },
    }
    preprocessor = Preprocessor(pipeline_config=PIPELINE_CONFIG)
    processed_data = preprocessor.fit_transform(df)
    print(processed_data.columns)
