import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Comprehensive feature engineering transformer for churn prediction

    This transformer creates various engineered features from the base dataset
    and can be easily integrated into scikit-learn pipelines.
    """

    def __init__(self):
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        """
        Fit the transformer. For this feature engineer, no fitting is required.

        Parameters:
        -----------
        X : pandas.DataFrame
            Input features
        y : array-like, optional
            Target values (ignored)

        Returns:
        --------
        self : ChurnFeatureEngineer
        """
        return self

    def transform(self, X):
        """
        Transform the input features by adding engineered features.

        Parameters:
        -----------
        X : pandas.DataFrame
            Input features

        Returns:
        --------
        X_transformed : pandas.DataFrame
            Transformed features with engineered columns
        """
        X = X.copy()

        # 1. Billing-based Features
        X["Average_Charges"] = X["Total_Charges"] / (X["tenure"] + 1)  # +1 to avoid division by 0
        X["Charge_to_Tenure_Ratio"] = X["Monthly_Charges"] / (X["tenure"] + 1)

        # 2. Tenure-based Features
        X["Is_New_Customer"] = (X["tenure"] < 6).astype(int)
        X["Is_Long_Term"] = (X["tenure"] > 24).astype(int)

        # 3. Contract and payment features
        auto_pay_methods = ["Credit card (automatic)", "Bank transfer (automatic)"]
        X["AutoPay"] = X["Payment_Method"].isin(auto_pay_methods).astype(int)

        X["Is_Paperless_and_Monthly"] = (
            (X["Paperless_Billing"] == "Yes") & (X["Contract"] == "Month-to-month")
        ).astype(int)

        contract_mapping = {"Month-to-month": 1, "One year": 12, "Two year": 24}
        X["Contract_Length"] = X["Contract"].map(contract_mapping)

        X["Payment_Method_Grouped"] = X["Payment_Method"].apply(
            lambda x: "Electronic_Check" if x == "Electronic check" else "Other"
        )

        # 4. Service bundling features
        X["Has_Internet"] = (X["Internet_Service"] != "No").astype(int)

        X["Entertainment_Package"] = (
            (X["Streaming_TV"] == "Yes") | (X["Streaming_Movies"] == "Yes")
        ).astype(int)

        security_services = ["Online_Security", "Tech_Support", "Device_Protection"]
        X["Security_Package"] = X[security_services].apply(
            lambda row: sum(val == "Yes" for val in row), axis=1
        )

        service_columns = [
            "Phone_Service",
            "Internet_Service",
            "Online_Security",
            "Online_Backup",
            "Device_Protection",
            "Tech_Support",
            "Streaming_TV",
            "Streaming_Movies",
        ]

        X["num_services"] = X[service_columns].apply(
            lambda row: sum("No" not in str(val) for val in row), axis=1
        )

        # 5. Interaction Features
        X["Contract_Charges_Interaction"] = X["Contract_Length"] * X["Monthly_Charges"]

        X["Engagement_Score"] = X["tenure"] * X["num_services"]

        X["Payment_Risk_Score"] = (X["Payment_Method"] == "Electronic check").astype(int) + (
            X["Paperless_Billing"] == "Yes"
        ).astype(int)

        X["tenure_group"] = pd.cut(
            X["tenure"],
            bins=[0, 12, 24, 48, 72],
            labels=["0-12 months", "13-24 months", "25-48 months", "49+ months"],
            include_lowest=True,
        )

        # Store feature names for later reference
        self.feature_names_out_ = X.columns.tolist()

        return X

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters:
        -----------
        input_features : array-like of str or None, default=None
            Input feature names

        Returns:
        --------
        feature_names_out : ndarray of str objects
            Transformed feature names
        """
        if self.feature_names_out_ is None:
            raise ValueError("This ChurnFeatureEngineer instance is not fitted yet.")
        return np.array(self.feature_names_out_)


if __name__ == "__main__":
    # Example usage
    from src.config import INTERIM_DATA_DIR

    df = pd.read_csv(f"{INTERIM_DATA_DIR}/telco_customer_churn.csv")
    df = df.head()

    feature_engineer = ChurnFeatureEngineer()
    df_transformed = feature_engineer.fit_transform(df)
    print(df_transformed.columns)
