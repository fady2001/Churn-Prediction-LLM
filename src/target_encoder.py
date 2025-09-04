from typing import List, Union

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target variable encoder for churn prediction.

    This encoder converts string target values to numerical values and provides
    methods to reverse the encoding for predictions.
    """

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.classes_ = None
        self.is_fitted_ = False

    def fit(self, y: Union[pd.Series, np.ndarray, List]) -> "TargetEncoder":
        """
        Fit the target encoder to the target variable.

        Parameters:
        -----------
        y : array-like
            Target variable values

        Returns:
        --------
        self : TargetEncoder
        """
        # Convert to numpy array if needed
        if isinstance(y, pd.Series):
            y = y.values
        elif isinstance(y, list):
            y = np.array(y)

        # Fit the label encoder
        self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.is_fitted_ = True

        logger.info(f"Target encoder fitted. Classes: {self.classes_}")
        logger.info(
            f"Encoding mapping: {dict(zip(self.classes_, self.label_encoder.transform(self.classes_)))}"
        )

        return self

    def transform(self, y: Union[pd.Series, np.ndarray, List]) -> np.ndarray:
        """
        Transform target variable to numerical values.

        Parameters:
        -----------
        y : array-like
            Target variable values to encode

        Returns:
        --------
        y_encoded : np.ndarray
            Numerically encoded target values
        """
        if not self.is_fitted_:
            raise ValueError("TargetEncoder must be fitted before transform. Call fit() first.")

        # Convert to numpy array if needed
        if isinstance(y, pd.Series):
            y = y.values
        elif isinstance(y, list):
            y = np.array(y)

        return self.label_encoder.transform(y)

    def fit_transform(self, y: Union[pd.Series, np.ndarray, List]) -> np.ndarray:
        """
        Fit the encoder and transform the target variable in one step.

        Parameters:
        -----------
        y : array-like
            Target variable values

        Returns:
        --------
        y_encoded : np.ndarray
            Numerically encoded target values
        """
        return self.fit(y).transform(y)

    def inverse_transform(self, y_encoded: Union[pd.Series, np.ndarray, List]) -> np.ndarray:
        """
        Convert numerical predictions back to original string labels.

        Parameters:
        -----------
        y_encoded : array-like
            Numerically encoded target values

        Returns:
        --------
        y_decoded : np.ndarray
            Original string labels
        """
        if not self.is_fitted_:
            raise ValueError(
                "TargetEncoder must be fitted before inverse_transform. Call fit() first."
            )

        # Convert to numpy array if needed
        if isinstance(y_encoded, pd.Series):
            y_encoded = y_encoded.values
        elif isinstance(y_encoded, list):
            y_encoded = np.array(y_encoded)

        # Handle both integer predictions and probability predictions
        if y_encoded.dtype == float:
            # If probabilities, convert to class predictions
            y_encoded = (y_encoded >= 0.5).astype(int)

        return self.label_encoder.inverse_transform(y_encoded)

    def get_encoding_mapping(self) -> dict:
        """
        Get the mapping from original labels to encoded values.

        Returns:
        --------
        mapping : dict
            Dictionary mapping original labels to encoded values
        """
        if not self.is_fitted_:
            raise ValueError("TargetEncoder must be fitted before getting mapping.")

        return dict(zip(self.classes_, self.label_encoder.transform(self.classes_)))

    def get_classes(self) -> np.ndarray:
        """
        Get the original class labels.

        Returns:
        --------
        classes : np.ndarray
            Array of original class labels
        """
        if not self.is_fitted_:
            raise ValueError("TargetEncoder must be fitted before getting classes.")

        return self.classes_


if __name__ == "__main__":
    # Example usage
    import pandas as pd

    # Sample data
    y_sample = pd.Series(["No", "Yes", "No", "Yes", "No"])

    # Create and use encoder
    encoder = TargetEncoder()
    y_encoded = encoder.fit_transform(y_sample)
    print(f"Original: {y_sample.values}")
    print(f"Encoded: {y_encoded}")
    print(f"Encoding mapping: {encoder.get_encoding_mapping()}")

    # Test inverse transform
    y_decoded = encoder.inverse_transform(y_encoded)
    print(f"Decoded: {y_decoded}")

    # Test with probabilities
    y_probs = np.array([0.2, 0.8, 0.3, 0.9, 0.1])
    y_decoded_probs = encoder.inverse_transform(y_probs)
    print(f"Probabilities: {y_probs}")
    print(f"Decoded from probs: {y_decoded_probs}")
