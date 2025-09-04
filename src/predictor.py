from pathlib import Path
import pickle
from typing import Tuple, Union

from loguru import logger
import numpy as np
import pandas as pd

from src.target_encoder import TargetEncoder


class ChurnPredictor:
    """
    A wrapper class for making churn predictions with automatic target encoding/decoding.
    """

    def __init__(self, model_pipeline, target_encoder: TargetEncoder):
        """
        Initialize the predictor with a trained pipeline and target encoder.

        Parameters:
        -----------
        model_pipeline : sklearn.pipeline.Pipeline
            Trained ML pipeline for feature processing and prediction
        target_encoder : TargetEncoder
            Fitted target encoder for converting predictions back to labels
        """
        self.model_pipeline = model_pipeline
        self.target_encoder = target_encoder

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions and return both encoded and decoded results.

        Parameters:
        -----------
        X : pd.DataFrame
            Input features for prediction

        Returns:
        --------
        predictions_encoded : np.ndarray
            Numerical predictions (0/1)
        predictions_decoded : np.ndarray
            String predictions ('No'/'Yes')
        """
        # Get encoded predictions from the pipeline
        predictions_encoded = self.model_pipeline.predict(X)

        # Decode to original labels
        predictions_decoded = self.target_encoder.inverse_transform(predictions_encoded)

        logger.info(f"Made predictions for {len(X)} samples")
        return predictions_encoded, predictions_decoded

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.

        Parameters:
        -----------
        X : pd.DataFrame
            Input features for prediction

        Returns:
        --------
        probabilities : np.ndarray
            Prediction probabilities for each class
        """
        return self.model_pipeline.predict_proba(X)

    def predict_churn_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get the probability of churn (positive class).

        Parameters:
        -----------
        X : pd.DataFrame
            Input features for prediction

        Returns:
        --------
        churn_probabilities : np.ndarray
            Probability of churn for each sample
        """
        probabilities = self.predict_proba(X)
        return probabilities[:, 1]  # Return probability of positive class (churn)

    def predict_with_interpretation(self, X: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        Make predictions with full interpretation including probabilities and labels.

        Parameters:
        -----------
        X : pd.DataFrame
            Input features for prediction
        threshold : float, default=0.5
            Probability threshold for classification

        Returns:
        --------
        results : pd.DataFrame
            DataFrame with predictions, probabilities, and interpretations
        """
        # Get probabilities
        probabilities = self.predict_proba(X)
        churn_proba = probabilities[:, 1]

        # Make predictions using threshold
        predictions_encoded = (churn_proba >= threshold).astype(int)
        predictions_decoded = self.target_encoder.inverse_transform(predictions_encoded)

        # Create results DataFrame
        results = pd.DataFrame(
            {
                "churn_probability": churn_proba,
                "no_churn_probability": probabilities[:, 0],
                "prediction_encoded": predictions_encoded,
                "prediction_label": predictions_decoded,
                "confidence": np.maximum(churn_proba, 1 - churn_proba),
                "threshold_used": threshold,
            }
        )

        return results

    def save(self, filepath: Union[str, Path]):
        """
        Save the predictor to a file.

        Parameters:
        -----------
        filepath : str or Path
            Path where to save the predictor
        """
        filepath = Path(filepath)
        with open(filepath, "wb") as f:
            pickle.dump(
                {"model_pipeline": self.model_pipeline, "target_encoder": self.target_encoder}, f
            )
        logger.info(f"Predictor saved to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ChurnPredictor":
        """
        Load a predictor from a file.

        Parameters:
        -----------
        filepath : str or Path
            Path to the saved predictor

        Returns:
        --------
        predictor : ChurnPredictor
            Loaded predictor instance
        """
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        predictor = cls(
            model_pipeline=data["model_pipeline"], target_encoder=data["target_encoder"]
        )
        logger.info(f"Predictor loaded from {filepath}")
        return predictor

    def get_target_mapping(self) -> dict:
        """
        Get the target encoding mapping.

        Returns:
        --------
        mapping : dict
            Dictionary showing how string labels map to numbers
        """
        return self.target_encoder.get_encoding_mapping()


if __name__ == "__main__":
    # Example usage - this would typically be used after training
    from src.full_pipeline import full_pipeline

    # Train the pipeline (this would normally be done separately)
    pipeline, encoder = full_pipeline()

    # Create predictor
    predictor = ChurnPredictor(pipeline, encoder)

    # Show encoding mapping
    print("Target encoding mapping:", predictor.get_target_mapping())

    # Example prediction on validation data
    val_df = pd.read_csv("data/raw/val.csv")
    X_val = val_df.drop(columns=["Churn"])

    # Make predictions
    encoded_preds, decoded_preds = predictor.predict(X_val.head(5))
    print("Encoded predictions:", encoded_preds)
    print("Decoded predictions:", decoded_preds)

    # Get detailed results
    detailed_results = predictor.predict_with_interpretation(X_val.head(5))
    print("Detailed results:")
    print(detailed_results)
