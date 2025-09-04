from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
import pandas as pd

from src.config import MODELS_DIR, RAW_DATA_DIR
from src.llm import CustomerInfoExtractor
from src.predictor import ChurnPredictor


def load_trained_model(model_path: Optional[str] = None) -> ChurnPredictor:
    """
    Load a trained churn prediction model.

    Parameters:
    -----------
    model_path : str, optional
        Path to the saved model. If None, looks for default model in models directory.

    Returns:
    --------
    ChurnPredictor
        Loaded predictor instance
    """
    if model_path is None:
        # Look for saved models in the models directory
        models_dir = Path(MODELS_DIR)
        model_files = list(models_dir.glob("*.pkl"))

        if not model_files:
            raise FileNotFoundError(
                f"No trained model found in {models_dir}. Please train a model first."
            )

        # Use the most recent model file
        model_path = max(model_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Using most recent model: {model_path}")

    return ChurnPredictor.load(model_path)


def predict_from_paragraph(
    paragraph: str, model: ChurnPredictor, llm_model: str = "gemma3:1b"
) -> Dict[str, Any]:
    """
    Extract features from a customer description paragraph using LLM and make churn prediction.

    Parameters:
    -----------
    paragraph : str
        Natural language description of the customer
    model : ChurnPredictor
        Trained prediction model
    llm_model : str
        LLM model name to use for extraction

    Returns:
    --------
    Dict containing extracted features, prediction results, and confidence
    """
    logger.info("Extracting customer features from paragraph using LLM")

    # Initialize LLM extractor
    extractor = CustomerInfoExtractor(model_name=llm_model)

    # Extract customer data
    customer_data = extractor.extract_customer_info(paragraph)
    logger.info("Customer features extracted successfully")

    # Convert to DataFrame
    customer_dict = customer_data.model_dump()
    df = pd.DataFrame([customer_dict])
    print(df.dtypes)
    print(df['tenure'])

    # Make prediction
    predictions_encoded, predictions_decoded = model.predict(df)
    churn_probabilities = model.predict_churn_probability(df)

    result = {
        "input_type": "paragraph",
        "extracted_features": customer_dict,
        "prediction": predictions_decoded[0],
        "prediction_encoded": int(predictions_encoded[0]),
        "churn_probability": float(churn_probabilities[0]),
        "no_churn_probability": float(1 - churn_probabilities[0]),
        "confidence": float(max(churn_probabilities[0], 1 - churn_probabilities[0])),
    }

    logger.info(
        f"Prediction completed: {predictions_decoded[0]} (confidence: {result['confidence']:.3f})"
    )
    return result


def predict_from_structured_data(data: Dict[str, Any], model: ChurnPredictor) -> Dict[str, Any]:
    """
    Make churn prediction from structured customer data.

    Parameters:
    -----------
    data : Dict
        Dictionary containing customer features
    model : ChurnPredictor
        Trained prediction model

    Returns:
    --------
    Dict containing prediction results and confidence
    """
    logger.info("Making prediction from structured data")

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Make prediction
    predictions_encoded, predictions_decoded = model.predict(df)
    churn_probabilities = model.predict_churn_probability(df)

    result = {
        "input_type": "structured_data",
        "input_features": data,
        "prediction": predictions_decoded[0],
        "prediction_encoded": int(predictions_encoded[0]),
        "churn_probability": float(churn_probabilities[0]),
        "no_churn_probability": float(1 - churn_probabilities[0]),
        "confidence": float(max(churn_probabilities[0], 1 - churn_probabilities[0])),
    }

    logger.info(
        f"Prediction completed: {predictions_decoded[0]} (confidence: {result['confidence']:.3f})"
    )
    return result


def predict_batch(
    data: Union[List[Dict], pd.DataFrame, str], model: ChurnPredictor
) -> Dict[str, Any]:
    """
    Make churn predictions for a batch of customers.

    Parameters:
    -----------
    data : Union[List[Dict], pd.DataFrame, str]
        Batch data - can be list of dictionaries, DataFrame, or path to CSV file
    model : ChurnPredictor
        Trained prediction model

    Returns:
    --------
    Dict containing batch prediction results
    """
    logger.info("Making batch predictions")

    # Handle different input types
    if isinstance(data, str):
        # Load from CSV file
        df = pd.read_csv(data)
        logger.info(f"Loaded {len(df)} records from {data}")
    elif isinstance(data, list):
        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame(data)
        logger.info(f"Processing {len(df)} records from list")
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
        logger.info(f"Processing {len(df)} records from DataFrame")
    else:
        raise ValueError("Unsupported data type. Use List[Dict], DataFrame, or CSV file path.")

    # store customer_ids
    customer_ids = df["customerID"].tolist()

    # Make predictions
    predictions_encoded, predictions_decoded = model.predict(df)
    churn_probabilities = model.predict_churn_probability(df)

    # Create results DataFrame
    results_df = pd.DataFrame(
        {
            "prediction": predictions_decoded,
            "prediction_encoded": predictions_encoded,
            "churn_probability": churn_probabilities,
            "no_churn_probability": 1 - churn_probabilities,
            "confidence": [max(p, 1 - p) for p in churn_probabilities],
        }
    )

    # Add customer IDs if available
    if customer_ids:
        results_df["customerID"] = customer_ids
        results_df = results_df[
            ["customerID"] + [col for col in results_df.columns if col != "customerID"]
        ]

    # Calculate summary statistics
    total_customers = len(results_df)
    churn_count = (predictions_encoded == 1).sum()
    no_churn_count = total_customers - churn_count
    avg_churn_probability = churn_probabilities.mean()

    result = {
        "input_type": "batch",
        "total_customers": total_customers,
        "churn_predictions": int(churn_count),
        "no_churn_predictions": int(no_churn_count),
        "churn_rate": float(churn_count / total_customers),
        "average_churn_probability": float(avg_churn_probability),
        "predictions": results_df.to_dict("records"),
    }

    logger.info(
        f"Batch prediction completed: {churn_count}/{total_customers} predicted to churn "
        f"(rate: {result['churn_rate']:.3f})"
    )
    return result


if __name__ == "__main__":
    model = load_trained_model(MODELS_DIR / "artifacts.pkl")
    # Example usage with a sample paragraph
    sample_paragraph = """
    Customer 4593 is a male senior citizen who has been with the company for 24 months.  
    He is married with no dependents. He has a phone service and is a dual customer.  
    His internet service is Fiber optic with online security and device protection, but no online backup.  
    He also receives tech support, streams TV and movies.  
    He has a one-year contract with paperless billing enabled, and pays by electronic check.  
    His monthly charges are 89.65 and his total charges so far are 2151.60.
    """
    result = predict_from_paragraph(sample_paragraph, model)
    logger.info(f"Sample Prediction Result: {result}")

    # Example usage with structured data
    test_df = pd.read_csv(RAW_DATA_DIR / "test.csv")
    structured_data = test_df.drop(columns=["Churn"]).iloc[0].to_dict()
    result = predict_from_structured_data(structured_data, model)
    logger.info(f"Structured Data Prediction Result: {result}")

    # Example usage with batch data from CSV
    batch_result = predict_batch(test_df, model)
    logger.info(f"Batch Prediction Summary: {batch_result}")
