from pathlib import Path
import traceback
from typing import Any, Dict, List

import litserve as ls
from loguru import logger
import pandas as pd

from src.llm import CustomerInfoExtractor
from src.predictor import ChurnPredictor


class ChurnPredictionAPI(ls.LitAPI):
    """
    LitServe API for churn prediction that handles:
    1. Text paragraphs (converted to customer data via LLM)
    2. Single customer dictionaries
    3. Batch of customer dictionaries
    """

    def setup(self, device):
        """Setup the API by loading the trained models"""
        try:
            # Load the trained churn predictor
            model_path = Path("models/artifacts.pkl")
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")

            self.predictor = ChurnPredictor.load(model_path)
            logger.info("Churn predictor loaded successfully")

            # Initialize the LLM-based customer info extractor
            self.extractor = CustomerInfoExtractor()
            logger.info("Customer info extractor initialized")

        except Exception as e:
            logger.error(f"Failed to setup API: {e}")
            raise

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode and validate the incoming request.

        Expected request formats:
        1. Paragraph: {"type": "paragraph", "data": "customer description text..."}
        2. Single customer: {"type": "single", "data": {"CustomerId": "123", "Gender": "Male", ...}}
        3. Batch: {"type": "batch", "data": [{"CustomerId": "123", ...}, {"CustomerId": "456", ...}]}
        """
        try:
            request_type = request.get("type")
            data = request.get("data")

            if not request_type or data is None:
                raise ValueError("Request must contain 'type' and 'data' fields")

            if request_type not in ["paragraph", "single", "batch"]:
                raise ValueError("Type must be one of: 'paragraph', 'single', 'batch'")

            return {"type": request_type, "data": data}

        except Exception as e:
            logger.error(f"Failed to decode request: {e}")
            raise ValueError(f"Invalid request format: {e}")

    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make churn predictions based on the request type.
        """
        try:
            request_type = request["type"]
            data = request["data"]

            if request_type == "paragraph":
                return self._predict_from_paragraph(data)
            elif request_type == "single":
                return self._predict_single_customer(data)
            elif request_type == "batch":
                return self._predict_batch_customers(data)
            else:
                raise ValueError(f"Unsupported request type: {request_type}")

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e), "type": request.get("type", "unknown")}

    def _predict_from_paragraph(self, paragraph: str) -> Dict[str, Any]:
        """Extract customer info from paragraph and make prediction"""
        try:
            # Extract customer information using LLM
            customer_data = self.extractor.extract_customer_info(paragraph)
            logger.info("Customer data extracted from paragraph")

            # Convert to DataFrame for prediction
            customer_dict = customer_data.model_dump()
            df = pd.DataFrame([customer_dict])

            # Make prediction
            result = self._make_prediction(df, customer_dict.get("CustomerId", "unknown"))

            return {
                "success": True,
                "type": "paragraph",
                "extracted_data": customer_dict,
                "prediction": result,
            }

        except Exception as e:
            logger.error(f"Failed to predict from paragraph: {e}")
            return {"success": False, "error": str(e), "type": "paragraph"}

    def _predict_single_customer(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction for a single customer dictionary"""
        try:
            # Validate and convert to DataFrame
            df = pd.DataFrame([customer_data])
            customer_id = customer_data.get("CustomerId", "unknown")

            # Make prediction
            result = self._make_prediction(df, customer_id)

            return {
                "success": True,
                "type": "single",
                "customer_id": customer_id,
                "prediction": result,
            }

        except Exception as e:
            logger.error(f"Failed to predict single customer: {e}")
            return {
                "success": False,
                "error": str(e),
                "type": "single",
                "customer_id": customer_data.get("CustomerId", "unknown"),
            }

    def _predict_batch_customers(self, customers_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make predictions for a batch of customers"""
        try:
            if not isinstance(customers_data, list):
                raise ValueError("Batch data must be a list of customer dictionaries")

            if len(customers_data) == 0:
                raise ValueError("Batch cannot be empty")

            # Convert to DataFrame
            df = pd.DataFrame(customers_data)
            customer_ids = [
                customer.get("CustomerId", f"unknown_{i}")
                for i, customer in enumerate(customers_data)
            ]

            # Make predictions
            detailed_results = self.predictor.predict_with_interpretation(df)

            # Format results for each customer
            predictions = []
            for i, customer_id in enumerate(customer_ids):
                result = {
                    "customer_id": customer_id,
                    "churn_probability": float(detailed_results.iloc[i]["churn_probability"]),
                    "prediction_label": detailed_results.iloc[i]["prediction_label"],
                    "confidence": float(detailed_results.iloc[i]["confidence"]),
                }
                predictions.append(result)

            return {
                "success": True,
                "type": "batch",
                "batch_size": len(customers_data),
                "predictions": predictions,
            }

        except Exception as e:
            logger.error(f"Failed to predict batch: {e}")
            return {
                "success": False,
                "error": str(e),
                "type": "batch",
                "batch_size": len(customers_data) if isinstance(customers_data, list) else 0,
            }

    def _make_prediction(self, df: pd.DataFrame, customer_id: str) -> Dict[str, Any]:
        """Helper method to make prediction for a single customer DataFrame"""
        try:
            # Get detailed prediction results
            detailed_results = self.predictor.predict_with_interpretation(df)
            result = detailed_results.iloc[0]

            return {
                "customer_id": customer_id,
                "churn_probability": float(result["churn_probability"]),
                "no_churn_probability": float(result["no_churn_probability"]),
                "prediction_label": result["prediction_label"],
                "prediction_encoded": int(result["prediction_encoded"]),
                "confidence": float(result["confidence"]),
                "threshold_used": float(result["threshold_used"]),
            }

        except Exception as e:
            logger.error(f"Failed to make prediction: {e}")
            raise

    def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Encode the response for sending back to client"""
        return output


def create_server():
    """
    Create and return a LitServe server instance.

    Parameters:
    -----------
    port : int, default=8000
        Port to run the server on
    workers : int, default=1
        Number of worker processes

    Returns:
    --------
    server : ls.LitServer
        Configured LitServe server
    """
    api = ChurnPredictionAPI()
    server = ls.LitServer(api, accelerator="cpu")
    return server


if __name__ == "__main__":
    # Start the server
    server = create_server()
    server.run(port=8000)
