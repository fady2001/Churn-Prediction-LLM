"""
Simple test script for the Churn Prediction API
Run this after starting the API server with: python api_server.py
"""

import requests
import json

# API Configuration
API_URL = "http://localhost:8000/predict"


def test_api():
    print("🧪 Testing Churn Prediction API")
    print("=" * 40)

    # Test 1: Single Customer
    print("\n1️⃣ Testing Single Customer Prediction:")

    customer = {
        "CustomerId": "TEST-001",
        "Gender": "Female",
        "Senior_Citizen": 0,
        "Is_Married": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "Phone_Service": "Yes",
        "Dual": "No",
        "Internet_Service": "DSL",
        "Online_Security": "Yes",
        "Online_Backup": "No",
        "Device_Protection": "No",
        "Tech_Support": "No",
        "Streaming_TV": "No",
        "Streaming_Movies": "No",
        "Contract": "Month-to-month",
        "Paperless_Billing": "Yes",
        "Payment_Method": "Electronic check",
        "Monthly_Charges": 55.75,
        "Total_Charges": 669.00,
    }

    payload = {"type": "single", "data": customer}

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        result = response.json()

        if result["success"]:
            pred = result["prediction"]
            print(f"✅ Success!")
            print(f"   Customer: {pred['customer_id']}")
            print(f"   Churn Probability: {pred['churn_probability']:.3f}")
            print(f"   Prediction: {pred['prediction_label']}")
            print(f"   Confidence: {pred['confidence']:.3f}")
        else:
            print(f"❌ Failed: {result.get('error')}")

    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 2: Paragraph Input
    print("\n2️⃣ Testing Paragraph Input:")

    paragraph = """
    John is a 45-year-old male customer who has been with us for 18 months.
    He is married but has no dependents. He uses our phone service and has
    fiber optic internet with streaming services. He pays monthly via credit card
    and his bills are around $85 per month.
    """

    payload = {"type": "paragraph", "data": paragraph.strip()}

    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        result = response.json()

        if result["success"]:
            pred = result["prediction"]
            print(f"✅ Success!")
            print(f"   Extracted ID: {pred['customer_id']}")
            print(f"   Churn Probability: {pred['churn_probability']:.3f}")
            print(f"   Prediction: {pred['prediction_label']}")
        else:
            print(f"❌ Failed: {result.get('error')}")

    except requests.exceptions.Timeout:
        print("⏰ Timeout - LLM processing takes time")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 3: Batch Processing
    print("\n3️⃣ Testing Batch Processing:")

    batch = [
        {**customer, "CustomerId": "BATCH-001"},
        {**customer, "CustomerId": "BATCH-002", "tenure": 36, "Contract": "Two year"},
    ]

    payload = {"type": "batch", "data": batch}

    try:
        response = requests.post(API_URL, json=payload, timeout=15)
        result = response.json()

        if result["success"]:
            print(f"✅ Success! Processed {result['batch_size']} customers")
            for i, pred in enumerate(result["predictions"]):
                print(
                    f"   Customer {i + 1}: {pred['customer_id']} -> {pred['prediction_label']} ({pred['churn_probability']:.3f})"
                )
        else:
            print(f"❌ Failed: {result.get('error')}")

    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n🎉 Testing complete!")
    print("\n💡 To run the full test suite, use: jupyter notebook notebooks/api_test.ipynb")


if __name__ == "__main__":
    test_api()
