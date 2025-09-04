import json
import re

from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from loguru import logger

from src.config import LLM_MODEL
from src.llm_models import CustomerData

system_prompt = """
You are an expert information extraction assistant specializing in customer data analysis.

Your task is to read a customer description paragraph and extract customer features in JSON format.

## IMPORTANT INSTRUCTIONS FOR MISSING DATA:
- If ANY feature is missing, not mentioned, or cannot be determined from the text, return null for that field
- Do NOT make assumptions or guess values
- Do NOT use placeholder values like "Unknown", "N/A", or empty strings
- If Customer ID is not mentioned, set it to null
- If demographic information is unclear, set those fields to null
- If service details are ambiguous, set them to null

## Features to extract:

### Customer Identification:
- CustomerId: Customer identifier or (null if not specified)

### Demographics:
- Gender: "Male" or "Female"
- Senior_Citizen: 1 for senior citizen, 0 for not senior citizen
- Is_Married: "Yes" or "No"
- Dependents: "Yes" if has children/dependents, "No" if there are no dependents

### Service Information:
- tenure: Number of months with company
- Phone_Service: "Yes" or "No"
- Dual: "Yes" if dual customer, "No" if not
- Internet_Service: "DSL", "Fiber optic", or "No"

### Internet Add-on Services (only relevant if has internet):
- Online_Security: "Yes", "No", or "No internet service"
- Online_Backup: "Yes", "No", or "No internet service"  
- Device_Protection: "Yes", "No", or "No internet service"
- Tech_Support: "Yes", "No", or "No internet service"
- Streaming_TV: "Yes", "No", or "No internet service"
- Streaming_Movies: "Yes", "No", or "No internet service"

### Contract and Billing:
- Contract: "Month-to-month", "One year", or "Two year"
- Paperless_Billing: "Yes" or "No"
- Payment_Method: "Electronic check", "Mailed check", "Bank transfer (automatic)", or "Credit card (automatic)"

### Financial Information:
- Monthly_Charges: Monthly charge amount as number
- Total_Charges: Total charges as number

## LOGICAL CONSISTENCY RULES:
1. If Internet_Service is "No", then all internet add-on services should be "No internet service"
2. If a service is explicitly mentioned as "not having" something, use "No" not null
3. Only use null when the information is completely absent from the text

## OUTPUT FORMAT:
Return ONLY a valid JSON object with this exact structure (no additional text):

{{
  "CustomerId": null,
  "Gender": null,
  "Senior_Citizen": null,
  "Is_Married": null,
  "Dependents": null,
  "tenure": null,
  "Phone_Service": null,
  "Dual": null,
  "Internet_Service": null,
  "Online_Security": null,
  "Online_Backup": null,
  "Device_Protection": null,
  "Tech_Support": null,
  "Streaming_TV": null,
  "Streaming_Movies": null,
  "Contract": null,
  "Paperless_Billing": null,
  "Payment_Method": null,
  "Monthly_Charges": null,
  "Total_Charges": null
}}
Remember: null indicates missing information, while "No" indicates explicitly stated absence of a service.
"""


class CustomerInfoExtractor:
    """
    Enhanced customer information extractor using LLM with structured output validation.
    """

    def __init__(self, model_name: str = LLM_MODEL):
        """
        Initialize the extractor with specified LLM model.

        Parameters:
        -----------
        model_name : str
            Name of the Ollama model to use
        """
        self.llm = ChatOllama(model=model_name)
        self.chat_template = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{customer_paragraph}")]
        )

    def extract_customer_info(self, customer_paragraph: str):
        """
        Extract customer information from a descriptive paragraph.

        Parameters:
        -----------
        customer_paragraph : str
            Text description of the customer

        Returns:
        --------
        result : dict
            Dictionary with extracted customer features
        """
        prompt = self.chat_template.format_messages(customer_paragraph=customer_paragraph)

        response = self.llm.invoke(prompt)
        raw_response = response.content
        try:
            match = re.search(r"\{.*?\}", raw_response, re.DOTALL)
            Customer_data = CustomerData(**json.loads(match.group()))
        except Exception as e:
            logger.error(f"Failed to parse LLM response or validate data: {e}")
            Customer_data = CustomerData()
        return Customer_data


if __name__ == "__main__":
    extractor = CustomerInfoExtractor()

    # Test Case 1: Complete information (original example)
    complete_prompt = """
    Customer 4593 is a male senior citizen who has been with the company for 24 months.  
    He is married with no dependents. He has a phone service and is a dual customer.  
    His internet service is Fiber optic with online security and device protection, but no online backup.  
    He also receives tech support, streams TV and movies.  
    He has a one-year contract with paperless billing enabled, and pays by electronic check.  
    His monthly charges are 89.65 and his total charges so far are 2151.60.
    """

    # Test Case 2: Missing Customer ID and partial information
    missing_id_prompt = """
    A female customer has been with the company for 12 months. She has phone service
    and uses fiber optic internet with streaming TV. She pays monthly charges of 65.50.
    """

    # Test Case 3: Minimal information with ambiguous details
    minimal_prompt = """
    A customer who might be elderly has some kind of internet service and pays around $50 monthly.
    """

    # Test Case 4: No internet service scenario
    no_internet_prompt = """
    Customer 7891 is a young married female with children. She has been with us for 6 months.
    She only has phone service, no internet. She has a month-to-month contract and pays by credit card.
    Her monthly charges are 25.00.
    """

    test_cases = [
        ("Complete Information", complete_prompt),
        ("Missing Customer ID", missing_id_prompt),
        ("Minimal Information", minimal_prompt),
        ("No Internet Service", no_internet_prompt),
    ]

    print("=== Testing Enhanced System Prompt with Missing Data Handling ===\n")

    for test_name, prompt in test_cases:
        print(f"--- {test_name} ---")
        result = extractor.extract_customer_info(prompt)
        print(result.model_dump_json(indent=2))
        print("\n")
