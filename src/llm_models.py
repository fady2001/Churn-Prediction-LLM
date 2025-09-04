from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field, field_validator


class GenderOption(str, Enum):
    """Gender enumeration"""

    MALE = "Male"
    FEMALE = "Female"


class YesNoOption(str, Enum):
    """Yes/No options"""

    YES = "Yes"
    NO = "No"


class YesNoInternetOption(str, Enum):
    """Yes/No/No internet service options"""

    YES = "Yes"
    NO = "No"
    NO_INTERNET_SERVICE = "No internet service"


class InternetService(str, Enum):
    """Internet service provider options"""

    DSL = "DSL"
    FIBER_OPTIC = "Fiber optic"
    NO = "No"


class ContractType(str, Enum):
    """Contract duration options"""

    MONTH_TO_MONTH = "Month-to-month"
    ONE_YEAR = "One year"
    TWO_YEAR = "Two year"


class PaymentMethod(str, Enum):
    """Payment method options"""

    ELECTRONIC_CHECK = "Electronic check"
    MAILED_CHECK = "Mailed check"
    BANK_TRANSFER = "Bank transfer (automatic)"
    CREDIT_CARD = "Credit card (automatic)"


class CustomerData(BaseModel):
    """
    Data model for customer information extracted from LLM output.

    This model validates and structures the JSON response from the LLM
    to ensure type safety and data consistency.
    """

    # Basic customer information
    CustomerId: Optional[str] = Field(None, description="Customer ID")
    Gender: Optional[GenderOption] = Field(None, description="Customer gender")
    Senior_Citizen: Optional[Union[int, str]] = Field(
        None, description="Senior citizen status (0/1 or Yes/No)"
    )
    Is_Married: Optional[YesNoOption] = Field(None, description="Marital status")
    Dependents: Optional[YesNoOption] = Field(None, description="Has dependents")

    # Service tenure
    tenure: Optional[Union[int, str]] = Field(None, description="Months with company")

    # Service information
    Phone_Service: Optional[YesNoOption] = Field(None, description="Has phone service")
    Dual: Optional[YesNoOption] = Field(None, description="Is dual customer")
    Internet_Service: Optional[InternetService] = Field(None, description="Internet service type")

    # Add-on services
    Online_Security: Optional[YesNoInternetOption] = Field(None, description="Has online security")
    Online_Backup: Optional[YesNoInternetOption] = Field(None, description="Has online backup")
    Device_Protection: Optional[YesNoInternetOption] = Field(
        None, description="Has device protection"
    )
    Tech_Support: Optional[YesNoInternetOption] = Field(None, description="Has tech support")
    Streaming_TV: Optional[YesNoInternetOption] = Field(None, description="Has streaming TV")
    Streaming_Movies: Optional[YesNoInternetOption] = Field(
        None, description="Has streaming movies"
    )

    # Contract and billing
    Contract: Optional[ContractType] = Field(None, description="Contract type")
    Paperless_Billing: Optional[YesNoOption] = Field(None, description="Uses paperless billing")
    Payment_Method: Optional[PaymentMethod] = Field(None, description="Payment method")

    # Financial information
    Monthly_Charges: Optional[Union[float, str]] = Field(None, description="Monthly charges")
    Total_Charges: Optional[Union[float, str]] = Field(None, description="Total charges")

    @field_validator("CustomerId")
    def validate_customer_id(cls, v):
        """Handle CustomerId validation"""
        if v is None or v == "" or str(v).strip() == "":
            return None
        return str(v).strip()

    @field_validator("Gender", mode="before")
    def validate_gender(cls, v):
        """Handle gender validation with flexible input"""
        if v is None or v == "":
            return None
        if isinstance(v, str):
            v = v.strip().lower()
            if v in ["male", "m"]:
                return GenderOption.MALE
            elif v in ["female", "f"]:
                return GenderOption.FEMALE
        return v

    @field_validator("Senior_Citizen")
    def validate_senior_citizen(cls, v):
        """Convert senior citizen values to standard format"""
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip()
            if v.lower() in ["yes", "true", "1"]:
                return 1
            elif v.lower() in ["no", "false", "0"]:
                return 0
            elif v == "":
                return None
        return v

    @field_validator("tenure")
    def validate_tenure(cls, v):
        """Convert tenure to integer if possible"""
        if v is None or v == "":
            return None
        if isinstance(v, str):
            try:
                return int(v.strip())
            except ValueError:
                return v
        return v

    @field_validator("Monthly_Charges", "Total_Charges")
    def validate_charges(cls, v):
        """Convert charge values to float if possible"""
        if v is None or v == "":
            return None
        if isinstance(v, str):
            try:
                return float(v.strip())
            except ValueError:
                return v
        return v

    @field_validator(
        "Online_Security",
        "Online_Backup",
        "Device_Protection",
        "Tech_Support",
        "Streaming_TV",
        "Streaming_Movies",
    )
    def validate_internet_dependent_services(cls, v, info):
        """Validate that internet-dependent services are consistent with internet service"""
        # Get the Internet_Service value from the context if available
        if hasattr(info, "data") and info.data:
            internet_service = info.data.get("Internet_Service")

            # If no internet service, all internet-dependent services should be "No internet service"
            if internet_service == InternetService.NO and v is not None:
                if v not in [YesNoInternetOption.NO_INTERNET_SERVICE]:
                    return YesNoInternetOption.NO_INTERNET_SERVICE

        return v

    @field_validator("Payment_Method", mode="before")
    def validate_payment_method(cls, v):
        """Handle Payment_Method validation with flexible input"""
        if v is None or v == "":
            return None
        if isinstance(v, str):
            v = v.strip().lower()
            if v in ["electronic check", "electronic", "e-check"]:
                return PaymentMethod.ELECTRONIC_CHECK
            elif v in ["mailed check", "mailed", "mail"]:
                return PaymentMethod.MAILED_CHECK
            elif v in ["bank transfer (automatic)", "bank transfer", "bank"]:
                return PaymentMethod.BANK_TRANSFER
            elif v in ["credit card (automatic)", "credit card", "credit"]:
                return PaymentMethod.CREDIT_CARD
        return v

    @field_validator("Contract", mode="before")
    def validate_contract(cls, v):
        """Handle Contract validation with flexible input"""
        if v is None or v == "":
            return None
        if isinstance(v, str):
            v = v.strip().lower()
            if v in ["month-to-month", "month to month", "monthly"]:
                return ContractType.MONTH_TO_MONTH
            elif v in ["one year", "1 year", "annual"]:
                return ContractType.ONE_YEAR
            elif v in ["two year", "2 year", "biennial"]:
                return ContractType.TWO_YEAR
        return v


if __name__ == "__main__":
    # Example usage and testing
    sample_llm_response = {
        "CustomerId": "4593",
        "Gender": "Male",
        "Senior_Citizen": "1",
        "Is_Married": "Yes",
        "Dependents": "No",
        "tenure": "24",
        "Phone_Service": "Yes",
        "Dual": "Yes",
        "Internet_Service": "Fiber optic",
        "Online_Security": "Yes",
        "Online_Backup": "No",
        "Device_Protection": "Yes",
        "Tech_Support": "Yes",
        "Streaming_TV": "Yes",
        "Streaming_Movies": "Yes",
        "Contract": "One year",
        "Paperless_Billing": "Yes",
        "Payment_Method": "e-check",
        "Monthly_Charges": "89.65",
        "Total_Charges": "2151.60",
    }

    # Test the model by creating a CustomerData instance
    try:
        customer_data = CustomerData(**sample_llm_response)
        print("CustomerData validation successful!")
        print(f"Customer ID: {customer_data.CustomerId}")
        print(f"Gender: {customer_data.Gender}")
        print(f"Monthly Charges: {customer_data.Monthly_Charges}")
        print(f"Internet Service: {customer_data.Internet_Service}")
        print("\nFull data:")
        print(customer_data.model_dump_json(indent=2))
    except Exception as e:
        print(f"Validation failed: {e}")
