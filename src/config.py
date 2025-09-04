from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

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

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
