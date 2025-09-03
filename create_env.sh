#!/bin/bash

# Script to create a .env file for the Churn Prediction LLM project
# Usage: ./create_env.sh

ENV_FILE=".env"

echo "Creating .env file..."

# Create .env file with common environment variables
cat > "$ENV_FILE" << EOF
# Environment variables go here, can be read by `python-dotenv` package:
#
#   `src/script.py`
#   ----------------------------------------------------------------
#    import dotenv
#
#    project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
#    dotenv_path = os.path.join(project_dir, '.env')
#    dotenv.load_dotenv(dotenv_path)
#   ----------------------------------------------------------------
#
# DO NOT ADD THIS FILE TO VERSION CONTROL!
EOF