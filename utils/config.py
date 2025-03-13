import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model configurations
COST_ESTIMATOR_MODEL = os.getenv("COST_ESTIMATOR_MODEL", "gpt-4-turbo")

# Validate that the required API keys are set
def validate_api_keys():
    missing_keys = []
    if not OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY")
    if not ANTHROPIC_API_KEY:
        missing_keys.append("ANTHROPIC_API_KEY")
    if not GOOGLE_API_KEY:
        missing_keys.append("GOOGLE_API_KEY")
    
    if missing_keys:
        raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
    
    return True
