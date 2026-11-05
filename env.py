import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
dotenv_path = Path(__file__).parent / ".env"

if dotenv_path.exists():
    load_dotenv(dotenv_path, override=True)

DATA_DIR = os.environ.get("DATA_DIR", "./data")
os.makedirs(DATA_DIR, exist_ok=True)
#os.environ["HF_HOME"] = DATA_DIR

HF_REPO_RESULTS = os.environ.get("HF_REPO_RESULTS", "eduagarcia/tokenizer_benchmark_results")
HF_REPO_BENCHMARK = os.environ.get("HF_REPO_BENCHMARK", "eduagarcia/multilingual_tokenizer_benchmark")
LIMIT_MODELS_SHOW = int(os.environ.get("LIMIT_MODELS_SHOW")) if os.environ.get("LIMIT_MODELS_SHOW") is not None else None

# Relative to DATA_DIR, EX: DATA_DIR/DEFAULT_BENCHMARK_PATH
DEFAULT_BENCHMARK_PATH = os.environ.get("DEFAULT_BENCHMARK_PATH", "benchmark")
DEFAULT_RESULTS_PATH = os.environ.get("DEFAULT_RESULTS_PATH", "results")

# Relative to DEFAULT_RESULTS_PATH, ex: DATA_DIR/DEFAULT_RESULTS_PATH/DEFAULT_MODEL_EVALUATIONS_PATH
DEFAULT_MODEL_EVALUATIONS_PATH = os.environ.get("DEFAULT_MODEL_EVALUATIONS_PATH", "evaluations")
DEFAULT_TOKENIZER_METADATA_PATH = os.environ.get("DEFAULT_TOKENIZER_METADATA_PATH", "tokenizer_metadata")

ENABLE_LOAD_CACHE = str(os.environ.get("ENABLE_LOAD_CACHE", "true")).lower() in ["true", "1", "yes", "y"]

APP_VERSION = os.environ.get("APP_VERSION", "1.1.0")