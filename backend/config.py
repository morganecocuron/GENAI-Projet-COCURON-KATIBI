"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# Council = sur TON PC (au début) donc localhost.
# En démo binôme: Council reste sur ton PC -> localhost côté orchestrator si orchestrator est sur ton PC.
COUNCIL_BASE_URL = os.getenv("COUNCIL_BASE_URL", "http://localhost:11434")

# Chairman = au début localhost, puis en démo: http://IP_PC_COPINE:11434
CHAIRMAN_BASE_URL = os.getenv("CHAIRMAN_BASE_URL", "http://localhost:11434")

# Noms des modèles OLLAMA (pas OpenRouter)
COUNCIL_MODELS = [
    os.getenv("COUNCIL_MODEL_1", "llama3"),
    os.getenv("COUNCIL_MODEL_2", "mistral"),
    os.getenv("COUNCIL_MODEL_3", "qwen2.5"),
]

CHAIRMAN_MODEL = os.getenv("CHAIRMAN_MODEL", "llama3")

# Data directory for conversation storage
DATA_DIR = "data/conversations"

# Concurrency / stability
MAX_PARALLEL_REQUESTS = int(os.getenv("MAX_PARALLEL_REQUESTS", "1"))  # 1 = safe
RETRY_COUNT = int(os.getenv("RETRY_COUNT", "1"))  # 0/1/2
RETRY_SLEEP_SECONDS = float(os.getenv("RETRY_SLEEP_SECONDS", "0.8"))
