import os
from pathlib import Path

_ENV_PATH = Path(".env")


def _load_env():
    if _ENV_PATH.exists():
        for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"\'')


def reload_config():
    """Re-read .env and update module-level variables. Called after /api/setup."""
    _load_env()
    global AI_NAME, NVIDIA_API_KEY, BASE_URL, MODEL
    global MAX_ITERATIONS, MAX_TOKENS, LLM_TEMPERATURE, SERVER_HOST, SERVER_PORT
    AI_NAME        = os.getenv("AI_NAME", "JARVIS")
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
    BASE_URL       = os.getenv("BASE_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
    MODEL          = os.getenv("MODEL", "mistralai/mistral-small-4-119b-2603")
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "30"))
    MAX_TOKENS     = int(os.getenv("MAX_TOKENS", "4096"))
    LLM_TEMPERATURE= float(os.getenv("LLM_TEMPERATURE", "0.2"))
    SERVER_HOST    = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT    = int(os.getenv("SERVER_PORT", "8000"))


# Initial load
_load_env()
AI_NAME        = os.getenv("AI_NAME", "JARVIS")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
BASE_URL       = os.getenv("BASE_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
MODEL          = os.getenv("MODEL", "mistralai/mistral-small-4-119b-2603")
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "30"))
MAX_TOKENS     = int(os.getenv("MAX_TOKENS", "4096"))
LLM_TEMPERATURE= float(os.getenv("LLM_TEMPERATURE", "0.2"))
SERVER_HOST    = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT    = int(os.getenv("SERVER_PORT", "8000"))
