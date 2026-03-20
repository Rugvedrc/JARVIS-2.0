import os
from pathlib import Path

_env = Path(".env")
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"\''))

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "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")
BASE_URL       = os.getenv("BASE_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
MODEL          = os.getenv("MODEL", "mistralai/mistral-small-4-119b-2603")
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "30"))
MAX_TOKENS     = int(os.getenv("MAX_TOKENS", "4096"))
LLM_TEMPERATURE= float(os.getenv("LLM_TEMPERATURE", "0.2"))
SERVER_HOST    = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT    = int(os.getenv("SERVER_PORT", "8000"))
