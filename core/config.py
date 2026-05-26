"""Read model config from .env — users change env, not code."""

import os
import re
from typing import Dict, Any
from dotenv import load_dotenv


env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=env_path)

def parse_ollama_tag(model: str) -> tuple[str, str | None]:
    """Extract quant from ollama tag: ollama/llama3.1:8b-q4_0 → (model, q4_0)"""
    if not model.startswith("ollama/"):
        return model, None

    match = re.match(r"(ollama/[^:]+):(.+)", model)
    if not match:
        return model, None

    base, tag = match.groups()
    quant_suffixes = [
        "q2_k", "q3_k_s", "q3_k_m", "q3_k_l",
        "q4_0", "q4_1", "q4_k", "q4_k_m", "q4_k_s",
        "q5_0", "q5_1", "q5_k", "q5_k_m", "q5_k_s",
        "q6_k", "q8_0", "fp16"
    ]

    for q in quant_suffixes:
        if tag.endswith(f"-{q}") or tag == q:
            return f"{base}:{tag}", q

    return f"{base}:{tag}", None


def load_model_config(prefix: str) -> Dict[str, Any]:
    """Load config from environment variables."""
    raw_model = os.getenv(f"{prefix}_MODEL", "")
    if not raw_model:
        raise ValueError(f"{prefix}_MODEL not set in .env")

    model, auto_quant = parse_ollama_tag(raw_model)

    return {
        "model": model,
        "quant": auto_quant or os.getenv(f"{prefix}_QUANT", "q4_k_m").lower(),
        "ctx": int(os.getenv(f"{prefix}_CTX", "8192")),
        "backend": os.getenv(f"{prefix}_BACKEND", "auto").lower(),
    }


# Global configs
TEXT_CFG = load_model_config("TEXT")
VISION_CFG = load_model_config("VISION")
EMBED_CFG = load_model_config("EMBED")