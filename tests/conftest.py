"""pytest fixtures — shared model instances."""

import os
import pytest
from pathlib import Path

# Load .env BEFORE any imports
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path)
    except ImportError:
        # Fallback: manual parser if python-dotenv not installed
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                os.environ.setdefault(key, value)

# Now safe to import
from models.model_router import ModelRouter
from models.llm import LLM
from core.config import TEXT_CFG, VISION_CFG, EMBED_CFG


@pytest.fixture(scope="session")
def text_router():
    """Text model from .env."""
    return ModelRouter(TEXT_CFG)


@pytest.fixture(scope="session")
def vision_router():
    """Vision model from .env."""
    return ModelRouter(VISION_CFG)


@pytest.fixture(scope="session")
def embed_router():
    """Embed model from .env."""
    return ModelRouter(EMBED_CFG)


@pytest.fixture(scope="session")
def llm():
    """Full LLM interface."""
    return LLM()


@pytest.fixture
def sample_prompts():
    """Standard test prompts."""
    return [
        "Explain quantum computing in simple terms",
        "Write a Python function to reverse a string",
        "What is the capital of France?",
        "Solve: 15 * 24 + 7",
        "Summarize the theory of relativity",
    ]