"""Verify all imports work."""

def test_core_config():
    from core.config import TEXT_CFG, VISION_CFG, EMBED_CFG
    assert TEXT_CFG["model"]
    assert VISION_CFG["model"]
    assert EMBED_CFG["model"]

def test_model_router():
    from models.model_router import ModelRouter

def test_llm():
    from models.llm import LLM

def test_memory():
    from memory import MemoryStore

def test_streamer():
    from inference.streamer import stream_chat