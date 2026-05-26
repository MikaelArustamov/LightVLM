"""SSE streaming via ModelRouter."""

import json
from typing import Generator, List, Dict

from models.llm import LLM


def stream_chat(llm: LLM, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
    """SSE format for FastAPI StreamingResponse."""
    full = ""
    for chunk in llm.chat_stream(messages, **kwargs):
        full += chunk
        yield f"data: {json.dumps({'token': full})}\n\n"
    yield "data: [DONE]\n\n"


def stream_image(llm: LLM, prompt: str, image_b64: str) -> Generator[str, None, None]:
    """Stream image generation result."""
    yield f"data: {json.dumps({'token': f'Generating: {prompt}...', 'generated': True})}\n\n"
    # Actual generation happens in api.py, this is just for SSE format
    yield "data: [DONE]\n\n"