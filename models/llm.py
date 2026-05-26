"""LLM interface — unified through ModelRouter."""

import base64
from typing import List, Dict, Any, Generator

from core.config import TEXT_CFG, VISION_CFG
from models.model_router import ModelRouter


class LLM:
    """Text, vision, streaming — all via ModelRouter."""

    def __init__(self):
        self.text = ModelRouter(TEXT_CFG)
        self.vision = ModelRouter(VISION_CFG)

    # ===== TEXT =====

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        return self.text.generate(messages, **kwargs)

    def chat_stream(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        yield from self.text.generate_stream(messages, **kwargs)

    # ===== VISION =====

    def vision_chat(self, image_b64: str, prompt: str, **kwargs) -> str:
        messages = [{
            "role": "user",
            "content": prompt,
            "images": [image_b64]
        }]
        return self.vision.generate(messages, **kwargs)

    # ===== IMAGE GENERATION DETECTION =====

    @staticmethod
    def is_image_request(text: str) -> tuple[bool, str]:
        triggers = ["draw", "generate image", "create image", "make a picture",
                    "paint", "sketch", "illustration", "generate a photo",
                    "нарисуй", "сгенерируй", "создай изображение"]
        t = text.lower()
        for trigger in triggers:
            if trigger in t:
                idx = t.find(trigger) + len(trigger)
                prompt = text[idx:].strip(" ,.:;!?")
                return True, prompt or text
        return False, text

    # ===== SYSTEM =====

    @staticmethod
    def system_prompt() -> str:
        return """You are LightVLM, a helpful AI assistant. You can:
- Answer questions and chat
- Analyze uploaded images
- Generate images when asked (draw, create, etc.)
Be concise but thorough."""

    @staticmethod
    def vision_prompt() -> str:
        return "Describe this image in detail. What do you see?"