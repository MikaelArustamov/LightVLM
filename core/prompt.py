"""Builds chat message arrays for LiteLLM."""


class PromptBuilder:
    def __init__(self):
        self.system = {
            "role": "system",
            "content": "You are LightVLM, a helpful AI assistant.",
        }

    def build(self, text: str) -> list[dict]:
        return [self.system, {"role": "user", "content": text}]


