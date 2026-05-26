"""Universal model router — powered by LiteLLM.

Provides a unified interface for chat completion, streaming, and embeddings
across multiple local and remote backends like Ollama, Llama.cpp, and HuggingFace.
"""

import os
from typing import List, Dict, Any, Generator
import litellm

# Disable telemetry data collection for privacy and performance reasons
litellm.telemetry = False


class ModelRouter:
    """Automates backend detection and routes payloads to targeted LLM providers.

    This class abstracts the underlying API differences between Ollama, native
    Llama.cpp GGUF execution, and HuggingFace transformer pipelines by utilizing LiteLLM.
    """

    def __init__(self, cfg: Dict[str, Any]):
        """Initializes the router configuration and resolves the destination model string.

        Args:
            cfg (Dict[str, Any]): Dictionary containing configuration keys:
                - "model" (str): The raw model name, path, or identifier.
                - "backend" (str, optional): Explicit backend override ("ollama", "llama_cpp", etc.).
                - "ctx" (int, optional): Context window size limit. Defaults to 8192.
        """
        self.cfg = cfg
        self.raw_model = cfg.get("model", "")
        self.backend = cfg.get("backend", "auto")
        self.ctx = cfg.get("ctx", 8192)

        # Parse the raw configuration into a LiteLLM-compatible format
        self.model_string = self._resolve_model_string()

    def _resolve_model_string(self) -> str:
        """Translates custom configuration heuristics into strict LiteLLM provider prefixes.

        Returns:
            str: Validated connection string formatted as 'provider/model_name'.
        """
        m = self.raw_model
        b = self.backend

        # 1. Handle explicit backend specifications
        if b == "ollama":
            return f"ollama/{m.replace('ollama/', '')}"
        if b == "llama_cpp":
            return f"llama_cpp/{m}"

        # FastEmbed is not natively supported by LiteLLM.
        # Rerouting to HuggingFace pipeline as a local execution alternative.
        if b == "fastembed" or m.startswith("fastembed/"):
            clean_name = m.replace("fastembed/", "")
            return f"huggingface/{clean_name}"

        # 2. Automated fallback routing based on string patterns or file system checks
        if m.startswith("ollama/"):
            return m
        if m.endswith(".gguf") or os.path.isfile(m):
            return f"llama_cpp/{m}"

        # Fallback default when provider cannot be confidently inferred
        return f"ollama/{m}"

    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Executes a synchronous chat completion request.

        Args:
            messages (List[Dict[str, Any]]): Standardized OpenAI-format message history.
            **kwargs: Extra parameters passed directly to the model (e.g., temperature, max_tokens).

        Returns:
            str: Generated text content from the assistant response.
        """
        response = litellm.completion(
            model=self.model_string,
            messages=messages,
            num_ctx=self.ctx,  # Propagated down to Ollama/Llama.cpp inference options
            **kwargs
        )
        return response.choices[0].message.content

    def generate_stream(self, messages: List[Dict[str, Any]], **kwargs) -> Generator[str, None, None]:
        """Executes an asynchronous-like streaming chat completion request via generators.

        Args:
            messages (List[Dict[str, Any]]): Standardized OpenAI-format message history.
            **kwargs: Extra runtime parameters for inference control.

        Yields:
            Generator[str, None, None]: Text tokens as they are emitted by the engine.
        """
        response = litellm.completion(
            model=self.model_string,
            messages=messages,
            stream=True,
            num_ctx=self.ctx,
            **kwargs
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                yield delta.content

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generates dense vector embeddings for text arrays.

        Args:
            texts (List[str]): Input documents or queries to be vectorized.

        Returns:
            List[List[float]]: Extracted pure multi-dimensional float arrays,
                               guaranteed to be structured for direct database ingestion.
        """
        response = litellm.embedding(
            model=self.model_string,
            input=texts
        )

        # Iterates over the raw API response schema to isolate and extract nested vectors
        return [item["embedding"] for item in response.data]
