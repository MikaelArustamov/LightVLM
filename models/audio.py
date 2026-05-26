"""Audio pipeline: faster-whisper локально. Без OpenAI, без Ollama для STT."""

import io
import tempfile
import os

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


class AudioPipeline:
    """Локальный Whisper. Не зависит от Ollama/OpenAI."""

    def __init__(self, model_size: str = "base"):
        if not WHISPER_AVAILABLE:
            raise RuntimeError("pip install faster-whisper")

        print(f"[Audio] Loading Whisper {model_size} locally...")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("[Audio] Whisper ready")

    def transcribe(self, audio_bytes: bytes, format: str = "webm") -> str:
        """Speech-to-text локально."""
        suffix = f".{format}" if format else ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            segments, info = self.model.transcribe(tmp_path, language=None, task="transcribe")
            text = " ".join([segment.text for segment in segments]).strip()
            print(f"[Audio] {info.language}: {text[:80]}...")
            return text
        finally:
            os.unlink(tmp_path)