"""Measure generation speed: TTFT, tok/s, % vs theoretical max."""

import time
import pytest
from inference.streamer import stream_chat


def count_tokens(text: str) -> int:
    """Rough token count (1 token ≈ 0.75 words for English)."""
    return int(len(text.split()) / 0.75)


def test_time_to_first_token(text_router, sample_prompts):
    """TTFT — how fast user sees first response."""
    messages = [{"role": "user", "content": sample_prompts[0]}]

    t0 = time.perf_counter()
    gen = text_router.generate_stream(messages)
    first = next(gen)
    ttft_ms = (time.perf_counter() - t0) * 1000

    print(f"\nTTFT: {ttft_ms:.1f}ms")
    assert ttft_ms < 5000, "TTFT too slow (>5s)"


def test_tokens_per_second(text_router, sample_prompts):
    """Throughput: tok/s for sustained generation."""
    messages = [{"role": "user", "content": sample_prompts[0]}]

    t0 = time.perf_counter()
    full = ""
    for chunk in text_router.generate_stream(messages):
        full += chunk

    elapsed = time.perf_counter() - t0
    tokens = count_tokens(full)
    tok_per_sec = tokens / elapsed if elapsed > 0 else 0

    print(f"\nGenerated: {tokens} tokens in {elapsed:.1f}s")
    print(f"Speed: {tok_per_sec:.1f} tok/s")

    # Baseline for llama3.1:8b on CPU ≈ 5-15 tok/s depending on quant
    # Q4 should be ~20-30% faster than Q8, ~50% faster than FP16
    assert tok_per_sec > 1.0, "Too slow, check if model loaded correctly"


def test_speed_comparison(text_router, sample_prompts):
    """Compare current config vs theoretical speeds in %."""
    messages = [{"role": "user", "content": "Count from 1 to 50"}]

    t0 = time.perf_counter()
    full = ""
    for chunk in text_router.generate_stream(messages):
        full += chunk

    elapsed = time.perf_counter() - t0
    tokens = count_tokens(full)
    actual_tok_s = tokens / elapsed

    # Theoretical baselines for llama3.1:8b on Mac CPU (M1/M2)
    # These are approximate
    BASELINE = {
        "fp16": 3.0,  # ~3 tok/s (heavy, memory-bound)
        "q8_0": 5.0,  # ~5 tok/s
        "q4_k_m": 8.0,  # ~8 tok/s (sweet spot)
        "q4_0": 10.0,  # ~10 tok/s (fastest, slightly worse quality)
    }

    current_quant = text_router.cfg["quant"]
    baseline = BASELINE.get(current_quant, 5.0)

    # Calculate % of theoretical max
    efficiency = (actual_tok_s / baseline) * 100

    print(f"\nQuant: {current_quant}")
    print(f"Actual: {actual_tok_s:.1f} tok/s")
    print(f"Baseline: {baseline:.1f} tok/s")
    print(f"Efficiency: {efficiency:.0f}%")

    # Should achieve at least 50% of theoretical
    assert efficiency > 50, f"Performance too low: {efficiency:.0f}%"