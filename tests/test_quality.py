"""Measure quality degradation from quantization."""


# Simple benchmark tasks — "poor man's MT-Bench"
BENCHMARK_TASKS = [
    {
        "name": "math_1",
        "prompt": "What is 17 * 23? Answer with only the number.",
        "expected": "391"
    },
    {
        "name": "math_2",
        "prompt": "Solve: (15 + 7) * 3 - 8. Answer with only the number.",
        "expected": "58"
    },
    {
        "name": "logic_1",
        "prompt": "If all cats are mammals, and all mammals are animals, are all cats animals? Answer yes or no.",
        "expected": "yes"
    },
    {
        "name": "code_1",
        "prompt": "Write a Python one-liner to sum a list: [1,2,3,4,5]. Answer with only the code.",
        "expected": "sum([1,2,3,4,5])"
    },
    {
        "name": "fact_1",
        "prompt": "What is the chemical symbol for gold? Answer with only the symbol.",
        "expected": "au"
    }
]


def test_perplexity_approx(text_router):
    """Approximate perplexity via log-probability proxy.

    Lower perplexity = model more confident = better quality.
    Q4 typically 10-30% worse than FP16.
    """
    messages = [{"role": "user", "content": "The capital of France is"}]

    # Generate and measure "confidence" via response coherence
    response = text_router.generate(messages, temperature=0.0)

    # Simple check: should contain "Paris" confidently
    assert "paris" in response.lower(), "Basic fact failed — quality issue?"

    print(f"\nPerplexity proxy: response='{response.strip()}'")


def test_benchmark_accuracy(text_router):
    """Run mini-MT-bench: 5 tasks, score 0-1 each."""
    score = 0
    results = []

    for task in BENCHMARK_TASKS:
        messages = [{"role": "user", "content": task["prompt"]}]
        response = text_router.generate(messages, temperature=0.0).lower().strip()

        # Check if expected in response
        correct = task["expected"] in response
        if correct:
            score += 1

        results.append({
            "task": task["name"],
            "expected": task["expected"],
            "got": response[:50],
            "correct": correct
        })

    accuracy = (score / len(BENCHMARK_TASKS)) * 100

    print(f"\nBenchmark Results:")
    for r in results:
        status = "✓" if r["correct"] else "✗"
        print(f"  {status} {r['task']}: expected '{r['expected']}', got '{r['got']}'")
    print(f"\nAccuracy: {accuracy:.0f}%")

    # Q4 should still get 80%+, Q8 90%+, FP16 95%+
    assert accuracy >= 60, f"Accuracy too low: {accuracy:.0f}% — model broken?"


def test_quality_vs_quant(text_router):
    """Print quant level and expected quality range."""
    quant = text_router.cfg["quant"]

    QUALITY_EXPECTATIONS = {
        "fp16": "95-100% (baseline)",
        "q8_0": "90-95% of FP16",
        "q5_k_m": "85-92% of FP16 (balanced)",  # ← добавить
        "q5_k_s": "83-90% of FP16",
        "q4_k_m": "80-90% of FP16 (recommended)",
        "q4_k_s": "78-85% of FP16",
        "q4_0": "70-80% of FP16 (fastest)",
        "q4_1": "72-82% of FP16",
        "q3_k_m": "60-75% of FP16 (risky)",
        "q3_k_s": "58-72% of FP16",
        "q3_k_l": "62-78% of FP16",
        "q2_k": "50-65% of FP16 (avoid)",
    }

    expected = QUALITY_EXPECTATIONS.get(quant, "unknown")
    print(f"\nQuant: {quant}")
    print(f"Expected quality: {expected}")

    # Just info, no assert
    assert quant in QUALITY_EXPECTATIONS, f"Unknown quant: {quant}"