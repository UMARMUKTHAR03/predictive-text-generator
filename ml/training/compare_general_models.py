import os
import time
import json
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CANDIDATES = [
    {
        "name": "distilgpt2",
        "model_id": "distilbert/distilgpt2",
        "description": "Distilled version of GPT-2 (82M params, 6 layers, 768 hidden, 12 heads)",
        "license": "Apache 2.0"
    },
    {
        "name": "pythia-70m",
        "model_id": "EleutherAI/pythia-70m",
        "description": "Pythia 70M autoregressive causal language model (70M params)",
        "license": "Apache 2.0"
    }
]

TEST_PROMPTS = [
    "The weather is very",
    "I am writing this email to",
    "The future of artificial intelligence",
    "In order to succeed in life you need to",
    "She opened the mysterious door and found"
]

def benchmark_model(model_info):
    model_id = model_info["model_id"]
    name = model_info["name"]
    print(f"\n==========================================")
    print(f"Benchmarking Candidate: {name} ({model_id})")
    print(f"==========================================")

    # Initial RAM
    process = psutil.Process()
    ram_before = process.memory_info().rss / (1024 * 1024)

    # Load Tokenizer & Model
    t0_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    t1_load = time.perf_counter()

    load_time_sec = t1_load - t0_load
    ram_after = process.memory_info().rss / (1024 * 1024)
    model_ram_mb = ram_after - ram_before
    param_count = sum(p.numel() for p in model.parameters())

    print(f"Parameters: {param_count:,}")
    print(f"Load time: {load_time_sec:.2f} s")
    print(f"Incremental RAM: {model_ram_mb:.1f} MB (Total: {ram_after:.1f} MB)")

    # Warmup
    inputs = tokenizer("Hello world", return_tensors="pt")
    with torch.no_grad():
        _ = model(**inputs)

    # Benchmark Generations & Next-word candidates
    results = []
    latencies = []

    for prompt in TEST_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        # Next-token logits
        t0_inf = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
        t1_inf = time.perf_counter()
        latencies.append((t1_inf - t0_inf) * 1000)

        # Top 5 candidate tokens
        top_probs, top_indices = probs.topk(10, dim=-1)
        candidates = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            tok_str = tokenizer.decode([idx.item()])
            candidates.append({"token": tok_str, "prob": round(prob.item(), 4)})

        # Multi-token generation (20 tokens, greedy & sampling)
        with torch.no_grad():
            gen_ids_greedy = model.generate(
                input_ids,
                max_new_tokens=15,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            greedy_text = tokenizer.decode(gen_ids_greedy[0], skip_special_tokens=True)

            gen_ids_sampled = model.generate(
                input_ids,
                max_new_tokens=15,
                do_sample=True,
                temperature=0.8,
                top_k=20,
                pad_token_id=tokenizer.eos_token_id
            )
            sampled_text = tokenizer.decode(gen_ids_sampled[0], skip_special_tokens=True)

        results.append({
            "prompt": prompt,
            "top_candidates": candidates[:5],
            "greedy_continuation": greedy_text,
            "sampled_continuation": sampled_text
        })
        print(f"\nPrompt: '{prompt}'")
        print(f"  Top-1: '{candidates[0]['token']}' ({candidates[0]['prob']*100:.1f}%)")
        print(f"  Greedy: '{greedy_text}'")

    avg_latency_ms = sum(latencies) / len(latencies)

    # Clean up memory
    del model
    del tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "name": name,
        "model_id": model_id,
        "parameters": param_count,
        "load_time_sec": round(load_time_sec, 2),
        "model_ram_mb": round(model_ram_mb, 1),
        "avg_next_token_latency_ms": round(avg_latency_ms, 2),
        "results": results
    }

if __name__ == "__main__":
    benchmark_data = {}
    for c in CANDIDATES:
        res = benchmark_model(c)
        benchmark_data[c["name"]] = res

    os.makedirs("ml/artifacts", exist_ok=True)
    out_file = "ml/artifacts/general_model_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"\nSaved general model comparison to {out_file}!")

