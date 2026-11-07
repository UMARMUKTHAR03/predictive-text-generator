import json
import time
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from backend.app.services.model_loader import ModelManager
from backend.app.services.prediction_service import TextGenerationService
from backend.app.services.general_service import GeneralLanguageModelService

# Test benchmark dataset across diverse general writing categories
GENERAL_BENCHMARK_PROMPTS = [
    {"category": "Everyday Conversation", "prompt": "The weather is very"},
    {"category": "Professional Email", "prompt": "I am writing this email to"},
    {"category": "Technology & AI", "prompt": "The future of artificial intelligence"},
    {"category": "Philosophy & Success", "prompt": "In order to succeed in life you need to"},
    {"category": "Creative Narrative", "prompt": "She opened the mysterious door and found"},
    {"category": "Education & Learning", "prompt": "The most effective way to learn a new language is to"},
    {"category": "Business Communication", "prompt": "Please let me know if you have any questions regarding"},
    {"category": "Daily Schedule", "prompt": "Tomorrow morning I am planning to"}
]

def run_general_evaluation():
    print("Running General English Evaluation on DistilGPT-2...")
    gen_svc = GeneralLanguageModelService()
    gen_svc.load()

    evaluated_samples = []
    latencies = []

    for item in GENERAL_BENCHMARK_PROMPTS:
        cat = item["category"]
        p = item["prompt"]

        cands, tokens, pred_lat = gen_svc.predict_next_words(p, top_k=5)
        gen_text, completed, tokens_gen, gen_lat = gen_svc.generate_continuation(
            prompt=p, max_tokens=15, strategy="greedy"
        )
        latencies.append(pred_lat)

        evaluated_samples.append({
            "category": cat,
            "prompt": p,
            "top_candidates": [{"word": c.word, "probability": c.probability} for c in cands],
            "greedy_continuation": gen_text,
            "completed_sentence": completed,
            "latency_ms": pred_lat
        })
        print(f"[{cat}] '{p}' -> Top: '{cands[0].word}' ({(cands[0].probability*100):.1f}%) | '{gen_text}'")

    avg_latency = sum(latencies) / len(latencies)
    eval_report = {
        "model_name": "DistilGPT-2",
        "publisher": "Hugging Face / DistilBERT team",
        "license": "Apache 2.0",
        "parameters": 81912576,
        "vocab_size": 50257,
        "context_length": 1024,
        "evaluation_domain": "General English open-domain sentences & communication",
        "test_prompts_count": len(GENERAL_BENCHMARK_PROMPTS),
        "avg_next_word_latency_ms": round(avg_latency, 2),
        "evaluated_samples": evaluated_samples
    }

    os.makedirs("ml/artifacts", exist_ok=True)
    out_path = "ml/artifacts/general_evaluation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    print(f"\nSaved general evaluation report to {out_path}!")
    return eval_report

if __name__ == "__main__":
    run_general_evaluation()

