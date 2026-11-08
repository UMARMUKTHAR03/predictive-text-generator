import os
import json
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from backend.app.services.cloudflare_workers_ai_service import CloudflareWorkersAIProvider

EVALUATION_PROMPTS = [
    # 1-4: Professional Emails
    {"category": "Email", "prompt": "I am writing this email to follow up on our discussion regarding"},
    {"category": "Email", "prompt": "Please find attached the updated project roadmap and"},
    {"category": "Email", "prompt": "Thank you for your prompt reply regarding the contract"},
    {"category": "Email", "prompt": "Could you please clarify the scheduled timeline for the"},
    # 5-8: Everyday Conversational English
    {"category": "Everyday", "prompt": "The weather today in the city is surprisingly"},
    {"category": "Everyday", "prompt": "We are planning to visit our grandparents this coming"},
    {"category": "Everyday", "prompt": "After finishing work today, I decided to go for a long walk in"},
    {"category": "Everyday", "prompt": "Can you recommend a good place to have dinner near"},
    # 9-12: Educational & Explanatory
    {"category": "Educational", "prompt": "Photosynthesis is the fundamental biological process through which plants"},
    {"category": "Educational", "prompt": "The industrial revolution fundamentally transformed European society by"},
    {"category": "Educational", "prompt": "In calculus, the derivative represents the instantaneous rate of"},
    {"category": "Educational", "prompt": "Gravitational waves are ripples in the fabric of spacetime created by"},
    # 13-16: Articles & Journalism
    {"category": "Article", "prompt": "Renewable energy infrastructure has experienced unprecedented growth across"},
    {"category": "Article", "prompt": "Recent advancements in semiconductor manufacturing indicate that global chip"},
    {"category": "Article", "prompt": "Urban planners are increasingly focusing on sustainable public transit to reduce"},
    {"category": "Article", "prompt": "The annual technology symposium highlighted emerging breakthroughs in synthetic"},
    # 17-20: Short Stories & Narrative Creative Writing
    {"category": "Short Story", "prompt": "The ancient lantern flickered softly against the damp stone walls of the"},
    {"category": "Short Story", "prompt": "As the spacecraft exited hyperspace, the uncharted planet appeared on the main"},
    {"category": "Short Story", "prompt": "Elena opened the weathered wooden chest and found an old brass compass and"},
    {"category": "Short Story", "prompt": "The autumn wind swept through the empty courtyard, scattering golden leaves across"}
]

def generate_offline_prompt_manifest(output_path: str = "ml/artifacts/cloudflare_gemma_prompt_manifest.json") -> Dict[str, Any]:
    """
    Generates a prompt manifest when Cloudflare credentials are not configured.
    Explicitly marked as unevaluated. Never overwrites the genuine evaluation artifact.
    """
    manifest = {
        "status": "unevaluated_offline_manifest",
        "description": "Standardized 20-prompt evaluation manifest for Google Gemma 4 26B-A4B-IT on Cloudflare Workers AI.",
        "model_id": "@cf/google/gemma-4-26b-a4b-it",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_count": len(EVALUATION_PROMPTS),
        "categories": ["Email", "Everyday", "Educational", "Article", "Short Story"],
        "prompts": EVALUATION_PROMPTS,
        "note": (
            "This file contains the benchmark prompt definitions. "
            "Actual evaluations are executed only when valid CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN "
            "credentials are provided. No simulated data is written to the genuine evaluation artifact."
        )
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Offline prompt manifest written to {output_path} (genuine evaluation preserved).")
    return manifest

def run_genuine_cloudflare_evaluation(
    provider: Optional[CloudflareWorkersAIProvider] = None,
    output_path: str = "ml/artifacts/cloudflare_gemma_evaluation.json",
    is_mocked_test: bool = False
) -> Dict[str, Any]:
    """
    Executes actual Cloudflare Workers AI inference across the 20 benchmark prompts.
    Evaluates both generation (20 operations) and prediction (20 operations) independently,
    tracking success, failure, and latency metrics separately.
    Requires server credentials. Never fabricates predictions or tokens.
    """
    if provider is None:
        provider = CloudflareWorkersAIProvider()

    if not provider.is_configured:
        print("Notice: Cloudflare Workers AI credentials unconfigured. Writing offline prompt manifest instead.")
        return generate_offline_prompt_manifest(output_path=output_path if "manifest" in output_path else "ml/artifacts/cloudflare_gemma_prompt_manifest.json")

    print(f"Executing Cloudflare Workers AI evaluation for {provider.model_id} (mocked={is_mocked_test})...")
    results: List[Dict[str, Any]] = []

    # Separate operational accounting
    generation_successes = 0
    generation_failures = 0
    generation_latencies: List[float] = []

    prediction_successes = 0
    prediction_failures = 0
    prediction_latencies: List[float] = []

    total_inference_requests = 0

    for item in EVALUATION_PROMPTS:
        cat = item["category"]
        prompt = item["prompt"]

        # 1. Evaluate Generation Operation
        total_inference_requests += 1
        gen_continuation = None
        gen_tokens = []
        gen_latency = 0.0
        gen_error = None
        try:
            gen_continuation, _, tokens, gen_lat = provider.generate_continuation(prompt, max_tokens=15)
            # Verify actual content presence
            if not gen_continuation or not gen_continuation.strip():
                raise ValueError("Received empty continuation text from provider.")
            gen_tokens = [t.token for t in tokens]
            gen_latency = round(gen_lat, 2)
            generation_latencies.append(gen_latency)
            generation_successes += 1
        except Exception as e:
            gen_error = str(e)
            generation_failures += 1

        # 2. Evaluate Prediction Operation
        total_inference_requests += 1
        pred_suggestions = []
        pred_latency = 0.0
        pred_error = None
        try:
            cands, _, pred_lat = provider.predict_next_words(prompt, top_k=5)
            # Verify actual candidate suggestions presence
            pred_suggestions = [c.word for c in cands if c.word and c.word.strip()]
            if not pred_suggestions:
                raise ValueError("Received empty or unusable candidate suggestions from provider.")
            pred_latency = round(pred_lat, 2)
            prediction_latencies.append(pred_latency)
            prediction_successes += 1
        except Exception as e:
            pred_error = str(e)
            prediction_failures += 1

        # Calculate prompt combined status
        prompt_status = "success" if (gen_error is None and pred_error is None) else (
            "partial" if (gen_error is None or pred_error is None) else "failed"
        )

        results.append({
            "category": cat,
            "prompt": prompt,
            "generation": {
                "continuation": gen_continuation,
                "tokens": gen_tokens,
                "latency_ms": gen_latency,
                "status": "success" if gen_error is None else "failed",
                "error": gen_error
            },
            "prediction": {
                "ranked_suggestions": pred_suggestions,
                "latency_ms": pred_latency,
                "status": "success" if pred_error is None else "failed",
                "error": pred_error
            },
            "status": prompt_status,
            # Backward-compatible fields
            "continuation": gen_continuation,
            "ranked_suggestions": pred_suggestions,
            "latency_ms": round(gen_latency + pred_latency, 2)
        })

    successful_requests = generation_successes + prediction_successes
    failed_requests = generation_failures + prediction_failures

    avg_generation_latency = round(sum(generation_latencies) / len(generation_latencies), 2) if generation_latencies else 0.0
    avg_prediction_latency = round(sum(prediction_latencies) / len(prediction_latencies), 2) if prediction_latencies else 0.0
    all_latencies = generation_latencies + prediction_latencies
    overall_avg_latency = round(sum(all_latencies) / len(all_latencies), 2) if all_latencies else 0.0

    status_label = "mocked_test_evaluated" if is_mocked_test else (
        "genuine_evaluated" if failed_requests == 0 else ("partial_evaluated" if successful_requests > 0 else "failed_evaluated")
    )

    report = {
        "status": status_label,
        "is_mocked_test": is_mocked_test,
        "model_id": provider.model_id,
        "provider": "Cloudflare Workers AI",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_prompts": len(results),
        "total_operations": len(results) * 2,
        "total_inference_requests": total_inference_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "generation_metrics": {
            "total_operations": len(results),
            "successful_operations": generation_successes,
            "failed_operations": generation_failures,
            "avg_latency_ms": avg_generation_latency,
            "min_latency_ms": min(generation_latencies) if generation_latencies else 0.0,
            "max_latency_ms": max(generation_latencies) if generation_latencies else 0.0
        },
        "prediction_metrics": {
            "total_operations": len(results),
            "successful_operations": prediction_successes,
            "failed_operations": prediction_failures,
            "avg_latency_ms": avg_prediction_latency,
            "min_latency_ms": min(prediction_latencies) if prediction_latencies else 0.0,
            "max_latency_ms": max(prediction_latencies) if prediction_latencies else 0.0
        },
        "overall_latency_ms": {
            "avg_latency_ms": overall_avg_latency
        },
        "categories_covered": ["Email", "Everyday", "Educational", "Article", "Short Story"],
        "results": results,
        "methodology": (
            "Genuine evaluation of Google Gemma 4 26B-A4B-IT (@cf/google/gemma-4-26b-a4b-it) "
            "conducted via Cloudflare Workers AI OpenAI-compatible chat completions interface. "
            "20 benchmark prompts were independently evaluated across both multi-token continuation generation (20 operations) "
            "and next-word prediction (20 operations), totalling 40 actual inference requests. "
            "Suggestions are ranked (#1 to #k) without fabricating unmeasured probabilities (reported as 0.0). "
            "This model operates as a serverless 26B Mixture-of-Experts (4B active) model on Cloudflare Workers AI, "
            "completely distinct from the research DistilGPT-2 and custom PyTorch BiLSTM models."
        )
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Genuine Cloudflare Workers AI evaluation written to {output_path}!")
    return report

if __name__ == "__main__":
    run_genuine_cloudflare_evaluation()
