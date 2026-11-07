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

def generate_offline_prompt_manifest() -> Dict[str, Any]:
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
    out_file = "ml/artifacts/cloudflare_gemma_prompt_manifest.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Offline prompt manifest written to {out_file} (genuine evaluation preserved).")
    return manifest

def run_genuine_cloudflare_evaluation(provider: Optional[CloudflareWorkersAIProvider] = None) -> Dict[str, Any]:
    """
    Executes actual Cloudflare Workers AI inference across the 20 benchmark prompts.
    Requires server credentials. Never fabricates predictions or tokens.
    """
    if provider is None:
        provider = CloudflareWorkersAIProvider()

    if not provider.is_configured:
        print("Notice: Cloudflare Workers AI credentials unconfigured. Writing offline prompt manifest instead.")
        return generate_offline_prompt_manifest()

    print(f"Executing genuine Cloudflare Workers AI evaluation for {provider.model_id}...")
    results: List[Dict[str, Any]] = []
    successful_requests = 0
    failed_requests = 0

    for item in EVALUATION_PROMPTS:
        cat = item["category"]
        prompt = item["prompt"]
        try:
            continuation, completed, tokens, lat_gen = provider.generate_continuation(prompt, max_tokens=15)
            cands, toks, lat_pred = provider.predict_next_words(prompt, top_k=5)
            suggestions = [c.word for c in cands]
            total_lat = round(lat_gen + lat_pred, 1)
            successful_requests += 1
            error_msg = None
        except Exception as e:
            continuation = None
            suggestions = []
            total_lat = 0.0
            failed_requests += 1
            error_msg = str(e)

        results.append({
            "category": cat,
            "prompt": prompt,
            "continuation": continuation,
            "ranked_suggestions": suggestions,
            "latency_ms": total_lat,
            "status": "success" if error_msg is None else "failed",
            "error": error_msg
        })

    report = {
        "status": "genuine_evaluated" if failed_requests == 0 else "partial_evaluated",
        "model_id": provider.model_id,
        "provider": "Cloudflare Workers AI",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_prompts": len(results),
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "categories_covered": ["Email", "Everyday", "Educational", "Article", "Short Story"],
        "results": results,
        "methodology": (
            "Genuine evaluation of Google Gemma 4 26B-A4B-IT (@cf/google/gemma-4-26b-a4b-it) "
            "conducted via Cloudflare Workers AI OpenAI-compatible chat completions interface. "
            "Suggestions are ranked (#1 to #k) without fabricating unmeasured probabilities. "
            "This model operates as a serverless 26B Mixture-of-Experts (4B active) model, "
            "completely distinct from the research DistilGPT-2 and custom BiLSTM models."
        )
    }

    out_file = "ml/artifacts/cloudflare_gemma_evaluation.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Genuine Cloudflare Workers AI evaluation written to {out_file}!")
    return report

if __name__ == "__main__":
    run_genuine_cloudflare_evaluation()
