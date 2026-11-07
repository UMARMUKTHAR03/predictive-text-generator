import os
import json
import time
from typing import List, Dict, Any
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

def generate_cloudflare_evaluation_report(provider: CloudflareWorkersAIProvider = None):
    """
    Evaluates 20 representative prompts across 5 key writing domains.
    If live Cloudflare credentials are configured in environment variables,
    queries the live API; otherwise utilizes the structured benchmark simulator.
    """
    if provider is None:
        provider = CloudflareWorkersAIProvider()

    results: List[Dict[str, Any]] = []

    for item in EVALUATION_PROMPTS:
        cat = item["category"]
        prompt = item["prompt"]
        
        # When unconfigured in offline/CI, provide simulated evaluation samples
        if not provider.is_configured:
            continuation = f"[Simulated continuation for '{prompt}']"
            suggestions = ["first", "second", "third", "fourth", "fifth"]
            lat = 120.0
        else:
            try:
                continuation, completed, tokens, lat_gen = provider.generate_continuation(prompt, max_tokens=15)
                cands, toks, lat_pred = provider.predict_next_words(prompt, top_k=5)
                suggestions = [c.word for c in cands]
                lat = round(lat_gen + lat_pred, 1)
            except Exception as e:
                continuation = f"[Provider error: {str(e)}]"
                suggestions = []
                lat = 0.0

        results.append({
            "category": cat,
            "prompt": prompt,
            "continuation": continuation,
            "top_suggestions": suggestions,
            "latency_ms": lat
        })

    report = {
        "model_id": provider.model_id,
        "provider": "Cloudflare Workers AI",
        "live_credentials_present": provider.is_configured,
        "prompt_count": len(results),
        "categories_covered": ["Email", "Everyday", "Educational", "Article", "Short Story"],
        "results": results,
        "methodology_notice": (
            "This evaluation measures qualitative generation quality and candidate forecasting "
            "for Llama 3.2 1B Instruct via Cloudflare Workers AI. "
            "It does NOT claim mathematical or architectural equivalence to DistilGPT-2 or the custom BiLSTM. "
            "All model benchmarks and parameter counts are reported independently."
        )
    }

    out_file = "ml/artifacts/cloudflare_gemma_evaluation.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Cloudflare Workers AI evaluation report written to {out_file}!")
    return report

if __name__ == "__main__":
    generate_cloudflare_evaluation_report()

