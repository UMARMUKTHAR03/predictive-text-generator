import os
import json
import time
import urllib.request
import urllib.error
from typing import List, Tuple, Dict, Any, Optional
from backend.app.schemas import PredictionCandidate, GenerationToken

CLOUDFLARE_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions"
DEFAULT_MODEL_ID = "@cf/meta/llama-3.2-1b-instruct"

class CloudflareWorkersAIProvider:
    """
    Managed Inference Provider adapter for Cloudflare Workers AI.
    Runs serverless Llama 3.2 1B Instruct without loading model weights into
    the 512 MB Render Free instance.
    
    Credentials (CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN) are read strictly
    from server-side environment variables and are never exposed to clients,
    logs, or git repositories.
    """
    def __init__(
        self,
        account_id: Optional[str] = None,
        api_token: Optional[str] = None,
        model_id: Optional[str] = None,
        timeout_seconds: float = 12.0
    ):
        self.account_id = account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
        self.api_token = api_token or os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
        self.model_id = model_id or os.environ.get("GENERAL_MODEL_ID", DEFAULT_MODEL_ID).strip()
        self.timeout_seconds = timeout_seconds

    @property
    def is_configured(self) -> bool:
        return bool(self.account_id and self.api_token)

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "Llama 3.2 1B Instruct (General English)",
            "provider": "Cloudflare Workers AI (Serverless Managed Inference)",
            "model_id": self.model_id,
            "architecture": "Llama 3.2 1B Autoregressive Transformer (16 layers, 2048 hidden, 32 heads)",
            "parameters": 1230000000,
            "vocab_size": 128256,
            "context_length": 8192,
            "license": "Llama 3.2 Community License",
            "domain": "General English (Open-domain conversations, emails, creative writing, articles)",
            "is_configured": self.is_configured,
            "status": "configured" if self.is_configured else "unconfigured_missing_credentials",
            "quota_notice": "Cloudflare Workers AI Free Tier provides 10,000 Neurons/day. Zero paid resources required."
        }

    def _call_chat_completions(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 30,
        temperature: float = 0.7,
        response_format: Optional[Dict[str, str]] = None
    ) -> Tuple[Dict[str, Any], float]:
        if not self.is_configured:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=503,
                detail=(
                    "Cloudflare Workers AI credentials not configured on the server. "
                    "Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in server environment variables."
                )
            )

        endpoint = CLOUDFLARE_BASE_URL.format(account_id=self.account_id)
        payload_dict: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": max(0.01, min(temperature, 2.0))
        }
        if response_format:
            payload_dict["response_format"] = response_format

        data_bytes = json.dumps(payload_dict).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data_bytes,
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
                "User-Agent": "NeuroText-AI-Cloudflare-Adapter/2.0"
            }
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                status_code = resp.getcode()
                resp_bytes = resp.read()
                latency_ms = (time.perf_counter() - t0) * 1000
                res_json = json.loads(resp_bytes.decode("utf-8"))
                return res_json, latency_ms
        except urllib.error.HTTPError as he:
            latency_ms = (time.perf_counter() - t0) * 1000
            err_body = he.read().decode("utf-8", errors="replace")
            from fastapi import HTTPException
            if he.code == 429:
                raise HTTPException(
                    status_code=429,
                    detail=f"Cloudflare Workers AI rate limit or daily free quota (10,000 Neurons/day) exceeded: {err_body}"
                )
            elif he.code in (401, 403):
                raise HTTPException(
                    status_code=503,
                    detail="Cloudflare Workers AI authorization error. Verify CLOUDFLARE_API_TOKEN and account permissions."
                )
            else:
                raise HTTPException(
                    status_code=503,
                    detail=f"Cloudflare Workers AI upstream error (HTTP {he.code}): {err_body}"
                )
        except urllib.error.URLError as ue:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=503,
                detail=f"Cloudflare Workers AI connection timeout or network failure: {str(ue.reason)}"
            )
        except Exception as ex:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=503,
                detail=f"Unexpected error communicating with Cloudflare Workers AI: {str(ex)}"
            )

    def check_health(self) -> bool:
        """
        Verify live connectivity with Cloudflare Workers AI using a 1-token minimal test.
        Returns True if successful, False if credentials missing or upstream unreachable.
        """
        if not self.is_configured:
            return False
        try:
            res, _ = self._call_chat_completions(
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0.1
            )
            return bool("choices" in res and len(res["choices"]) > 0)
        except Exception:
            return False

    def predict_next_words(
        self,
        text: str,
        top_k: int = 5,
        temperature: float = 1.0
    ) -> Tuple[List[PredictionCandidate], List[str], float]:
        """
        Produce ranked next-word or next-phrase suggestions using structured schema instruction.
        
        Note on probability semantics:
        Cloudflare Workers AI Chat Completions does not return raw token log-probabilities.
        In strict adherence to project standards, probability is reported as 0.0, and candidates
        are explicitly ranked (rank=1, 2, ... top_k). No fabricated probabilities are returned.
        """
        norm_text = text.rstrip()
        if not norm_text:
            return [], [], 0.0

        words = norm_text.split()
        tokens = words[-10:] if len(words) > 10 else words

        system_prompt = (
            "You are a predictive text language model engine. "
            "Given the user's unfinished sentence, forecast up to {top_k} distinct, most likely next words or short phrases that continue the sentence naturally. "
            "Respond ONLY with a valid JSON array of strings, e.g. [\"word1\", \"word2\", \"word3\"]. "
            "Do not include explanation, markdown formatting, or prefix."
        ).format(top_k=top_k)

        user_content = f"Sentence: \"{norm_text}\"\nNext candidate continuations:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        res_json, latency_ms = self._call_chat_completions(
            messages=messages,
            max_tokens=40,
            temperature=temperature
        )

        choices = res_json.get("choices", [])
        if not choices:
            return [], tokens, round(latency_ms, 2)

        raw_reply = choices[0].get("message", {}).get("content", "").strip()
        parsed_words: List[str] = []

        # Attempt JSON parse of array
        try:
            start_bracket = raw_reply.find("[")
            end_bracket = raw_reply.rfind("]")
            if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
                extracted_json = raw_reply[start_bracket:end_bracket + 1]
                items = json.loads(extracted_json)
                if isinstance(items, list):
                    parsed_words = [str(item).strip() for item in items if str(item).strip()]
        except Exception:
            pass

        # Fallback line-by-line / bullet extraction if JSON parse fails
        if not parsed_words:
            lines = raw_reply.splitlines()
            for line in lines:
                clean_line = line.lstrip("0123456789.-*• \t\"'").rstrip("\"',. ")
                if clean_line:
                    parsed_words.append(clean_line)

        # Sanitize and deduplicate candidates
        candidates: List[PredictionCandidate] = []
        seen = set()
        for word in parsed_words:
            # Strip outer quotes or brackets if present
            w = word.strip("[]\"' \t\n")
            if not w:
                continue
            lower_w = w.lower()
            if lower_w in seen:
                continue
            seen.add(lower_w)
            candidates.append(PredictionCandidate(
                word=w,
                probability=0.0,  # 0.0 indicates unmeasured model probability; ranking provided by rank
                rank=len(candidates) + 1
            ))
            if len(candidates) >= top_k:
                break

        return candidates, tokens, round(latency_ms, 2)

    def generate_continuation(
        self,
        prompt: str,
        max_tokens: int = 15,
        temperature: float = 0.8,
        strategy: str = "greedy",
        top_k: int = 20,
        repetition_penalty: float = 1.1
    ) -> Tuple[str, str, List[GenerationToken], float]:
        """
        Generate multi-token sequential continuation using Cloudflare Workers AI.
        """
        system_prompt = (
            "You are a writing completion assistant. "
            "Continue the following prompt directly and naturally without repeating the prompt, without adding conversational filler (like 'Here is...'), "
            "and without commentary. Output only the immediate continuation."
        )

        effective_temp = 0.1 if strategy == "greedy" else max(0.1, min(temperature, 1.5))
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        res_json, latency_ms = self._call_chat_completions(
            messages=messages,
            max_tokens=max_tokens,
            temperature=effective_temp
        )

        choices = res_json.get("choices", [])
        if not choices:
            return "", prompt, [], round(latency_ms, 2)

        generated_text = choices[0].get("message", {}).get("content", "").strip()

        # Clean any leading quote marks or echoes of the prompt if returned
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        completed_text = (prompt.strip() + " " + generated_text).strip()

        # Break into readable token chunks
        words = generated_text.split()
        tokens_generated: List[GenerationToken] = [
            GenerationToken(token=w, probability=0.0) for w in words
        ]

        return generated_text, completed_text, tokens_generated, round(latency_ms, 2)
