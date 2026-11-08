import os
import json
import time
import urllib.request
import urllib.error
from typing import List, Tuple, Dict, Any, Optional
from backend.app.schemas import PredictionCandidate, GenerationToken

CLOUDFLARE_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions"
DEFAULT_MODEL_ID = "@cf/google/gemma-4-26b-a4b-it"

DEFAULT_PREDICT_MAX_TOKENS = int(os.environ.get("CLOUDFLARE_PREDICT_MAX_TOKENS", "96"))
DEFAULT_GENERATE_MAX_TOKENS = int(os.environ.get("CLOUDFLARE_GENERATE_MAX_TOKENS", "30"))

class CloudflareWorkersAIProvider:
    """
    Managed Inference Provider adapter for Cloudflare Workers AI.
    Runs serverless Google Gemma 4 26B-A4B-IT (Mixture-of-Experts with 4B active parameters)
    without loading model weights into the 512 MB Render Free instance.
    
    Model Lifecycle Status:
      - Active, fully supported on Cloudflare Workers AI catalog.
      - Zero planned deprecation date (unlike @cf/meta/llama-3.2-1b-instruct which is deprecated Nov 20, 2026).
    
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
        # Dynamic state tracking: 'unconfigured', 'configured', 'online', 'unavailable'
        self.has_verified_online = False
        self.last_verified_timestamp: Optional[float] = None
        self.last_error: Optional[str] = None
        self._cached_health: Optional[bool] = None
        self._cached_health_time: float = 0.0
        self._health_cache_ttl: float = 60.0  # 60s TTL for lightweight non-inference health check

    @property
    def is_configured(self) -> bool:
        return bool(self.account_id and self.api_token)

    @property
    def provider_state(self) -> str:
        if not self.is_configured:
            return "unconfigured"
        if self.last_error:
            return "unavailable"
        if self.has_verified_online:
            return "online"
        return "configured"

    @property
    def metadata(self) -> Dict[str, Any]:
        is_gemma = "gemma-4" in self.model_id.lower() or "gemma" in self.model_id.lower()
        model_name = "Gemma 4 26B-A4B-IT (General English)" if is_gemma else "Llama 3.2 1B Instruct (General English)"
        arch = "Gemma 4 Mixture-of-Experts (26B total, 4B active parameters)" if is_gemma else "Llama 3.2 1B Autoregressive Transformer (16 layers, 2048 hidden)"
        params = 26000000000 if is_gemma else 1230000000
        active_params = 4000000000 if is_gemma else 1230000000
        context_len = 256000 if is_gemma else 8192
        license_str = "Apache 2.0 (Google Gemma Terms & Apache 2.0)" if is_gemma else "Llama 3.2 Community License"
        license_url = "https://www.apache.org/licenses/LICENSE-2.0" if is_gemma else "https://llama.meta.com/llama3/license/"
        model_card_url = "https://developers.cloudflare.com/workers-ai/models/gemma-4-26b-a4b-it/" if is_gemma else "https://developers.cloudflare.com/workers-ai/models/"
        lifecycle_status = "Active Supported (No planned deprecation)" if is_gemma else "Deprecated on Nov 20, 2026"

        return {
            "name": model_name,
            "provider": "Cloudflare Workers AI (Serverless Managed Inference)",
            "model_id": self.model_id,
            "architecture": arch,
            "parameters": params,
            "active_parameters": active_params,
            "vocab_size": 256000 if is_gemma else 128256,
            "context_length": context_len,
            "license": license_str,
            "license_url": license_url,
            "model_card_url": model_card_url,
            "lifecycle_status": lifecycle_status,
            "domain": "General English (Open-domain conversations, emails, creative writing, articles)",
            "is_configured": self.is_configured,
            "provider_state": self.provider_state,
            "last_verified_timestamp": self.last_verified_timestamp,
            "probabilities_available": False,
            "ranking_mode": "rank_only",
            "quota_notice": "Cloudflare Workers AI Free Tier provides 10,000 Neurons/day. Zero paid resources required."
        }

    def _call_chat_completions(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 96,
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
            "temperature": max(0.01, min(temperature, 2.0)),
            # Gemma 4 reasoning mode causes reasoning thought chains to consume token budget
            # or return empty content when enable_thinking is true. Explicitly disable thinking.
            "chat_template_kwargs": {
                "enable_thinking": False
            }
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
            self.last_error = f"HTTP {he.code}: {err_body}"
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
            self.last_error = str(ue.reason)
            from fastapi import HTTPException
            raise HTTPException(
                status_code=503,
                detail=f"Cloudflare Workers AI connection timeout or network failure: {str(ue.reason)}"
            )
        except Exception as ex:
            self.last_error = str(ex)
            from fastapi import HTTPException
            raise HTTPException(
                status_code=503,
                detail=f"Unexpected error communicating with Cloudflare Workers AI: {str(ex)}"
            )

    def verify_token_connectivity(self) -> bool:
        """
        Genuinely verifies API token validity using Cloudflare's non-inference verify endpoint:
        GET https://api.cloudflare.com/client/v4/user/tokens/verify
        Zero billable Neurons or model tokens consumed.
        """
        if not self.is_configured:
            return False

        endpoint = "https://api.cloudflare.com/client/v4/user/tokens/verify"
        req = urllib.request.Request(
            endpoint,
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
                "User-Agent": "NeuroText-AI-Token-Verify/2.0"
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=min(self.timeout_seconds, 5.0)) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("success") is True:
                    self.has_verified_online = True
                    self.last_verified_timestamp = time.time()
                    self.last_error = None
                    return True
                else:
                    self.last_error = f"Token verification unsuccessful: {data.get('errors')}"
                    return False
        except urllib.error.HTTPError as he:
            err_body = he.read().decode("utf-8", errors="replace")
            # If 401/403/invalid token, mark unavailable
            if he.code in (400, 401, 403):
                self.last_error = f"HTTP {he.code}: Unauthorized or invalid API token"
            else:
                self.last_error = f"HTTP {he.code}: {err_body}"
            return False
        except Exception as ex:
            self.last_error = str(ex)
            return False

    def check_health(self) -> bool:
        """
        Check health status WITHOUT consuming billable Neurons on routine polls.
        Returns True if previously verified online; False otherwise.
        """
        return self.provider_state == "online"

    def _extract_text_content(self, res_json: Any) -> str:
        """
        Safely extracts text content from various Cloudflare / OpenAI chat completion response shapes.
        Handles nulls, missing keys, string responses, and array contents.
        """
        if not isinstance(res_json, dict):
            return ""

        # Primary OpenAI-compatible structure: choices[0].message.content
        choices = res_json.get("choices")
        if isinstance(choices, list) and len(choices) > 0 and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()
                elif isinstance(content, list):
                    # In some multi-part formats, content is a list of blocks
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(str(part.get("text", "")))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    return " ".join(text_parts).strip()
            # Some responses have 'text' directly on choice
            choice_text = choices[0].get("text")
            if isinstance(choice_text, str):
                return choice_text.strip()

        # Alternative Workers AI native structure: result.response
        result = res_json.get("result")
        if isinstance(result, dict):
            resp = result.get("response")
            if isinstance(resp, str):
                return resp.strip()
        elif isinstance(result, str):
            return result.strip()

        # Top-level 'response'
        top_resp = res_json.get("response")
        if isinstance(top_resp, str):
            return top_resp.strip()

        return ""

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
        from fastapi import HTTPException

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
            max_tokens=DEFAULT_PREDICT_MAX_TOKENS,
            temperature=temperature
        )

        raw_reply = self._extract_text_content(res_json)
        if not raw_reply:
            self.last_error = "Upstream Cloudflare Workers AI returned empty prediction content."
            raise HTTPException(
                status_code=502,
                detail="Cloudflare Workers AI returned an empty prediction response. The model did not produce candidate tokens."
            )

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

        if not candidates:
            self.last_error = f"Failed to parse usable candidates from Cloudflare response: '{raw_reply[:100]}'"
            raise HTTPException(
                status_code=502,
                detail=f"Unable to extract candidate suggestions from Cloudflare response: '{raw_reply[:80]}'"
            )

        # Mark provider as genuinely verified online since usable inference completed
        self.has_verified_online = True
        self.last_verified_timestamp = time.time()
        self.last_error = None
        self._cached_health = True
        self._cached_health_time = time.time()

        return candidates, tokens, round(latency_ms, 2)

    def generate_continuation(
        self,
        prompt: str,
        max_tokens: int = DEFAULT_GENERATE_MAX_TOKENS,
        temperature: float = 0.8,
        strategy: str = "greedy",
        top_k: int = 20,
        repetition_penalty: float = 1.1
    ) -> Tuple[str, str, List[GenerationToken], float]:
        """
        Generate multi-token sequential continuation using Cloudflare Workers AI.
        """
        from fastapi import HTTPException

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

        generated_text = self._extract_text_content(res_json)
        if not generated_text:
            self.last_error = "Upstream Cloudflare Workers AI returned empty generation content."
            raise HTTPException(
                status_code=502,
                detail="Cloudflare Workers AI returned an empty continuation response."
            )

        # Clean any leading quote marks or echoes of the prompt if returned
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        completed_text = (prompt.strip() + " " + generated_text).strip()

        # Break into readable token chunks
        words = generated_text.split()
        tokens_generated: List[GenerationToken] = [
            GenerationToken(token=w, probability=0.0) for w in words
        ]

        # Mark provider as genuinely verified online since usable inference completed
        self.has_verified_online = True
        self.last_verified_timestamp = time.time()
        self.last_error = None
        self._cached_health = True
        self._cached_health_time = time.time()

        return generated_text, completed_text, tokens_generated, round(latency_ms, 2)

