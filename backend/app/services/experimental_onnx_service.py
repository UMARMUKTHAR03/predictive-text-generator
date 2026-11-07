import time
import os
import psutil
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
from backend.app.schemas import PredictionCandidate, GenerationToken

PINNED_DISTILGPT2_REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"

class ExperimentalONNXGeneralService:
    """
    Experimental General English Language Model Service powered by DistilGPT-2 (82M params)
    quantized to dynamic INT8 using ONNX Runtime and Optimum.
    
    Evaluated as a memory-reduction candidate to run within constrained RAM (<450 MB target).
    Preserves exact model architecture, pinned revision, tokenizer vocabulary,
    mathematical probability calculations, and whole-word BPE candidate extraction.
    """
    def __init__(
        self,
        onnx_model_dir: str = "ml/artifacts/onnx/distilgpt2_int8",
        file_name: str = "model_quantized.onnx",
        revision: str = PINNED_DISTILGPT2_REVISION
    ):
        self.onnx_model_dir = onnx_model_dir
        self.file_name = file_name
        self.model_id = "distilbert/distilgpt2"
        self.revision = revision
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.metadata = {
            "name": "DistilGPT-2 ONNX INT8 (General English Experimental)",
            "model_id": self.model_id,
            "revision": self.revision,
            "quantization": "Dynamic INT8 (ONNX Runtime CPU / AVX512-VNNI)",
            "weights_format": "ONNX (model_quantized.onnx ~82.4 MB)",
            "parameters": 81912576,
            "vocab_size": 50257,
            "license": "Apache 2.0",
            "context_length": 1024,
            "domain": "General English (WebText, everyday sentences, emails, articles, creative writing)"
        }

    def load(self):
        if self.is_loaded:
            return
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        import onnxruntime as ort
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 2
        session_options.inter_op_num_threads = 1
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.model = ORTModelForCausalLM.from_pretrained(
            self.onnx_model_dir,
            file_name=self.file_name,
            session_options=session_options
        )
        self.is_loaded = True
        self.load_time_sec = time.perf_counter() - t0

    def unload(self):
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        import gc
        gc.collect()

    def predict_next_words(
        self,
        text: str,
        top_k: int = 5,
        temperature: float = 1.0
    ) -> Tuple[List[PredictionCandidate], List[str], float]:
        """
        Extract whole-word candidate completions using GPT-2 byte-level BPE boundaries.
        Matches the exact logic of GeneralLanguageModelService.
        """
        if not self.is_loaded:
            self.load()

        t0 = time.perf_counter()
        ends_with_space = len(text) > 0 and text[-1].isspace()
        norm_text = text.rstrip()
        if not norm_text:
            return [], [], 0.0

        parts = norm_text.split()
        last_word = parts[-1] if parts else ""

        inputs = self.tokenizer(norm_text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        tokens = [self.tokenizer.decode([t]) for t in input_ids[0]]

        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]  # [1, vocab_size]
        scaled_logits = logits / max(temperature, 1e-4)
        probs = F.softmax(scaled_logits, dim=-1)[0]

        top_probs, top_indices = probs.topk(min(100, probs.size(-1)), dim=-1)

        candidates: List[PredictionCandidate] = []
        seen_words = set()

        for prob, idx in zip(top_probs, top_indices):
            raw_token = self.tokenizer.decode([idx.item()])
            clean_token = raw_token.strip()

            if not clean_token or clean_token.startswith("<|") or clean_token.endswith("|>"):
                continue
            if not any(c.isalnum() for c in clean_token):
                continue

            starts_new_word = raw_token.startswith(" ") or raw_token.startswith("\n")

            if ends_with_space:
                if not starts_new_word:
                    continue
                cand_word = clean_token
            else:
                if starts_new_word:
                    cand_word = clean_token
                else:
                    if clean_token[0].isupper() and last_word.islower():
                        continue
                    cand_word = last_word + clean_token

            lower_word = cand_word.lower()
            if lower_word not in seen_words:
                seen_words.add(lower_word)
                candidates.append(PredictionCandidate(
                    word=cand_word,
                    probability=round(prob.item(), 4),
                    rank=len(candidates) + 1
                ))

            if len(candidates) >= top_k:
                break

        latency_ms = (time.perf_counter() - t0) * 1000
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
        Generate multi-token sequential continuation with authentic probability extraction.
        Matches the exact logic of GeneralLanguageModelService.
        """
        if not self.is_loaded:
            self.load()

        t0 = time.perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        do_sample = strategy != "greedy"
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": repetition_penalty,
            "attention_mask": attention_mask,
            "return_dict_in_generate": True,
            "output_scores": True
        }

        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 0.1)
            if strategy == "top_k":
                gen_kwargs["top_k"] = top_k

        gen_outputs = self.model.generate(input_ids, **gen_kwargs)
        output_ids = gen_outputs.sequences
        scores = gen_outputs.scores
        generated_ids = output_ids[0][input_ids.shape[-1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        completed_text = (prompt.strip() + " " + generated_text).strip()

        tokens_generated = []
        for score_step, token_id in zip(scores, generated_ids):
            step_probs = F.softmax(score_step, dim=-1)
            actual_prob = float(step_probs[0, token_id].item())
            tok_str = self.tokenizer.decode([token_id.item()])
            tokens_generated.append(GenerationToken(token=tok_str, probability=round(actual_prob, 4)))

        latency_ms = (time.perf_counter() - t0) * 1000
        return generated_text, completed_text, tokens_generated, round(latency_ms, 2)
