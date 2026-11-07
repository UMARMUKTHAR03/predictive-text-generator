import os
import json
import torch
from typing import Optional
from ml.training.model import BiLSTMLanguageModel
from ml.preprocessing.tokenizer import TextTokenizer

class ModelManager:
    _instance: Optional["ModelManager"] = None

    def __init__(self):
        self.device = torch.device("cpu")
        self.model: Optional[BiLSTMLanguageModel] = None
        self.tokenizer: Optional[TextTokenizer] = None
        self.model_metadata: dict = {}
        self.metrics: dict = {}
        self.max_sequence_len: int = 40
        self.is_loaded: bool = False

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self, models_dir: str = "models/current"):
        if not os.path.exists(models_dir):
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "current")

        tok_path = os.path.join(models_dir, "tokenizer.json")
        weights_path = os.path.join(models_dir, "model.pt")
        meta_path = os.path.join(models_dir, "model_metadata.json")
        metrics_path = os.path.join(models_dir, "metrics.json")

        # Fallback to ml/artifacts if needed
        if not os.path.exists(weights_path):
            alt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ml", "artifacts")
            if os.path.exists(os.path.join(alt_dir, "best_model.pt")):
                weights_path = os.path.join(alt_dir, "best_model.pt")
                tok_path = os.path.join(alt_dir, "tokenizer.json")

        if not os.path.exists(tok_path) or not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model artifacts not found in {models_dir} or ml/artifacts")

        self.tokenizer = TextTokenizer.load_json(tok_path)
        vocab_size = self.tokenizer.total_words

        self.model = BiLSTMLanguageModel(vocab_size=vocab_size, embedding_dim=100, hidden_dim=150)
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.model_metadata = json.load(f)

        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)
                self.max_sequence_len = self.metrics.get("parameters", {}).get("max_sequence_len", 40)

        self.is_loaded = True
        print(f"ModelManager successfully loaded model with vocab size {vocab_size}")

