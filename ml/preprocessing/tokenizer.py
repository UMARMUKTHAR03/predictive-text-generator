import re
from typing import List, Dict, Any, Optional
import json

DEFAULT_FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

def clean_text(text: str) -> str:
    """Normalize whitespace and non-breaking spaces identical to original notebook."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\xa0", " ").replace("\u200a", " ")
    return text.strip()

def tokenize_text(text: str, filters: str = DEFAULT_FILTERS, lower: bool = True) -> List[str]:
    """Tokenize text matching Keras Tokenizer default behavior."""
    cleaned = clean_text(text)
    if lower:
        cleaned = cleaned.lower()
    
    # Translate filter characters to spaces
    translation_table = str.maketrans({char: " " for char in filters})
    cleaned = cleaned.translate(translation_table)
    tokens = cleaned.split()
    return tokens

class TextTokenizer:
    """
    Standard lightweight tokenizer fully compatible with Keras Tokenizer word_index.
    Supports serialization to and from JSON for universal deployment without Keras/TensorFlow runtime.
    """
    def __init__(self, oov_token: str = "<oov>", filters: str = DEFAULT_FILTERS, lower: bool = True):
        self.oov_token = oov_token
        self.filters = filters
        self.lower = lower
        self.word_counts: Dict[str, int] = {}
        self.word_index: Dict[str, int] = {}
        self.index_word: Dict[int, str] = {}
        self.total_words: int = 1

    def fit_on_texts(self, texts: List[str]):
        """Fit tokenizer on a list of texts (e.g. training set titles)."""
        counts: Dict[str, int] = {}
        for text in texts:
            tokens = tokenize_text(text, filters=self.filters, lower=self.lower)
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1

        self.word_counts = counts
        # Sort words by frequency (descending), then alphabetically for determinism
        sorted_words = sorted(counts.items(), key=lambda x: (-x[1], x[0]))

        self.word_index = {}
        self.index_word = {}

        if self.oov_token:
            self.word_index[self.oov_token] = 1
            self.index_word[1] = self.oov_token
            current_id = 2
        else:
            current_id = 1

        for word, _ in sorted_words:
            if word == self.oov_token:
                continue
            self.word_index[word] = current_id
            self.index_word[current_id] = word
            current_id += 1

        self.total_words = len(self.word_index) + 1  # 1-indexed plus 0 padding

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """Convert a list of text strings into lists of token integer indices."""
        oov_id = self.word_index.get(self.oov_token, 0)
        sequences = []
        for text in texts:
            tokens = tokenize_text(text, filters=self.filters, lower=self.lower)
            seq = [self.word_index.get(token, oov_id) for token in tokens]
            sequences.append(seq)
        return sequences

    def sequence_to_text(self, sequence: List[int]) -> str:
        """Convert a sequence of token IDs back to a string."""
        words = [self.index_word.get(idx, self.oov_token) for idx in sequence if idx != 0]
        return " ".join(words)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "oov_token": self.oov_token,
            "filters": self.filters,
            "lower": self.lower,
            "total_words": self.total_words,
            "vocab_size": len(self.word_index),
            "word_index": self.word_index,
            "index_word": {str(k): v for k, v in self.index_word.items()},
        }

    def save_json(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, filepath: str) -> "TextTokenizer":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(oov_token=data.get("oov_token", "<oov>"),
                  filters=data.get("filters", DEFAULT_FILTERS),
                  lower=data.get("lower", True))
        tok.total_words = data["total_words"]
        tok.word_index = data["word_index"]
        tok.index_word = {int(k): v for k, v in data["index_word"].items()}
        return tok

