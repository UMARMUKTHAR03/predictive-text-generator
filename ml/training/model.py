import torch
import torch.nn as nn
from typing import Dict, Any

class BiLSTMLanguageModel(nn.Module):
    """
    Bidirectional LSTM Language Model architecture matching the original Keras design:
    - Embedding: vocab_size -> embedding_dim (100)
    - Bidirectional LSTM: hidden_dim (150 per direction, total 300)
    - Linear Head: 300 -> vocab_size (logits)
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 100, hidden_dim: int = 150, padding_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        # BiLSTM outputs hidden_dim * 2 features
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len]
        embeds = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        lstm_out, _ = self.lstm(embeds)  # [batch_size, seq_len, hidden_dim * 2]
        # Language modeling next-token prediction uses the final sequence step
        last_step = lstm_out[:, -1, :]  # [batch_size, hidden_dim * 2]
        logits = self.fc(last_step)  # [batch_size, vocab_size]
        return logits

    def get_config(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "padding_idx": self.padding_idx,
            "architecture": "Bidirectional LSTM (Embedding -> BiLSTM(150) -> Linear(vocab))",
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

