import math

import torch.nn as nn

from aidd_codebase.utils.typescripts import Tensor


class TokenEmbedding(nn.Module):
    """Helper Module to convert input tensor to corresponding
    tensor of token embeddings."""

    def __init__(self, vocab_size: int, emb_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor) -> Tensor:
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
