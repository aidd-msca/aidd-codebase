import math
from typing import Optional

import torch
import torch.nn as nn
from aidd_codebase.utils.typescripts import Tensor

# Positional Encoding Options
# adding vs concatonating | static vs learned |
# relative (clipping at k) vs absolute | Type: Image vs Graph vs Sequence


# class PositionalEncoding(nn.Module):
#     def __init__(self, positional: str, combination: str, learned_option: str) -> None:
#         super().__init__()
#         self.pos_encoding = nn.Sequential(
#             PositionalChoice.get_choice(positional),
#             CombinationChoice.get_choice(combination),
#             LearnedChoice.get_choice(learned_option),
#         )


class SequencePositionalEncoding(nn.Module):
    """Helper Module that adds positional encoding to sequence embedding."""

    def __init__(self, emb_size: int, dropout: float, maxlen: Optional[int] = None) -> None:
        super().__init__()
        if not maxlen:
            maxlen = 5000

        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor) -> Tensor:
        return self.dropout(  # Pos embedding behind buffer
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class ImagePositionalEncoding(nn.Module):
    """Helper Module that adds positional encoding to sequence embedding."""

    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000) -> None:
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor) -> Tensor:
        return self.dropout(  # Pos embedding behind buffer
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )
