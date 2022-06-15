from typing import List

import torch

from aidd_codebase.utils.tools import compose
from aidd_codebase.utils.typescripts import Tensor


class Tokenizer:
    def __init__(
        self,
        vocab: str,
        pad_idx: int,
        bos_idx: int,
        eos_idx: int,
        max_seq_len: int,
    ) -> None:
        self.VOCAB_SIZE = len(vocab)
        self.char_to_ix = {ch: i for i, ch in enumerate(vocab)}
        self.ix_to_char = {i: ch for i, ch in enumerate(vocab)}
        self.PAD_IDX = pad_idx
        self.BOS_IDX = bos_idx
        self.EOS_IDX = eos_idx
        self.MAX_SEQ_LEN = max_seq_len
        self.MAX_TENSOR_LEN = self.MAX_SEQ_LEN + 2

        self.smile_prep = compose(self.smile_tokenizer, self.tensor_transform)
        self.smile_return = compose(
            self.reverse_tensor_transform, self.reverse_tokenizer
        )

    def smile_tokenizer(self, smile: str) -> List:
        return [self.char_to_ix[ch] for ch in smile]

    def reverse_tokenizer(self, tokens: List[int]) -> List:
        return [self.ix_to_char[token] for token in tokens]

    def tensor_transform(self, token_ids: List[int]) -> Tensor:
        """Function to add BOS/EOS, padding and create tensor for
        input sequence indices."""
        return torch.cat(
            (
                torch.tensor(
                    [self.BOS_IDX]
                    + token_ids
                    + [self.EOS_IDX]
                    + [self.PAD_IDX] * (self.MAX_SEQ_LEN - len(token_ids))
                ),
            )
        )

    def reverse_tensor_transform(self, tensor: Tensor):
        """Transforms a tensor back into a list of tokens."""
        tensor = tensor[
            torch.logical_and(
                torch.logical_and(
                    tensor != self.BOS_IDX, tensor != self.EOS_IDX
                ),
                tensor != self.PAD_IDX,
            )
        ]
        return tensor.tolist()
