from dataclasses import dataclass
from typing import List, Optional

import torch
from aidd_codebase.datamodules.datachoice import DataChoice
from aidd_codebase.utils.config import _ABCDataClass
from aidd_codebase.utils.metacoding import CreditType
from aidd_codebase.utils.tools import compose
from aidd_codebase.utils.typescripts import Tensor


@DataChoice.register_arguments(call_name="sequence_tokenizer")
@dataclass(unsafe_hash=True)
class TokenArguments(_ABCDataClass):
    # Define special symbols and indices
    pad_idx: int = 0  # Padding
    bos_idx: int = 1  # Beginning of Sequence
    eos_idx: int = 2  # End of Sequence
    unk_idx: int = 3  # Unknown Value
    msk_idx: Optional[int] = None  # Mask

    # Our vocabulary
    vocab: str = (
        " ^$?#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy"
    )
    max_seq_len: int = 250


@DataChoice.register_choice(
    call_name="sequence_tokenizer",
    author="Peter Hartog",
    github_handle="PeterHartog",
    credit_type=CreditType.NONE,
)
class Tokenizer:
    def __init__(self, token_args: TokenArguments) -> None:
        self.vocab_size = len(token_args.vocab)
        self.char_to_ix = {ch: i for i, ch in enumerate(token_args.vocab)}
        self.ix_to_char = {i: ch for i, ch in enumerate(token_args.vocab)}
        self.pad_idx = token_args.pad_idx
        self.bos_idx = token_args.bos_idx
        self.eos_idx = token_args.eos_idx
        self.max_seq_len = token_args.max_seq_len
        self.max_tensor_len = self.max_seq_len + 2

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
                    [self.bos_idx]
                    + token_ids
                    + [self.eos_idx]
                    + [self.pad_idx] * (self.max_seq_len - len(token_ids))
                ),
            )
        )

    def reverse_tensor_transform(self, tensor: Tensor):
        """Transforms a tensor back into a list of tokens."""
        tensor = tensor[
            torch.logical_and(
                torch.logical_and(
                    tensor != self.bos_idx, tensor != self.eos_idx
                ),
                tensor != self.pad_idx,
            )
        ]
        return tensor.tolist()
