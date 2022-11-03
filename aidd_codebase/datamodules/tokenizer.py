import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from aidd_codebase.datamodules.datachoice import DataChoice
from aidd_codebase.datamodules.vocab import DEEPCHEM_PubChem_1M_VOCAB
from aidd_codebase.utils.config import _ABCDataClass
from aidd_codebase.utils.metacoding import CreditType
from aidd_codebase.utils.tools import compose
from aidd_codebase.utils.typescripts import Tensor


class _ABCTokenizer(ABC):
    @abstractmethod
    def tokenize(self, smile: str) -> List:
        raise NotImplementedError

    @abstractmethod
    def detokenize(self, tokens: List) -> str:
        raise NotImplementedError

    @abstractmethod
    def tensorize(self, tokens: List) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def detensorize(self, tensor: Tensor) -> List:
        raise NotImplementedError


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
class Tokenizer(_ABCTokenizer):
    def __init__(self, token_args: Dict) -> None:

        args = TokenArguments(**token_args)

        self.vocab_size = len(args.vocab)
        self.char_to_ix = {ch: i for i, ch in enumerate(args.vocab)}
        self.ix_to_char = {i: ch for i, ch in enumerate(args.vocab)}
        self.pad_idx = args.pad_idx
        self.bos_idx = args.bos_idx
        self.eos_idx = args.eos_idx
        self.max_seq_len = args.max_seq_len
        self.max_tensor_len = self.max_seq_len + 2

        self.smile_prep = compose(self.tokenize, self.tensorize)
        self.smile_return = compose(self.detensorize, self.detokenize)

    def tokenize(self, smile: str) -> List:
        """Generates integers for to the tokens for a smiles string."""
        return [self.char_to_ix[ch] for ch in smile]

    def detokenize(self, tokens: List) -> str:
        """Generates smiles string from tokens."""
        return "".join([self.ix_to_char[token] for token in tokens])

    def sequentialize(self, tokens: List) -> List:
        """Adds BOS, EOS and padding to tokens."""
        return (
            [self.bos_idx]
            + tokens
            + [self.eos_idx]
            + [self.pad_idx] * (self.max_seq_len - len(tokens))
        )

    def desequentialize(self, tokens: List) -> List:
        """Removes BOS, EOS and padding from tokens."""
        return [
            token
            for token in tokens
            if token not in [self.bos_idx, self.eos_idx, self.pad_idx]
        ]

    def tensorize(self, tokens: List[int]) -> Tensor:
        """Create tensor from the tokens."""
        return torch.tensor(self.sequentialize(tokens))

    def detensorize(self, tensor: Tensor):
        """Transforms a tensor back into a list of tokens."""
        return self.desequentialize(tensor.tolist())


@DataChoice.register_arguments(call_name="regex_tokenizer")
@dataclass(unsafe_hash=True)
class RegexTokenArguments(_ABCDataClass):
    # Define special symbols and indices
    pad_idx: int = 0  # Padding
    bos_idx: int = 1  # Beginning of Sequence
    eos_idx: int = 2  # End of Sequence
    unk_idx: int = 3  # Unknown Value
    msk_idx: Optional[int] = None  # Mask

    pad_tok: str = "[PAD]"  # Padding
    bos_tok: str = "[BOS]"  # Beginning of Sequence
    eos_tok: str = "[EOS]"  # End of Sequence
    unk_tok: str = "[UNK]"  # Unknown Value
    msk_tok: str = "[MASK]"  # Mask

    bad_tok: List[str] = field(
        default_factory=lambda: ["[CLS]", "[SEP]"]  # Default Bad Tokens
    )

    # Our vocabulary
    vocab: Dict[str, int] = field(
        default_factory=lambda: DEEPCHEM_PubChem_1M_VOCAB
    )
    regex: str = (
        r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|"
        + r"=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
    )
    max_seq_len: int = 250


@DataChoice.register_choice(
    call_name="regex_tokenizer",
    author="Peter Hartog",
    github_handle="PeterHartog",
    credit_type=CreditType.NONE,
)
class RegexTokenizer:
    def __init__(self, token_args: Dict) -> None:

        args = RegexTokenArguments(**token_args)

        self.regex = re.compile(args.regex)

        self.vocab_size = len(args.vocab)
        self.char_to_ix = {ch: i for i, ch in enumerate(args.vocab)}
        self.ix_to_char = {i: ch for i, ch in enumerate(args.vocab)}
        self.pad_idx = args.pad_idx
        self.bos_idx = args.bos_idx
        self.eos_idx = args.eos_idx
        self.max_seq_len = args.max_seq_len
        self.max_tensor_len = self.max_seq_len + 2

        self.smile_prep = compose(self.tokenize, self.tensorize)
        self.smile_return = compose(self.detokenize, self.detensorize)

    def tokenize(self, smile: str) -> List:
        return [token for token in self.regex.findall(smile)]

    def detokenize(self, tokens: List[int]) -> str:
        return "".join([self.ix_to_char[token] for token in tokens])

    def sequentialize(self, tokens: List) -> List:
        """Adds BOS, EOS and padding to tokens."""
        return (
            [self.bos_idx]
            + tokens
            + [self.eos_idx]
            + [self.pad_idx] * (self.max_seq_len - len(tokens))
        )

    def desequentialize(self, tokens: List) -> List:
        """Removes BOS, EOS and padding from tokens."""
        return [
            token
            for token in tokens
            if token not in [self.bos_idx, self.eos_idx, self.pad_idx]
        ]

    def tensorize(self, tokens: List[int]) -> Tensor:
        """Create tensor from the tokens."""
        return torch.cat(torch.tensor(tokens))

    def detensorize(self, tensor: Tensor):
        """Transforms a tensor back into a list of tokens."""
        return self.desequentialize(tensor.tolist())
