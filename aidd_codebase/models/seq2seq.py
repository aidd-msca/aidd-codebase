from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from aidd_codebase.models.beamsearch import BeamSearch

from ..utils.config import _ABCDataClass
from ..utils.metacoding import CreditType
from ..utils.typescripts import Tensor
from .decoder import Decoder
from .encoder import Encoder
from .modelchoice import ModelChoice


@ModelChoice.register_arguments(call_name="pl_seq2seq")
@dataclass(unsafe_hash=True)
class Seq2SeqArguments(_ABCDataClass):
    src_vocab_size: int = 112
    tgt_vocab_size: int = 112
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    emb_size: int = 512
    num_heads: int = 8
    dim_feedforward: int = 512
    dropout: float = 0.1
    weight_sharing: bool = False
    pad_idx: int = 0
    bos_idx: int = 1
    eos_idx: int = 2
    unk_idx: int = 3
    max_seq_len: int = 252


@ModelChoice.register_choice(
    call_name="pl_seq2seq",
    author="Peter Hartog",
    github_handle="PeterHartog",
    credit_type=CreditType.NONE,
)
class Seq2Seq(pl.LightningModule):
    def __init__(self, model_args: Dict) -> None:
        super().__init__()

        # self.save_hyperparameters()

        args = Seq2SeqArguments(**model_args)

        self.max_seq_len = args.max_seq_len
        self.pad_idx = args.pad_idx
        self.bos_idx = args.bos_idx
        self.eos_idx = args.eos_idx
        self.unk_idx = args.unk_idx

        self.encoder = Encoder(model_args=args)
        self.decoder = Decoder(model_args=args)
        self.fc_out = nn.Linear(args.emb_size, args.tgt_vocab_size)

    # (ignore mypy error using type: ignore)
    def forward(  # type: ignore
        self,
        src: Tensor,
        tgt: Tensor,
    ):
        src_mask = self.create_src_mask(src)
        tgt_mask = self.create_tgt_mask(tgt)
        src_padding_mask = (
            self.create_padding_mask(src, self.pad_idx)
            if self.pad_idx
            else None
        )
        tgt_padding_mask = (
            self.create_padding_mask(tgt, self.pad_idx)
            if self.pad_idx
            else None
        )

        memory = self.encoder(src, src_mask, src_padding_mask)

        x = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=None,
            tgt_padding_mask=tgt_padding_mask,
            memory_padding_mask=src_padding_mask,
        )
        out = self.fc_out(x)
        return out

    def beam_search_predict(
        self, src: Tensor, alg: str = "greedy", k: int = 1
    ) -> Tensor:
        bs = BeamSearch(
            self.decoder,
            self.fc_out,
            self.max_seq_len,
            self.pad_idx,
            self.bos_idx,
            self.eos_idx,
            self.unk_idx,
            self.device,
        )

        src_mask = self.create_src_mask(src)
        src_padding_mask = (
            self.create_padding_mask(src, self.pad_idx)
            if self.pad_idx
            else None
        )

        memory = self.encoder(src, src_mask, src_padding_mask)
        return bs.sample_search(memory, alg=alg, k=k)

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """Generates an upper-trianghular matrix of -inf with zeros on diag."""
        return torch.triu(
            torch.ones(sz, sz, device=self.device) * float("-inf"), diagonal=1
        )

    def create_src_mask(self, src: Tensor) -> Tensor:
        return torch.zeros(
            (src.shape[0], src.shape[0]), device=self.device
        ).type(torch.bool)

    def create_tgt_mask(self, tgt: Tensor) -> Tensor:
        return self.generate_square_subsequent_mask(tgt.shape[0])

    @staticmethod
    def create_padding_mask(
        seq: Tensor, pad_idx: int, eos_idx: int = 2
    ) -> Tensor:
        mask = torch.where((seq == pad_idx) | (seq == eos_idx), True, False)
        return mask.transpose(0, 1)
