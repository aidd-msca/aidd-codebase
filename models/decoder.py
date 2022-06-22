from dataclasses import dataclass
from typing import Optional

import pytorch_lightning as pl
import torch.nn as nn

from ..utils.config import _ABCDataClass
from ..utils.metacoding import CreditType
from ..utils.typescripts import Tensor
from .embedding.embedding import TokenEmbedding
from .embedding.positional import SequencePositionalEncoding
from .modelchoice import ModelChoice
from .modules.modules import WeightSharingDecoder


@ModelChoice.register_arguments(call_name="pl_decoder")
@dataclass(unsafe_hash=True)
class DecoderArguments(_ABCDataClass):
    tgt_vocab_size = 112
    num_decoder_layers = 3
    emb_size = 512
    num_heads = 8
    dim_feedforward = 512
    dropout: float = 0.1
    weight_sharing: bool = False
    max_seq_len: Optional[int] = None


@ModelChoice.register_choice(
    call_name="pl_decoder",
    author="Peter Hartog",
    github_handle="PeterHartog",
    credit_type=CreditType.NONE,
)
class Decoder(pl.LightningModule):
    def __init__(
        self,
        model_args: DecoderArguments,
    ) -> None:
        super().__init__()

        self.positional_encoding = SequencePositionalEncoding(
            emb_size=model_args.emb_size,
            dropout=model_args.dropout,
            maxlen=model_args.max_seq_len,
        )
        self.tgt_tok_emb = TokenEmbedding(
            vocab_size=model_args.tgt_vocab_size, emb_size=model_args.emb_size
        )
        norm = nn.LayerNorm(model_args.emb_size)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_args.emb_size,
            nhead=model_args.num_heads,
            dim_feedforward=model_args.dim_feedforward,
            dropout=model_args.dropout,
            activation="relu",
        )

        if model_args.weight_sharing:
            self.decoder = WeightSharingDecoder(
                decoder_layer=decoder_layer,
                num_layers=model_args.num_decoder_layers,
                norm=norm,
            )
        else:
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=model_args.num_decoder_layers,
                norm=norm,
            )

    # (ignore mypy error using type: ignore)
    def forward(  # type: ignore
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        memory_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        tgt_embedded = self.positional_encoding(self.tgt_tok_emb(tgt))
        return self.decoder(
            tgt=tgt_embedded,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
