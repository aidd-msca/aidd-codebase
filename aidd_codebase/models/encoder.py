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
from .modules.modules import WeightSharingEncoder


@ModelChoice.register_arguments(call_name="pl_encoder")
@dataclass(unsafe_hash=True)
class EncoderArguments(_ABCDataClass):
    src_vocab_size = 112
    num_encoder_layers = 3
    emb_size = 512
    num_heads = 8
    dim_feedforward = 512
    dropout: float = 0.1
    weight_sharing: bool = False
    max_seq_len: Optional[int] = None


@ModelChoice.register_choice(
    call_name="pl_encoder",
    author="Peter Hartog",
    github_handle="PeterHartog",
    credit_type=CreditType.NONE,
)
class Encoder(pl.LightningModule):
    def __init__(self, model_args: EncoderArguments) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.src_vocab_size = model_args.src_vocab_size
        self.num_encoder_layers = model_args.num_encoder_layers
        self.emb_size = model_args.emb_size
        self.num_heads = model_args.num_heads
        self.dim_feedforward = model_args.dim_feedforward
        self.dropout = model_args.dropout

        self.positional_encoding = SequencePositionalEncoding(
            emb_size=model_args.emb_size,
            dropout=model_args.dropout,
            maxlen=model_args.max_seq_len,
        )
        self.src_tok_emb = TokenEmbedding(
            vocab_size=model_args.src_vocab_size, emb_size=model_args.emb_size
        )

        norm = nn.LayerNorm(model_args.emb_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_args.emb_size,
            nhead=model_args.num_heads,
            dim_feedforward=model_args.dim_feedforward,
            dropout=model_args.dropout,
            activation="relu",
        )

        if model_args.weight_sharing:
            self.encoder = WeightSharingEncoder(
                encoder_layer=encoder_layer,
                num_layers=model_args.num_encoder_layers,
                norm=norm,
            )
        else:
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=model_args.num_encoder_layers,
                norm=norm,
            )

    # (ignore mypy error using type: ignore)
    def forward(  # type: ignore
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        src_embedded = self.positional_encoding(self.src_tok_emb(src))
        return self.encoder(src_embedded, mask, src_padding_mask)
