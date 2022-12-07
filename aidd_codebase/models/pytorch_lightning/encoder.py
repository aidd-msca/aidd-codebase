import inspect
from dataclasses import dataclass
from typing import Dict, Optional
from aidd_codebase.authors import Peter

import torch.nn as nn
from aidd_codebase.models.embedding.embedding import TokenEmbedding
from aidd_codebase.models.embedding.positional import SequencePositionalEncoding
from aidd_codebase.registries import ModelRegistry
from aidd_codebase.models.modules.modules import WeightSharingEncoder
from aidd_codebase.models.pytorch.transformer_custom import (
    CustomTransformerEncoderLayer,
)
from abstract_codebase.accreditation import CreditType
from aidd_codebase.utils.typescripts import Tensor

import pytorch_lightning as pl


@ModelRegistry.register_arguments(key="pl_encoder")
@dataclass(unsafe_hash=True)
class EncoderArguments:
    src_vocab_size = 112
    num_encoder_layers = 3
    emb_size = 512
    num_heads = 8
    dim_feedforward = 512
    dropout: float = 0.1

    batch_first: bool = False
    weight_sharing: bool = False
    need_attention_weights: bool = True
    average_attention_weights: bool = True

    max_seq_len: Optional[int] = None


@ModelRegistry.register(
    key="pl_encoder",
    credit=Peter,
    credit_type=CreditType.NONE,
)
class Encoder(pl.LightningModule):
    def __init__(self, model_args: Dict) -> None:
        super().__init__()

        args = EncoderArguments(
            **{k: v for k, v in model_args.items() if k in inspect.signature(EncoderArguments).parameters}
        )
        self.batch_first = args.batch_first

        self.src_vocab_size = args.src_vocab_size
        self.num_encoder_layers = args.num_encoder_layers
        self.emb_size = args.emb_size
        self.num_heads = args.num_heads
        self.dim_feedforward = args.dim_feedforward
        self.dropout = args.dropout

        self.positional_encoding = SequencePositionalEncoding(
            emb_size=args.emb_size,
            dropout=args.dropout,
            maxlen=args.max_seq_len,
        )
        self.src_tok_emb = TokenEmbedding(vocab_size=args.src_vocab_size, emb_size=args.emb_size)

        norm = nn.LayerNorm(args.emb_size)

        if args.need_attention_weights or args.average_attention_weights:
            encoder_layer = CustomTransformerEncoderLayer(
                batch_first=args.batch_first,
                d_model=args.emb_size,
                nhead=args.num_heads,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
                activation="relu",
                need_weights=args.need_attention_weights,
                average_attn=args.average_attention_weights,
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=args.emb_size,
                nhead=args.num_heads,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
                activation="relu",
            )

        if args.weight_sharing:
            self.encoder = WeightSharingEncoder(
                encoder_layer=encoder_layer,
                num_layers=args.num_encoder_layers,
                norm=norm,
            )
        else:
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=args.num_encoder_layers,
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
