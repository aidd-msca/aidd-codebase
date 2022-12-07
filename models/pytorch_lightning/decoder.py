import inspect
from dataclasses import dataclass
from typing import Dict, Optional

import torch.nn as nn
from aidd_codebase.models.embedding.embedding import TokenEmbedding
from aidd_codebase.models.embedding.positional import SequencePositionalEncoding
from aidd_codebase.registries import ModelRegistry
from aidd_codebase.models.modules.modules import WeightSharingDecoder
from aidd_codebase.models.pytorch.transformer_custom import (
    CustomTransformerDecoderLayer,
)
from abstract_codebase.accreditation import CreditType
from aidd_codebase.utils.typescripts import Tensor
from aidd_codebase.authors import Peter

import pytorch_lightning as pl


@ModelRegistry.register_arguments(key="pl_decoder")
@dataclass(unsafe_hash=True)
class DecoderArguments:
    tgt_vocab_size = 112
    num_decoder_layers = 3
    emb_size = 512
    num_heads = 8
    dim_feedforward = 512
    dropout: float = 0.1

    weight_sharing: bool = False
    need_attention_weights: bool = False
    average_attention_weights: bool = True

    max_seq_len: Optional[int] = None


@ModelRegistry.register(
    key="pl_decoder",
    credit=Peter,
    credit_type=CreditType.NONE,
)
class Decoder(pl.LightningModule):
    def __init__(
        self,
        model_args: Dict,
    ) -> None:
        super().__init__()

        args = DecoderArguments(
            **{k: v for k, v in model_args.items() if k in inspect.signature(DecoderArguments).parameters}
        )
        # args = DecoderArguments(**model_args)

        self.positional_encoding = SequencePositionalEncoding(
            emb_size=args.emb_size,
            dropout=args.dropout,
            maxlen=args.max_seq_len,
        )
        self.tgt_tok_emb = TokenEmbedding(vocab_size=args.tgt_vocab_size, emb_size=args.emb_size)
        norm = nn.LayerNorm(args.emb_size)

        if args.need_attention_weights or args.average_attention_weights:
            decoder_layer = CustomTransformerDecoderLayer(
                d_model=args.emb_size,
                nhead=args.num_heads,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
                activation="relu",
                need_weights=args.need_attention_weights,
                average_attn=args.average_attention_weights,
            )
        else:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=args.emb_size,
                nhead=args.num_heads,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
                activation="relu",
            )

        if args.weight_sharing:
            self.decoder = WeightSharingDecoder(
                decoder_layer=decoder_layer,
                num_layers=args.num_decoder_layers,
                norm=norm,
            )
        else:
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=args.num_decoder_layers,
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
