from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config import _ABCDataClass
from ..utils.metacoding import CreditType
from ..utils.typescripts import Tensor
from .embedding.embedding import TokenEmbedding
from .embedding.positional import SequencePositionalEncoding
from .modelchoice import ModelChoice
from .modules.modules import WeightSharingDecoder, WeightSharingEncoder


@ModelChoice.register_choice("pl_decoder", "Peter Hartog", CreditType.NONE)
class Decoder(pl.LightningModule):
    def __init__(
        self,
        tgt_vocab_size: int,
        num_decoder_layers: int,
        emb_size: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        weight_sharing: bool = False,
        max_seq_len: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.positional_encoding = SequencePositionalEncoding(
            emb_size=emb_size, dropout=dropout, maxlen=max_seq_len
        )
        self.tgt_tok_emb = TokenEmbedding(
            vocab_size=tgt_vocab_size, emb_size=emb_size
        )
        norm = nn.LayerNorm(emb_size)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=F.relu,
        )

        if weight_sharing:
            self.decoder = WeightSharingDecoder(
                decoder_layer=decoder_layer,
                num_layers=num_decoder_layers,
                norm=norm,
            )
        else:
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=num_decoder_layers,
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
