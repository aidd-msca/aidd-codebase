from typing import Optional

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config import _ABCDataClass
from ..utils.metacoding import CreditType
from ..utils.typescripts import Tensor
from .embedding.embedding import TokenEmbedding
from .embedding.positional import SequencePositionalEncoding
from .modelchoice import ModelChoice
from .modules.modules import WeightSharingEncoder


@ModelChoice.register_arguments("pl_encoder")
class EncoderArguments(_ABCDataClass):
    NAME: str = "pl_encoder"

    SHARE_WEIGHT: bool = False

    EMB_SIZE: int = 512

    NHEAD: int = 8
    DROPOUT: float = 0.1
    FFN_HID_DIM: int = 512
    NUM_ENCODER_LAYERS: int = 3
    NUM_DECODER_LAYERS: int = 3


@ModelChoice.register_choice("pl_encoder", "Peter Hartog", CreditType.NONE)
class Encoder(pl.LightningModule):
    def __init__(
        self,
        src_vocab_size: int,
        num_encoder_layers: int,
        emb_size: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        weight_sharing: bool = False,
        max_seq_len: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.src_vocab_size = src_vocab_size
        self.num_encoder_layers = num_encoder_layers
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.positional_encoding = SequencePositionalEncoding(
            emb_size=emb_size, dropout=dropout, maxlen=max_seq_len
        )
        self.src_tok_emb = TokenEmbedding(
            vocab_size=src_vocab_size, emb_size=emb_size
        )

        norm = nn.LayerNorm(emb_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=F.relu,
        )

        if weight_sharing:
            self.encoder = WeightSharingEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_encoder_layers,
                norm=norm,
            )
        else:
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_encoder_layers,
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
