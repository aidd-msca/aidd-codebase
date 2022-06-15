from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from aidd_codebase.utils.metacoding import CreditType
from aidd_codebase.utils.typescripts import Tensor
from .embedding.embedding import TokenEmbedding
from .embedding.positional import SequencePositionalEncoding
from .modelchoice import ModelChoice
from .modules.modules import WeightSharingDecoder, WeightSharingEncoder


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


@ModelChoice.register_choice("pl_seq2seq", "Peter Hartog", CreditType.NONE)
class Seq2Seq(pl.LightningModule):
    def __init__(
        self,
        encoder: Encoder,
        tgt_vocab_size: int,
        num_decoder_layers: int,
        emb_size: int,
        num_heads: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        weight_sharing: bool = False,
        pad_idx: Optional[int] = None,
        max_seq_len: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["Encoder"])

        self.pad_idx = pad_idx

        self.encoder = encoder
        self.decoder = Decoder(
            tgt_vocab_size=tgt_vocab_size,
            num_decoder_layers=num_decoder_layers,
            emb_size=emb_size,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            weight_sharing=weight_sharing,
            max_seq_len=max_seq_len,
        )
        self.fc_out = nn.Linear(emb_size, tgt_vocab_size)

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
