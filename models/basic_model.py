import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer

from models.embedding.embedding import TokenEmbedding
from models.embedding.positional import SequencePositionalEncoding


class Perceptron(torch.nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.fc = torch.nn.Linear(1, 1)
        self.relu = torch.nn.ReLU()  # instead of Heaviside step fn

    def forward(self, x):
        output = self.fc(x)
        output = self.relu(x)  # instead of Heaviside step fn
        return output


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, 1),
            torch.nn.ReLU(),
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.model(x)


class Seq2SeqTransformer(nn.Module):
    """Sequence to Sequence transformer used for smile to smile
    classification."""

    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = SequencePositionalEncoding(
            emb_size, dropout=dropout
        )

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )


def generate_square_subsequent_mask(sz: int, device: str) -> Tensor:
    """Generates an upper-trianghular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1).to(
        device
    )


def create_src_mask(src: Tensor, device: str) -> Tensor:
    src_seq_len = src.shape[0]
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(
        torch.bool
    )
    return src_mask


def create_tgt_mask(tgt: Tensor, device: str) -> Tensor:
    tgt_seq_len = tgt.shape[0]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    return tgt_mask


def create_padding_mask(seq: Tensor, PAD_IDX) -> Tensor:
    return (seq == PAD_IDX).transpose(0, 1)
