import warnings
from typing import Optional

import torch.nn as nn

from ...math_utils.cross_product import ScaledDotProductChoice
from aidd_codebase.utils.metacoding import DictChoiceFactory
from aidd_codebase.utils.typescripts import Tensor


class AttentionChoice(DictChoiceFactory):
    pass


AttentionChoice.register_prebuilt_choice(
    call_name="torch_multihead_attn", callable_cls=nn.MultiHeadAttention
)


# TODO build versions for skip_con|identical|batch_first


@AttentionChoice.register_choice("multihead_attn")
class MultiheadAttention(nn.Module):
    def __init__(
        self,
        qdim: int,
        kdim: int,
        vdim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        batch_first: bool = False,
        dot_product_method: str = "matmul",
    ):
        super().__init__()
        assert (
            qdim % num_heads == 0
        ), "Embedding dimension must be divisible by the number of heads."

        self.qdim = qdim
        self.kdim = kdim
        self.vdim = vdim
        self.num_heads = num_heads
        self.head_dim = qdim // num_heads

        if qdim == kdim and qdim == vdim:
            self.qkv_proj = nn.Linear(qdim, 3 * qdim, bias)
        else:
            self.q_proj = nn.Linear(qdim, qdim, bias)
            self.k_proj = nn.Linear(kdim, qdim, bias)
            self.v_proj = nn.Linear(vdim, qdim, bias)

        self.o_proj = nn.Linear(qdim, qdim)

        self.dot_product = ScaledDotProductChoice.get_choice(
            dot_product_method
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # query: [B,Q] (hidden state, decoder output, etc.)
        # keys: [T,B,K] (encoder outputs)
        # values: [T,B,V] (encoder outputs)
        # assume Q == K

        # not_batched = False
        # not_batch_first = False
        # if not batch_first_check(tensor=x):
        #     x = convert_to_batch_first(x)
        #     not_batch_first = True
        # if not is_batch(tensor=x):
        #     x = convert_unbatched(x)
        #     not_batched = True

        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(
            batch_size, seq_length, self.num_heads, 3 * self.head_dim
        )
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = self.dot_product(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout
        )
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        return o, attention


@AttentionChoice.register_choice("self_attn")
class SelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.attention = AttentionChoice.get_choice("self_attn")
        self.layernorm = nn.LayerNorm(input_dim)

    # layernorm then multihead

    def forward(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        residual = x
        x = self.attention(x, x, x)
        x += residual
        x = self.layernorm(x)
        return x

    pass


@AttentionChoice.register_choice("cross_attn")
class CrossAttention(nn.Module):
    # qv are from latent vector
    pass


@AttentionChoice.register_choice("geom_attn")
class GeometricAttention(nn.Module):
    pass


class Attention(nn.Module):
    def __init__(
        self,
        attention_type: str,
        embed_dim: int,
        kdim: int,
        vdim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        batch_first: bool = False,
        dot_product_method: str = "matmul",
    ):
        super().__init__()
        AttentionChoice.validate_choice(attention_type)
        ScaledDotProductChoice.validate_choice(dot_product_method)

        if embed_dim != kdim and not attention_type == "cross_attn":
            warnings.warn("embed_dim must match kdim.")
            warnings.warn(f"Using cross_attn instead of {attention_type}.")
            attention = AttentionChoice.get_choice("cross_attn")
        else:
            attention = AttentionChoice.get_choice(attention_type)

        self.attention = attention(
            embed_dim=embed_dim,
            kdim=kdim,
            vdim=vdim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            batch_first=batch_first,
        )

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        pad_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        return self.attention(
            q, k, v, key_padding_mask=pad_mask, attn_mask=attn_mask
        )
