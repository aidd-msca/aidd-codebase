import math
from typing import Optional

import torch
import torch.nn.functional as F

from aidd_codebase.utils.metacoding import DictChoiceFactory
from aidd_codebase.utils.typescripts import Tensor


class ScaledDotProductChoice(DictChoiceFactory):
    pass


@ScaledDotProductChoice.register_choice("matmul")
def scaled_dot_product_matmul(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tensor:
    # attn = (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    # output = (B, Nt, E) x (B, E, Ns) -> (B, Nt, E)
    # q = [B, Nt, E]; k = [B, Ns, E]; v = [B, Ns, E]
    # B: batch; E: embedding; Nt: target size; Ns: source size
    batch_size, target_seq_length, embed_dim = q.shape
    q = q / math.sqrt(embed_dim)

    attn = torch.matmul(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn = attn.masked_fill(attn_mask == 0, -1e9)

    attn = F.softmax(attn, dim=-1)

    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)

    output = torch.matmul(attn, v.transpose(-2, -1))
    return output, attn


@ScaledDotProductChoice.register_choice("einsum")
def scaled_dot_product_einsum(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tensor:
    # attn = (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    # output = (B, Nt, E) x (B, E, Ns) -> (B, Nt, E)
    # q = [B, Nt, E]; k = [B, Ns, E]; v = [B, Ns, E]
    # B: batch; E: embedding; Nt: target size; Ns: source size
    batch_size, target_seq_length, embed_dim = q.shape
    q = q / math.sqrt(embed_dim)

    attn = torch.einsum("BtE,BsE->Bts", q, k)
    if attn_mask is not None:
        attn = attn.masked_fill(attn_mask == 0, -1e9)

    attn = F.softmax(attn, dim=-1)

    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)

    output = torch.einsum("Bts,BsE->BtE", attn, v)
    return output, attn


@ScaledDotProductChoice.register_choice("bmm")
def scaled_dot_product_bmm(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tensor:
    # attn = (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    # output = (B, Nt, E) x (B, E, Ns) -> (B, Nt, E)
    # q = [B, Nt, E]; k = [B, Ns, E]; v = [B, Ns, E]
    # B: batch; E: embedding; Nt: target size; Ns: source size
    batch_size, target_seq_length, embed_dim = q.shape
    q = q / math.sqrt(embed_dim)

    if attn_mask is not None:
        attn = torch.baddbmm(attn_mask, q, k.transpose(-2, -1))
    else:
        attn = torch.bmm(q, k.transpose(-2, -1))

    attn = F.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)

    output = torch.bmm(attn, v)
    return output, attn
