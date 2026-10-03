import torch
from torch import nn
from einops import einsum
from jaxtyping import Float, Bool
from .operat0r import softmax
import math


def scaled_dot_product_attention(
    query: Float[torch.Tensor, "batch_size ... seq_len d_k"],
    key: Float[torch.Tensor, "batch_size ... seq_len d_k"],
    value: Float[torch.Tensor, "batch_size ... seq_len d_v"],
    mask: Bool[torch.Tensor, "batch_size ... seq_len seq_len"] = None,
) -> Float[torch.Tensor, "batch_size ... seq_len d_v"]:
    """
    Compute the scaled dot-product attention.

    Args:
        query: Float[Tensor, "batch_size, ..., seq_len, d_k"]: The query tensor.
        key: Float[Tensor, "batch_size, ..., seq_len, d_k"]: The key tensor.
        value: Float[Tensor, "batch_size, ..., seq_len, d_v"]: The value tensor.
        mask: Bool[Tensor, "batch_size, ..., seq_len, seq_len"]: Optional mask tensor.

    Returns:
        softmax(QK^T / sqrt(d_k)) V with mask
    """

    d_k = query.shape[-1]
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores.masked_fill_(~mask, float("-inf"))

    attn_weights = softmax(scores, dim=-1)
    return attn_weights @ value
