import torch
from torch import nn
from einops import einsum, rearrange
from jaxtyping import Float, Bool
from .operat0r import softmax
from .linear import Linear
from .rope import RoPE

import math


def scaled_dot_product_attention(
    query: Float[torch.Tensor, "batch_size ... seq_len d_k"],
    key: Float[torch.Tensor, "batch_size ... seq_len d_k"],
    value: Float[torch.Tensor, "batch_size ... seq_len d_v"],
    mask: Bool[torch.Tensor, "... seq_len seq_len"] = None,
) -> Float[torch.Tensor, "batch_size ... seq_len d_v"]:
    """
    softmax(QK^T / sqrt(d_k)) V with mask
    """

    d_k = query.shape[-1]
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores.masked_fill_(~mask, float("-inf"))

    attn_weights = softmax(scores, dim=-1)
    return attn_weights @ value


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_out: int = 0,
        rope: RoPE = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        """
        d_model: embedding dimension
        num_heads: number of attention heads
        d_out: output dimension, if 0, then d_out = d_model
        rope: RoPE instance for rotary positional encoding
        **default causal is True**
        """
        super().__init__()
        assert d_model % num_heads == 0, "[MultiHeadSelfAttention]: d_model must be divisible by num_heads"
        assert d_out % num_heads == 0, "[MultiHeadSelfAttention]: d_out must be divisible by num_heads"

        self.d_model = d_model
        self.d_out = d_out if d_out != 0 else d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_out // num_heads
        self.rope = rope

        self.W_q = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.W_k = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.W_v = Linear(d_model, num_heads * self.d_v, device=device, dtype=dtype)
        self.W_o = Linear(num_heads * self.d_v, self.d_out, device=device, dtype=dtype)

    def forward(
        self, x: Float[torch.Tensor, "batch_size seq_len d_model"], causal: bool = True
    ) -> Float[torch.Tensor, "batch_size seq_len d_out"]:
        seq_len = x.shape[-2]

        Q = rearrange(self.W_q(x), "b s (h d) -> b h s d", h=self.num_heads)
        K = rearrange(self.W_k(x), "b s (h d) -> b h s d", h=self.num_heads)
        V = rearrange(self.W_v(x), "b s (h d) -> b h s d", h=self.num_heads)

        if self.rope is not None:
            Q = self.rope(Q)
            K = self.rope(K)

        mask = None
        if causal:
            mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))

        attn = scaled_dot_product_attention(Q, K, V, mask)
        attn = rearrange(attn, "b h s d -> b s (h d)")
        return self.W_o(attn)
