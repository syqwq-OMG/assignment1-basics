import torch
from torch import nn
from jaxtyping import Float
from .attention import MultiHeadSelfAttention
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU
from .rope import RoPE


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RoPE = None):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, rope=rope)
        self.norm1 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: Float[torch.Tensor, "... seq_len d_model"]) -> Float[torch.Tensor, "... seq_len d_model"]:
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
