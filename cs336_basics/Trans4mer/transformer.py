import torch
from torch import nn
from jaxtyping import Float
from .attention import MultiHeadSelfAttention
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU
from .rope import RoPE
from .embedding import Embedding
from .linear import Linear


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


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        context_length: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.rope = RoPE(theta=rope_theta, d_k=d_model // num_heads, max_seq_len=context_length)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, rope=self.rope) for _ in range(num_layers)]
        )
        self.norm = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)

    def forward(self, x: Float[torch.Tensor, "... seq_len"]) -> Float[torch.Tensor, "... seq_len vocab_size"]:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.linear(x)
        return x
