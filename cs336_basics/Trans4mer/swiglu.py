import torch
from torch import nn
from jaxtyping import Float
from .operat0r import silu
from .linear import Linear


class SwiGLU(nn.Module):
    """
    SwiGLU module:
    FFN(x) = W2 * (SiLU(W1 * x) * W3 * x)
    """

    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else (int(8 * d_model / 3) // 64 + 1) * 64
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: Float[torch.Tensor, "batch_size seq_len d_model"]) -> Float[torch.Tensor, "batch_size seq_len d_model"]:
        return self.w2(silu(self.w1(x)) * self.w3(x))
