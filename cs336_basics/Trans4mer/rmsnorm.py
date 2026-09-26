import torch
from torch import nn
from einops import einsum
from jaxtyping import Float


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.eps = eps
        self.weights: Float[torch.tensor, "d_model"] = nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x.to(torch.float32)
        
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        
        result = x / rms
        result = result * self.weights
        return result.to(in_type)
