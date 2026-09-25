import torch
from torch import nn
from einops import einsum
from jaxtyping import Float


class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps:float=1e-5, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass