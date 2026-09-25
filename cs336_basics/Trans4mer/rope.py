import torch
from torch import nn
from einops import einsum
from jaxtyping import Float


class RoPE(nn.Module):
    def __init__(self, theta:float, d_k:int, max_seq_len:int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        pass
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        pass