import torch
from torch import nn
from math import sqrt
from einops import einsum
from jaxtyping import Float


class Linear(nn.Module):
    """
    linear module
    y=x@W^T
    """

    def __init__(self, in_features: int, out_features: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        sigma = sqrt(2.0 / (in_features + out_features))
        self.weights: Float[torch.Tensor, "out_feat in_feat"] = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty((out_features, in_features), device=device, dtype=dtype), mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "... in_feat, out_feat in_feat -> ... out_feat")
