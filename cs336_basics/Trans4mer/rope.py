import torch
from torch import nn


class RoPE(nn.Module):
    def __get_buffer(
        self, theta: float, d_k: int, seq_len: int, device: torch.device = None, dtype: torch.dtype = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        i = torch.arange(seq_len, device=device, dtype=dtype)
        k = torch.arange(d_k // 2, device=device, dtype=dtype)
        angles = i.unsqueeze(-1) * (1.0 / (theta ** (2 * k / d_k)))
        return torch.cos(angles), torch.sin(angles)

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        assert d_k % 2 == 0, "[RoPE]: d_k must be even"

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        for x, y in zip(["cos_theta_", "sin_theta_"], self.__get_buffer(theta, d_k, max_seq_len, device=device, dtype=dtype)):
            self.register_buffer(x, y, persistent=False)
        # cos_theta_[i:seq_len][k:d_k//2] = cos(i / theta^(2k/d_k))
        # sin_theta_[i:seq_len][k:d_k//2] = sin(i / theta^(2k/d_k))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_k, "[RoPE]: input tensor last dimension must be equal to d_k"
        cos_theta_i = self.cos_theta_[token_positions]
        sin_theta_i = self.sin_theta_[token_positions]
        x_even, x_odd = x[..., ::2], x[..., 1::2]
        r_even = x_even * cos_theta_i - x_odd * sin_theta_i
        r_odd = x_even * sin_theta_i + x_odd * cos_theta_i
        # use torch.stack + flatten
        return torch.stack((r_even, r_odd), dim=-1).flatten(-2)
