import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        pass

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        pass
