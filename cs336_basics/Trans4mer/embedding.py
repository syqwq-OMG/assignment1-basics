import torch
from torch import nn
from jaxtyping import Float

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device = None, dtype: torch.dtype = None):
        """
        num_embeddings: the size of vocab

        embedding_dim: the dimension of embedding
        """
        super().__init__()
        self.weights: Float[torch.Tensor, "num_embeddings embedding_dim"] = nn.Parameter(
            nn.init.trunc_normal_(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype), mean=0.0, std=1.0, a=-3.0, b=3.0)
        )

    def forward(self, token_ids: torch.Tensor) -> Float[torch.Tensor, " ... d_model"]:
        return self.weights[token_ids]
