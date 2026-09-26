import torch

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the sigmoid of x.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Sigmoid of the input tensor.
    """
    return 1 / (1 + torch.exp(-x))

def silu(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the SiLU = x * sigmoid(x) activation function of x.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: SiLU of the input tensor.
    """
    return x * sigmoid(x)