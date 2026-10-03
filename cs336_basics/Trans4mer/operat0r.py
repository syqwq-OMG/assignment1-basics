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


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute the softmax of x along the specified dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension along which to compute the softmax. Default is -1.

    Returns:
        torch.Tensor: Softmax of the input tensor along the specified dimension.
    """
    exp_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def cross_entropy_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute the cross-entropy loss between logits and targets.
    """
    inputs = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    t = torch.log(torch.sum(torch.exp(inputs), dim=-1, keepdim=True)) - inputs.gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    return torch.mean(t)
