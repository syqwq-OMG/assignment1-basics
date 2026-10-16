import numpy as np
import torch


def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:

    starts = np.random.randint(
        0,
        len(x) - context_length,
        size=batch_size,
    )

    offsets = np.arange(context_length)

    input_indices = starts[:, None] + offsets[None, :]
    target_indices = input_indices + 1

    inputs = x[input_indices]
    targets = x[target_indices]

    inputs = torch.as_tensor(inputs, dtype=torch.long, device=device)
    targets = torch.as_tensor(targets, dtype=torch.long, device=device)

    return inputs, targets
