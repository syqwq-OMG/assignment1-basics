from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from typing import IO, Any, BinaryIO

import torch


CheckpointDestination = str | PathLike[str] | BinaryIO | IO[bytes]
CheckpointSource = str | PathLike[str] | BinaryIO | IO[bytes]


def save_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: CheckpointDestination
) -> None:
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(checkpoint, out)


def load_checkpoint(src: CheckpointSource, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    checkpoint: Any = torch.load(src, weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must contain a mapping of training state")

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])
