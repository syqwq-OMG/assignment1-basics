from __future__ import annotations

import argparse
import random
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.BPE.bpe import BPE
from cs336_basics.Trans4mer.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.Trans4mer.data import get_batch
from cs336_basics.Trans4mer.operat0r import cross_entropy_loss, gradient_clipping
from cs336_basics.Trans4mer.optim1zer import AdamW, learning_rate_schedule
from cs336_basics.Trans4mer.transformer import TransformerLM
from cs336_basics.decode import decode


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a decoder-only Transformer language model.")

    data = parser.add_argument_group("data")
    data.add_argument("--train-data", type=Path, required=True, help="Tokenized training data (.npy or raw binary).")
    data.add_argument("--val-data", type=Path, required=True, help="Tokenized validation data (.npy or raw binary).")
    data.add_argument(
        "--data-dtype",
        default="uint16",
        choices=("uint16", "uint32", "int32", "int64"),
        help="Token dtype for raw binary files; ignored for .npy files.",
    )

    model = parser.add_argument_group("model")
    model.add_argument("--vocab-size", type=int, required=True)
    model.add_argument("--context-length", type=int, default=256)
    model.add_argument("--d-model", type=int, default=512)
    model.add_argument("--num-layers", type=int, default=4)
    model.add_argument("--num-heads", type=int, default=16)
    model.add_argument("--d-ff", type=int, default=1344)
    model.add_argument("--rope-theta", type=float, default=10_000.0)

    training = parser.add_argument_group("training")
    training.add_argument("--batch-size", type=int, default=32)
    training.add_argument("--max-iters", type=int, default=10_000)
    training.add_argument("--max-lr", type=float, default=1e-3)
    training.add_argument("--min-lr", type=float, default=1e-4)
    training.add_argument("--warmup-iters", type=int, default=100)
    training.add_argument("--cosine-cycle-iters", type=int, default=None)
    training.add_argument("--weight-decay", type=float, default=0.01)
    training.add_argument("--beta1", type=float, default=0.9)
    training.add_argument("--beta2", type=float, default=0.999)
    training.add_argument("--eps", type=float, default=1e-8)
    training.add_argument("--max-grad-norm", type=float, default=1.0)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument(
        "--device",
        default="auto",
        help="PyTorch device (for example cpu, cuda, cuda:0, or mps); auto selects the best available device.",
    )

    reporting = parser.add_argument_group("reporting and checkpoints")
    reporting.add_argument("--log-interval", type=int, default=10)
    reporting.add_argument("--eval-interval", type=int, default=100)
    reporting.add_argument("--eval-iters", type=int, default=20)
    reporting.add_argument("--checkpoint-interval", type=int, default=1_000)
    reporting.add_argument("--checkpoint-path", type=Path, required=True)
    reporting.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from.")

    args = parser.parse_args(argv)
    if args.cosine_cycle_iters is None:
        args.cosine_cycle_iters = args.max_iters
    validate_args(args, parser)
    return args


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    positive_names = (
        "vocab_size",
        "context_length",
        "d_model",
        "num_layers",
        "num_heads",
        "d_ff",
        "batch_size",
        "max_iters",
        "log_interval",
        "eval_interval",
        "eval_iters",
        "checkpoint_interval",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.d_model % args.num_heads != 0:
        parser.error("--d-model must be divisible by --num-heads")
    if args.warmup_iters < 0:
        parser.error("--warmup-iters cannot be negative")
    if args.cosine_cycle_iters is not None and args.cosine_cycle_iters <= args.warmup_iters:
        parser.error("--cosine-cycle-iters must be greater than --warmup-iters")
    if not 0.0 <= args.min_lr <= args.max_lr:
        parser.error("learning rates must satisfy 0 <= --min-lr <= --max-lr")
    if args.max_grad_norm <= 0:
        parser.error("--max-grad-norm must be positive")


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_token_data(path: Path, raw_dtype: str) -> np.ndarray:
    """Open tokenized data without loading the complete array into RAM."""
    if not path.is_file():
        raise FileNotFoundError(f"token data does not exist: {path}")

    if path.suffix == ".npy":
        data = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        data = np.memmap(path, dtype=np.dtype(raw_dtype), mode="r")

    if data.ndim != 1:
        raise ValueError(f"token data must be one-dimensional, got shape {data.shape} from {path}")
    if not np.issubdtype(data.dtype, np.integer):
        raise TypeError(f"token data must have an integer dtype, got {data.dtype} from {path}")
    return data


def language_model_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Average next-token cross entropy over batch and sequence dimensions."""
    return cross_entropy_loss(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


@torch.inference_mode()
def estimate_loss(
    model: TransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    eval_iters: int,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for _ in range(eval_iters):
        inputs, targets = get_batch(dataset, batch_size, context_length, str(device))
        total_loss += language_model_loss(model(inputs), targets).item()
    model.train()
    return total_loss / eval_iters


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    train_data = load_token_data(args.train_data, args.data_dtype)
    val_data = load_token_data(args.val_data, args.data_dtype)
    minimum_tokens = args.context_length + 1
    if len(train_data) < minimum_tokens or len(val_data) < minimum_tokens:
        raise ValueError(f"each dataset must contain at least {minimum_tokens} tokens")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        context_length=args.context_length,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)
    optimizer = AdamW(
        model.parameters(), lr=args.max_lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay
    )

    start_iteration = 0
    if args.resume is not None:
        start_iteration = load_checkpoint(args.resume, model, optimizer)
        if start_iteration > args.max_iters:
            raise ValueError(f"checkpoint iteration {start_iteration} exceeds requested --max-iters {args.max_iters}")

    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.train()
    started_at = time.perf_counter()
    running_loss = 0.0
    running_steps = 0

    print(
        f"device={device} parameters={sum(parameter.numel() for parameter in model.parameters()):,} "
        f"train_tokens={len(train_data):,} val_tokens={len(val_data):,} start_iteration={start_iteration}"
    )

    for iteration in range(start_iteration, args.max_iters):
        lr = learning_rate_schedule(iteration, args.max_lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, str(device))
        optimizer.zero_grad(set_to_none=True)
        loss = language_model_loss(model(inputs), targets)
        loss.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        completed_iterations = iteration + 1
        running_loss += loss.item()
        running_steps += 1

        if completed_iterations % args.log_interval == 0 or completed_iterations == args.max_iters:
            elapsed = time.perf_counter() - started_at
            tokens_seen = completed_iterations * args.batch_size * args.context_length
            print(
                f"iter={completed_iterations}/{args.max_iters} train_loss={running_loss / running_steps:.4f} "
                f"lr={lr:.6g} tokens={tokens_seen:,} elapsed={elapsed:.1f}s"
            )
            running_loss = 0.0
            running_steps = 0

        if completed_iterations % args.eval_interval == 0 or completed_iterations == args.max_iters:
            val_loss = estimate_loss(model, val_data, args.batch_size, args.context_length, args.eval_iters, device)
            elapsed = time.perf_counter() - started_at
            print(f"iter={completed_iterations}/{args.max_iters} val_loss={val_loss:.4f} elapsed={elapsed:.1f}s")

        if completed_iterations % args.checkpoint_interval == 0 or completed_iterations == args.max_iters:
            save_checkpoint(model, optimizer, completed_iterations, args.checkpoint_path)
            print(f"checkpoint={args.checkpoint_path} iteration={completed_iterations}")


def parse_decode_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained Transformer language model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-prefix",
        required=True,
        help="Prefix used with BPE.save(), without .vocab.json/.merges.json suffixes.",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0, help="Use 0 for greedy decoding.")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--end-token", default="<|endoftext|>")
    parser.add_argument("--completion-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")

    model = parser.add_argument_group("model (must match training)")
    model.add_argument("--vocab-size", type=int, default=None, help="Defaults to the tokenizer vocabulary size.")
    model.add_argument("--context-length", type=int, default=256)
    model.add_argument("--d-model", type=int, default=512)
    model.add_argument("--num-layers", type=int, default=4)
    model.add_argument("--num-heads", type=int, default=16)
    model.add_argument("--d-ff", type=int, default=1344)
    model.add_argument("--rope-theta", type=float, default=10_000.0)

    args = parser.parse_args(argv)
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens cannot be negative")
    if args.temperature < 0:
        parser.error("--temperature cannot be negative")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must satisfy 0 < --top-p <= 1")
    for name in ("context_length", "d_model", "num_layers", "num_heads", "d_ff"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.d_model % args.num_heads != 0:
        parser.error("--d-model must be divisible by --num-heads")
    return args


def load_model_checkpoint(checkpoint_path: Path, model: TransformerLM, device: torch.device) -> int | None:
    checkpoint: Any = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must contain a mapping of training state")
    if "model" not in checkpoint:
        raise KeyError("checkpoint does not contain model state")
    model.load_state_dict(checkpoint["model"])
    iteration = checkpoint.get("iteration")
    return None if iteration is None else int(iteration)


def run_decode(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    tokenizer = BPE()
    tokenizer.load(args.tokenizer_prefix)

    tokenizer_vocab_size = len(tokenizer.get_vocab())
    vocab_size = tokenizer_vocab_size if args.vocab_size is None else args.vocab_size
    if vocab_size != tokenizer_vocab_size:
        raise ValueError(
            f"model vocabulary size ({vocab_size}) does not match tokenizer vocabulary size ({tokenizer_vocab_size})"
        )

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        context_length=args.context_length,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)
    iteration = load_model_checkpoint(args.checkpoint, model, device)

    prompt_tokens = tokenizer.encode(args.prompt)
    if not prompt_tokens:
        raise ValueError("the prompt must encode to at least one token")
    end_token_id = tokenizer.bytes2int_dict.get(args.end_token.encode("utf-8"))

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_tokens = decode(
        model=model,
        prompt_tokens=prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        context_length=args.context_length,
        temperature=args.temperature,
        top_p=args.top_p,
        end_token_id=end_token_id,
    )

    tokens_to_print = output_tokens[len(prompt_tokens) :] if args.completion_only else output_tokens
    if iteration is not None:
        print(f"Loaded checkpoint at iteration {iteration}.", file=sys.stderr)
    print(tokenizer.decode(tokens_to_print))


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "decode":
        run_decode(parse_decode_args(arguments[1:]))
        return
    if arguments and arguments[0] == "train":
        arguments = arguments[1:]
    train(parse_args(arguments))


if __name__ == "__main__":
    main()
