from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import wandb

from cs336_basics.BPE.bpe import BPE
from wandb_record import experiment


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "tests" / "fixtures" / "TinyStories-train.txt"
DEFAULT_SPECIAL_TOKEN = "<|endoftext|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the BPE tokenizer and record the experiment in W&B.")
    parser.add_argument(
        "--input-path", type=Path, default=DEFAULT_INPUT, help=f"Training corpus (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10_000,
        help="Total vocabulary size, including special tokens (default: 10000)",
    )
    parser.add_argument(
        "--special-token",
        action="append",
        dest="special_tokens",
        help=(
            f"Special token to reserve. Repeat this option for multiple tokens. Defaults to {DEFAULT_SPECIAL_TOKEN!r}."
        ),
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help=(
            "Explicitly use this many pre-tokenization threads. If omitted, "
            "BPE.train selects its size-aware default strategy."
        ),
    )
    parser.add_argument("--valid-path", type=Path, help="Complete validation text for the official BPE score.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_path.expanduser().resolve()
    special_tokens = args.special_tokens or [DEFAULT_SPECIAL_TOKEN]

    if not input_path.is_file():
        raise FileNotFoundError(f"Training corpus does not exist: {input_path}")
    if args.valid_path is not None and not args.valid_path.is_file():
        raise FileNotFoundError(f"Validation corpus does not exist: {args.valid_path}")
    if not 256 + len(special_tokens) <= args.vocab_size <= 32_000:
        raise ValueError("Vocabulary must contain all bytes and special tokens and be at most 32000.")

    config = {
        "input_path": str(input_path),
        "valid_file": str(args.valid_path.resolve()) if args.valid_path else None,
        "input_size_bytes": input_path.stat().st_size,
        "vocab_size": args.vocab_size,
        "special_tokens": special_tokens,
        "num_threads": args.num_threads,
        "parallel_strategy": (f"threads:{args.num_threads}" if args.num_threads is not None else "automatic"),
    }

    with experiment("train-bpe", config, root=PROJECT_ROOT) as run:
        tokenizer = BPE(special_tokens=special_tokens)

        started_at = time.perf_counter()
        tokenizer.train(input_path=input_path, vocab_size=args.vocab_size, num_threads=args.num_threads)
        train_seconds = time.perf_counter() - started_at

        vocab = tokenizer.get_vocab()
        merges = tokenizer.get_merges()
        actual_vocab_size = len(vocab)

        metrics = {
            "bpe/train_seconds": train_seconds,
            "bpe/input_size_bytes": input_path.stat().st_size,
            "bpe/actual_vocab_size": actual_vocab_size,
            "bpe/num_merges": len(merges),
            "bpe/throughput_mib_per_second": (input_path.stat().st_size / (1024**2) / train_seconds),
        }
        run.log(metrics)
        run.summary.update(metrics)

        output_dir = PROJECT_ROOT / "outputs" / run.id / "bpe"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_prefix = output_dir / "tokenizer"
        tokenizer.save(str(output_prefix))

        tokenizer_artifact = wandb.Artifact(
            name=f"bpe-tokenizer-{run.id}",
            type="tokenizer",
            metadata={
                "vocab_size": actual_vocab_size,
                "num_merges": len(merges),
                "special_tokens": special_tokens,
                "training_corpus": input_path.name,
                "training_seconds": train_seconds,
            },
        )
        for path in sorted(output_dir.glob("tokenizer.*.json")):
            tokenizer_artifact.add_file(str(path), name=path.name)
        run.log_artifact(tokenizer_artifact)

        if args.valid_path is not None:
            # The scored interval starts with a freshly loaded, cold tokenizer.
            tokenizer = BPE()
            tokenizer.load(str(output_prefix))
            with args.valid_path.open(encoding="utf-8", newline="") as handle:
                text = handle.read()
            started_at = time.perf_counter()
            ids = tokenizer.encode(text)
            decoded = tokenizer.decode(ids)
            passed = decoded == text
            valid_seconds = time.perf_counter() - started_at

            raw = text.encode("utf-8")
            first_ids = ids[:200]
            rows, offset = [], 0
            for position, token_id in enumerate(first_ids):
                piece = tokenizer.get_vocab()[token_id]
                end = offset + len(piece)
                try:
                    piece.decode("utf-8", errors="strict")
                    standalone_valid = True
                except UnicodeDecodeError:
                    standalone_valid = False
                rows.append(
                    [
                        position,
                        int(token_id),
                        piece.hex(),
                        repr(piece),
                        repr(tokenizer.decode([token_id])),
                        standalone_valid,
                        offset,
                        end,
                        piece == raw[offset:end],
                    ]
                )
                offset = end
            columns = [
                "position",
                "token_id",
                "token_hex",
                "token_bytes",
                "decoded_token",
                "standalone_utf8_valid",
                "byte_start",
                "byte_end",
                "matches_input_bytes",
            ]
            result = {
                "bpe/valid_seconds": valid_seconds,
                "bpe/valid_token_count": len(ids),
                "bpe/roundtrip_passed": passed,
                "bpe/vocab_size": actual_vocab_size,
            }
            print("First 200 token IDs:", first_ids, flush=True)
            print("First 200 tokens decoded:", repr(tokenizer.decode(first_ids)), flush=True)
            print("Token columns:", columns, flush=True)
            for row in rows:
                print(row, flush=True)
            run.log({"bpe/first_200_tokens": wandb.Table(columns=columns, data=rows)})
            run.summary.update(result)
            result_path = output_dir / "result.json"
            result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            run.save(str(result_path), base_path=str(output_dir), policy="now")
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if len(ids) < 200 or not passed:
                raise ValueError("Validation must contain at least 200 tokens and roundtrip exactly.")

        print(
            f"BPE training complete: {actual_vocab_size} vocabulary entries, "
            f"{len(merges)} merges, {train_seconds:.2f} seconds."
        )
        print(f"Tokenizer files: {output_dir}")
        print(f"Transformer --tokenizer-prefix {output_prefix}")
        print(f"W&B run: {run.url}")


if __name__ == "__main__":
    main()
