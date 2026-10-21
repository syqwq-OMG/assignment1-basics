from __future__ import annotations

# ruff: noqa: E402 -- course timing must start before importing ML dependencies.
import time

PROGRAM_STARTED_UNIX = time.time()
PROGRAM_STARTED_CLOCK = time.perf_counter()

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from cs336_basics.BPE.bpe import BPE
from cs336_basics.encoding import encode_file
from cs336_basics.main import parse_args as parse_training_args, train
from wandb_record import experiment

ROOT = Path(__file__).resolve().parent


def read_previous_run(path, tokenizer_hashes, args):
    prior = json.loads(path.read_text(encoding="utf-8"))
    previous = float(prior["time/total_seconds"])
    if prior["config"]["tokenizer_sha256"] != tokenizer_hashes:
        raise ValueError("Cannot change tokenizer when resuming")
    checkpoint_path = Path(prior["checkpoint_path"])
    if args.resume is None:
        # An interrupted encoding run has consumed time but no model to load.
        if checkpoint_path.exists() or prior["status"] != "failed":
            raise ValueError("Restart without --resume requires a failed run with no checkpoint")
    else:
        for key in ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff", "rope_theta"):
            if prior["config"][key] != getattr(args, key):
                raise ValueError(f"Resumed model configuration differs: {key}")
        if args.resume.resolve() != checkpoint_path.resolve():
            raise ValueError("--resume must point to the checkpoint in the previous result.json")
    return previous, prior["run_id"]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Official OWT Transformer run with source snapshots, W&B, full validation and timing.",
        epilog="Additional model/training flags are forwarded to cs336_basics.main (python -m cs336_basics.main --help).",
    )
    parser.add_argument("--tokenizer-prefix", type=Path, required=True)
    parser.add_argument("--train-text", type=Path, default=Path("/ai/lmfs2026/data/assignment1-data/owt_train.txt"))
    parser.add_argument("--val-text", type=Path, default=Path("/ai/lmfs2026/data/assignment1-data/owt_valid.txt"))
    parser.add_argument(
        "--previous-timing",
        type=Path,
        help="Previous result.json; also supports restarting a failed run that has no checkpoint.",
    )
    parser.add_argument("--run-name", help="Distinct W&B experiment name; defaults to WANDB_NAME or student-task.")
    parser.add_argument("--max-train-seconds", type=float, default=21_600)
    parser.add_argument("--finalize-reserve-seconds", type=float, default=1_800)
    parser.add_argument(
        "--encoding-workers", type=int, default=min(8, getattr(os, "process_cpu_count", os.cpu_count)() or 1)
    )
    parser.add_argument("--encoding-cache-size", type=int, default=65_536)
    parser.add_argument("--encoding-chunk-mib", type=int, default=1)
    options, forwarded = parser.parse_known_args(argv)
    if options.encoding_workers < 1 or options.encoding_cache_size < 0 or options.encoding_chunk_mib < 1:
        parser.error("Encoding workers/chunk size must be positive; cache size cannot be negative")
    if not math.isfinite(options.max_train_seconds) or not 0 < options.max_train_seconds <= 21_600:
        parser.error("--max-train-seconds must be in (0, 21600]")
    if (
        not math.isfinite(options.finalize_reserve_seconds)
        or not 0 < options.finalize_reserve_seconds < options.max_train_seconds
    ):
        parser.error("--finalize-reserve-seconds must be positive and below the total budget")
    for path in (options.train_text, options.val_text):
        if not path.is_file():
            parser.error(f"Text corpus does not exist: {path}")

    tokenizer = BPE()
    tokenizer.load(str(options.tokenizer_prefix))
    vocab = tokenizer.get_vocab()
    if sorted(vocab) != list(range(len(vocab))):
        parser.error("Tokenizer IDs must be contiguous starting at zero")
    if len(vocab) > 32_000:
        parser.error("Official vocabulary limit is 32000, including special tokens")
    tokenizer_hashes = {
        suffix: hashlib.sha256(Path(f"{options.tokenizer_prefix}.{suffix}.json").read_bytes()).hexdigest()
        for suffix in ("vocab", "merges", "special_tokens")
    }
    # These temporary argument values are replaced with this run's private outputs.
    args = parse_training_args(
        [
            "--train-data",
            str(options.train_text),
            "--val-data",
            str(options.val_text),
            "--vocab-size",
            str(len(vocab)),
            "--checkpoint-path",
            str(ROOT / "outputs" / "last.pt"),
            "--log-interval",
            "100",
            *forwarded,
        ]
    )
    if args.log_interval != 100:
        parser.error("Official training PPL is logged every 100 optimizer updates")
    if args.vocab_size != len(vocab):
        parser.error("--vocab-size must match the actual saved BPE vocabulary")
    previous = 0.0
    resume_from_run = None
    if options.previous_timing:
        try:
            previous, resume_from_run = read_previous_run(options.previous_timing, tokenizer_hashes, args)
        except ValueError as error:
            parser.error(str(error))
    elif args.resume is not None:
        parser.error("--resume requires --previous-timing so cumulative budget is preserved")
    if not math.isfinite(previous) or not 0 <= previous < options.max_train_seconds:
        parser.error("Prior elapsed time is invalid or the cumulative budget is exhausted")

    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(
        {
            "train_file": str(options.train_text.resolve()),
            "valid_file": str(options.val_text.resolve()),
            "bpe_prefix": str(options.tokenizer_prefix.resolve()),
            "tokenizer_sha256": tokenizer_hashes,
            "special_tokens": tokenizer.get_special_tokens(),
            "previous_elapsed_seconds": previous,
            "resume_from_run": resume_from_run,
            "restart_without_checkpoint": options.previous_timing is not None and args.resume is None,
            "run_name": options.run_name,
            "max_train_seconds": options.max_train_seconds,
            "finalize_reserve_seconds": options.finalize_reserve_seconds,
            "encoding_workers": options.encoding_workers,
            "encoding_cache_size": options.encoding_cache_size,
            "encoding_chunk_mib": options.encoding_chunk_mib,
            "validation_policy": "consecutive non-overlapping context blocks; every target once; include final partial block",
            "extra_tokens_inserted": False,
        }
    )
    deadline = PROGRAM_STARTED_CLOCK + options.max_train_seconds - previous
    stop_training = deadline - options.finalize_reserve_seconds
    with experiment("transformer", config, root=ROOT, name=options.run_name) as run:
        out = ROOT / "outputs" / run.id
        args.checkpoint_path = out / "last.pt"
        args.program_started_clock = PROGRAM_STARTED_CLOCK
        args.previous_elapsed_seconds = previous
        report = {
            "run_id": run.id,
            "run_name": run.name,
            "resume_from_run": resume_from_run,
            "config": config,
            "checkpoint_path": str(args.checkpoint_path),
            "status": "failed",
        }
        run.define_metric("train/step")
        run.define_metric("train/ppl", step_metric="train/step")
        run.define_metric("time/elapsed_seconds", step_metric="train/step")
        print(f"Run ID: {run.id}\nW&B name: {run.name}\nW&B: {run.url}\nLocal outputs: {out}", flush=True)
        try:
            if time.perf_counter() >= stop_training:
                raise TimeoutError("No time remains before the finalization reserve.")
            for name, source in (("train", options.train_text), ("val", options.val_text)):
                destination = out / f"{name}.bin"
                dtype, count = encode_file(
                    tokenizer,
                    source,
                    destination,
                    stop_training,
                    chunk_size=options.encoding_chunk_mib * 1024 * 1024,
                    workers=options.encoding_workers,
                    cache_size=options.encoding_cache_size,
                )
                setattr(args, f"{name}_data", destination)
                args.data_dtype = dtype
                print(f"Encoded {source}: {count:,} tokens -> {destination}", flush=True)
            result = train(args, run=run, stop_training_clock=stop_training)
            report.update(result)
            report["status"] = "completed"
        finally:
            # Training synchronizes CUDA before returning; uploads below are excluded.
            ended_unix = time.time()
            current = time.perf_counter() - PROGRAM_STARTED_CLOCK
            total = previous + current
            report.update(
                {
                    "time/program_started_utc": datetime.fromtimestamp(PROGRAM_STARTED_UNIX, UTC).isoformat(),
                    "time/experiment_ended_utc": datetime.fromtimestamp(ended_unix, UTC).isoformat(),
                    "time/previous_elapsed_seconds": previous,
                    "time/current_run_seconds": current,
                    "time/total_seconds": total,
                    "time/difference_from_6h_seconds": total - 21_600,
                    "time/within_6h": total <= 21_600,
                }
            )
            if report["status"] == "completed":
                count = report["val_token_count"]
                nll = report["val_nll_sum"] / count if count > 0 else float("nan")
                if math.isfinite(nll) and 0 <= nll < 709:
                    report.update(
                        {
                            "val/ppl": math.exp(nll),
                            "val/token_count": count,
                            "train/final_step": report["step"],
                            "train/total_seconds": total,
                        }
                    )
                else:
                    report["status"] = "invalid_validation"
            if total > options.max_train_seconds:
                report["status"] = "over_budget"
            path = out / "result.json"
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            run.summary.update({key: value for key, value in report.items() if key != "config"})
            run.save(str(path), base_path=str(out), policy="now")
            print(json.dumps({key: value for key, value in report.items() if key != "config"}), flush=True)
        if report["status"] != "completed":
            raise RuntimeError(f"Run did not produce a valid in-budget score: {report['status']}")


if __name__ == "__main__":
    main(sys.argv[1:])
