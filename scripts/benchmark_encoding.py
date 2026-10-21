"""Benchmark encoding on a private OWT prefix without W&B or GPU training."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

from cs336_basics.BPE.bpe import BPE
from cs336_basics.encoding import encode_file


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-prefix", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("/ai/lmfs2026/data/assignment1-data/owt_train.txt"))
    parser.add_argument("--sample-mib", type=int, default=32)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.sample_mib < 1 or any(workers < 1 for workers in args.workers):
        parser.error("Sample size and workers must be positive")
    with tempfile.TemporaryDirectory(prefix="encoding-benchmark-") as directory:
        root = Path(directory)
        with args.source.open("rb") as handle:
            sample = handle.read(args.sample_mib * 1024**2)
        # End the sample after a full reserved marker, with no UTF-8 truncation.
        marker = b"<|endoftext|>"
        position = sample.rfind(marker)
        if position < 0:
            raise ValueError("Benchmark prefix must contain an endoftext document boundary")
        sample = sample[: position + len(marker)]
        source = root / "sample.txt"
        source.write_bytes(sample)
        rows = []
        baseline = None
        for workers in args.workers:
            tokenizer = BPE()
            tokenizer.load(str(args.tokenizer_prefix))
            cache_size = 2048 if workers == 1 else 65_536
            destination = root / f"tokens-{workers}.bin"
            started = time.perf_counter()
            dtype, count = encode_file(
                tokenizer, source, destination, started + 600, workers=workers, cache_size=cache_size
            )
            elapsed = time.perf_counter() - started
            digest = file_digest(destination)
            if baseline is None:
                baseline = (digest, count, dtype)
            if (digest, count, dtype) != baseline:
                raise AssertionError("Serial and parallel token output differs")
            row = {
                "workers": workers,
                "cache_size": cache_size,
                "seconds": elapsed,
                "mib_per_second": len(sample) / 1024**2 / elapsed,
                "token_count": count,
                "sha256": digest,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        report = {"source": str(args.source), "sample_bytes": len(sample), "results": rows}
        if args.output:
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
