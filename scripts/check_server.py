"""Read-only server preflight; --wandb additionally tests online experiment logging."""

import argparse
import importlib.metadata
import os
from pathlib import Path
import sys

import torch
import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/ai/lmfs2026/data/assignment1-data"))
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="Local debugging only.")
    args = parser.parse_args()
    failures = []
    print("Python:", sys.version.split()[0], "Executable:", sys.executable)
    for name in ("torch", "wandb", "numpy", "regex", "einops", "jaxtyping"):
        try:
            print(f"{name}: {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"Missing dependency: {name}")
    print("CUDA build:", torch.version.cuda, "Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            gpu = torch.cuda.get_device_properties(index)
            print(f"GPU {index}: {gpu.name}, {gpu.total_memory / 1024**3:.1f} GiB")
    elif not args.allow_cpu:
        failures.append("CUDA is unavailable. Activate /lmfs2026/env/a1/bin/activate and check GPU allocation.")
    for filename in ("owt_train.txt", "owt_valid.txt"):
        path = args.data_dir / filename
        if path.is_file() and os.access(path, os.R_OK):
            print(f"Data: {path}, {path.stat().st_size:,} bytes")
        else:
            failures.append(f"Cannot read data: {path}")
    for name in ("STUDENT_ID", "WANDB_ENTITY", "WANDB_PROJECT", "WANDB_MODE"):
        value = os.environ.get(name)
        print(f"{name}: {value or '(unset)'}")
        if not value:
            failures.append(f"Missing {name}; source w-b.sh first.")
    Path("outputs").mkdir(exist_ok=True)
    if failures:
        raise SystemExit("\n".join(failures))
    if args.wandb:
        with wandb.init(
            entity=os.environ["WANDB_ENTITY"],
            project=os.environ["WANDB_PROJECT"],
            name=f"{os.environ['STUDENT_ID']}-environment-test",
            job_type="environment-test",
        ) as run:
            run.log({"test/value": 1})
            print("Entity:", run.entity, "Run URL:", run.url)
    print("Preflight passed.")


if __name__ == "__main__":
    main()
