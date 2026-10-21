"""Launch the saved best configuration through the official timed entry point."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "owt_best.json")
    parser.add_argument("--tokenizer-prefix", type=Path, required=True)
    parser.add_argument("--print-command", action="store_true", help="Show the command without starting training.")
    options, forwarded = parser.parse_known_args()
    config = json.loads(options.config.read_text(encoding="utf-8"))
    command = [sys.executable, str(ROOT / "transformer_main.py"),
               "--tokenizer-prefix", str(options.tokenizer_prefix.expanduser().resolve())]
    for section in ("data", "model", "training", "encoding", "budget"):
        for name, value in config[section].items():
            if name == "compile_model":
                if value:
                    command.append("--compile")
            else:
                command.extend(["--" + name.replace("_", "-"), str(value)])
    command.extend(forwarded)
    if options.print_command:
        print(shlex.join(command))
        return
    os.chdir(ROOT)
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
