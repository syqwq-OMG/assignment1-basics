"""Sequential GPU-3 OWT search; each independent model has its own <=6h clock.

This launches the existing official entry point without changing model code.
All outputs, commands, selection decisions and timing estimates remain on disk.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


TRAIN_LINE = re.compile(r"iter=(\d+)/\d+ train_loss=([\d.]+).*elapsed=([\d.]+)s")


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--end", required=True, help="ISO 8601 GPU allocation end with UTC offset")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--pilot-seconds", type=int, default=5400)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--width-batch", action="store_true")
    parser.add_argument("--import-results", type=Path)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--adopt-result", type=Path)
    parser.add_argument("--adopt-log", type=Path)
    parser.add_argument("--runtime-train-data", type=Path)
    parser.add_argument("--runtime-val-data", type=Path)
    options = parser.parse_args()
    root = Path(__file__).resolve().parent
    end = datetime.fromisoformat(options.end).timestamp()
    if options.pilot_seconds < 3600 or options.pilot_seconds > 21600:
        parser.error("pilot budget must be between 3600 and 21600 seconds")
    pilots = [
        {"label": "4L-lr12e4", "num_layers": 4, "max_lr": 0.0012},
        {"label": "8L-lr6e4", "num_layers": 8, "max_lr": 0.0006},
        {"label": "8L-lr12e4", "num_layers": 8, "max_lr": 0.0012},
        {"label": "8L-lr24e4", "num_layers": 8, "max_lr": 0.0024},
        {"label": "12L-lr12e4", "num_layers": 12, "max_lr": 0.0012},
    ]
    if options.width_batch:
        pilots = [
            {"label": "8L-d512-b32-lr6e4", "num_layers": 8, "d_model": 512, "d_ff": 1344,
             "num_heads": 8, "batch_size": 32, "max_lr": 0.0006},
            {"label": "8L-d768-b16-lr6e4", "num_layers": 8, "d_model": 768, "d_ff": 2048,
             "num_heads": 12, "batch_size": 16, "max_lr": 0.0006},
            {"label": "8L-d768-b32-lr6e4", "num_layers": 8, "d_model": 768, "d_ff": 2048,
             "num_heads": 12, "batch_size": 32, "max_lr": 0.0006},
            {"label": "8L-d768-b64-lr6e4", "num_layers": 8, "d_model": 768, "d_ff": 2048,
             "num_heads": 12, "batch_size": 64, "max_lr": 0.0006},
        ]
    plan = {
        "gpu": 3, "code_commit": ((root / "CODE_COMMIT").read_text().strip()
                                      if (root / "CODE_COMMIT").exists() else "train/precision-compile"),
        "allocation_end": options.end,
        "pilot_seconds": options.pilot_seconds, "pilots": pilots,
        "finalists": 2, "final_seconds_each": 21600,
        "selection": "lowest finite full-validation PPL among valid independent pilots",
        "shared": {"d_model": 512, "d_ff": 1344, "context_length": 512,
                   "num_heads": 8, "batch_size": 16, "beta2": 0.95,
                   "weight_decay": 0.1, "seed": 42},
        "runtime": {"CUDA_VISIBLE_DEVICES": "3", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
                    "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"},
        "budget_policy": "Fresh random initialization per run; no pilot checkpoint resumes. "
                         "Encoding, saving and complete validation count in every run.",
    }
    if options.dry_run:
        print(json.dumps(plan, indent=2))
        return
    options.output.mkdir(parents=True, exist_ok=True)
    lock = (options.output / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (options.output / "experiments.json").exists():
        raise RuntimeError("Output already used; refusing to silently repeat or reset experiments")
    write_json(options.output / "plan.json", plan)
    env = os.environ.copy()
    env.update(plan["runtime"])
    env.update(WANDB_MODE="online", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
               WANDB_ENTITY="syqwq-east-china-normal-university",
               WANDB_PROJECT="lmfs-assignment1", STUDENT_ID="10234900421")
    env.update(TORCHINDUCTOR_COMPILE_THREADS="4", MAX_JOBS="4")
    results = json.loads(options.import_results.read_text()) if options.import_results else []
    # Imported independent models retain their original runtime in any finalist.
    for entry in results:
        entry["candidate"].setdefault("precision", "tf32")
        entry["candidate"].setdefault("compile_model", False)
    runtime_choice = {"precision": "tf32", "compile_model": False}

    def status(message):
        line = f"{datetime.now().astimezone().isoformat()} {message}"
        print(line, flush=True)
        (options.output / "status.txt").write_text(line + "\n")

    def execute(candidate, stage, index):
        label = f"{stage}-{index}-{candidate['label']}"
        # Leave 15 minutes outside each run for uploads/exit, plus 10 minutes
        # before the allocation ends. Never increase an individual 6h budget.
        available = math.floor(end - time.time() - 1500)
        budget = min(options.pilot_seconds if stage == "pilot" else 21600, available)
        if budget < 3600:
            status(f"SKIPPED {label}: insufficient time before GPU allocation ends")
            return None
        minimum_reserve = 1800 if stage == "pilot" else 900
        reserve = int(min(budget * 0.48, max(minimum_reserve, candidate.get("finalize_estimate", 1800))))
        if stage == "pilot":
            cycle, warmup = 1_000_000, 100
        else:
            # Cosine decay should finish before the time-based stop, using the
            # pilot's measured update speed and preprocessing duration.
            train_seconds = max(300, budget - reserve - candidate["preparation_seconds"] - 180)
            cycle = max(1000, int(train_seconds / candidate["seconds_per_step"] * 0.95))
            warmup = min(500, max(100, cycle // 100))
        command = [options.python, "transformer_main.py",
                   "--run-name", f"10234900421-owt-gpu3-20261020-{label}",
                   "--tokenizer-prefix", str(options.tokenizer_prefix.resolve()),
                   "--encoding-workers", "8", "--encoding-cache-size", "65536",
                   "--device", "cuda", "--max-iters", "1000000",
                   "--warmup-iters", str(warmup), "--cosine-cycle-iters", str(cycle),
                   "--max-lr", str(candidate["max_lr"]),
                   "--min-lr", str(candidate["max_lr"] * 0.1),
                   "--num-layers", str(candidate["num_layers"]),
                   "--eval-interval", "1000", "--eval-iters", "10",
                   "--checkpoint-interval", "10000",
                   "--max-train-seconds", str(budget),
                   "--finalize-reserve-seconds", str(reserve)]
        shared = dict(plan["shared"])
        shared.update({key: candidate[key] for key in shared if key in candidate})
        precision = candidate.get("precision", runtime_choice["precision"])
        compiled = candidate.get("compile_model", runtime_choice["compile_model"])
        candidate = dict(candidate, precision=precision, compile_model=compiled)
        command += ["--precision", precision]
        if compiled:
            command += ["--compile"]
        env["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0" if precision == "fp32" else "1"
        for key, value in shared.items():
            command += ["--" + key.replace("_", "-"), str(value)]
        write_json(options.output / f"{label}-command.json", {"command": command, "runtime": plan["runtime"]})
        log = options.output / f"{label}.log"
        status(f"RUNNING {label} budget={budget}s reserve={reserve}s cosine={cycle} GPU=3")
        started = time.monotonic()
        observed = ""
        preparation = None
        # No training subprocess is killed at its budget boundary: the official
        # entry point reserves time for atomic save and full validation itself.
        # The only external guard is the GPU allocation's absolute end.
        with log.open("w") as writer:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=writer, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            with log.open() as reader:
                while child.poll() is None:
                    text = reader.read()
                    observed += text
                    if preparation is None:
                        match = TRAIN_LINE.search(observed)
                        if match:
                            preparation = max(0.0, time.monotonic() - started - float(match[3]))
                    if time.time() >= end - 600:
                        # Only this child's process group, never other GPU users.
                        import signal
                        os.killpg(child.pid, signal.SIGINT)
                        try:
                            child.wait(timeout=300)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                        break
                    time.sleep(15)
                observed += reader.read()
        run_match = re.search(r"^Run ID: ([a-f0-9]+)$", observed, re.MULTILINE)
        result_path = root / "outputs" / run_match[1] / "result.json" if run_match else None
        entry = {"label": label, "stage": stage, "candidate": candidate, "returncode": child.returncode,
                 "log": str(log), "budget": budget, "reserve": reserve, "cosine_cycle_iters": cycle,
                 "wall_seconds": time.monotonic() - started, "valid": False}
        if result_path and result_path.exists():
            report = json.loads(result_path.read_text())
            ppl = report.get("val/ppl", float("inf"))
            entry.update(run_id=report["run_id"], result=str(result_path), ppl=ppl,
                         url=f"https://wandb.ai/{env['WANDB_ENTITY']}/{env['WANDB_PROJECT']}/runs/{report['run_id']}",
                         elapsed=report["time/total_seconds"], report_status=report["status"])
            entry["valid"] = (child.returncode == 0 and report["status"] == "completed"
                              and report["time/within_6h"] and report["time/total_seconds"] <= budget
                              and report.get("val_token_count") == 66_401_097
                              and math.isfinite(ppl) and ppl > 0)
            samples = list(TRAIN_LINE.finditer(observed))
            if samples and entry["valid"]:
                first, last = samples[0], samples[-1]
                steps = int(last[1]) - int(first[1])
                seconds_per_step = ((float(last[3]) - float(first[3])) / steps if steps > 0
                                    else float(last[3]) / int(last[1]))
                preparation = preparation or max(0, report["time/total_seconds"] - float(last[3]) - reserve)
                finalization = max(0, report["time/total_seconds"] - preparation - float(last[3]))
                entry.update(seconds_per_step=seconds_per_step, preparation_seconds=preparation,
                             finalize_estimate=max(900, int(finalization * 1.7 + 180)))
        results.append(entry)
        write_json(options.output / "experiments.json", results)
        valid = sorted((r for r in results if r["valid"]), key=lambda r: r["ppl"])
        write_json(options.output / "ranking.json", valid)
        if valid:
            write_json(options.output / "best.json", valid[0])
        status(f"FINISHED {label} valid={entry['valid']} PPL={entry.get('ppl')} exit={child.returncode}")
        return entry

    try:
        if options.wait_pid:
            status(f"WAITING for existing pilot PID={options.wait_pid}; original training remains untouched")
            while True:
                proc = Path(f"/proc/{options.wait_pid}/stat")
                if not proc.exists() or proc.read_text().split(") ", 1)[1].startswith("Z"):
                    break
                if time.time() >= end - 1500:
                    raise TimeoutError("Allocation ended while waiting for existing pilot")
                time.sleep(10)
        if options.adopt_result:
            report = json.loads(options.adopt_result.read_text())
            config = report["config"]
            samples = list(TRAIN_LINE.finditer(options.adopt_log.read_text()))
            first, last = samples[0], samples[-1]
            per_step = (float(last[3])-float(first[3]))/(int(last[1])-int(first[1]))
            # The existing pilot's trained model is never restarted or relabeled.
            entry = {"label": "pilot-3-8L-lr12e4", "stage": "pilot",
                     "candidate": {"label": "8L-lr12e4", "num_layers": 8, "max_lr": 0.0012,
                                   "precision": "tf32", "compile_model": False},
                     "returncode": None, "adopted_existing_run": True, "run_id": report["run_id"],
                     "result": str(options.adopt_result), "log": str(options.adopt_log),
                     "budget": config["max_train_seconds"], "elapsed": report["time/total_seconds"],
                     "ppl": report.get("val/ppl", float("inf")), "seconds_per_step": per_step,
                     "preparation_seconds": 530, "finalize_estimate": 900,
                     "url": f"https://wandb.ai/{env['WANDB_ENTITY']}/{env['WANDB_PROJECT']}/runs/{report['run_id']}"}
            entry["valid"] = (report["status"] == "completed" and report["time/within_6h"]
                              and report["time/total_seconds"] <= config["max_train_seconds"]
                              and report.get("val_token_count") == 66401097 and math.isfinite(entry["ppl"]))
            results.append(entry)
        write_json(options.output / "experiments.json", results)
        if options.width_batch:
            bench_results = []
            for precision in ("fp32", "tf32", "bf16"):
                for compiled in (False, True):
                    label = f"runtime-{precision}-compile{int(compiled)}"
                    status(f"BENCHMARK {label} GPU=3")
                    destination = options.output / f"{label}.json"
                    command = [options.python, "benchmark_runtime.py", "--precision", precision,
                               "--train-data", str(options.runtime_train_data),
                               "--val-data", str(options.runtime_val_data), "--result", str(destination)]
                    if compiled:
                        command += ["--compile"]
                    bench_env = dict(env, TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0" if precision == "fp32" else "1")
                    try:
                        with (options.output / f"{label}.log").open("w") as stream:
                            completed = subprocess.run(command, cwd=root, env=bench_env, stdout=stream,
                                                       stderr=subprocess.STDOUT, timeout=420)
                        if completed.returncode == 0 and destination.exists():
                            bench_results.append(json.loads(destination.read_text()))
                    except subprocess.TimeoutExpired:
                        status(f"BENCHMARK TIMEOUT {label}; skipping this runtime")
            reference = next((r for r in bench_results if r["precision"] == "fp32" and not r["compile_model"]), None)
            stable = [r for r in bench_results if r["stable"] and reference is not None
                      and abs(r["first_loss"]-reference["first_loss"]) < 0.05]
            if not stable:
                raise RuntimeError("Runtime checks failed; refusing unverified precision/compile selection")
            fastest = min(stable, key=lambda r: r["steady_step_seconds"])
            runtime_choice = {key: fastest[key] for key in ("precision", "compile_model")}
            write_json(options.output / "runtime-results.json", bench_results)
            write_json(options.output / "runtime-choice.json", fastest)
            status(f"RUNTIME SELECTED {runtime_choice}")
        for index, candidate in enumerate(pilots, 1):
            execute(candidate, "pilot", index)
        ranked = sorted((r for r in results if r["valid"] and "seconds_per_step" in r), key=lambda r: r["ppl"])
        if not ranked:
            raise RuntimeError("No valid pilot; refusing to choose an unverified configuration")
        finalists = ranked[:2]
        write_json(options.output / "selection.json", finalists)
        for index, pilot in enumerate(finalists, 1):
            candidate = dict(pilot["candidate"])
            candidate.update({key: pilot[key] for key in
                              ("seconds_per_step", "preparation_seconds", "finalize_estimate")})
            execute(candidate, "final", index)
        status("COMPLETED: ranking.json and best.json contain full-validation results and W&B links")
    except BaseException as error:
        status(f"FAILED: {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
