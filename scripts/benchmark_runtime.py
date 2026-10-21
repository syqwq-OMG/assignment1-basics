"""Real-token runtime comparison, recorded separately from official PPL experiments."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.main import configure_runtime, evaluate_full, forward_logits, language_model_loss, load_token_data
from cs336_basics.Trans4mer.optim1zer import AdamW
from cs336_basics.Trans4mer.operat0r import gradient_clipping
from cs336_basics.Trans4mer.transformer import TransformerLM
from wandb_record import experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("fp32", "tf32", "bf16"), required=True)
    parser.add_argument("--compile", dest="compile_model", action="store_true")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    config = vars(args).copy()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    config.update(d_model=512, num_layers=8, d_ff=1344, num_heads=8, batch_size=16,
                  context_length=512, steps=60, seed=42, scoring_run=False,
                  token_source="unchanged own-BPE OWT files from pilot-2; runtime benchmark only")
    started = time.perf_counter()
    name = f"10234900421-runtime-{args.precision}-compile{int(args.compile_model)}-20261020"
    with experiment("runtime-benchmark", config, root=Path(__file__).resolve().parents[1], name=name) as run:
        torch.manual_seed(42)
        np.random.seed(42)
        train_data = load_token_data(args.train_data, "uint16")
        val_data = load_token_data(args.val_data, "uint16")
        model = TransformerLM(32000, 512, 512, 8, 8, 1344, 10000).cuda()
        configure_runtime(model, args.precision, args.compile_model)
        optimizer = AdamW(model.parameters(), lr=0.0006, betas=(0.9, 0.95), weight_decay=0.1)
        # Identical real-text batches in each independent process.
        batches = []
        for i in range(4):
            start = i * 8192
            x = torch.tensor(np.array(train_data[start:start+8192]), dtype=torch.long, device="cuda").reshape(16, 512)
            y = torch.tensor(np.array(train_data[start+1:start+8193]), dtype=torch.long, device="cuda").reshape(16, 512)
            batches.append((x, y))
        times, losses = [], []
        torch.cuda.reset_peak_memory_stats()
        for step in range(60):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            x, y = batches[step % len(batches)]
            loss = language_model_loss(forward_logits(model, x), y)
            loss.backward()
            gradient_clipping(model.parameters(), 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - tick)
            losses.append(float(loss))
            if not math.isfinite(losses[-1]):
                raise ValueError("Non-finite benchmark loss")
        nll, count = evaluate_full(model, val_data[:8230], 16, 512, torch.device("cuda"))
        torch.cuda.synchronize()
        result = {"precision": args.precision, "compile_model": args.compile_model,
                  "run_id": run.id, "url": run.url, "status": "completed",
                  "first_loss": losses[0], "last_loss": sum(losses[-4:])/4,
                  "steady_step_seconds": float(np.median(times[10:])),
                  "tokens_per_second": 8192 / float(np.median(times[10:])),
                  "first_step_seconds": times[0], "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                  "probe_val_nll": nll/count, "probe_val_tokens": count,
                  "wall_seconds": time.perf_counter()-started,
                  "finite_gradients": all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                  "fp32_parameters": all(p.dtype == torch.float32 for p in model.parameters())}
        result["stable"] = (result["finite_gradients"] and result["fp32_parameters"]
                            and math.isfinite(result["probe_val_nll"]) and result["last_loss"] < result["first_loss"])
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, indent=2))
        run.summary.update(result)
        run.save(str(args.result), base_path=str(args.result.parent), policy="now")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
