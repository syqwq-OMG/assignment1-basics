import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pytest

import server_search


def test_timing_ignores_training_examples_in_printed_source():
    source = '00045 | print("iter=100/1000 train_loss=5.0 elapsed=10.0s")\n'
    actual = 'iter=300/1000 train_loss=4.0 elapsed=25.0s\n'
    assert server_search.TRAIN_LINE.search(source) is None
    assert server_search.TRAIN_LINE.search(source + actual)[1] == "300"


@pytest.mark.parametrize("fail_final", [False, True])
def test_recovery_preserves_runs_carries_failed_time_and_rejects_false_completion(tmp_path, monkeypatch, fail_final):
    monkeypatch.setattr(server_search, "__file__", str(tmp_path / "server_search.py"))
    text = tmp_path / "owt.txt"
    text.write_text("unchanged corpus")
    tokenizer = tmp_path / "tokenizer"
    for suffix in ("vocab", "merges", "special_tokens"):
        Path(f"{tokenizer}.{suffix}.json").write_text("{}")
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({"stable": True, "precision": "bf16", "compile_model": True}))
    labels = ["8L-d512-b32-lr6e4", "8L-d768-b16-lr6e4", "8L-d768-b32-lr6e4", "8L-d768-b64-lr6e4"]
    rows = []
    for index, (label, ppl, width, batch) in enumerate(zip(labels[:3], [52, 59, 47], [512, 768, 768], [32, 16, 32]), 1):
        rows.append({"label": f"pilot-{index}-{label}", "stage": "pilot", "valid": True, "ppl": ppl,
                     "seconds_per_step": 0.1, "preparation_seconds": 500, "finalize_estimate": 900,
                     "candidate": {"label": label, "num_layers": 8, "max_lr": 0.0006,
                                   "d_model": width, "d_ff": 1344 if width == 512 else 2048,
                                   "num_heads": 8 if width == 512 else 12, "batch_size": batch,
                                   "precision": "bf16", "compile_model": True}})
    previous = tmp_path / "failed-pilot.json"
    previous.write_text(json.dumps({"status": "failed", "checkpoint_path": str(tmp_path / "never.pt"),
                                    "time/total_seconds": 494}))
    rows.append({"label": f"pilot-4-{labels[3]}", "stage": "pilot", "valid": False,
                 "result": str(previous), "candidate": {}, "wall_seconds": 500})
    for index, label in enumerate([labels[2], labels[0]], 1):
        rows.append({"label": f"final-{index}-{label}", "stage": "final", "valid": False,
                     "candidate": {}, "wall_seconds": 15, "result": None})
    imported = tmp_path / "imported.json"
    imported.write_text(json.dumps(rows))
    launched = []

    class Child:
        def __init__(self, command, **kwargs):
            launched.append(command)
            failed = fail_final and "final-1-" in command[command.index("--run-name") + 1]
            self.returncode = 1 if failed else 0
            rid = f"{len(launched):012x}"
            out = tmp_path / "outputs" / rid
            out.mkdir(parents=True)
            report = {"run_id": rid, "status": "failed" if failed else "completed",
                      "time/total_seconds": 600, "time/current_run_seconds": 200,
                      "time/within_6h": True, "val_token_count": 66401097, "val/ppl": 60}
            (out / "result.json").write_text(json.dumps(report))
            kwargs["stdout"].write(f"Run ID: {rid}\niter=100/1000000 train_loss=5.0 elapsed=10.0s\niter=200/1000000 train_loss=4.0 elapsed=20.0s\n")
            kwargs["stdout"].flush()

        def poll(self):
            return self.returncode

    monkeypatch.setattr(server_search.subprocess, "Popen", Child)
    output = tmp_path / "search"
    monkeypatch.setattr(sys, "argv", ["server_search.py", "--width-batch", "--continue-search",
                       "--runtime-choice", str(runtime), "--import-results", str(imported),
                       "--tokenizer-prefix", str(tokenizer), "--train-text", str(text), "--val-text", str(text),
                       "--output", str(output), "--end", (datetime.now(timezone.utc)+timedelta(days=1)).isoformat()])
    if fail_final:
        with pytest.raises(RuntimeError, match="Campaign incomplete"):
            server_search.main()
        assert "FAILED" in (output / "status.txt").read_text()
    else:
        server_search.main()
        assert "COMPLETED" in (output / "status.txt").read_text()
    assert len(launched) == 3  # one failed pilot retry and two finals; no completed pilot repeats
    assert launched[0][launched[0].index("--previous-timing")+1] == str(previous)
    for command in launched:
        assert command[command.index("--train-text")+1] == str(text)
        assert command[command.index("--val-text")+1] == str(text)
    for command in launched[1:]:
        ledger = json.loads(Path(command[command.index("--previous-timing")+1]).read_text())
        assert ledger["time/total_seconds"] == 15 and ledger["run_id"] is None
