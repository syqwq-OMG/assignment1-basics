import time
import argparse
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cs336_basics.BPE.bpe import BPE
from cs336_basics.encoding import encode_file
from cs336_basics.main import evaluate_full
from transformer_main import read_previous_run
from wandb_record import experiment


@pytest.mark.parametrize("length", [2, 4, 5, 9, 20, 33])
def test_full_validation_includes_every_target_and_weights_partial_batches(length):
    class Model(torch.nn.Module):
        def forward(self, inputs):
            assert not self.training
            assert not torch.is_grad_enabled()
            classes = torch.arange(7, dtype=torch.float32)
            return -(inputs[..., None] - classes).square()

    model = Model()
    data = np.arange(length) % 7
    nll, count = evaluate_full(model, data, batch_size=3, context_length=4, device=torch.device("cpu"))
    logits = -(torch.tensor(data[:-1, None]) - torch.arange(7)).float().square()
    expected = F.cross_entropy(logits, torch.tensor(data[1:]), reduction="sum")
    assert count == length - 1
    assert nll == pytest.approx(expected.item(), rel=1e-6)
    assert model.training


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 31])
@pytest.mark.parametrize("special_tokens", [["<|endoftext|>"], []])
@pytest.mark.parametrize("workers", [1, 2])
def test_stream_encoding_preserves_special_and_whitespace_boundaries(tmp_path, chunk_size, special_tokens, workers):
    vocab = {i: bytes([i]) for i in range(256)}
    merges = [(b"a", b"b"), (b"\n", b"\n")]
    for piece in (b"ab", b"\n\n"):
        vocab[len(vocab)] = piece
    for token in special_tokens:
        vocab[len(vocab)] = token.encode()
    tokenizer = BPE(vocab=vocab, merges=merges, special_tokens=special_tokens)
    text = "ab 你好\r\n\n<|endoftext|>ab\n\n<|endoftext|>ab 😀 tail  "
    source, destination = tmp_path / "text.txt", tmp_path / "tokens.bin"
    source.write_bytes(text.encode())
    dtype, count = encode_file(tokenizer, source, destination, time.perf_counter() + 60, chunk_size, workers=workers)
    actual = np.fromfile(destination, dtype=dtype).tolist()
    assert actual == tokenizer.encode(text)
    assert count == len(actual)
    assert tokenizer.decode(actual) == text


def test_encoding_stops_when_budget_reserve_is_reached(tmp_path):
    source, destination = tmp_path / "text.txt", tmp_path / "tokens.bin"
    source.write_text("abc", encoding="utf-8")
    tokenizer = BPE(vocab={i: bytes([i]) for i in range(256)})
    with pytest.raises(TimeoutError):
        encode_file(tokenizer, source, destination, time.perf_counter() - 1)
    assert not destination.exists()
    assert not destination.with_suffix(".tmp").exists()


def test_overlapping_special_token_is_not_split_at_embedded_marker(tmp_path):
    special_tokens = ["<|endoftext|>", "prefix<|endoftext|>suffix"]
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode()
    tokenizer = BPE(vocab=vocab, special_tokens=special_tokens)
    text = "before prefix<|endoftext|>suffix after<|endoftext|>end"
    source, destination = tmp_path / "text.txt", tmp_path / "tokens.bin"
    source.write_text(text, encoding="utf-8")
    dtype, count = encode_file(tokenizer, source, destination, time.perf_counter() + 60, chunk_size=1, workers=2)
    actual = np.fromfile(destination, dtype=dtype).tolist()
    assert actual == tokenizer.encode(text)
    assert count == len(actual)


@pytest.mark.parametrize("workers", [1, 2])
def test_failed_encoding_preserves_existing_destination(tmp_path, workers):
    tokenizer = BPE(vocab={i: bytes([i]) for i in range(256)}, special_tokens=["<|endoftext|>"])
    source, destination = tmp_path / "text.txt", tmp_path / "tokens.bin"
    source.write_text("abc<|endoftext|>", encoding="utf-8")
    destination.write_bytes(b"previous-complete-result")
    # Special token intentionally absent from vocab: fail in the actual encoder.
    with pytest.raises(KeyError):
        encode_file(tokenizer, source, destination, time.perf_counter() + 60, workers=workers)
    assert destination.read_bytes() == b"previous-complete-result"
    assert not destination.with_suffix(".tmp").exists()


def test_restart_after_encoding_keeps_prior_time_and_run_link(tmp_path):
    prior_path = tmp_path / "result.json"
    hashes = {"vocab": "v", "merges": "m", "special_tokens": "s"}
    prior = {
        "run_id": "old-run",
        "status": "failed",
        "checkpoint_path": str(tmp_path / "last.pt"),
        "time/total_seconds": 6657.47,
        "config": {"tokenizer_sha256": hashes},
    }
    prior_path.write_text(json.dumps(prior), encoding="utf-8")
    args = argparse.Namespace(resume=None)
    assert read_previous_run(prior_path, hashes, args) == (6657.47, "old-run")
    with pytest.raises(ValueError, match="Cannot change tokenizer"):
        read_previous_run(prior_path, {"vocab": "different"}, args)
    (tmp_path / "last.pt").write_bytes(b"checkpoint-exists")
    with pytest.raises(ValueError, match="no checkpoint"):
        read_previous_run(prior_path, hashes, args)


def test_checkpoint_resume_requires_matching_model_and_checkpoint(tmp_path):
    prior_path = tmp_path / "result.json"
    checkpoint = tmp_path / "last.pt"
    config = dict(
        vocab_size=32000, context_length=256, d_model=512, num_layers=4, num_heads=16, d_ff=1344, rope_theta=10000.0
    )
    hashes = {"vocab": "v"}
    prior = {
        "run_id": "old-run",
        "status": "completed",
        "checkpoint_path": str(checkpoint),
        "time/total_seconds": 100.0,
        "config": {**config, "tokenizer_sha256": hashes},
    }
    prior_path.write_text(json.dumps(prior), encoding="utf-8")
    args = argparse.Namespace(resume=checkpoint, **config)
    assert read_previous_run(prior_path, hashes, args) == (100.0, "old-run")
    args.d_model = 256
    with pytest.raises(ValueError, match="d_model"):
        read_previous_run(prior_path, hashes, args)
    args.d_model = 512
    args.resume = tmp_path / "another.pt"
    with pytest.raises(ValueError, match="checkpoint in the previous"):
        read_previous_run(prior_path, hashes, args)


@pytest.mark.parametrize(
    "explicit_name,environment_name,expected",
    [
        ("owt-parallel8-restart", "environment-name", "owt-parallel8-restart"),
        (None, "environment-name", "environment-name"),
        (None, None, "student-transformer"),
    ],
)
def test_source_inventory_does_not_overwrite_experiment_name(
    tmp_path, monkeypatch, explicit_name, environment_name, expected
):
    import wandb

    (tmp_path / "source.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "z-last-file.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("STUDENT_ID", "student")
    monkeypatch.setenv("WANDB_ENTITY", "test-entity")
    if environment_name is None:
        monkeypatch.delenv("WANDB_NAME", raising=False)
    else:
        monkeypatch.setenv("WANDB_NAME", environment_name)

    class InitReached(Exception):
        pass

    def capture_init(**kwargs):
        assert kwargs["name"] == expected
        raise InitReached

    monkeypatch.setattr(wandb, "init", capture_init)
    with pytest.raises(InitReached), experiment("transformer", {}, root=tmp_path, name=explicit_name):
        pass
