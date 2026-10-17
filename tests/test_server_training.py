import time

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cs336_basics.BPE.bpe import BPE
from cs336_basics.encoding import encode_file
from cs336_basics.main import evaluate_full


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
