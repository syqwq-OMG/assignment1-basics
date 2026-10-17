"""Ordered, bounded dataset encoding using the project's own BPE implementation."""

from collections import deque
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from cs336_basics.BPE.bpe import BPE

_worker_tokenizer = None
_worker_dtype = None


def _initialize_encoder(vocab, merges, special_tokens, cache_size, dtype):
    global _worker_tokenizer, _worker_dtype
    _worker_tokenizer = BPE(vocab=vocab, merges=merges, special_tokens=special_tokens, cache_size=cache_size)
    _worker_dtype = np.dtype(dtype)


def _encode_chunk(text):
    return np.asarray(_worker_tokenizer.encode(text), dtype=_worker_dtype)


def _text_chunks(source, special_tokens, chunk_size, deadline_clock):
    marker = "<|endoftext|>"
    # A marker contained in another special token is not a safe cut point.
    safe_marker = marker in special_tokens and not any(marker in token and token != marker for token in special_tokens)
    with source.open(encoding="utf-8", newline="") as reader:
        pending = ""
        while chunk := reader.read(chunk_size):
            _check_deadline(deadline_clock)
            pending += chunk
            if safe_marker and marker in pending:
                end = pending.rfind(marker) + len(marker)
                yield pending[:end]
                pending = pending[end:]
        if pending:
            _check_deadline(deadline_clock)
            yield pending


def _check_deadline(deadline_clock):
    if time.perf_counter() >= deadline_clock:
        raise TimeoutError("Budget reserve reached while encoding data; no training started.")


def encode_file(
    tokenizer,
    source,
    destination,
    deadline_clock,
    chunk_size=1024 * 1024,
    *,
    workers=1,
    cache_size=None,
    progress_interval=10.0,
):
    """Encode at complete reserved-marker boundaries and write results in order.

    Uses spawn (safe alongside W&B/CUDA), at most 2*workers in-flight chunks,
    and worker-local caches. No marker means a single complete-text task, since
    arbitrary character, line, or byte cuts can change the BPE token sequence.
    The destination is published atomically only after all chunks succeed.
    """
    source, destination = Path(source), Path(destination)
    if workers < 1 or chunk_size < 1:
        raise ValueError("workers and chunk_size must be positive")
    if cache_size is not None and cache_size < 0:
        raise ValueError("cache_size cannot be negative")
    _check_deadline(deadline_clock)
    dtype = np.dtype("uint16" if len(tokenizer.get_vocab()) <= 65_536 else "uint32")
    temporary = destination.with_suffix(".tmp")
    count, completed_bytes = 0, 0
    total_bytes = source.stat().st_size
    started_at = last_progress = time.perf_counter()
    special_tokens = tokenizer.get_special_tokens()
    marker = "<|endoftext|>"
    can_split = marker in special_tokens and not any(marker in token and token != marker for token in special_tokens)
    if not can_split:
        workers = 1
    if cache_size is None:
        cache_size = tokenizer.CACHE_SIZE_LIMIT
    print(f"Encoding {source}: workers={workers} cache_size={cache_size:,} input={total_bytes:,} bytes", flush=True)

    def write_result(writer, result, byte_count):
        nonlocal count, completed_bytes, last_progress
        _check_deadline(deadline_clock)
        result.tofile(writer)
        count += len(result)
        completed_bytes += byte_count
        now = time.perf_counter()
        if now - last_progress >= progress_interval or completed_bytes == total_bytes:
            elapsed = now - started_at
            rate = completed_bytes / max(elapsed, 1e-9)
            eta = max(0, total_bytes - completed_bytes) / max(rate, 1e-9)
            print(
                f"Encoding {source.name}: {completed_bytes:,}/{total_bytes:,} bytes "
                f"({100 * completed_bytes / max(total_bytes, 1):.1f}%) "
                f"tokens={count:,} rate={rate / 1024**2:.2f} MiB/s elapsed={elapsed:.1f}s ETA={eta:.1f}s",
                flush=True,
            )
            last_progress = now

    chunks = _text_chunks(source, special_tokens, chunk_size, deadline_clock)
    try:
        with temporary.open("wb") as writer:
            if workers == 1:
                # Preserve the caller's tokenizer cache setting and cache contents.
                original_cache_size = tokenizer.CACHE_SIZE_LIMIT
                tokenizer.CACHE_SIZE_LIMIT = cache_size
                try:
                    for text in chunks:
                        result = np.asarray(tokenizer.encode(text), dtype=dtype)
                        write_result(writer, result, len(text.encode("utf-8")))
                finally:
                    tokenizer.CACHE_SIZE_LIMIT = original_cache_size
            else:
                with mp.get_context("spawn").Pool(
                    processes=workers,
                    initializer=_initialize_encoder,
                    initargs=(tokenizer.get_vocab(), tokenizer.get_merges(), special_tokens, cache_size, dtype.name),
                ) as pool:
                    pending = deque()

                    def drain_one():
                        future, byte_count = pending.popleft()
                        _check_deadline(deadline_clock)
                        try:
                            result = future.get(timeout=max(0.001, deadline_clock - time.perf_counter()))
                        except mp.TimeoutError as error:
                            raise TimeoutError("Budget reserve reached waiting for an encoding worker.") from error
                        write_result(writer, result, byte_count)

                    for text in chunks:
                        _check_deadline(deadline_clock)
                        pending.append((pool.apply_async(_encode_chunk, (text,)), len(text.encode("utf-8"))))
                        if len(pending) >= 2 * workers:
                            drain_one()
                    while pending:
                        drain_one()
        _check_deadline(deadline_clock)
        temporary.replace(destination)
    finally:
        chunks.close()
        # A failed/expired operation must not leave a plausible reusable token file.
        temporary.unlink(missing_ok=True)
    return dtype.name, count
