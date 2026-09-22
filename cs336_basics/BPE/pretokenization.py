import os
from typing import BinaryIO
import regex as re
from collections.abc import Iterable, Iterator

PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes | Iterable[bytes] | None = None,
) -> list[int]:
    """
    Chunk the file at the start of a special token.

    ``split_special_token`` accepts either one token (for backwards
    compatibility) or an iterable of tokens. Reads overlap by one less than
    the longest token, so a token split across two reads is still found.
    May return fewer chunks if the boundaries end up overlapping.
    """
    if desired_num_chunks <= 0:
        raise ValueError("desired_num_chunks must be positive")

    if split_special_token is None:
        special_tokens = ()
    elif isinstance(split_special_token, bytes):
        special_tokens = (split_special_token,)
    else:
        special_tokens = tuple(dict.fromkeys(split_special_token))

    if any(not isinstance(token, bytes) or not token for token in special_tokens):
        raise ValueError("special tokens must be non-empty bytestrings")

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size == 0:
        return [0]

    # With no safe delimiter, keep the complete text in one chunk. Splitting
    # at an arbitrary byte offset could break UTF-8 or one pre-token.
    if not special_tokens:
        return [0, file_size]

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
    overlap_size = max(len(token) for token in special_tokens) - 1

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        carry = b""
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Include the end of the previous read: otherwise a special token
            # spanning two mini chunks would be missed.
            search_buffer = carry + mini_chunk
            search_buffer_start = initial_position - len(carry)
            positions = [
                found_at
                for token in special_tokens
                if (found_at := search_buffer.find(token)) != -1
            ]
            if positions:
                chunk_boundaries[bi] = search_buffer_start + min(positions)
                break

            carry = search_buffer[-overlap_size:] if overlap_size else b""
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


## Usage
# with open(..., "rb") as f:
#     num_processes = 4
#     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

#     # The following is a serial implementation, but you can parallelize this
#     # by sending each start/end pair to a set of processes.
#     for start, end in zip(boundaries[:-1], boundaries[1:]):
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")
#         # Run pre-tokenization on your chunk and store the counts for each pre-token


def pretokenization(
    text: str,
    special_tokens: Iterable[str] | None = None,
    *,
    keep_special_tokens: bool = False,
) -> Iterator[str]:
    tokens = tuple(sorted(set(special_tokens or ()), key=lambda x:-len(x)))
    if not tokens:
        yield from (match.group() for match in PAT.finditer(text))
        return

    special_set = set(tokens)
    special_pattern = re.compile(
        "(" + "|".join(re.escape(token) for token in tokens) + ")"
    )
    for piece in special_pattern.split(text):
        if not piece:
            continue
        if piece in special_set:
            if keep_special_tokens:
                yield piece
            continue
        yield from (match.group() for match in PAT.finditer(piece))
