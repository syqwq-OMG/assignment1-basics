from collections.abc import Iterable, Iterator
import os.path
from collections import Counter

from .pretokenization import find_chunk_boundaries, pretokenization


def bytify(s: str) -> tuple[bytes, ...]:
    """Convert a string to a tuple containing one byte per element."""
    return tuple(bytes([b]) for b in s.encode("utf-8"))


def apply_merge(
    byptk: tuple[bytes, ...], merge: tuple[bytes, bytes]
) -> tuple[bytes, ...]:
    """Apply a merge operation to a word represented as a tuple of bytes."""
    merged_word = []
    i = 0
    while i < len(byptk):
        if i + 1 < len(byptk) and (byptk[i], byptk[i + 1]) == merge:
            merged_word.append(byptk[i] + byptk[i + 1])
            i += 2
        else:
            merged_word.append(byptk[i])
            i += 1
    return tuple(merged_word)


class BPE:
    def __init__(
        self,
        vocab: dict[int, bytes] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None,
        special_tokens: list[str] | None = None,
    ):
        self.dkd = vocab
        self.ekd = {v: k for k, v in vocab.items()} if vocab is not None else {}
        self.merges = merges if merges is not None else []
        self.special_tokens = list(dict.fromkeys(special_tokens or []))

    def train(self, input_path: str, vocab_size: int) -> None:
        # 训练后设置 self.vocab 和 self.merges，并重建编码查找表。
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file {input_path} does not exist")

        minimum_vocab_size = 256 + len(self.special_tokens)
        if vocab_size < minimum_vocab_size:
            raise ValueError(f"vocab_size must be at least {minimum_vocab_size}")

        pretoken_counter = Counter()  # {" hello": 10, " world": 5, ...}
        byte_pretoken_counter = Counter()  # (b"l", b"o", b"w"): 5,

        # get pretoken counts
        with open(input_path, "rb") as f:
            # get chunks
            chunk_boundaries = find_chunk_boundaries(
                f,
                desired_num_chunks=10,
                split_special_token=(
                    token.encode("utf-8") for token in self.special_tokens
                ),
            )

            # process within each chunk
            for i in range(len(chunk_boundaries) - 1):
                f.seek(chunk_boundaries[i])
                chunk_data = f.read(
                    chunk_boundaries[i + 1] - chunk_boundaries[i]
                )  # bytes info
                chunk_text = chunk_data.decode("utf-8")  # decode to str
                # count the frequency of pretoken to more conveniently count byte level frequencies
                for pretoken in pretokenization(chunk_text, self.special_tokens):
                    pretoken_counter[pretoken] += 1

        # convert pretoken's key to bytes
        for pretoken, cnt in pretoken_counter.items():
            byte_pretoken_counter[bytify(pretoken)] += cnt



        self.dkd = {i: bytes([i]) for i in range(256)}

        # Special tokens are part of the requested vocabulary size.
        for special_token in self.special_tokens:
            self.dkd[len(self.dkd)] = special_token.encode("utf-8")

        self.ekd = {v: k for k, v in self.dkd.items()}
        self.merges = []

        while len(self.dkd) < vocab_size:
            pairs_counter = Counter()  # (b"l", b"o"): 5, (b"o", b"w"): 5, ...

            for byte_ptk, cnt in byte_pretoken_counter.items():
                for pair in zip(byte_ptk, byte_ptk[1:]):
                    pairs_counter[pair] += cnt

            # first freq second lexicographically
            best_pair = max(
                pairs_counter, key=lambda x: (pairs_counter[x], x), default=None
            )
            if best_pair is None:
                break

            self.merges.append(best_pair)
            new_token = best_pair[0] + best_pair[1]
            new_token_id = len(self.dkd)
            self.dkd[new_token_id] = new_token
            self.ekd[new_token] = new_token_id

            # cannot mutate a Counter while iterating over it
            updated_counter = Counter()
            for byte_ptk, cnt in byte_pretoken_counter.items():
                new_byte_ptk = apply_merge(byte_ptk, best_pair)
                updated_counter[new_byte_ptk] += cnt

            byte_pretoken_counter = updated_counter

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            yield from self.encode(s)

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    def save(self, path_prefix: str) -> None:
        # 词表、merges 分开保存习惯，并另存特殊 token 等配置。
        raise NotImplementedError

    def load(self, path_prefix: str) -> None:
        raise NotImplementedError

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        raise NotImplementedError
