from collections.abc import Iterable, Iterator
import os.path
from collections import Counter, defaultdict
import json
from typing import Self

from .pretokenization import find_chunk_boundaries, pretokenization


def bytify(s: str) -> tuple[bytes, ...]:
    """Convert a string to a tuple containing one byte per element."""
    return tuple(bytes([b]) for b in s.encode("utf-8"))


def apply_merge(byptk: tuple[bytes, ...], merge: tuple[bytes, bytes]) -> tuple[bytes, ...]:
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
        self, vocab: dict[int, bytes] | None = None, merges: list[tuple[bytes, bytes]] | None = None, special_tokens: list[str] | None = None
    ):
        self.int2bytes_dict = vocab
        self.bytes2int_dict = {v: k for k, v in vocab.items()} if vocab is not None else {}
        self.merges = merges if merges is not None else []
        self.special_tokens = list(dict.fromkeys(special_tokens or []))

    def train(self, input_path: str, vocab_size: int) -> None:
        # 训练后设置 self.vocab 和 self.merges，并重建编码查找表。
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file {input_path} does not exist")

        minimum_vocab_size = 256 + len(self.special_tokens)
        if vocab_size < minimum_vocab_size:
            raise ValueError(f"vocab_size must be at least {minimum_vocab_size}")

        pretoken_tid: dict[str, int] = defaultdict(int)  # {"low": 0, "lowe": 1, ...}
        tid_pretoken_bytes: dict[int, tuple[bytes, ...]] = defaultdict(tuple)  # {0: (b"l", b"o", b"w"), 1: (b"o", b"w"), ...}
        tid_counter: dict[int, int] = defaultdict(int)  # {0: 5, 1: 5, ...}
        pairs_tids: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)  # (b"l", b"o"): {0, 1, 2}, (b"o", b"w"): {0, 1, 2}, ... where the pair occurs
        pairs_counter: dict[tuple[bytes, bytes], int] = Counter()  # (b"l", b"o"): 5, (b"o", b"w"): 5, ...

        # get pretoken counts
        with open(input_path, "rb") as f:
            # get chunks
            chunk_boundaries = find_chunk_boundaries(
                f, desired_num_chunks=10, split_special_token=(token.encode("utf-8") for token in self.special_tokens)
            )

            # process within each chunk
            for i in range(len(chunk_boundaries) - 1):
                f.seek(chunk_boundaries[i])
                chunk_data = f.read(chunk_boundaries[i + 1] - chunk_boundaries[i])  # bytes info
                chunk_text = chunk_data.decode("utf-8")  # decode to str
                # count the frequency of pretoken to more conveniently count byte level frequencies
                for pretoken in pretokenization(chunk_text, self.special_tokens):
                    if pretoken not in pretoken_tid:
                        tid = len(pretoken_tid)
                        pretoken_tid[pretoken] = tid
                        tid_pretoken_bytes[tid] = bytify(pretoken)
                        tid_counter[tid] = 1
                    else:
                        tid = pretoken_tid[pretoken]
                        tid_counter[tid] += 1

        # get pairs_tids and pairs_counter
        for tid, byte_ptk in enumerate(tid_pretoken_bytes.values()):
            current_count = tid_counter[tid]
            for pair in zip(byte_ptk, byte_ptk[1:]):
                if pair not in pairs_tids:
                    pairs_tids[pair] = set()
                pairs_tids[pair].add(tid)
                pairs_counter[pair] += current_count

        self.int2bytes_dict = {i: bytes([i]) for i in range(256)}

        # Special tokens are part of the requested vocabulary size.
        for special_token in self.special_tokens:
            self.int2bytes_dict[len(self.int2bytes_dict)] = special_token.encode("utf-8")

        self.bytes2int_dict = {v: k for k, v in self.int2bytes_dict.items()}
        self.merges = []

        while len(self.int2bytes_dict) < vocab_size:
            # first freq second lexicographically
            best_pair = max(pairs_counter, key=lambda x: (pairs_counter[x], x), default=None)
            if best_pair is None or pairs_counter[best_pair] <= 0:
                break

            self.merges.append(best_pair)
            new_token = best_pair[0] + best_pair[1]
            new_token_id = len(self.int2bytes_dict)
            self.int2bytes_dict[new_token_id] = new_token
            self.bytes2int_dict[new_token] = new_token_id

            # update
            pairs_discard_ids:dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
            pairs_add_ids:dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
            
            for tid in pairs_tids[best_pair]:
                old_pretoken_bytes = tid_pretoken_bytes[tid]
                new_pretoken_bytes = apply_merge(old_pretoken_bytes, best_pair)
                tid_pretoken_bytes[tid] = new_pretoken_bytes
                
                # TODO: can be optimized by only update the ajacent pairs
                for p in zip(old_pretoken_bytes, old_pretoken_bytes[1:]):
                    pairs_discard_ids[p].add(tid)
                    pairs_counter[p] -= tid_counter[tid]
                
                for p in zip(new_pretoken_bytes, new_pretoken_bytes[1:]):
                    pairs_add_ids[p].add(tid)
                    pairs_counter[p] += tid_counter[tid]
                
            for p, tids in pairs_discard_ids.items():
                pairs_tids[p] -= tids
            for p, tids in pairs_add_ids.items():
                pairs_tids[p] |= tids

            pairs_tids.pop(best_pair)
            pairs_counter.pop(best_pair)

    def encode(self, text: str) -> list[int]:
        ids = []
        merges_rank = {m: i for i, m in enumerate(self.merges)}

        pretoken_seq = pretokenization(text, self.special_tokens, keep_special_tokens=True)
        for pretoken_str in pretoken_seq:
            # pretoken_str: single str splited
            # special token
            if pretoken_str in self.special_tokens:
                ids.append(self.bytes2int_dict[pretoken_str.encode("utf-8")])
                continue
            # not special token
            bytes_token = bytify(pretoken_str)

            # apply BPE merges to the bytes token
            while len(bytes_token) > 1:
                # list all pairs that is in the merges dict
                all_pairs = [p for p in zip(bytes_token, bytes_token[1:]) if p in merges_rank]
                if len(all_pairs) == 0:
                    break

                best_pair = min(all_pairs, key=lambda x: merges_rank[x])
                bytes_token = apply_merge(bytes_token, best_pair)

            ids.extend(self.bytes2int_dict[i] for i in bytes_token)

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            yield from self.encode(s)

    def decode(self, ids: list[int]) -> str:
        byte_str = b"".join(self.int2bytes_dict[id] for id in ids)
        return byte_str.decode("utf-8", errors="replace")  # !WARN

    def save(self, path_prefix: str) -> None:
        # 词表、merges 分开保存习惯，并另存特殊 token 等配置。
        vocab_serialized = {i: b.hex() for i, b in self.int2bytes_dict.items()}
        merges_serialized = [[x.hex(), y.hex()] for (x, y) in self.merges]
        special_tokens_serialized = {"special_tokens": self.special_tokens}

        with open(f"{path_prefix}.vocab.json", "w") as vf:
            json.dump(vocab_serialized, vf, ensure_ascii=False, indent=2)
        with open(f"{path_prefix}.merges.json", "w") as mf:
            json.dump(merges_serialized, mf, ensure_ascii=False, indent=2)
        with open(f"{path_prefix}.special_tokens.json", "w") as sf:
            json.dump(special_tokens_serialized, sf, ensure_ascii=False, indent=2)

    def load(self, path_prefix: str) -> None:
        with open(f"{path_prefix}.vocab.json") as vf:
            vocab_serialized = json.load(vf)
        with open(f"{path_prefix}.merges.json") as mf:
            merges_serialized = json.load(mf)
        with open(f"{path_prefix}.special_tokens.json") as sf:
            special_tokens_serialized = json.load(sf)

        self.int2bytes_dict = {int(i): bytes.fromhex(b) for i, b in vocab_serialized.items()}
        self.bytes2int_dict = {b: int(i) for i, b in self.int2bytes_dict.items()}
        self.merges = [(bytes.fromhex(x), bytes.fromhex(y)) for x, y in merges_serialized]
        self.special_tokens = special_tokens_serialized["special_tokens"]

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> Self:
        with open(vocab_filepath, encoding="utf-8") as vf:
            vocab_serialized = json.load(vf)
        with open(merges_filepath, encoding="utf-8") as mf:
            merges_serialized = json.load(mf)

        if not isinstance(vocab_serialized, dict):
            raise ValueError("vocab file must contain a JSON object")
        if not isinstance(merges_serialized, list):
            raise ValueError("merges file must contain a JSON list")

        vocab = {int(token_id): bytes.fromhex(token_hex) for token_id, token_hex in vocab_serialized.items()}
        merges = [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in merges_serialized]

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
