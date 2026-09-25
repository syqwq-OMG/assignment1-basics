from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import heapq
import multiprocessing
import os
from collections import Counter, defaultdict, OrderedDict
import json
from typing import Self
from tqdm.auto import tqdm

from .pretokenization import find_chunk_boundaries, pretokenization


class ReverseOrderPair:
    """Reverse byte-pair ordering for use in Python's min-heap."""

    __slots__ = ("pair",)

    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other: "ReverseOrderPair") -> bool:
        return self.pair > other.pair

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ReverseOrderPair) and self.pair == other.pair


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


def count_pretokens_in_chunk(task: tuple[str | os.PathLike, int, int, tuple[str, ...]]) -> Counter[str]:
    """Read and pre-tokenize one byte range of the training corpus."""
    input_path, start, end, special_tokens = task
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_text = f.read(end - start).decode("utf-8")
    return Counter(pretokenization(chunk_text, list(special_tokens)))


class BPE:
    def __init__(
        self,
        vocab: dict[int, bytes] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None,
        special_tokens: list[str] | None = None,
        cache_size: int = 2048,
    ):
        self.int2bytes_dict = vocab
        self.bytes2int_dict = {v: k for k, v in vocab.items()} if vocab is not None else {}
        self.merges_rank: dict[tuple[bytes, bytes], int] = {m: i for i, m in enumerate(merges)} if merges is not None else {}
        self.special_tokens = list(dict.fromkeys(special_tokens or []))
        self.encode_cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        self.CACHE_SIZE_LIMIT: int = cache_size  # Limit the cache size to avoid excessive memory usage

    def get_vocab(self) -> dict[int, bytes]:
        return self.int2bytes_dict

    def get_merges(self) -> list[tuple[bytes, bytes]]:
        return list(self.merges_rank.keys())

    def get_special_tokens(self) -> list[str]:
        return self.special_tokens

    def query_cache(self, text: str) -> tuple[int, ...] | None:
        if text in self.encode_cache:
            self.encode_cache.move_to_end(text)  # mark as recently used
            return self.encode_cache[text]
        return None

    def into_cache(self, text: str, ids: tuple[int, ...]) -> None:
        self.encode_cache[text] = ids
        if len(self.encode_cache) > self.CACHE_SIZE_LIMIT:
            # Remove the oldest item from the cache
            self.encode_cache.popitem(last=False)

    def train(self, input_path: str | os.PathLike, vocab_size: int, num_threads: int | None = None) -> None:
        # 训练后设置 self.vocab 和 self.merges，并重建编码查找表。
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file {input_path} does not exist")

        self.encode_cache.clear()  # new train means potential new encoding result

        minimum_vocab_size = 256 + len(self.special_tokens)
        if vocab_size < minimum_vocab_size:
            raise ValueError(f"vocab_size must be at least {minimum_vocab_size}")
        if num_threads is not None and num_threads < 1:
            raise ValueError("num_threads must be at least 1")

        pretoken_tid: dict[str, int] = defaultdict(int)  # {"low": 0, "lowe": 1, ...}
        tid_pretoken_bytes: dict[int, tuple[bytes, ...]] = defaultdict(tuple)  # {0: (b"l", b"o", b"w"), 1: (b"o", b"w"), ...}
        tid_counter: dict[int, int] = defaultdict(int)  # {0: 5, 1: 5, ...}
        pairs_tids: dict[tuple[bytes, bytes], set[int]] = defaultdict(
            set
        )  # (b"l", b"o"): {0, 1, 2}, (b"o", b"w"): {0, 1, 2}, ... where the pair occurs
        pairs_counter: dict[tuple[bytes, bytes], int] = Counter()  # (b"l", b"o"): 5, (b"o", b"w"): 5, ...

        # Split the corpus once, then independently read and pre-tokenize each
        # byte range in a worker thread or process.
        with open(input_path, "rb") as f:
            chunk_boundaries = find_chunk_boundaries(
                f, desired_num_chunks=10, split_special_token=(token.encode("utf-8") for token in self.special_tokens)
            )

        special_tokens = tuple(self.special_tokens)
        chunk_tasks = [(input_path, chunk_boundaries[i], chunk_boundaries[i + 1], special_tokens) for i in range(len(chunk_boundaries) - 1)]
        pretoken_counter: Counter[str] = Counter()

        # determine whether to use threads or processes based on the file size and user preference
        if num_threads is not None:
            executor_type = ThreadPoolExecutor
            worker_count = min(num_threads, len(chunk_tasks))
            executor_kwargs = {}
        elif os.path.getsize(input_path) >= 64 * 1024 * 1024:
            executor_type = ProcessPoolExecutor
            worker_count = min(4, os.cpu_count() or 1, len(chunk_tasks))
            start_method = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
            executor_kwargs = {"mp_context": multiprocessing.get_context(start_method)}
        else:
            executor_type = ThreadPoolExecutor
            worker_count = 1
            executor_kwargs = {}

        # do pretokenization in parallel using the chosen executor type
        with executor_type(max_workers=worker_count, **executor_kwargs) as executor:
            chunk_counters = executor.map(count_pretokens_in_chunk, chunk_tasks)
            
            for chunk_counter in tqdm(chunk_counters, total=len(chunk_tasks), desc="Pretokenizing", unit="chunk", disable=None):
                pretoken_counter.update(chunk_counter)

        for pretoken, count in pretoken_counter.items():
            tid = len(pretoken_tid)
            pretoken_tid[pretoken] = tid
            tid_pretoken_bytes[tid] = bytify(pretoken)
            tid_counter[tid] = count

        # get pairs_tids and pairs_counter
        for tid, byte_ptk in tid_pretoken_bytes.items():
            current_count = tid_counter[tid]
            for pair in zip(byte_ptk, byte_ptk[1:]):
                if pair not in pairs_tids:
                    pairs_tids[pair] = set()
                pairs_tids[pair].add(tid)
                pairs_counter[pair] += current_count

        # =======================================================================
        # ||               pretoken finish, now training, merging               ||
        # =======================================================================
        self.int2bytes_dict = {i: bytes([i]) for i in range(256)}
        # Special tokens are part of the requested vocabulary size.
        for special_token in self.special_tokens:
            self.int2bytes_dict[len(self.int2bytes_dict)] = special_token.encode("utf-8")

        self.bytes2int_dict = {v: k for k, v in self.int2bytes_dict.items()}
        self.merges_rank = {}

        pair_heap = [(-count, ReverseOrderPair(pair)) for pair, count in pairs_counter.items()]
        heapq.heapify(pair_heap)

        progress = tqdm(total=vocab_size - len(self.int2bytes_dict), desc="Training BPE", unit="merge", disable=None)
        while len(self.int2bytes_dict) < vocab_size:
            best_pair = None
            while pair_heap:
                neg_count, wrapped_pair = heapq.heappop(pair_heap)
                candidate = wrapped_pair.pair
                # lazy update, use the counter to check if the count is still valid
                if pairs_counter.get(candidate) == -neg_count:
                    best_pair = candidate
                    break
            if best_pair is None:
                break

            self.merges_rank[best_pair] = len(self.merges_rank)

            new_token = best_pair[0] + best_pair[1]
            new_token_id = len(self.int2bytes_dict)
            self.int2bytes_dict[new_token_id] = new_token
            self.bytes2int_dict[new_token] = new_token_id

            # update
            pairs_discard_ids: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
            pairs_add_ids: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

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

            # lazy update pairs_tids and pairs_counter
            for p, tids in pairs_discard_ids.items():
                pairs_tids[p] -= tids
            for p, tids in pairs_add_ids.items():
                pairs_tids[p] |= tids

            affected_pairs = pairs_discard_ids.keys() | pairs_add_ids.keys()

            for p in affected_pairs:
                count = pairs_counter.get(p, 0)
                if count > 0:
                    heapq.heappush(pair_heap, (-count, ReverseOrderPair(p)))
                else:
                    # remove 0 count pairs from the counter
                    pairs_counter.pop(p, None)
                    pairs_tids.pop(p, None)

            progress.update(1)

        progress.close()

    def encode(self, text: str) -> list[int]:
        ids = []

        pretoken_seq = pretokenization(text, self.special_tokens, keep_special_tokens=True)
        for pretoken_str in pretoken_seq:
            # pretoken_str: single str splited

            # special token
            if pretoken_str in self.special_tokens:
                ids.append(self.bytes2int_dict[pretoken_str.encode("utf-8")])
                continue

            # not special token
            # check if the pretoken is in the cache
            cached = self.query_cache(pretoken_str)
            if cached is not None:
                ids.extend(cached)
                continue

            bytes_token = bytify(pretoken_str)

            # apply BPE merges to the bytes token
            while len(bytes_token) > 1:
                # list all pairs that is in the merges dict
                all_pairs = [p for p in zip(bytes_token, bytes_token[1:]) if p in self.merges_rank]
                if len(all_pairs) == 0:
                    break

                best_pair = min(all_pairs, key=lambda x: self.merges_rank[x])
                bytes_token = apply_merge(bytes_token, best_pair)

            pretoken_bytes_encoded = [self.bytes2int_dict[b] for b in bytes_token]
            ids.extend(pretoken_bytes_encoded)
            self.into_cache(pretoken_str, tuple(pretoken_bytes_encoded))

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
        merges_serialized = [[x.hex(), y.hex()] for (x, y) in self.merges_rank.keys()]
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
        self.merges_rank = {(bytes.fromhex(x), bytes.fromhex(y)): i for i, (x, y) in enumerate(merges_serialized)}
        self.special_tokens = special_tokens_serialized["special_tokens"]
        self.encode_cache.clear()  # clear cache after loading new model

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
