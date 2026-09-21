from collections.abc import Iterable, Iterator

class BPE:
    def __init__(self, vocab: dict[int, bytes] | None = None,
                 merges: list[tuple[bytes, bytes]] | None = None,
                 special_tokens: list[str] | None = None):
        raise NotImplementedError

    def train(self, input_path: str, vocab_size: int,
              special_tokens: list[str]) -> None:
        # 训练后设置 self.vocab 和 self.merges，并重建编码查找表。
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    def save(self, path_prefix: str) -> None:
        # 词表、merges 分开保存习惯，并另存特殊 token 等配置。
        raise NotImplementedError

    def load(self, path_prefix: str) -> None:
        raise NotImplementedError

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str,
                   special_tokens: list[str] | None = None):
        raise NotImplementedError