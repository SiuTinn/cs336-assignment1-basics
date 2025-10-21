import os
from typing import BinaryIO, Dict, List, Tuple, Iterable, Iterator
import regex as re
from tqdm import tqdm
import time
from collections import Counter


class PreTokenizer:
    def __init__(self, special_tokens: list[str] | None = None):
        """
        init pretokenizer

        Args:
            special_tokens
        """
        if special_tokens is None:
            special_tokens = []
        self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        # Use capturing group to preserve special tokens in split
        escaped_tokens = [re.escape(token) for token in self.special_tokens]
        if escaped_tokens:
            pattern = "(" + "|".join(escaped_tokens) + ")"
        else:
            pattern = r"(?!)"
        self.special_tokens_pattern = pattern
        self.word_pattern = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    def find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    def read_corpus(self, input_path: str) -> Iterable[List[str]]:
        with open(input_path, 'rb') as f:
            boundaries = self.find_chunk_boundaries(f, 100, "<|endoftext|>".encode('utf-8'))
            for start, end in tqdm(list(zip(boundaries[:-1], boundaries[1:])), desc="读取语料"):
                f.seek(start)
                chunk = f.read(end - start).decode('utf-8', errors='ignore')
                chunk = chunk.replace('\r\n', '\n')
                yield re.split(self.special_tokens_pattern, chunk)

    def build_word_frequency(self, docs: Iterable[str]) -> Counter:
        """
        构建词频率字典
        """
        word_freq = Counter()
        str_freq = Counter()

        for doc in docs:
            if not doc:
                continue
            # Skip special tokens - they should not be tokenized further
            if doc in self.special_tokens:
                str_freq[doc] += 1
            else:
                matches = [word.group(0) for word in self.word_pattern.finditer(doc)]
                str_freq.update(matches)
        # 将字符串词频转换为字节词频
        for word, freq in str_freq.items():
            word_freq[word.encode("utf-8")] = freq

        return word_freq

    def pretokenize(self, text: str) -> List[bytes]:
        """
        Args:
            input text

        Returns:
            预分词bytes列表
        """
        parts = re.split(self.special_tokens_pattern, text)

        result = []

        for part in parts:
            if part in self.special_tokens:
                result.append(part.encode('utf-8'))
            elif part:
                tokens = [match.group(0).encode('utf-8') for match in self.word_pattern.finditer(part)]
                result.extend(tokens)
        return result

    def pretokenize_iter(self, texts: Iterable[str]) -> Iterable[bytes]:
        """
        Args:
            texts:可迭代的字符串对象

        Returns:
            生成预分词结果bytes的迭代器
        """
        for text in texts:
            parts = re.split(self.special_tokens_pattern, text)

            for part in parts:
                if part in self.special_tokens:
                    yield part.encode('utf-8')
                elif part:
                    for match in self.word_pattern.finditer(part):
                        yield match.group(0).encode('utf-8')
