import os
import numpy as np
from typing import Dict, Iterable, List, Tuple
from tqdm import tqdm
from cs336_basics.bpe_tokenizer.pre_tokenizer import PreTokenizer


class BPETokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: List[str] = None,
    ) -> None:
        self.token_vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.token_to_id: Dict[bytes, int] = {token: idx for idx, token in self.token_vocab.items()}
        self.pre_tokenizer = PreTokenizer(self.special_tokens)
        self.word_to_ids: Dict[bytes, List[int]] = {} #缓存表 缓存已经计算过的词对应id序列

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] = None) -> 'BPETokenizer':
        """
        Args:
            vocab_filepath
            merges_filepath
            special_tokens

        Returns:
            BPETokenizer 实例
        """
        vocab = {}
        with open(vocab_filepath, 'rb') as f:
            vocab_size_bytes = f.read(4)
            vocab_size = int.from_bytes(vocab_size_bytes, byteorder='little')

            for _ in range(vocab_size):
                token_id_bytes = f.read(4)
                token_id = int.from_bytes(token_id_bytes, byteorder='little')
                token_len_bytes = f.read(4)
                token_len = int.from_bytes(token_len_bytes, byteorder='little')
                token = f.read(token_len)
                vocab[token_id] = token
            
        merges = []
        with open(merges_filepath, 'rb') as f:
            merges_count_bytes = f.read(4)
            merges_count = int.from_bytes(merges_count_bytes, byteorder='little')

            for _ in range(merges_count):
                first_len_bytes = f.read(4)
                first_len = int.from_bytes(first_len_bytes, byteorder='little')
                first = f.read(first_len)
                second_len_bytes = f.read(4)
                second_len = int.from_bytes(second_len_bytes, byteorder='little')
                second = f.read(second_len)
                merges.append((first, second))
        return cls(vocab, merges, special_tokens)

    def calculate_token_ids(self, word: bytes) -> List[int]:
        """
        将一个bytes根据merges不断合并
        得到token ID序列
        """
        token_ids = []
        bytes_list = [bytes([b]) for b in word]

        while len(bytes_list) > 1:
            min_rule_idx = None
            min_merge_pos = None

            for i, pair in enumerate(zip(bytes_list[:-1], bytes_list[1:])):
                idx = self.token_to_id.get(pair[0] + pair[1])
                if (idx is not None) and ((min_rule_idx is None) or (idx < min_rule_idx)):
                    min_rule_idx = idx
                    min_merge_pos = i
            
            if min_rule_idx is None:
                break

            bytes_list[min_merge_pos:min_merge_pos + 2] = [bytes_list[min_merge_pos] + bytes_list[min_merge_pos + 1]]

        for part in bytes_list:
            try:
                id = self.token_to_id[part]
                token_ids.append(id)
            except KeyError:
                print(f"Warning: Token {part} not found in vocabulary.")
                pass
        
        return token_ids
    
    def encode(self, text: str) -> List[int]:
        """
        text -> BPE token ID list
        """
        words = self.pre_tokenizer.pretokenize(text)
        ids = []
        for word in words:
            if word in self.token_to_id:
                ids.append(self.token_to_id[word])
            elif word in self.word_to_ids:
                ids.extend(self.word_to_ids[word])
            else:
                token_ids = self.calculate_token_ids(word)
                self.word_to_ids[word] = token_ids
                ids.extend(token_ids)
        return ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        对可迭代对象中的每个文本进行编码 每次调用返回一个token ID
        """
        word_iter = self.pre_tokenizer.pretokenize_iter(iterable)
        for word in word_iter:
            if word in self.token_to_id:
                yield self.token_to_id[word]
            elif word in self.word_to_ids:
                yield from self.word_to_ids[word]
            else:
                token_ids = self.calculate_token_ids(word)
                self.word_to_ids[word] = token_ids
                yield from token_ids
    
    def encode_to_npfile(self, input_path: os.PathLike, output_path: os.PathLike) -> None:
        """
        把输入文件中的文本编码为BPE token ID列表 并保存为numpy数组文件 .npy
        """
        token_ids = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for token_id in tqdm(self.encode_iterable(f), desc="编码进度", unit="tokens"):
                token_ids.append(token_id)
        
        # 转换为 numpy 数组并保存
        token_array = np.array(token_ids, dtype=np.int32)
        np.save(output_path, token_array)
        print(f"文件已经保存到{output_path}\n总token数:{len(token_ids)}")
    
    def decode(self, ids: Iterable[int], end_token_id: int = None) -> str:
        """
        将BPE token ID列表解码为文本
        """
        text_bytes = b""
        for id in ids:
            if id in self.token_vocab:
                text_bytes += self.token_vocab[id]
            else:
                print(f"Warning: ID {id} not found in vocabulary.")
                continue

            if (end_token_id is not None) and (id == end_token_id):
                break
            
        return text_bytes.decode('utf-8', errors='ignore')


if __name__ == "__main__":
    # Example usage
    special_tokens = ["<|endoftext|>"]
    vocab_path = "./data/output/TinyStories_train_10000_token_vocab.bin"
    mergers_path = "./data/output/TinyStories_train_10000_merges.bin"
    input_path = "./data/TinyStoriesV2-GPT4-valid.txt"
    # input_path = "./data/TinyStoriesV2-GPT4-train.txt"
    tokenizer = BPETokenizer.from_files(vocab_path, mergers_path, special_tokens)
    output_path = "./data/output/TinyStories_valid_10000_token_ids.npy"
    tokenizer.encode_to_npfile(input_path, output_path)

    # with open(input_path, 'r', encoding='utf-8') as f:
    #     text = f.read()

    # encoded_ids = tokenizer.encode(text)
    # print("Encoded IDs:", encoded_ids[:1000])

    # decoded_text = tokenizer.decode(encoded_ids[:1000])
    # print("Decoded Text:", decoded_text)

    # # 迭代器使用
    # token_list = []
    # with open(input_path, 'r', encoding='utf-8') as f:
    #     for id in tokenizer.encode_iterable(f):
    #         print(id, end=' ')
    #         token_list.append(id)
    #         if id == 999:
    #             break
    # decoded_text = tokenizer.decode(token_list)
    # print("\nDecoded Text:", decoded_text)