import os
import regex as re
import time
from collections import Counter
from typing import Dict, List, Set, Tuple
import cProfile
import pstats
import heapq

from tqdm import tqdm
from cs336_basics.bpe_tokenizer.pre_tokenizer import PreTokenizer


class BPETrainer:
    def __init__(self, vocab_size: int, special_tokens: List[str]):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.preprocessor = PreTokenizer(special_tokens)
        self.token_vocab: Dict[int, bytes] = {}
        self.merges: List[Tuple[bytes, bytes]] = []
        self.splits: Dict[bytes, List[bytes]] = {}
        self.pair_freqs: Dict[Tuple[bytes, bytes], int] = {}
        self.pair_to_words: Dict[Tuple[bytes, bytes], Set[bytes]] = {}
        self.freq_max_heap = []
    
    def _push_pair_to_heap(self, pair: Tuple[bytes, bytes], freq: int) -> None:
        heapq.heappush(self.freq_max_heap, (-freq, pair))

    def _pop_pair_from_heap(self) -> Tuple[bytes, bytes]:
        """
        从最大堆中弹出频率最高的字节对
        """
        while self.freq_max_heap:
            freq, pair = heapq.heappop(self.freq_max_heap)
            freq = -freq
            if pair in self.pair_freqs and self.pair_freqs[pair] == freq:
                # 因为pair_freqs删除pair/减少某个pair的freq后，最大堆不立刻同步更新（使用懒惰删除策略）,所以要在弹出时进行检测
                # 如果不一样说明对应频率已经被减小/被删除
                return pair
        raise ValueError("堆没有返回频率最大的字节对")

    def initialize_splits_and_pairs(self, word_freqs: Counter) -> None:
        for word, word_freq in word_freqs.items():
            # Special tokens should not be split into bytes
            special_token_bytes = {token.encode('utf-8') for token in self.special_tokens}
            if word in special_token_bytes:
                # Keep special tokens as atomic units
                self.splits[word] = [word]
            else:
                self.splits[word] = [bytes([b]) for b in word]
            
            word_pieces = self.splits[word]
            if len(word_pieces) == 1:
                continue
            for j, pair in enumerate(zip(word_pieces[:-1],word_pieces[1:])):
                self.pair_freqs[pair] = self.pair_freqs.get(pair, 0) + word_freq
                if pair not in self.pair_to_words:
                    self.pair_to_words[pair] = set()
                self.pair_to_words[pair].add(word)
        
        for pair, freq in self.pair_freqs.items():
            self._push_pair_to_heap(pair, freq)
    
    def find_best_pairs(self) -> Tuple[bytes, bytes]:
        """
        从堆中弹出频率最高的字节对（优化版）
        使用堆避免每次都遍历整个pair_freqs字典
        时间复杂度从O(n)降低到O(log n) per iteration
        """
        special_token_bytes = {token.encode('utf-8') for token in self.special_tokens}
        
        # 从堆中取出频率最高的对，跳过包含特殊token的对
        while self.freq_max_heap:
            neg_freq, pair = heapq.heappop(self.freq_max_heap)
            freq = -neg_freq
            
            # 检查该对是否仍然有效（懒惰删除策略）
            if pair not in self.pair_freqs or self.pair_freqs[pair] != freq:
                continue
            
            # 跳过包含特殊token的对
            if pair[0] in special_token_bytes or pair[1] in special_token_bytes:
                continue
                
            return pair
        
        raise ValueError("没有找到频率最高的字节对")

    def _update_pair_freqs(self, new_pair, old_pair, word, word_freq) -> None:
        self.pair_to_words.setdefault(new_pair, set()).add(word)
        self.pair_freqs[new_pair] = self.pair_freqs.get(new_pair, 0) + word_freq
        self._push_pair_to_heap(new_pair, self.pair_freqs[new_pair])

        if old_pair in self.pair_freqs:
            self.pair_freqs[old_pair] -= word_freq  
            if self.pair_freqs[old_pair] <= 0:
                del self.pair_freqs[old_pair]
            else:
                self._push_pair_to_heap(old_pair, self.pair_freqs[old_pair])

    def update_splits_and_pairs(
            self,
            best_pair: Tuple[bytes, bytes],
            new_token: bytes,
            word_freqs: Counter) -> None:
        """
        包含best_pair的单词需要更新
        直接从反向索引获得
        """
        affected_words = list(self.pair_to_words.get(best_pair, set()))
        best_pair_deleted = False  # ✅ 标记是否已删除 best_pair
        
        for word in affected_words:
            word_freq = word_freqs[word]
            word_pieces = self.splits[word]
            i = 0
            while i < len(word_pieces) - 1:
                if word_pieces[i] == best_pair[0] and word_pieces[i+1] == best_pair[1]:
                    word_pieces[i] = new_token
                    word_pieces.pop(i+1)
                    
                    # ✅ 只删除一次
                    if not best_pair_deleted:
                        if best_pair in self.pair_freqs:
                            del self.pair_freqs[best_pair]
                        if best_pair in self.pair_to_words:
                            del self.pair_to_words[best_pair]
                        best_pair_deleted = True
                    
                    if i > 0:
                        new_pair_left = (word_pieces[i-1], new_token)
                        old_pair_left = (word_pieces[i-1], best_pair[0])
                        self._update_pair_freqs(new_pair_left, old_pair_left, word, word_freq)
                    if i < len(word_pieces) - 1:
                        new_pair_right = (new_token, word_pieces[i+1])
                        old_pair_right = (best_pair[1], word_pieces[i+1])
                        self._update_pair_freqs(new_pair_right, old_pair_right, word, word_freq)
                else:
                    i += 1

    def add_special_tokens(self) -> None:
        for m, token in enumerate(self.special_tokens):
            self.token_vocab[self.vocab_size - len(self.special_tokens) + m] = token.encode('utf-8')

    def train(self, input_path: str) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        print("[1/4] 读取语料库...")
        word_freq = Counter()
        for docs in self.preprocessor.read_corpus(input_path):
            chunk_word_freq = self.preprocessor.build_word_frequency(docs)
            word_freq.update(chunk_word_freq)
        print(f"✓ 加载完成，词汇量: {len(word_freq)}")

        self.token_vocab = {i: bytes([i]) for i in range(256)}
        num_merges = self.vocab_size - 256 - len(self.special_tokens)
        self.merges = []

        print("[2/4] 初始化字节对和频率...")
        self.initialize_splits_and_pairs(word_freq)
        print(f"✓ 初始化完成，字节对数: {len(self.pair_freqs)}")

        print(f"[3/4] 执行 {num_merges} 次 BPE 合并...")
        for num_merge in tqdm(range(num_merges), desc="合并进度", unit="merge"):
            if not self.pair_freqs:
                print(f"⚠ 在第 {num_merge} 次合并时用尽了所有字节对")
                break

            best_pair = self.find_best_pairs()
            self.merges.append(best_pair)

            new_token = best_pair[0] + best_pair[1]
            self.token_vocab[256 + num_merge] = new_token
            self.update_splits_and_pairs(best_pair, new_token, word_freq)

        print(f"✓ 完成 {len(self.merges)} 次合并")

        print("[4/4] 添加特殊 token...")
        self.add_special_tokens()
        print(f"✓ 词汇表大小: {len(self.token_vocab)}")

        return self.token_vocab, self.merges

    def to_file(self, vocab_filepath: str, merges_filepath: str) -> None:
        # 创建输出目录（如果不存在）
        os.makedirs(os.path.dirname(vocab_filepath) or '.', exist_ok=True)
        os.makedirs(os.path.dirname(merges_filepath) or '.', exist_ok=True)
        
        with open(vocab_filepath, 'wb') as f:
            f.write(len(self.token_vocab).to_bytes(4, byteorder='little'))
            for token_id, token in self.token_vocab.items():
                f.write(token_id.to_bytes(4, byteorder='little'))
                f.write(len(token).to_bytes(4, byteorder='little'))
                f.write(token)

        with open(merges_filepath, 'wb') as f:
            f.write(len(self.merges).to_bytes(4, byteorder='little'))
            for first, second in self.merges:
                f.write(len(first).to_bytes(4, byteorder='little'))
                f.write(first)
                f.write(len(second).to_bytes(4, byteorder='little'))
                f.write(second)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str]
    ) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    tokenizer = BPETrainer(vocab_size, special_tokens)
    return tokenizer.train(input_path)


if __name__ == "__main__":
    # Example usage
    input_path = "./data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    trainer = BPETrainer(vocab_size, special_tokens)
    token_vocab, merges = trainer.train(input_path)
    # cProfile.run('trainer.train(input_path)', 'tokenizer_stats') 
    # p = pstats.Stats('tokenizer_stats')
    # p.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)
    trainer.to_file(
        "./data/output/TinyStories_train_10000_token_vocab.bin",
        "./data/output/TinyStories_train_10000_merges.bin"
    )

    # print("Token Vocabulary:")
    # for idx, token in token_vocab.items():
    #     if isinstance(token, bytes):
    #         try:
    #             decoded = token.decode('utf-8')
    #             print(f"{idx}: {decoded}")
    #         except UnicodeDecodeError:
    #             print(f"{idx}: {list(token)}")  # 以字节列表形式显示
    #     else:
    #         print(f"{idx}: {token}")
    
    # print("\nMerges:")
    # for merge in merges:
    #     try:
    #         first = merge[0].decode('utf-8')
    #     except UnicodeDecodeError:
    #         first = list(merge[0])
        
    #     try:
    #         second = merge[1].decode('utf-8')
    #     except UnicodeDecodeError:
    #         second = list(merge[1])
        
    #     print(f"{first}{second}")
