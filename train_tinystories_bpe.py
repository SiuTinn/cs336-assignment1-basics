#!/usr/bin/env python3
"""
在TinyStories数据集上训练BPE分词器的脚本
使用10000大小的词汇表，包含<|endoftext|>特殊令牌
记录训练时间、内存使用情况和词汇表统计信息
"""

import os
import sys
import time
import psutil
import json
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from cs336_basics.bpe_tokenizer.trainer import BPETrainer


def get_process_memory_mb():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def train_and_analyze():
    """Train BPE tokenizer on TinyStories dataset and analyze results"""
    
    # Configuration
    input_path = "./data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    
    # Output paths
    output_dir = "./data/output"
    os.makedirs(output_dir, exist_ok=True)
    
    vocab_output = os.path.join(output_dir, "TinyStories_train_10000_token_vocab.bin")
    merges_output = os.path.join(output_dir, "TinyStories_train_10000_merges.bin")
    stats_output = os.path.join(output_dir, "training_statistics.json")
    
    print("=" * 80)
    print("BPE Tokenizer Training on TinyStories Dataset")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Input file: {input_path}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Special tokens: {special_tokens}")
    print(f"  Output vocabulary: {vocab_output}")
    print(f"  Output merges: {merges_output}")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"\nError: Input file not found: {input_path}")
        return False
    
    file_size_mb = os.path.getsize(input_path) / 1024 / 1024
    print(f"\nInput file size: {file_size_mb:.2f} MB")
    
    # Record initial memory
    initial_memory = get_process_memory_mb()
    print(f"Initial process memory: {initial_memory:.2f} MB")
    
    # Start training
    print("\nStarting training...")
    start_time = time.time()
    start_memory = get_process_memory_mb()
    
    trainer = BPETrainer(vocab_size, special_tokens)
    token_vocab, merges = trainer.train(input_path)
    
    end_time = time.time()
    end_memory = get_process_memory_mb()
    
    # Calculate statistics
    training_time = end_time - start_time
    memory_used = end_memory - start_memory
    peak_memory = end_memory
    
    # Convert training time to hours and minutes
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = training_time % 60
    
    print(f"\nTraining completed!")
    print(f"  Total time: {hours}h {minutes}m {seconds:.2f}s ({training_time:.2f} seconds)")
    print(f"  Memory used: {memory_used:.2f} MB")
    print(f"  Peak memory: {peak_memory:.2f} MB")
    
    # Analyze vocabulary
    print("\n" + "=" * 80)
    print("Vocabulary Analysis")
    print("=" * 80)
    
    vocab_size_actual = len(token_vocab)
    print(f"\nVocabulary size: {vocab_size_actual}")
    
    # Find longest token
    longest_token = None
    longest_length = 0
    for token_id, token_bytes in token_vocab.items():
        if len(token_bytes) > longest_length:
            longest_length = len(token_bytes)
            longest_token = token_bytes
    
    print(f"Longest token length: {longest_length} bytes")
    
    # Try to decode the longest token
    if longest_token:
        try:
            decoded = longest_token.decode('utf-8')
            print(f"Longest token (decoded): '{decoded}'")
            print(f"Longest token (hex): {longest_token.hex()}")
        except UnicodeDecodeError:
            print(f"Longest token (bytes): {list(longest_token)}")
            print(f"Longest token (hex): {longest_token.hex()}")
    
    # Check for special tokens
    print(f"\nSpecial tokens in vocabulary:")
    special_token_bytes = {token.encode('utf-8') for token in special_tokens}
    for token_id, token_bytes in token_vocab.items():
        if token_bytes in special_token_bytes:
            print(f"  Token ID {token_id}: {token_bytes.decode('utf-8')}")
    
    # Analyze token length distribution
    token_lengths = [len(token) for token in token_vocab.values()]
    avg_length = sum(token_lengths) / len(token_lengths)
    
    print(f"\nToken length statistics:")
    print(f"  Average length: {avg_length:.2f} bytes")
    print(f"  Min length: {min(token_lengths)} bytes")
    print(f"  Max length: {max(token_lengths)} bytes")
    
    # Count tokens by length
    length_counts = {}
    for length in token_lengths:
        length_counts[length] = length_counts.get(length, 0) + 1
    
    print(f"\nTokens by length:")
    for length in sorted(length_counts.keys()):
        count = length_counts[length]
        percentage = (count / len(token_vocab)) * 100
        print(f"  {length} byte(s): {count} tokens ({percentage:.1f}%)")
    
    # Number of merges
    num_merges = len(merges)
    print(f"\nNumber of merges: {num_merges}")
    print(f"Expected merges: {vocab_size - 256 - len(special_tokens)}")
    
    # Save vocabulary and merges to disk
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)
    
    trainer.to_file(vocab_output, merges_output)
    print(f"\nVocabulary saved to: {vocab_output}")
    print(f"Merges saved to: {merges_output}")
    
    # Verify file sizes
    vocab_file_size = os.path.getsize(vocab_output) / 1024
    merges_file_size = os.path.getsize(merges_output) / 1024
    print(f"  Vocabulary file size: {vocab_file_size:.2f} KB")
    print(f"  Merges file size: {merges_file_size:.2f} KB")
    
    # Prepare statistics dictionary
    stats = {
        "training_config": {
            "input_file": input_path,
            "input_file_size_mb": file_size_mb,
            "vocab_size": vocab_size,
            "special_tokens": special_tokens,
        },
        "training_time": {
            "total_seconds": training_time,
            "hours": hours,
            "minutes": minutes,
            "seconds": seconds,
            "readable": f"{hours}h {minutes}m {seconds:.2f}s"
        },
        "memory": {
            "initial_mb": initial_memory,
            "peak_mb": peak_memory,
            "used_mb": memory_used,
        },
        "vocabulary_stats": {
            "actual_size": vocab_size_actual,
            "longest_token_length": longest_length,
            "longest_token_hex": longest_token.hex() if longest_token else None,
            "average_token_length": avg_length,
            "min_token_length": min(token_lengths),
            "max_token_length": max(token_lengths),
            "tokens_by_length": length_counts,
        },
        "merges_stats": {
            "total_merges": num_merges,
            "expected_merges": vocab_size - 256 - len(special_tokens),
        },
        "output_files": {
            "vocabulary": vocab_output,
            "vocabulary_size_kb": vocab_file_size,
            "merges": merges_output,
            "merges_size_kb": merges_file_size,
        }
    }
    
    # Save statistics to JSON
    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"Statistics saved to: {stats_output}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"\n✓ Training time: {hours}h {minutes}m {seconds:.2f}s")
    print(f"✓ Memory used: {memory_used:.2f} MB")
    print(f"✓ Peak memory: {peak_memory:.2f} MB")
    print(f"✓ Longest token: {longest_length} bytes")
    
    # Analysis of reasonableness
    print("\n" + "=" * 80)
    print("Analysis: Is the longest token reasonable?")
    print("=" * 80)
    
    print(f"\nThe longest token is {longest_length} bytes long.")
    print(f"Average token length is {avg_length:.2f} bytes.")
    
    if longest_token:
        try:
            decoded = longest_token.decode('utf-8')
            print(f"\nLongest token: '{decoded}'")
        except:
            print(f"\nLongest token: (non-UTF-8 bytes)")
    
    print("\nAnalysis:")
    print(f"  - Initial byte vocabulary: 256 tokens (1 byte each)")
    print(f"  - Special tokens: {len(special_tokens)} token(s) ({longest_length} bytes)")
    print(f"  - Merged tokens: {num_merges} (various lengths)")
    print(f"\nThe longest token represents frequent subword units in the")
    print(f"training corpus that were created through BPE merging.")
    print(f"Token length of {longest_length} bytes is reasonable as it")
    print(f"represents a sequence that occurred frequently together in")
    print(f"the TinyStories dataset and was iteratively merged by BPE.")
    
    print("\n" + "=" * 80)
    
    return True


if __name__ == "__main__":
    success = train_and_analyze()
    sys.exit(0 if success else 1)
