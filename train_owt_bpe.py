#!/usr/bin/env python3
"""
在OWT数据集上训练BPE分词器的脚本
使用32,000大小的词汇表
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
    """Train BPE tokenizer on OpenWebText dataset and analyze results"""
    
    # Configuration
    input_path = "./data/owt_train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]
    
    # Output paths
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    vocab_output = os.path.join(output_dir, "owt_train_32000_token_vocab.bin")
    merges_output = os.path.join(output_dir, "owt_train_32000_merges.bin")
    stats_output = os.path.join(output_dir, "owt_training_statistics.json")
    
    print("=" * 80)
    print("BPE Tokenizer Training on OpenWebText Dataset")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Input file: {input_path}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Special tokens: {special_tokens}")
    print(f"  Output vocabulary: {vocab_output}")
    print(f"  Output merges: {merges_output}")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        print("Please download the OpenWebText dataset first:")
        print("  wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz")
        print("  gunzip owt_train.txt.gz")
        sys.exit(1)
    
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
    longest_token_id = None
    for token_id, token_bytes in token_vocab.items():
        if len(token_bytes) > longest_length:
            longest_length = len(token_bytes)
            longest_token = token_bytes
            longest_token_id = token_id
    
    print(f"Longest token length: {longest_length} bytes")
    
    # Try to decode the longest token
    if longest_token:
        try:
            decoded = longest_token.decode('utf-8')
            print(f"  Token ID: {longest_token_id}")
            print(f"  Bytes: {longest_token}")
            print(f"  Decoded: {repr(decoded)}")
            print(f"  Interpretation: {decoded}")
        except UnicodeDecodeError as e:
            print(f"  Token ID: {longest_token_id}")
            print(f"  Bytes: {longest_token}")
            print(f"  Cannot decode as UTF-8: {e}")
    
    # Check for special tokens
    print(f"\nSpecial tokens in vocabulary:")
    special_token_bytes = {token.encode('utf-8') for token in special_tokens}
    found_special_tokens = 0
    for token_id, token_bytes in token_vocab.items():
        if token_bytes in special_token_bytes:
            print(f"  Token ID {token_id}: {token_bytes.decode('utf-8')}")
            found_special_tokens += 1
    if found_special_tokens == 0:
        print("  No special tokens found in vocabulary")
    
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
        print(f"  {length} bytes: {count} tokens ({percentage:.2f}%)")
    
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
    statistics = {
        "dataset": "OpenWebText",
        "vocab_size": vocab_size_actual,
        "training_time_seconds": training_time,
        "training_time_formatted": f"{hours}h {minutes}m {seconds:.2f}s",
        "memory_used_mb": memory_used,
        "peak_memory_mb": peak_memory,
        "initial_memory_mb": initial_memory,
        "input_file_size_mb": file_size_mb,
        "vocabulary_file_size_kb": vocab_file_size,
        "merges_file_size_kb": merges_file_size,
        "num_merges": num_merges,
        "token_length_stats": {
            "min": min(token_lengths),
            "max": max(token_lengths),
            "avg": avg_length
        },
        "longest_token": {
            "id": longest_token_id,
            "length": longest_length,
            "bytes": longest_token.hex() if longest_token else None,
            "decoded": longest_token.decode('utf-8', errors='replace') if longest_token else None
        },
        "special_tokens": special_tokens
    }
    
    # Save statistics to JSON
    with open(stats_output, 'w') as f:
        json.dump(statistics, f, indent=2)
    print(f"\nStatistics saved to: {stats_output}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"\nDataset: OpenWebText")
    print(f"Vocabulary size: {vocab_size_actual}")
    print(f"Training time: {hours}h {minutes}m {seconds:.2f}s")
    print(f"Peak memory: {peak_memory:.2f} MB")
    print(f"\nLongest token analysis:")
    print(f"  Length: {longest_length} bytes")
    if longest_token:
        try:
            decoded = longest_token.decode('utf-8')
            print(f"  Decoded value: {repr(decoded)}")
            print(f"  Interpretation: This token represents the sequence: {decoded}")
        except UnicodeDecodeError:
            print(f"  Note: Token cannot be decoded as valid UTF-8")


if __name__ == "__main__":
    train_and_analyze()
