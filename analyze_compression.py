#!/usr/bin/env python3
"""
从TinyStories中抽取10份文档样本
使用已训练的分词器（词汇量10K）进行编码
计算压缩比（字节/令牌）
"""
import os
import sys
import random
from pathlib import Path
import json

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from cs336_basics.bpe_tokenizer.tokenizer import BPETokenizer


def extract_documents(input_path: str, num_documents: int = 10, random_seed: int = 42) -> list[str]:
    """
    从输入文件中提取指定数量的文档
    假设文档由 <|endoftext|> 特殊标记分隔
    
    Args:
        input_path: 输入文件路径
        num_documents: 要提取的文档数量
        random_seed: 随机种子
        
    Returns:
        文档列表
    """
    random.seed(random_seed)
    
    # 读取文件并按特殊标记分隔
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 分割文档
    documents = content.split('<|endoftext|>')
    # 移除空文档
    documents = [doc.strip() for doc in documents if doc.strip()]
    
    print(f"Total documents in file: {len(documents)}")
    
    # 随机抽取指定数量的文档
    if len(documents) < num_documents:
        print(f"Warning: Only {len(documents)} documents available, less than requested {num_documents}")
        selected_docs = documents
    else:
        selected_docs = random.sample(documents, num_documents)
    
    return selected_docs


def analyze_compression(tokenizer: BPETokenizer, documents: list[str]) -> dict:
    """
    分析文档的压缩统计
    
    Args:
        tokenizer: BPE分词器实例
        documents: 文档列表
        
    Returns:
        包含压缩统计的字典
    """
    results = {
        "total_documents": len(documents),
        "documents": [],
        "summary": {}
    }
    
    total_bytes = 0
    total_tokens = 0
    document_compression_ratios = []
    
    print("\n" + "=" * 100)
    print("Document Encoding and Compression Analysis")
    print("=" * 100)
    
    for idx, doc in enumerate(documents, 1):
        # 获取原始字节数
        doc_bytes = doc.encode('utf-8')
        num_bytes = len(doc_bytes)
        
        # 编码文档
        token_ids = tokenizer.encode(doc)
        num_tokens = len(token_ids)
        
        # 计算压缩比
        if num_tokens > 0:
            compression_ratio = num_bytes / num_tokens  # 字节/令牌
        else:
            compression_ratio = 0
        
        document_compression_ratios.append(compression_ratio)
        total_bytes += num_bytes
        total_tokens += num_tokens
        
        # 文档信息
        doc_preview = doc[:100].replace('\n', ' ')  # 预览（前100字符）
        
        print(f"\nDocument {idx}:")
        print(f"  Preview: {doc_preview}..." if len(doc) > 100 else f"  Content: {doc_preview}")
        print(f"  Bytes: {num_bytes:>8}")
        print(f"  Tokens: {num_tokens:>8}")
        print(f"  Compression ratio (bytes/token): {compression_ratio:>8.4f}")
        
        # 显示前10个和后10个token IDs
        if num_tokens > 0:
            preview_tokens = token_ids[:10] if num_tokens > 10 else token_ids
            print(f"  First tokens: {preview_tokens}")
        
        # 记录文档结果
        results["documents"].append({
            "index": idx,
            "bytes": num_bytes,
            "tokens": num_tokens,
            "compression_ratio": compression_ratio,
            "preview": doc_preview[:100],
            "token_ids": token_ids[:20] if len(token_ids) > 20 else token_ids  # 保存前20个token ID
        })
    
    # 计算总体统计
    if total_tokens > 0:
        overall_compression_ratio = total_bytes / total_tokens
    else:
        overall_compression_ratio = 0
    
    avg_compression_ratio = sum(document_compression_ratios) / len(document_compression_ratios) if document_compression_ratios else 0
    min_compression_ratio = min(document_compression_ratios) if document_compression_ratios else 0
    max_compression_ratio = max(document_compression_ratios) if document_compression_ratios else 0
    
    results["summary"] = {
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "overall_compression_ratio": overall_compression_ratio,
        "average_compression_ratio": avg_compression_ratio,
        "min_compression_ratio": min_compression_ratio,
        "max_compression_ratio": max_compression_ratio,
        "avg_bytes_per_document": total_bytes / len(documents) if documents else 0,
        "avg_tokens_per_document": total_tokens / len(documents) if documents else 0,
    }
    
    # 打印总体统计
    print("\n" + "=" * 100)
    print("Summary Statistics")
    print("=" * 100)
    print(f"\nTotal:")
    print(f"  Bytes: {total_bytes:>15}")
    print(f"  Tokens: {total_tokens:>15}")
    print(f"  Overall compression ratio (bytes/token): {overall_compression_ratio:>10.4f}")
    print(f"\nPer-document average:")
    print(f"  Bytes: {total_bytes / len(documents):>15.2f}")
    print(f"  Tokens: {total_tokens / len(documents):>15.2f}")
    print(f"\nCompression ratio statistics:")
    print(f"  Average: {avg_compression_ratio:>15.4f} bytes/token")
    print(f"  Min: {min_compression_ratio:>15.4f} bytes/token")
    print(f"  Max: {max_compression_ratio:>15.4f} bytes/token")
    
    return results


def main():
    """主函数"""
    
    # 配置
    input_path = "./data/TinyStoriesV2-GPT4-train.txt"
    vocab_path = "./output/TinyStories_train_10000_token_vocab.bin"
    merges_path = "./output/TinyStories_train_10000_merges.bin"
    special_tokens = ["<|endoftext|>"]
    
    # 输出路径
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    output_json = os.path.join(output_dir, "compression_analysis_10k.json")
    
    print("=" * 100)
    print("TinyStories Document Compression Analysis")
    print("=" * 100)
    print(f"\nConfiguration:")
    print(f"  Input file: {input_path}")
    print(f"  Vocabulary path: {vocab_path}")
    print(f"  Merges path: {merges_path}")
    print(f"  Vocabulary size: 10,000")
    print(f"  Special tokens: {special_tokens}")
    
    # 检查文件是否存在
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        print(f"\nError: Tokenizer files not found!")
        print(f"  Vocab: {vocab_path} (exists: {os.path.exists(vocab_path)})")
        print(f"  Merges: {merges_path} (exists: {os.path.exists(merges_path)})")
        print(f"\nPlease run 'python train_tinystories_bpe.py' first to train the tokenizer.")
        sys.exit(1)
    
    if not os.path.exists(input_path):
        print(f"\nError: Input file not found: {input_path}")
        sys.exit(1)
    
    # 加载分词器
    print(f"\nLoading tokenizer...")
    tokenizer = BPETokenizer.from_files(vocab_path, merges_path, special_tokens)
    print(f"Tokenizer loaded successfully")
    print(f"Vocabulary size: {len(tokenizer.token_vocab)}")
    
    # 提取文档样本
    print(f"\nExtracting {10} document samples...")
    documents = extract_documents(input_path, num_documents=10)
    print(f"Successfully extracted {len(documents)} documents")
    
    # 分析压缩
    results = analyze_compression(tokenizer, documents)
    
    # 保存结果到JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_json}")
    
    # 打印关键发现
    print("\n" + "=" * 100)
    print("Key Findings")
    print("=" * 100)
    print(f"\n压缩比分析 (字节/令牌):")
    print(f"  - 整体压缩比: {results['summary']['overall_compression_ratio']:.4f}")
    print(f"  - 平均压缩比: {results['summary']['average_compression_ratio']:.4f}")
    print(f"  - 最小压缩比: {results['summary']['min_compression_ratio']:.4f} (最高效)")
    print(f"  - 最大压缩比: {results['summary']['max_compression_ratio']:.4f} (最低效)")
    
    print(f"\n解释:")
    print(f"  - 压缩比 < 1.0: 不可能 (至少每个token占1字节)")
    print(f"  - 压缩比 ≈ 1.0: 平均每个token占1字节（主要由单字节token组成）")
    print(f"  - 压缩比 ≈ 4.0: 平均每个token占4字节（较多多字节token）")
    print(f"  - 压缩比越高: BPE效率越好（更少令牌表示相同内容）")
    
    compression_ratio = results['summary']['overall_compression_ratio']
    print(f"\n总结:")
    print(f"  该分词器的整体压缩比为 {compression_ratio:.4f} 字节/令牌")
    print(f"  这意味着平均每个令牌代表约 {compression_ratio:.2f} 字节的文本")
    if compression_ratio > 4.0:
        print(f"  这是一个很好的压缩率，表明BPE产生了有效的子词单位")
    elif compression_ratio > 3.0:
        print(f"  这是一个良好的压缩率")
    else:
        print(f"  这相对较低，可能需要更大的词汇量")


if __name__ == "__main__":
    main()
