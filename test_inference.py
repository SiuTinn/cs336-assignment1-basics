"""
简单的模型推理测试脚本
用于测试训练好的 TransformerLM 模型生成文本的效果
"""

import torch
from cs336_basics.bpe_tokenizer.tokenizer import BPETokenizer
from cs336_basics.transformer.module import TransformerLM, softmax
from cs336_basics.transformer.train_utils import load_checkpoint


def nucleus_sampling(probs, top_p):
    """
    Top-p (nucleus) 采样
    Args:
        probs: (batch_size, vocab_size) 概率分布
        top_p: 累计概率阈值
    Returns:
        过滤后的概率分布
    """
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 创建nucleus mask
    nucleus_mask = cumulative_probs <= top_p
    nucleus_mask[:, 0] = True  # 至少保留概率最高的token
    
    # 过滤概率
    sorted_probs_filtered = sorted_probs * nucleus_mask
    sorted_probs_sum = torch.sum(sorted_probs_filtered, dim=-1, keepdim=True)
    sorted_probs_normalized = sorted_probs_filtered / sorted_probs_sum
    
    # 恢复原始顺序
    probs_filtered = torch.zeros_like(probs)
    probs_filtered.scatter_(1, sorted_indices, sorted_probs_normalized)
    
    return probs_filtered


def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=128,
    temperature=1.0,
    top_p=0.9,
    device=None
):
    """
    生成文本
    Args:
        model: TransformerLM 模型
        tokenizer: BPE tokenizer
        prompt: 输入提示文本
        max_new_tokens: 最大生成token数量
        temperature: 温度参数（控制随机性）
        top_p: nucleus采样参数
        device: 设备
    Returns:
        生成的文本
    """
    # 编码输入
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.int32).to(device)
    
    end_token_id = tokenizer.encode("<|endoftext|>")[0]
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 获取logits
            logits = model(input_tensor)  # (1, seq_len, vocab_size)
            
            # 取最后一个位置的logits
            last_logits = logits[0, -1, :] / temperature  # (vocab_size,)
            
            # 计算概率
            probs = softmax(last_logits.unsqueeze(0), dim=-1)  # (1, vocab_size)
            
            # nucleus采样
            probs = nucleus_sampling(probs, top_p)
            
            # 采样下一个token
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
            
            # 拼接到序列
            input_tensor = torch.cat([input_tensor, next_token.to(torch.int32)], dim=1)
            
            # 检查是否生成了结束token
            if next_token.item() == end_token_id:
                break
    
    # 解码生成的序列
    generated_ids = input_tensor[0].cpu().numpy()
    generated_text = tokenizer.decode(generated_ids, end_token_id=end_token_id)
    
    return generated_text


if __name__ == "__main__":
    # 配置
    model_config = {
        "vocab_size": 10000,
        "context_length": 256,
        "num_layers": 4,
        "num_heads": 16,
        "d_model": 512,
        "d_ff": 1344,
        "rope_theta": 10000,
    }
    
    # 推理参数
    generation_config = {
        "max_new_tokens": 128,
        "temperature": 1.0,      # 1.0 = 标准采样，< 1.0 更确定，> 1.0 更随机
        "top_p": 0.9,            # nucleus采样阈值
    }
    
    # 测试提示
    test_prompts = [
        "Once upon a time,",
        "Tom and Lily are best friends.",
        "The cat was very",
        "In a big forest,",
    ]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载分词器
    print("\n=== 加载分词器 ===")
    tokenizer = BPETokenizer.from_files(
        vocab_filepath="./output/TinyStories_train_10000_token_vocab.bin",
        merges_filepath="./output/TinyStories_train_10000_merges.bin",
        special_tokens=["<|endoftext|>"]
    )
    print("分词器加载成功")
    
    # 初始化模型
    print("\n=== 初始化模型 ===")
    model = TransformerLM(
        vocab_size=model_config["vocab_size"],
        context_length=model_config["context_length"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        d_model=model_config["d_model"],
        d_ff=model_config["d_ff"],
        rope_theta=model_config["rope_theta"],
        device=device
    )
    
    # 加载训练好的权重
    print("\n=== 加载模型权重 ===")
    checkpoint_path = "./data/model/final_model_v0.pt"
    load_checkpoint(
        src=checkpoint_path,
        model=model,
        optimizer=None
    )
    print(f"模型权重加载成功: {checkpoint_path}")
    
    model.to(device)
    
    # 生成文本
    print("\n" + "="*60)
    print("开始生成文本")
    print("="*60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- 测试 {i}/{len(test_prompts)} ---")
        print(f"提示: {prompt}")
        
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=generation_config["max_new_tokens"],
            temperature=generation_config["temperature"],
            top_p=generation_config["top_p"],
            device=device
        )
        
        print(f"生成: {generated}")
        print()
    
    print("="*60)
    print("测试完成！")
    print("="*60)
    
    # 交互式生成
    print("\n您可以尝试自定义提示（输入 'quit' 退出）:")
    while True:
        user_prompt = input("\n请输入提示文本: ")
        if user_prompt.lower() == 'quit':
            break
        
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=user_prompt,
            max_new_tokens=generation_config["max_new_tokens"],
            temperature=generation_config["temperature"],
            top_p=generation_config["top_p"],
            device=device
        )
        
        print(f"生成: {generated}")
