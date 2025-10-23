"""
Transformer 模块中的基础神经网络层实现。

本模块包含了 Transformer 架构中的核心组件：
- Linear: 线性变换层
- Embedding: 词嵌入层
- RMSNorm: Root Mean Square 层归一化
"""

import torch
import torch.nn as nn
from math import sqrt
from einops import einsum, reduce, rearrange
from jaxtyping import Float


class Linear(nn.Module):
    """
    自定义的线性变换层，使用截断正态分布初始化权重。
    
    该实现使用 einops 库进行高效的张量操作，支持多维输入。
    """
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        初始化线性层。
        
        Args:
            in_features (int): 输入特征的维度
            out_features (int): 输出特征的维度
            device: 张量所在的设备（CPU 或 CUDA）
            dtype: 张量的数据类型
        """
        super().__init__()
        self.d_in = in_features
        self.d_out = out_features
        # 初始化权重矩阵，形状为 (out_features, in_features)
        self.W = nn.Parameter(torch.empty((out_features, in_features),device=device,dtype=dtype))
        # 使用 Xavier-like 初始化，标准差基于输入和输出维度
        std = sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.W, mean=0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：执行线性变换。
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch, ..., in_features)
                              支持任意中间维度
        
        Returns:
            torch.Tensor: 输出张量，形状为 (batch, ..., out_features)
        """
        return einsum(x, self.W, 'batch ... input, output input -> batch ... output')


class Embedding(nn.Module):
    """
    词嵌入层，将离散的词索引映射到连续的向量空间。
    
    使用学习的权重矩阵存储每个词的嵌入向量。
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """
        初始化嵌入层。
        
        Args:
            num_embeddings (int): 词汇表的大小（不同词的个数）vocab_size
            embedding_dim (int): 每个嵌入向量的维度 d_model
            device: 张量所在的设备（CPU 或 CUDA）
            dtype: 张量的数据类型
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # 初始化嵌入权重矩阵，形状为 (num_embeddings, embedding_dim)
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        # 使用截断正态分布初始化
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：将词索引转换为嵌入向量。
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch, seq_len)
                              每个元素是词汇表中的索引值 [0, num_embeddings)
        
        Returns:
            torch.Tensor: 嵌入向量张量，形状为 (batch, seq_len, embedding_dim)
        """
        batch_size = x.shape[0]
        # 为批处理中的每个样本查找对应的嵌入向量
        return torch.stack([torch.index_select(self.weight, dim=0, index=x[i]) for i in range(batch_size)])


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)。
    
    相比于 LayerNorm，RMSNorm 省略了减去均值的步骤，直接对均方根进行归一化，
    具有更高的计算效率，同时保持了良好的性能。
    
    常用于现代 Transformer 模型中（如 LLaMA）。
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):  
        """
        初始化 RMSNorm 层。
        
        Args:
            d_model (int): 输入特征的维度
            eps (float): 防止除零的极小值，默认为 1e-5
            device: 张量所在的设备（CPU 或 CUDA）
            dtype: 张量的数据类型
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # 初始化可学习的缩放参数（gamma），初值为全 1
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：对输入进行 RMSNorm 归一化。
        
        步骤：
        1. 计算沿着特征维度的均方
        2. 计算均方根 (RMS)
        3. 将输入除以 RMS 进行归一化
        4. 乘以可学习的缩放参数
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch, seq_len, d_model)
                              或任何以 d_model 作为最后维度的形状
        
        Returns:
            torch.Tensor: 归一化后的张量，形状与输入相同，dtype 与输入相同
        """
        # 保存原始数据类型
        in_dtype = x.dtype
        # 转换为 float32 进行计算以提高数值稳定性
        x = x.to(torch.float32)
        # 计算沿最后一维的均方，保持维度以便于广播
        mean_square = torch.mean(x ** 2, dim=-1, keepdim=True)
        # 计算均方根，加上 eps 防止除零
        rms = torch.sqrt(mean_square + self.eps)
        # 归一化并乘以缩放参数
        x = x / rms * self.weight
        # 转换回原始数据类型
        return x.to(in_dtype)


class SwiGLUFeedForward(nn.Module):
    """
    Feedforward neural network using the SWiGLU activation function
    """
    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        """
        Args:
            d_model: 输入特征的维度
            d_ff:SWiGLU中间层维度， 如果为None则等于 8/3*d_model(需要四舍五入)
        """
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            self.d_ff = int(8 / 3 * d_model)
            self.d_ff = (self.d_ff + 63) // 64 * 64
        else:
            self.d_ff = d_ff
        self.weight1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.weight2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.weight3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        w1_x = self.weight1(x)
        w3_x = self.weight3(x)
        silu = w1_x * torch.sigmoid(w1_x)
        swiglu = silu * w3_x
        return self.weight2(swiglu)


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.register_buffer("rope", self._precompute_freqs_cis(), persistent=False)
    
    def _precompute_freqs_cis(self) -> torch.Tensor:
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2, device=self.device)[:(self.d_k // 2)] / self.d_k))
        seq_idx = torch.arange(0, self.max_seq_len, device=self.device)
        freqs = einsum(seq_idx, freqs, "seq, d -> seq d")
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_ = rearrange(x, "... seq (d two) -> ... seq d two", two=2).float()
        x_ = torch.view_as_complex(x_)

        rope_pos = self.rope[token_positions]
        x_out = rearrange(torch.view_as_real(x_ * rope_pos), "... seq d two -> ...seq (d two)", two=2)
        return x_out.to(x.dtype)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    计算缩放点积注意力
    Args:
        Q: 查询张量，形状为(batch, ..., seq_len_q, d_k)
        K: 键张量，形状为(batch, ..., seq_len_k, d_k)
        V: 值张量，形状为(batch, ..., seq_len_k, d_v)
        mask: 可选的掩码张量，形状为(seq_len_q, seq_len_k)
    Returns:
        注意力输出张量，形状为(batch, ..., seq_len_q, d_v)
    """
    scores = einsum(Q, K, 'batch ... q d_k, batch ... k d_k -> batch ... q k')
    d_k = Q.shape[-1]
    scores /= sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('inf'))
    
    attn_weights = softmax(scores, dim=-1)
    output = einsum(attn_weights, V, 'batch ... q k, batch ... k d_v -> batch ... q d_v')
    return output


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.device = device
        self.dtype = dtype

        self.w_qkv = Linear(d_model, self.num_heads * self.d_k * 3, device=device, dtype=dtype)
        self.w_o = Linear(self.num_heads * self.d_k, self.d_model, device=device,dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量，形状为(batch, ..., seq_len, d_model)
        Returns:
            输出张量，形状为(batch, ..., seq_len, d_model)
        """
        seq_len = x.shape[-2]
        QKV = self.w_qkv(x)
        Q, K, V = rearrange(QKV, "... seq_len (three head d_k) -> three ... head seq_len d_k", three=3, head=self.num_heads)
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool)).to(self.device)
        atten = scaled_dot_product_attention(Q, K, V,mask)
        atten = rearrange(atten, "... head seq_len d_k -> ... seq_len (head d_k)")
        return self.w_o(atten)


class MultiheadSelfAttentionWithRoPE(MultiheadSelfAttention):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, device=None, dtype=None):
        super().__init__(d_model, num_heads, device=device, dtype=dtype)
        self.rope = RoPE(theta, self.d_k, max_seq_len, device=device)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量，形状为(batch, ..., seq_len, d_model)
            token_positions: 位置索引，形状为(batch, ..., seq_len)
        Returns:
            输出张量，形状为(batch, ..., seq_len, d_model)
        """
        seq_len = x.shape[-2]
        QKV = self.w_qkv(x)
        Q, K, V = rearrange(QKV, "... seq_len (three head d_k) -> three ... head seq_len d_k", three=3, head=self.num_heads)
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool)).to(self.device)
        
        atten = scaled_dot_product_attention(Q, K, V, mask)
        atten = rearrange()