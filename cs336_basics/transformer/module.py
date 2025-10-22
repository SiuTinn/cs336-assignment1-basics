import torch
import torch.nn as nn
from math import sqrt
from einops import einsum, reduce, rearrange
from jaxtyping import Float


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super.__init__()
        self.d_in = in_features
        self.d_out = out_features
        self.W = nn.Parameter(torch.empty((out_features, in_features),device=device,dtype=dtype))
        std = sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.W, mean=0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Arg:
            x: 输入张量，形状为(batch, ..., in_features)
        """
        return einsum(x, self.W, 'batch ... input, output input -> batch ... output')


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为(batch, seq_len)，每个元素是词汇表中的索引
        """
        batch_size = x.shape[0]
        return torch.stack([torch.index_select(self.weight, dim=0, index=x[i]) for i in range(batch_size)])


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):  
        """
        Args:
            d_model:输入特征的维度
            eps:防止除0的极小值
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            输入tensor(batch,seq_len,d_modle)
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_square = torch.mean(x ** 2, dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)
        x = x / rms * self.weight
        return x.to(in_dtype)


