import math
import os
from typing import IO, BinaryIO, Callable, Iterable, Iterator, Optional
import torch
from torch import nn
import numpy.typing as npt
import numpy as np
from einops import einsum, rearrange


def cross_entropy(logits_i: torch.Tensor, target_i: torch.Tensor) -> torch.Tensor:
    """
    计算单个样本中单个词的交叉熵

    Args:
        logits_i: [1:x_{i}]计算出的未归一化logits向量 (batch_size, ..., vocab_size)
        target_i: 表示logits中第几个是正确答案 (batch_size, ...)

    Returns:
        torch.Tensor: 平均交叉熵
    """
    # 原式：-log{ softmax(logits_i)[target_i] } 
    # 拆开softmax并化简：-logits[target_i] + log(sum(exp(logits_i)))
    logits_i_reshaped = rearrange(logits_i, "b ... v -> (b ...) v")
    target_i_reshaped = rearrange(target_i, "b ... -> (b ...)")

    logits_i_stable = logits_i_reshaped - logits_i_reshaped.max(dim=-1, keepdim=True).values

    targets_logit = logits_i_stable.gather(1, target_i_reshaped.unsqueeze(1)).squeeze(1)
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_i_stable), dim=-1))
    loss = -targets_logit + log_sum_exp
    return loss.mean()


class SGDoptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0 :
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr
        }
        super().__init__(params, defaults=defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss


class AdamWOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8):
        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 1
                    state["exp_avg"] = torch.zeros_like(param.data)
                    state["exp_avg_sq"] = torch.zeros_like(param.data)

                step = state["step"]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = betas
                grad = param.grad.data
                state["exp_avg"] = beta1 * exp_avg + (1 - beta1) * grad
                state["exp_avg_sq"] = beta2 * exp_avg_sq + (1 - beta2) * grad ** 2
                lr_t = lr * math.sqrt(1 - beta2 ** step) / (1 - beta1 ** step)
                param.data -= lr_t * state["exp_avg"] / (torch.sqrt(state["exp_avg_sq"]) + eps)
                param.data -= lr * weight_decay * param.data
                state["step"] += 1
     
        return loss


def learning_rate_cosine_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
) -> float:
    """
    计算余弦退火学习率,使得在训练前中后期动态调整学习率
    （优化器是针对某个参数的大小特调学习率，调度器则是全局迭代进度调整学习率）

    Args:
        it: 当前迭代次数
        max_learning_rate: 最大学习率
        min_learning_rate: 最小学习率
        warmup_iters: 预热结束时迭代次数
        cosine_cycle_iters: 余弦周期结束时迭代次数

    Returns:
        float: 当前学习率
    """
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif it < cosine_cycle_iters:
        cos_percent = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (
            1 + math.cos(math.pi * cos_percent)
        )
    else:
        return min_learning_rate


def clip_grad(params: Iterable[torch.nn.Parameter], max_norm: float = 1.0, eps: float = 1e-6):
    """
    梯度裁剪，防止梯度爆炸

    Args:
        params: 模型参数列表
        max_norm: 最大梯度范数
    """
    total_norm = 0.0
    # 计算参数梯度的L2范数
    for param in params:
        if param.grad is not None:
            total_norm += torch.sum(param.grad ** 2)
    total_norm = total_norm ** 0.5

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        # 如果总范数超过最大范数，则进行裁剪
        for param in params:
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)


def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将一个token序列数据集，通过滑动窗口法，采样batch条长度为context_length的token序列数据
    （每一条数据不一定要在<|endoftext|>截断）

    Args:
        dataset: 输入的token序列数据集，形状为一个一维数组
        batch_size: 每个batch的大小，相当于要采样的样本数量
        context_length: 上下文长度，即batch中每条数据的长度

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 返回一个元组，包含两个Tensor
            - 输入序列Tensor，形状为(batch_size, context_length)
            - 目标序列Tensor，形状为(batch_size, context_length)，每个输入序列的下一个token作为目标
    """
    dataset_len = dataset.shape[0]
    if dataset_len < context_length:
        raise ValueError(f"Dataset length {dataset_len} is less than the context length {context_length}.")

    starts = np.random.randint(0, dataset_len - context_length, size=batch_size)
    inputs = np.stack([dataset[start:start + context_length] for start in starts], dtype=np.int64)
    targets = np.stack([dataset[start + 1:start + context_length + 1] for start in starts], dtype=np.int64)

    return (
        torch.from_numpy(inputs).to(device),
        torch.from_numpy(targets).to(device)
    )


def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | BinaryIO | IO[bytes]
):
    """
    将模型参数、优化器状态和迭代次数储存在一个字典中，存入指定的文件或文件对象中。
    """
    return torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }, out)


def load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer
) -> int:
    """
    从指定的文件或文件对象中加载模型参数、优化器状态，并返回迭代次数。
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


def evaluate_model(model: nn.Module, dataset, device, batch_size, context_length, num_batches=10):
    """
    在验证集上评估模型性能

    Args:
        model: 要评估的模型
        dataset: 验证数据集，一维token序列
        device: 计算设备
        batch_size: 批次大小
        context_length: 上下文长度
        num_batches: 要评估的批次数量

    Returns:
        float: 验证集上的平均损失
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets = get_batch(
                dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device
            )
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            total_loss += loss.item()

    model.train()
    return total_loss / num_batches


def compute_grad_norm(parameters: Iterator[nn.parameter]):
    """
    计算参数梯度的 L2 范数

    参数:
        parameters: 模型参数
  
    返回:
        梯度的 L2 范数
    """
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10,10)))
    opt = AdamWOptimizer([weights], lr=1e1, weight_decay=0.01)
    for t in range(100):
        opt.zero_grad()
        loss = (weights ** 2).mean()
        print(f"t={t}, loss={loss.cpu().item()}")
        loss.backward()
        opt.step()
