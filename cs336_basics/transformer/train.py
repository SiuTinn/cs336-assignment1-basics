import torch
import numpy as np
import numpy.typing as npt
import wandb
from tqdm import tqdm
from loguru import logger

import cs336_basics.transformer.module as module
import cs336_basics.transformer.train_utils as utils

if __name__ == "__main__":
    logger.add("./output/log/train_v0.log", rotation="1 day", retention="7 days", level="INFO")

    model_config = {
        "vocab_size": 10000,
        "context_length": 256,
        "num_layers": 4,
        "num_heads": 16,
        "d_model": 512,
        "d_ff": 1344,
        "rope_theta": 10000,
    }

    optim_config = {
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "betas": (0.9, 0.999),
        "max_norm": 1.0,
    }

    train_config = {
        "batch_size": 16,
        "total_epochs": 0.5,
        "checkpoint_freq": 2000,
        "log_freq": 10,           # 每隔多少步记录一次日志
        "val_freq": 400,          # 每隔多少步在验证集上评估
        "val_batch_size": 16,     # 验证时的批次大小
        "val_batches": 20,        # 验证时使用的批次数量
    }   

    data_paths = {
        "training_dataset_path": "./data/token/TinyStories_train_10000_token_ids.npy",
        "validation_dataset_path": "./data/token/TinyStories_valid_10000_token_ids.npy",  # 验证集路径
        "checkpoint_load_path": None,  # 模型检查点路径
        "checkpoint_save_format": "./data/model/checkpoint_v0_{}.pt",  # 检查点保存路径格式
        "final_model_path": "./data/model/final_model_v0.pt",  # 最终模型保存路径
    }

    run = wandb.init(
        project="cs336-assignment-1",
        name="train_v1",
        config={
            "model": model_config,
            "optimizer": optim_config,
            "training": train_config,
        }
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("Initing model ...")
    model = module.TransformerLM(
        vocab_size=model_config["vocab_size"],
        context_length=model_config["context_length"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        d_model=model_config["d_model"],
        d_ff=model_config["d_ff"],
        rope_theta=model_config["rope_theta"],
        device=device,
    )
    logger.info("Finished initing model.")

    logger.info("initing optimizer...")
    optimizer = utils.AdamWOptimizer(
        model.parameters(),
        lr=optim_config["lr"],
        weight_decay=optim_config["weight_decay"],
        betas=optim_config["betas"],
    )
    logger.info("Finished initing optimizer")

    start_iter = 1
    if data_paths["checkpoint_load_path"]:
        logger.info(f"开始加载模型检查点:{data_paths['checkpoint_load_path']}")
        start_iter = utils.load_checkpoint(
            data_paths["checkpoint_load_path"],
            model=model,
            optimizer=optimizer
        )
        start_iter += 1
        logger.info(f"模型检查点加载成功，当前迭代次数: {start_iter}")
    else:
        logger.info("没有提供模型检查点，开始从头训练。")
    
    # 加载数据集
    logger.info(f"开始加载数据集，训练集：{data_paths['training_dataset_path']}, 验证集：{data_paths['validation_dataset_path']}")
    training_dataset = np.load(data_paths['training_dataset_path'], mmap_mode='r+') # 使用内存映射
    validation_dataset = None
    if data_paths['validation_dataset_path']:
        validation_dataset = np.load(data_paths['validation_dataset_path'], mmap_mode='r+')
    logger.info("数据集加载完成")

    # 计算训练所需step
    total_tokens = training_dataset.shape[0]
    total_steps = int(train_config["total_epochs"] * total_tokens) // (train_config["batch_size"] * model_config["context_length"])
    logger.info(f"总token数: {total_tokens}, 训练轮数: {train_config['total_epochs']}, batch大小: {train_config['batch_size']}, 上下文长度: {model_config['context_length']}")
    logger.info(f"总训练步数: {total_steps}")

    logger.info("开始训练模型...")
    for step in tqdm(range(start_iter, total_steps + 1), desc="训练进度", unit="step"):
        optimizer.zero_grad()

        lr_now = utils.learning_rate_cosine_schedule(
            it=step,
            max_learning_rate=optim_config["lr"],
            min_learning_rate=optim_config["lr"] * 0.01,
            warmup_iters=int(0.05 * total_steps),
            cosine_cycle_iters=total_steps,
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_now
        
        inputs, targets = utils.get_batch(
            training_dataset,
            batch_size=train_config["batch_size"],
            context_length=model_config["context_length"],
            device=device
        )

        logits = model(inputs)

        loss = utils.cross_entropy(logits, targets)

        loss.backward()

        if step % train_config["log_freq"] == 0:
            grad_norm = utils.compute_grad_norm(model.parameters())

        utils.clip_grad(model.parameters(), max_norm=optim_config["max_norm"])  # 梯度裁剪
        optimizer.step()
         # 日志记录
        if step % train_config["log_freq"] == 0:
            logger.info(f"Step {step}, Loss: {loss.item()}, Grad L2 Norm: {grad_norm}")

            # 使用wandb记录损失和梯度范数
            wandb.log({"train_loss": loss.item(), "lr": lr_now, "grad_l2_norm": grad_norm, "step": step})

# 在验证集上评估模型
        if validation_dataset is not None and step % train_config["val_freq"] == 0:
            logger.info(f"在验证集上评估模型...")
            val_loss = utils.evaluate_model(
                model=model,
                dataset=validation_dataset,
                device=device,
                batch_size=train_config["val_batch_size"],
                context_length=model_config["context_length"],
                num_batches=train_config["val_batches"]
            )
            logger.info(f"验证集损失: {val_loss}")
            wandb.log({"val_loss": val_loss, "step": step})
        
        # 保存检查点
        if step % train_config["checkpoint_freq"] == 0:
            checkpoint_save_path = data_paths["checkpoint_save_format"].format(step)
            logger.info(f"正在保存模型检查点到: {checkpoint_save_path}")
            utils.save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=step,
                out=checkpoint_save_path
            )
            logger.info("模型检查点保存成功。")
    logger.info("模型训练完成。")
    
    # 保存最终模型
    logger.info(f"正在保存最终模型到: {data_paths['final_model_path']}")
    utils.save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=total_steps,
        out=data_paths["final_model_path"],
    )
    logger.info("最终模型保存成功。")
    
    # 关闭wandb
    wandb.finish()