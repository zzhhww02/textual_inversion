import math
import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm.auto import tqdm
from utils.helpers import freeze_params


logger = get_logger(__name__)


def save_progress(text_encoder, placeholder_token_id, accelerator, save_path):
    """保存训练得到的嵌入向量"""
    logger.info("保存嵌入向量...")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
    learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)


def train(text_encoder, vae, unet, tokenizer, train_dataset, noise_scheduler,
          placeholder_token_id, hyperparameters, placeholder_token):
    """训练主函数"""
    # 初始化Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=hyperparameters["gradient_accumulation_steps"],
        mixed_precision=hyperparameters["mixed_precision"]
    )

    # 配置梯度检查点
    if hyperparameters["gradient_checkpointing"]:
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    # 创建数据加载器
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hyperparameters["train_batch_size"],
        shuffle=True
    )

    # 调整学习率
    if hyperparameters["scale_lr"]:
        learning_rate = (
            hyperparameters["learning_rate"]
            * hyperparameters["gradient_accumulation_steps"]
            * hyperparameters["train_batch_size"]
            * accelerator.num_processes
        )
    else:
        learning_rate = hyperparameters["learning_rate"]

    # 优化器（仅优化文本编码器的嵌入层）
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),
        lr=learning_rate
    )

    # 准备训练组件
    text_encoder, optimizer, train_dataloader = accelerator.prepare(
        text_encoder, optimizer, train_dataloader
    )

    # 设备和数据类型配置
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 移动VAE和UNet到设备
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.eval()  # VAE不训练，固定为评估模式
    unet.train()  # UNet用于计算梯度

    # 计算训练轮次
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / hyperparameters["gradient_accumulation_steps"])
    num_train_epochs = math.ceil(hyperparameters["max_train_steps"] / num_update_steps_per_epoch)

    # 训练日志
    total_batch_size = (
        hyperparameters["train_batch_size"]
        * accelerator.num_processes
        * hyperparameters["gradient_accumulation_steps"]
    )
    logger.info("***** 开始训练 *****")
    logger.info(f"  训练样本数 = {len(train_dataset)}")
    logger.info(f"  单设备batch size = {hyperparameters['train_batch_size']}")
    logger.info(f"  总batch size（含并行） = {total_batch_size}")
    logger.info(f"  梯度累积步数 = {hyperparameters['gradient_accumulation_steps']}")
    logger.info(f"  总优化步数 = {hyperparameters['max_train_steps']}")

    # 进度条
    progress_bar = tqdm(range(hyperparameters["max_train_steps"]), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("训练步数")
    global_step = 0

    # 开始训练
    for epoch in range(num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder):
                # 图片转 latent
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * 0.18215  # VAE缩放因子

                # 生成噪声
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps, (bsz,),
                    device=latents.device
                ).long()

                # 添加噪声到latent
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 文本编码
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # 预测噪声
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states.to(weight_dtype)).sample

                # 计算损失
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"不支持的预测类型: {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(noise_pred, target, reduction="none").mean([1, 2, 3]).mean()
                accelerator.backward(loss)

                # 优化器步骤
                optimizer.step()
                optimizer.zero_grad()

            # 进度更新
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # 保存中间结果
                if global_step % hyperparameters["save_steps"] == 0:
                    os.makedirs(hyperparameters["output_dir"], exist_ok=True)
                    save_path = os.path.join(hyperparameters["output_dir"], f"learned_embeds-step-{global_step}.bin")
                    save_progress(text_encoder, placeholder_token_id, accelerator, save_path)

            # 达到最大步数则停止
            if global_step >= hyperparameters["max_train_steps"]:
                break
        accelerator.wait_for_everyone()

    # 训练结束保存最终结果
    final_save_path = os.path.join(hyperparameters["output_dir"], "learned_embeds-final.bin")
    save_progress(text_encoder, placeholder_token_id, accelerator, final_save_path)
    logger.info(f"训练完成！最终嵌入向量保存至: {final_save_path}")