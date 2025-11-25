from peft import LoraConfig, get_peft_model
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm.auto import tqdm
import math
import os
def train_lora_on_trained_embedding(
    output_dir,          # LoRA 输出目录
    train_dataset,       # 训练数据集
    embeds_path,
    lora_configs,
    text_encoder,
    vae,
    unet,
    tokenizer,
    noise_scheduler
):
    train_batch_size = lora_configs["train_batch_size"]
    gradient_accumulation_steps = lora_configs["gradient_accumulation_steps"]
    learning_rate = lora_configs["learning_rate"]
    max_train_steps = lora_configs["max_train_steps"]
    save_steps = lora_configs["save_steps"]
    mixed_precision = lora_configs["mixed_precision"]
    lora_rank=lora_configs["lora_rank"]
    lora_alpha=lora_configs["lora_alpha"]
    lora_dropout=lora_configs["lora_dropout"]
    # 然后加载你训练好的 embedding
    learned_embeds = torch.load(embedding_path, map_location="cpu")
    placeholder_token = list(learned_embeds.keys())[0]
    embed_weight = learned_embeds[placeholder_token]

    # 添加 token 并替换 embedding
    tokenizer.add_tokens(placeholder_token)
    text_encoder.resize_token_embeddings(len(tokenizer))
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

    with torch.no_grad():
        text_encoder.get_input_embeddings().weight[placeholder_token_id] = embed_weight

    # 初始化accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    # 配置LoRA
    # 为UNet创建LoRA配置
    unet_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["to_q", "to_k", "to_v"],  # 对于UNet
        lora_dropout=lora_dropout,
        bias="none",
    )

    # 为Text Encoder创建LoRA配置
    text_encoder_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj"],  # 对于Text Encoder
        lora_dropout=lora_dropout,
        bias="none",
    )

    # 分别应用LoRA
    unet = get_peft_model(unet, unet_lora_config)
    text_encoder = get_peft_model(text_encoder, text_encoder_lora_config)

    print("Text Encoder trainable parameters:")
    text_encoder.print_trainable_parameters()
    print("UNet trainable parameters:")
    unet.print_trainable_parameters()

    # 准备数据加载器
    train_dataloader = create_dataloader(train_batch_size)

    # 优化器
    optimizer = torch.optim.AdamW(
        list(text_encoder.parameters()) + list(unet.parameters()),
        lr=learning_rate,
    )

    # 准备训练
    text_encoder, unet, optimizer, train_dataloader = accelerator.prepare(
        text_encoder, unet, optimizer, train_dataloader
    )

    # 设置数据类型
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 设置VAE - 确保使用正确的数据类型
    vae.to(accelerator.device, dtype=weight_dtype)
    vae.eval()

    # 训练循环
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(num_train_epochs):
        text_encoder.train()
        unet.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder, unet):
                # 图像编码到潜在空间 - 确保数据类型一致
                with torch.no_grad():
                    # 确保输入数据与模型数据类型一致
                    pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * 0.18215

                # 采样噪声和时间步 - 确保数据类型一致
                noise = torch.randn_like(latents, dtype=weight_dtype)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device
                ).long()

                # 添加噪声
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 文本编码和噪声预测
                input_ids = batch["input_ids"].to(accelerator.device)
                encoder_hidden_states = text_encoder(input_ids)[0]

                # 确保数据类型一致
                if encoder_hidden_states.dtype != weight_dtype:
                    encoder_hidden_states = encoder_hidden_states.to(weight_dtype)

                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # 计算损失
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # 确保目标数据类型与预测值一致
                if target.dtype != noise_pred.dtype:
                    target = target.to(noise_pred.dtype)

                loss = F.mse_loss(noise_pred, target, reduction="mean")
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % save_steps == 0 and accelerator.is_main_process:
                    # 保存LoRA权重
                    save_lora_weights(text_encoder, unet, output_dir, global_step)

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    # 保存最终LoRA权重
    if accelerator.is_main_process:
        save_lora_weights(text_encoder, unet, output_dir, "final")

    return output_dir

def save_lora_weights(text_encoder, unet, output_dir, step):
    """保存LoRA权重"""
    os.makedirs(output_dir, exist_ok=True)

    # 收集LoRA权重
    lora_state_dict = {}

    for name, param in text_encoder.named_parameters():
        if 'lora' in name and param.requires_grad:
            lora_state_dict[f"text_encoder.{name}"] = param.data.cpu()

    for name, param in unet.named_parameters():
        if 'lora' in name and param.requires_grad:
            lora_state_dict[f"unet.{name}"] = param.data.cpu()

    save_path = os.path.join(output_dir, f"lora_weights_{step}.bin")
    torch.save(lora_state_dict, save_path)
    print(f"Saved LoRA weights to {save_path}")



