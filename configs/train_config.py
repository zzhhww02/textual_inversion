# 模型配置
pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"

# 数据配置
images_path = r"D:\桌面\实验\pic"  # 训练图片目录
what_to_teach = "style"  # "object" 或 "style"
placeholder_token = "<van-gogh>"  # 自定义占位符token
initializer_token = "painting"  # 初始化token
embeds_path="embedding-output/learned_embeds.bin"
# 训练超参数
hyperparameters = {
    "learning_rate": 5e-04,
    "scale_lr": True,
    "max_train_steps": 2000,
    "save_steps": 250,
    "train_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True,
    "mixed_precision": "fp16",
    "seed": 42,
    "output_dir": "embedding-output"
}
lora_configs = {
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-4,
    "max_train_steps": 500,
    "save_steps": 100,
    "mixed_precision": "fp16",
    "lora_rank": 4,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
}