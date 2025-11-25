# 模型配置
pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"  # 可替换为其他模型

# 数据配置
images_path = r"D:\桌面\实验\pic"  # 训练图片目录（运行时需指定）
what_to_teach = "object"  # "object" 或 "style"
placeholder_token = "<cat-toy>"  # 自定义占位符token
initializer_token = "toy"  # 初始化token

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