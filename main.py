#%%
import os
from PIL import Image
import torch
from huggingface_hub import notebook_login
from configs.train_config import (
    images_path, what_to_teach, placeholder_token, initializer_token,
    hyperparameters,embeds_path, lora_configs
)
from utils.helpers import image_grid, check_image_path
from data.dataset import TextualInversionDataset
from models.model_loader import (
    load_tokenizer, check_initializer_token, load_models, load_scheduler
)
from trainer.trainer import train
from trainer.lora_train import train_lora_on_trained_embedding
#%%
#登录HuggingFace
notebook_login()
#%%
# 加载并显示训练图片
images = []
for f in os.listdir(images_path):
    try:
        img = Image.open(os.path.join(images_path, f)).resize((512, 512))
        images.append(img)
    except:
        print(f"警告：{f} 不是有效图片，已跳过")
if images:
    grid = image_grid(images, 1, len(images))
    grid.show()  # 显示图片网格（本地运行时）
else:
    raise ValueError("图片目录中未找到有效图片")
#%%
# 加载tokenizer
tokenizer = load_tokenizer(placeholder_token)

# 检查初始化token
initializer_token_id = check_initializer_token(tokenizer, initializer_token)
placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

# 加载模型
text_encoder, vae, unet = load_models()

# 调整文本编码器嵌入层大小（添加新token的嵌入）
text_encoder.resize_token_embeddings(len(tokenizer))
token_embeds = text_encoder.get_input_embeddings().weight.data
token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]  # 初始化新token嵌入
#%%
# 冻结不需要训练的参数
from utils.helpers import freeze_params
freeze_params(vae.parameters())
freeze_params(unet.parameters())
params_to_freeze = [
    *text_encoder.text_model.encoder.parameters(),
    *text_encoder.text_model.final_layer_norm.parameters(),
    *text_encoder.text_model.embeddings.position_embedding.parameters()
]
freeze_params(params_to_freeze)

# 创建数据集
train_dataset = TextualInversionDataset(
    data_root=images_path,
    tokenizer=tokenizer,
    size=vae.sample_size,
    placeholder_token=placeholder_token,
    repeats=100,
    learnable_property=what_to_teach,
    center_crop=False,
    set="train",
)

# 加载调度器
noise_scheduler = load_scheduler()
#%%
#开始训练
train(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        noise_scheduler=noise_scheduler,
        placeholder_token_id=placeholder_token_id,
        hyperparameters=hyperparameters,
        placeholder_token=placeholder_token
    )
#%%
#lora训练
train_lora_on_trained_embedding(
    output_dir=hyperparameters["output_dir"],
    train_dataset=train_dataset,
    embeds_path=embeds_path,
    lora_configs=lora_configs,
    text_encoder=text_encoder,
    vae=vae,
    unet=unet,
    tokenizer=tokenizer,
    noise_scheduler=noise_scheduler
    )