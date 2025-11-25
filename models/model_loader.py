from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from configs.train_config import pretrained_model_name_or_path


def load_tokenizer(placeholder_token):
    """加载CLIP tokenizer并添加占位符token"""
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(f"tokenizer已包含{placeholder_token}，请更换占位符")
    return tokenizer


def check_initializer_token(tokenizer, initializer_token):
    """检查初始化token是否为单个token"""
    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    if len(token_ids) > 1:
        raise ValueError("初始化token必须是单个词（如'toy'，不能是'cat toy'）")
    return token_ids[0]


def load_models():
    """加载Stable Diffusion的文本编码器、VAE、UNet"""
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet"
    )
    return text_encoder, vae, unet


def load_scheduler():
    """加载噪声调度器"""
    return DDPMScheduler.from_config(pretrained_model_name_or_path, subfolder="scheduler")