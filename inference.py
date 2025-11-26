#%%
from huggingface_hub import notebook_login
import os
import torch
import PIL
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from configs.train_config import pretrained_model_name_or_path,embeds_path
from utils.helpers import image_grid
#%%
repo_id_embeds="embedding-output/learned_embeds.bin"
#%%
#设定pipe
pipe = StableDiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path,
    torch_dtype=torch.float16
).to("cuda")
pipe.load_textual_inversion(repo_id_embeds)
#%%
#可选lora
lora_repo_id = "embedding-output/pytorch_lora_weights.safetensors"
pipe.load_lora_weights(lora_repo_id, weight_name="pytorch_lora_weights.safetensors")
#%%
#查看placeholder_token
learned_embeds = torch.load("embedding-output/learned_embeds.bin", map_location="cpu")
placeholder_token = list(learned_embeds.keys())[0]
print(placeholder_token)
#%%
prompt = "a dog in the style of <van-gogh>"

num_samples = 2
num_rows = 2

all_images = []
for _ in range(num_rows):
    images = pipe(prompt, num_images_per_prompt=num_samples, num_inference_steps=50, guidance_scale=7.5).images
    all_images.extend(images)

grid = image_grid(all_images, num_samples, num_rows)
grid