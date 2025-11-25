import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# 训练提示模板
imagenet_templates_small = [
    "a photo of a {}", "a rendering of a {}", "a cropped photo of the {}",
    "the photo of a {}", "a photo of a clean {}", "a photo of a dirty {}",
    "a dark photo of the {}", "a photo of my {}", "a photo of the cool {}",
    "a close-up photo of a {}", "a bright photo of the {}", "a cropped photo of a {}",
    "a photo of the {}", "a good photo of the {}", "a photo of one {}",
    "a close-up photo of the {}", "a rendition of the {}", "a photo of the clean {}",
    "a rendition of a {}", "a photo of a nice {}", "a good photo of a {}",
    "a photo of the nice {}", "a photo of the small {}", "a photo of the weird {}",
    "a photo of the large {}", "a photo of a cool {}", "a photo of a small {}"
]

imagenet_style_templates_small = [
    "a painting in the style of {}", "a rendering in the style of {}",
    "a cropped painting in the style of {}", "the painting in the style of {}",
    "a clean painting in the style of {}", "a dirty painting in the style of {}",
    "a dark painting in the style of {}", "a picture in the style of {}",
    "a cool painting in the style of {}", "a close-up painting in the style of {}",
    "a bright painting in the style of {}", "a cropped painting in the style of {}",
    "a good painting in the style of {}", "a close-up painting in the style of {}",
    "a rendition in the style of {}", "a nice painting in the style of {}",
    "a small painting in the style of {}", "a weird painting in the style of {}",
    "a large painting in the style of {}"
]


class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, f) for f in os.listdir(self.data_root)]
        self.num_images = len(self.image_paths)
        self._length = self.num_images * repeats if set == "train" else self.num_images

        self.interpolation = {
            "linear": Image.LINEAR,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        # 加载图片
        image = Image.open(self.image_paths[i % self.num_images])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # 生成提示文本
        text = random.choice(self.templates).format(self.placeholder_token)
        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # 图片预处理
        img = np.array(image).astype(np.uint8)
        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img).resize((self.size, self.size), resample=self.interpolation)
        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)

        return example