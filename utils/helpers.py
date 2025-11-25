from PIL import Image
import torch
import os


def image_grid(imgs, rows, cols):
    """生成图片网格"""
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def check_image_path(images_path):
    """检查图片目录是否存在"""
    while not os.path.exists(str(images_path)):
        print('图片目录不存在，请重新输入：')
        images_path = input("")
    return images_path


def freeze_params(params):
    """冻结模型参数"""
    for param in params:
        param.requires_grad = False