
# Textual Inversion 项目结构说明

本项目用于实现Textual Inversion训练，通过少量图像学习特定概念的嵌入向量，以下是项目核心目录与文件的功能说明：


## 配置与依赖
- `requirements.txt`：项目依赖清单。



## 核心配置
- `configs/train_config.py`：训练参数配置文件，定义预训练模型路径、数据集目录、学习率、训练步数等超参数。


## 数据处理
- `data/dataset.py`：数据集处理模块，实现`TextualInversionDataset`类，负责加载图像、生成提示文本（基于预设模板）及图像预处理（裁剪、缩放、翻转等）。


## 模型加载
- `models/model_loader.py`：模型组件加载工具，包含CLIP分词器（添加占位符token）、文本编码器、VAE、UNet及噪声调度器的加载逻辑。


## 训练逻辑
- `trainer/trainer.py`：训练主模块，实现训练循环、损失计算（MSE损失）、梯度优化及嵌入向量保存等核心功能，支持混合精度训练与梯度累积。
- `trainer/lora_train.py`：lora训练模块。


## 工具函数
- `utils/helpers.py`：辅助功能模块，提供图像网格生成、目录路径检查、模型参数冻结等工具函数。


