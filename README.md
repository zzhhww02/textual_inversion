本项目用于实现文本反转（Textual Inversion）训练，通过少量图像学习特定概念的嵌入向量，以下是项目结构及各部分功能说明：
.idea/：IDE（如 PyCharm）的项目配置文件夹，包含模块信息、Python 环境、版本控制等配置文件。
requirements.txt：项目依赖清单，列出运行所需的 Python 库及版本（如 diffusers、transformers、torch 等）。
configs/train_config.py：训练配置文件，定义预训练模型路径、数据目录、训练超参数（学习率、迭代步数等）。
data/dataset.py：数据集处理模块，实现 TextualInversionDataset 类，用于加载图像、生成提示文本及图像预处理。
models/model_loader.py：模型加载模块，负责加载 CLIP tokenizer、文本编码器、VAE、UNet 及噪声调度器。
trainer/trainer.py：训练主模块，包含训练循环、损失计算、梯度优化及嵌入向量保存等核心逻辑。
utils/helpers.py：工具函数模块，提供图像网格生成、路径检查、参数冻结等辅助功能。
.gitignore：Git 版本控制忽略文件配置，指定无需跟踪的文件 / 文件夹（如 IDE 临时文件）。# Textual Inversion 项目结构说明
本项目用于实现文本反转（Textual Inversion）训练，通过少量图像学习特定概念的嵌入向量，以下是项目结构及各部分功能说明：
.idea/：IDE（如 PyCharm）的项目配置文件夹，包含模块信息、Python 环境、版本控制等配置文件。
requirements.txt：项目依赖清单，列出运行所需的 Python 库及版本（如 diffusers、transformers、torch 等）。
configs/train_config.py：训练配置文件，定义预训练模型路径、数据目录、训练超参数（学习率、迭代步数等）。
data/dataset.py：数据集处理模块，实现 TextualInversionDataset 类，用于加载图像、生成提示文本及图像预处理。
models/model_loader.py：模型加载模块，负责加载 CLIP tokenizer、文本编码器、VAE、UNet 及噪声调度器。
trainer/trainer.py：训练主模块，包含训练循环、损失计算、梯度优化及嵌入向量保存等核心逻辑。
utils/helpers.py：工具函数模块，提供图像网格生成、路径检查、参数冻结等辅助功能。
.gitignore：Git 版本控制忽略文件配置，指定无需跟踪的文件 / 文件夹（如 IDE 临时文件）。
