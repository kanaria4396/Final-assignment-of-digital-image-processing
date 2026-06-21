# ChordEdit 模型权重下载指南

本项目包含 ChordEdit 一步式图像编辑算法的完整代码实现，但**模型权重文件因体积较大未包含在 Git 仓库中**。

## 必需模型文件

### 1. SD-Turbo（必需）
- **模型大小**: 约 4GB
- **下载地址**: [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo)
- **用途**: 基础一步式图像生成模型
- **下载方式**:
  ```bash
  # 方法1: 使用项目脚本（推荐国内用户）
  python scripts/download_models.py
  
  # 方法2: 使用 HF-Mirror 镜像加速
  HF_ENDPOINT=https://hf-mirror.com python scripts/download_models.py
  
  # 方法3: 手动下载
  huggingface-cli download stabilityai/sd-turbo --local-dir SD-Turbo
  ```

### 2. InstaFlow（可选）
- **模型大小**: 约 4GB
- **下载地址**: [XCLiu/instaflow_0_9B_from_sd_1_5](https://huggingface.co/XCLiu/instaflow_0_9B_from_sd_1_5)
- **用途**: 备选基础模型，用于对比实验
- **下载方式**: 与 SD-Turbo 相同

### 3. EditBench / PIE-Bench 数据集（评估必需）
- **数据集大小**: 约 200MB
- **下载地址**: [google/pie-bench](https://huggingface.co/datasets/google/pie-bench)
- **用途**: 图像编辑任务评估基准
- **下载方式**:
  ```bash
  python scripts/download_models.py
  ```

## 快速下载脚本

项目根目录下的 `scripts/download_models.py` 支持自动下载所有模型和数据：

```bash
# 安装依赖
pip install -r requirements.txt

# 下载所有模型（使用 HF-Mirror 镜像加速）
HF_ENDPOINT=https://hf-mirror.com python scripts/download_models.py
```

## 手动下载说明

如果自动下载失败，可以手动从 HuggingFace 下载：

1. **访问模型页面**: https://huggingface.co/stabilityai/sd-turbo
2. **下载模型文件**: 点击 "Files" 标签，下载所有 .safetensors 文件
3. **放置位置**: 将下载的文件放入对应的 `SD-Turbo/` 或 `InstaFlow/` 文件夹
4. **注意**: 请保留原有的 `config.json` 和 `model_index.json` 配置文件

## 验证模型完整性

下载完成后，可以运行以下命令验证：

```bash
# 检查模型文件是否存在
ls SD-Turbo/*.safetensors

# 运行演示脚本测试
python scripts/demo.py --help
```

## 国内镜像源

如果 HuggingFace 访问困难，可以使用以下镜像：

- **HF-Mirror**: https://hf-mirror.com
- 设置方法: `export HF_ENDPOINT=https://hf-mirror.com`

## 模型文件说明

下载的模型文件应包含：

### SD-Turbo/
```
SD-Turbo/
├── model_index.json          # 模型索引
├── unet/
│   └── *.safetensors         # UNet 权重（主要文件，约 3.5GB）
├── vae/
│   └── *.safetensors         # VAE 权重
├── text_encoder/
│   └── *.safetensors         # 文本编码器权重
└── scheduler/
    └── scheduler_config.json # 调度器配置
```

### EditBench/
```
EditBench/
├── images/                   # 编辑源图像
├── prompts/                  # 编辑提示词
└── metadata.json             # 数据集元信息
```

## 技术支持

如果下载过程中遇到问题，请：

1. 检查网络连接
2. 确认磁盘空间充足（至少 10GB）
3. 尝试使用代理或 VPN
4. 参考 [HuggingFace 官方文档](https://huggingface.co/docs/huggingface_hub)

## 注意事项

- 模型权重文件受各自原始许可协议保护
- 请勿将模型文件上传至公共仓库
- 项目代码遵循 MIT 许可
