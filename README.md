# ChordEdit 图像编辑项目

本项目实现了 **ChordEdit: One-Step Low-Energy Transport for Image Editing** (CVPR 2026) 的核心算法，集成了 SD-Turbo 与 InstaFlow 作为基础一步式生成模型，并提供了基于 EditBench 数据集的完整评估流程。

## 项目结构

```
.
├── EditBench/              # 图像编辑评估数据集
├── SD-Turbo/               # Stability AI SD-Turbo 模型权重
├── InstaFlow/              # InstaFlow 模型权重
├── chordedit/              # ChordEdit 核心代码
│   ├── __init__.py
│   ├── chordedit_core.py   # 核心算法 (CCF + Proximal Refinement)
│   ├── model_wrappers.py   # SD-Turbo / InstaFlow 封装
│   ├── pipeline.py         # 面向用户的高阶编辑流程
│   └── utils.py            # 评估指标与工具函数
├── scripts/
│   ├── download_models.py  # 模型与数据集下载脚本
│   ├── demo.py             # 单图编辑演示
│   └── evaluate_editbench.py # EditBench 批量评估
├── outputs/                # 编辑结果输出目录
├── requirements.txt        # Python 依赖
└── README.md               # 项目说明
```

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载模型与数据集

由于模型权重文件较大 (SD-Turbo 约 4GB，InstaFlow 约 4GB)，请运行下载脚本：

```bash
python scripts/download_models.py
```

或手动从 HuggingFace 下载：
- **SD-Turbo**: `stabilityai/sd-turbo`
- **InstaFlow**: `XCLiu/instaflow_0_9B_from_sd_1_5`
- **EditBench/PIE-Bench**: `google/pie-bench`

## 核心算法说明

### Chord Control Field (CCF)

论文核心创新在于通过动态最优传输理论，将高能量的瞬时漂移差转化为稳定的低能量控制场：

```
û_t(x_t) = (t * R(x_t, t-δ) + δ * R(x_t, t)) / (t + δ)
```

其中 `R(x_t, t) = v(x_t, t, c_tar) - v(x_t, t, c_src)` 为条件速度场差。

### Proximal Refinement

在单步传输后，通过重参数化加噪与目标条件前向步，补充高频细节：

```
x_t*   = (1-t) * x_tar* + t * ε
x_tar** = x_t* + t * v(x_t*, t, c_tar)
```

### 超参数 (论文默认值)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `t` | 编辑时间步 | 0.9 |
| `delta` | 和弦窗口 | 0.15 |
| `lambda_scale` | 步长缩放 | 1.0 |
| `use_proximal` | 启用近端精修 | True |

## 快速开始

### 单图编辑演示

```bash
python scripts/demo.py \
    --image path/to/image.jpg \
    --src_prompt "a photo of a dog" \
    --tar_prompt "a photo of a dog wearing sunglasses" \
    --model_path SD-Turbo \
    --output outputs/result.png
```

### Python API

```python
from chordedit.pipeline import ChordEditPipeline
from PIL import Image

pipe = ChordEditPipeline(
    model_path="SD-Turbo",
    model_type="sd-turbo",
    device="cuda",
    t=0.9,
    delta=0.15,
    use_proximal=True,
)

src = Image.open("photo.jpg")
edited = pipe.edit(
    src_image=src,
    src_prompt="a red car",
    tar_prompt="a blue car",
)
edited.save("edited.png")
```

### EditBench 评估

```bash
python scripts/evaluate_editbench.py \
    --model_path SD-Turbo \
    --model_type sd-turbo \
    --dataset_dir EditBench \
    --output_dir outputs/eval \
    --t 0.9 \
    --delta 0.15 \
    --use_proximal \
    --save_images
```

评估完成后，`outputs/eval/evaluation_results.json` 将包含：
- **SSIM**: 结构相似度 (越高表示背景保持越好)
- **PSNR**: 峰值信噪比
- **LPIPS**: 感知距离 (越低越好)
- **Background Consistency**: 非编辑区域一致性

## 实验验证要点

为验证 ChordEdit 对一步式模型不稳定性的改善，建议进行以下对比实验：

1. **Baseline 对比**: 与 Naive Single-Step Editing (简单漂移差单步编辑) 对比
2. **消融实验**:
   - `delta=0`: 退化为简单漂移差，验证和弦窗口的作用
   - `use_proximal=False`: 禁用近端精修，验证精修阶段贡献
   - 不同 `t` 值: 分析编辑时间步对结果的影响
3. **模型泛化**: 分别在 SD-Turbo 与 InstaFlow 上测试，验证模型无关性
4. **编辑类型**: 覆盖风格迁移、对象替换、属性修改、场景变换等任务

## 技术细节

### 模型封装 (`model_wrappers.py`)

- `SDTurboWrapper`: 针对 SD-Turbo (Rectified Flow 参数化) 的 velocity 接口封装
- `InstaFlowWrapper`: 针对 InstaFlow 的 velocity 接口封装
- 统一接口: `encode_prompt`, `encode_image`, `decode_latents`, `velocity`, `drift_difference`

### 稳定性优化策略

1. **时间窗口平滑**: 通过 `t` 与 `t-delta` 两点的加权平均，抑制高频噪声
2. **低能量传输**: CCF 的方差低于瞬时漂移场，降低单步大跨度积分的误差累积
3. **近端对齐**: 在噪声空间重新注入目标语义，修复单步传输可能丢失的细节
4. **免训练 / 免反演**: 无需额外训练编辑网络或精确的 DDIM Inversion

## 引用

若本项目对您的研究有所帮助，请引用原始论文：

```bibtex
@inproceedings{lu2026chordedit,
  title={ChordEdit: One-Step Low-Energy Transport for Image Editing},
  author={Lu, Liangsi and Chen, Xuhang and Guo, Minzhe and Li, Shichu and Wang, Jingchao and Shi, Yang},
  booktitle={CVPR},
  year={2026}
}
```

## 许可

本项目代码仅供学术研究使用。模型权重遵循各自原始许可协议 (SD-Turbo: Stability AI 许可，InstaFlow: 对应项目许可)。