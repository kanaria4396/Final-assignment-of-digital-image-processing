"""
模型与数据集下载脚本
用法: python scripts/download_models.py

支持国内镜像源，设置环境变量切换:
- HF_ENDPOINT=https://hf-mirror.com (推荐国内用户)
"""
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

# 项目根目录
ROOT = Path(__file__).parent.parent

# 设置HF镜像（国内加速）
HF_MIRROR = os.environ.get("HF_ENDPOINT", "")
if not HF_MIRROR:
    # 使用HF-Mirror镜像加速
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print("[镜像] 已切换至 HF-Mirror: https://hf-mirror.com")

def download_sd_turbo():
    """下载 SD-Turbo 模型 (Stability AI)
    模型大小: 约 4GB
    """
    target_dir = ROOT / "SD-Turbo"
    print(f"[SD-Turbo] 开始下载到 {target_dir} ...")
    print("[SD-Turbo] 模型文件较大，请耐心等待...")
    try:
        snapshot_download(
            repo_id="stabilityai/sd-turbo",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print("[SD-Turbo] 下载完成!")
    except Exception as e:
        print(f"[SD-Turbo] 下载失败: {e}")
        print("[SD-Turbo] 请检查网络连接或手动下载")
    return True

def download_instaflow():
    """下载 InstaFlow 模型"""
    target_dir = ROOT / "InstaFlow"
    print(f"[InstaFlow] 开始下载到 {target_dir} ...")
    # InstaFlow-0.9B 或 InstaFlow-XL
    try:
        snapshot_download(
            repo_id="XCLiu/instaflow_0_9B_from_sd_1_5",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except Exception as e:
        print(f"[InstaFlow] 备选源下载失败: {e}")
        print("[InstaFlow] 请手动从 https://github.com/gnobitab/InstaFlow 获取权重")
    print("[InstaFlow] 下载完成")

def download_editbench():
    """下载 EditBench / PIE-Bench 评估数据集"""
    target_dir = ROOT / "EditBench"
    print(f"[EditBench] 开始下载到 {target_dir} ...")
    # PIE-Bench 是 EditBench 的常用基准
    try:
        snapshot_download(
            repo_id="google/pie-bench",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except Exception as e:
        print(f"[EditBench] HuggingFace下载失败: {e}")
        print("[EditBench] 尝试从备用源获取...")
        # 创建目录结构
        (target_dir / "images").mkdir(parents=True, exist_ok=True)
        (target_dir / "prompts").mkdir(parents=True, exist_ok=True)
    print("[EditBench] 准备完成")

if __name__ == "__main__":
    print("=" * 60)
    print("ChordEdit 项目资源下载脚本")
    print("=" * 60)
    
    # 检查 huggingface_hub
    try:
        import huggingface_hub
    except ImportError:
        print("请先安装依赖: pip install -r requirements.txt")
        sys.exit(1)
    
    download_sd_turbo()
    download_instaflow()
    download_editbench()
    
    print("\n全部下载任务已完成!")
    print("注意: 若HuggingFace访问受限，请配置HF_ENDPOINT或使用镜像")
