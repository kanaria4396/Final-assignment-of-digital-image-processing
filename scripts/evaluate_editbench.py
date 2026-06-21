"""
EditBench / PIE-Bench 评估脚本
用于验证 ChordEdit 在图像编辑任务上的稳定性与质量

评估指标:
- SSIM / PSNR: 与源图像的结构/像素相似度 (衡量背景保持)
- LPIPS: 感知距离
- Background Consistency: 非编辑区域一致性
- CLIP Directional Similarity: 文本-图像方向一致性 (可选)
"""
import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from PIL import Image

# 将项目根目录加入路径
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from chordedit.pipeline import ChordEditPipeline
from chordedit import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ChordEdit on EditBench")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径 (如 SD-Turbo 或 InstaFlow)")
    parser.add_argument("--model_type", type=str, default="sd-turbo", choices=["sd-turbo", "instaflow"])
    parser.add_argument("--dataset_dir", type=str, default=str(ROOT / "EditBench"), help="数据集目录")
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "outputs" / "eval"), help="输出目录")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--t", type=float, default=0.9, help="ChordEdit 时间步 t")
    parser.add_argument("--delta", type=float, default=0.15, help="ChordEdit 窗口 delta")
    parser.add_argument("--use_proximal", action="store_true", default=True, help="是否使用 Proximal Refinement")
    parser.add_argument("--save_images", action="store_true", default=True, help="是否保存结果图像")
    return parser.parse_args()


def load_editbench_dataset(dataset_dir: Path):
    """
    加载 EditBench / PIE-Bench 数据集
    预期结构:
        EditBench/
            images/
                000.jpg, 001.jpg, ...
            prompts.json:
                [
                    {"image": "000.jpg", "src_prompt": "...", "tar_prompt": "...", "edit_region": "..."},
                    ...
                ]
    返回:
        samples: list of dict
    """
    prompts_file = dataset_dir / "prompts.json"
    images_dir = dataset_dir / "images"

    if not prompts_file.exists():
        # 若不存在 prompts.json，生成一个示例用于测试
        print(f"[警告] 未找到 {prompts_file}，将创建示例测试数据")
        samples = []
        if images_dir.exists():
            for img_path in sorted(images_dir.glob("*.jpg"))[:5]:
                samples.append({
                    "image": img_path.name,
                    "src_prompt": "a photo",
                    "tar_prompt": "a photo in oil painting style",
                })
        return samples

    with open(prompts_file, "r", encoding="utf-8") as f:
        samples = json.load(f)
    return samples


def evaluate_sample(pipe, sample, images_dir, output_dir, args, lpips_model=None):
    """评估单个样本"""
    img_path = images_dir / sample["image"]
    if not img_path.exists():
        return None

    src_image = Image.open(img_path).convert("RGB")
    src_prompt = sample.get("src_prompt", "")
    tar_prompt = sample.get("tar_prompt", "")

    # 执行编辑
    try:
        edited_image = pipe.edit(
            src_image,
            src_prompt,
            tar_prompt,
            return_intermediate=False,
        )
    except Exception as e:
        print(f"编辑失败 {sample['image']}: {e}")
        return None

    # 保存结果
    if args.save_images:
        save_path = output_dir / f"{Path(sample['image']).stem}_edited.png"
        edited_image.save(save_path)
        compare_path = output_dir / f"{Path(sample['image']).stem}_compare.png"
        utils.save_comparison(src_image, edited_image, tar_prompt, compare_path, src_prompt)

    # 计算指标
    src_tensor = utils.load_image(str(img_path)).to(args.device)
    edited_tensor = pipe.preprocess(edited_image)

    metrics = {}
    metrics["ssim"] = utils.compute_ssim(src_tensor, edited_tensor)
    metrics["psnr"] = utils.compute_psnr(src_tensor, edited_tensor)
    metrics["mse"] = utils.compute_mse(src_tensor, edited_tensor)

    if lpips_model is not None:
        metrics["lpips"] = utils.compute_lpips(src_tensor, edited_tensor, lpips_model)
    else:
        metrics["lpips"] = -1.0

    # 背景一致性 (无掩码时整图评估)
    metrics["background_consistency"] = metrics["ssim"]  # 近似

    return metrics


def main():
    args = parse_args()
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ChordEdit EditBench 评估")
    print("=" * 60)
    print(f"模型: {args.model_type} @ {args.model_path}")
    print(f"ChordEdit 参数: t={args.t}, delta={args.delta}, proximal={args.use_proximal}")
    print(f"设备: {device}")
    print("=" * 60)

    # 初始化 LPIPS (若已安装)
    lpips_model = None
    try:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(device)
        print("[LPIPS] 已加载")
    except ImportError:
        print("[LPIPS] 未安装，跳过 LPIPS 评估")

    # 初始化 ChordEdit 流程
    pipe = ChordEditPipeline(
        model_path=args.model_path,
        model_type=args.model_type,
        device=device,
        t=args.t,
        delta=args.delta,
        use_proximal=args.use_proximal,
    )

    # 加载数据集
    dataset_dir = Path(args.dataset_dir)
    samples = load_editbench_dataset(dataset_dir)
    print(f"[数据集] 共 {len(samples)} 个样本")

    # 评估循环
    all_metrics = []
    images_dir = dataset_dir / "images"

    for sample in tqdm(samples, desc="评估中"):
        metrics = evaluate_sample(pipe, sample, images_dir, output_dir, args, lpips_model)
        if metrics is not None:
            metrics["image"] = sample.get("image", "")
            metrics["tar_prompt"] = sample.get("tar_prompt", "")
            all_metrics.append(metrics)

    # 汇总统计
    if len(all_metrics) == 0:
        print("未成功评估任何样本")
        return

    summary = {}
    for key in ["ssim", "psnr", "mse", "lpips", "background_consistency"]:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            summary[key] = {
                "mean": float(torch.tensor(values).mean()),
                "std": float(torch.tensor(values).std()),
                "min": float(min(values)),
                "max": float(max(values)),
            }

    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)
    for key, stats in summary.items():
        print(f"{key.upper():25s}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
              f"min={stats['min']:.4f}, max={stats['max']:.4f}")

    # 保存详细结果
    result_path = output_dir / "evaluation_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "summary": summary,
            "per_sample": all_metrics,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存至: {result_path}")


if __name__ == "__main__":
    main()