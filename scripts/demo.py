"""
ChordEdit 快速演示脚本
展示对单张图像进行一步式编辑的效果
"""
import sys
from pathlib import Path
import argparse

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
from chordedit.pipeline import ChordEditPipeline


def main():
    parser = argparse.ArgumentParser(description="ChordEdit Demo")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--src_prompt", type=str, default="a photo", help="源描述")
    parser.add_argument("--tar_prompt", type=str, required=True, help="目标描述")
    parser.add_argument("--model_path", type=str, default=str(ROOT / "SD-Turbo"))
    parser.add_argument("--model_type", type=str, default="sd-turbo", choices=["sd-turbo", "instaflow"])
    parser.add_argument("--output", type=str, default=str(ROOT / "outputs" / "demo_result.png"))
    parser.add_argument("--t", type=float, default=0.9)
    parser.add_argument("--delta", type=float, default=0.15)
    parser.add_argument("--no_proximal", action="store_true", help="禁用 Proximal Refinement")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 初始化流程
    pipe = ChordEditPipeline(
        model_path=args.model_path,
        model_type=args.model_type,
        device=device,
        t=args.t,
        delta=args.delta,
        use_proximal=not args.no_proximal,
    )

    # 加载图像
    src_image = Image.open(args.image).convert("RGB")
    print(f"源图像: {args.image}")
    print(f"源提示: {args.src_prompt}")
    print(f"目标提示: {args.tar_prompt}")

    # 执行编辑
    edited_image = pipe.edit(src_image, args.src_prompt, args.tar_prompt)

    # 保存
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    edited_image.save(args.output)
    print(f"结果已保存: {args.output}")


if __name__ == "__main__":
    main()