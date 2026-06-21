"""
工具函数: 图像处理、评估指标、可视化
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Union, Tuple
from skimage.metrics import structural_similarity as ssim_skimage

def compute_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    计算 SSIM (Structural Similarity Index)
    输入: [B, 3, H, W] 或 [3, H, W], range [0, 1]
    """
    if img1.dim() == 4:
        img1 = img1[0]
    if img2.dim() == 4:
        img2 = img2[0]
    img1_np = img1.permute(1, 2, 0).cpu().numpy()
    img2_np = img2.permute(1, 2, 0).cpu().numpy()
    return ssim_skimage(img1_np, img2_np, channel_axis=2, data_range=1.0)

def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    计算 PSNR (Peak Signal-to-Noise Ratio)
    输入: range [0, 1]
    """
    mse = F.mse_loss(img1, img2).item()
    if mse < 1e-10:
        return 100.0
    return -10 * np.log10(mse)

def compute_lpips(img1: torch.Tensor, img2: torch.Tensor, lpips_model) -> float:
    """
    计算 LPIPS (Learned Perceptual Image Patch Similarity)
    输入: range [0, 1], device 需与 lpips_model 一致
    """
    # lpips 需要 [-1, 1]
    img1 = img1 * 2 - 1
    img2 = img2 * 2 - 1
    with torch.no_grad():
        dist = lpips_model(img1, img2)
    return dist.item()

def compute_mse(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """计算 MSE"""
    return F.mse_loss(img1, img2).item()

def compute_background_consistency(
    edited: torch.Tensor,
    original: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """
    计算背景一致性 (Background Consistency)
    mask: 编辑区域掩码, 1 表示编辑区域, 0 表示背景/非编辑区域
    在非编辑区域计算 MSE
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # [B, 1, H, W]
    if mask.max() > 1:
        mask = mask / 255.0
    bg_mask = 1.0 - mask
    diff = (edited - original) ** 2
    if bg_mask.sum() < 1e-6:
        return 1.0
    bg_mse = (diff * bg_mask).sum() / (bg_mask.sum() * edited.shape[1])
    # 转换为一致性分数 (越高越好)
    consistency = 1.0 - min(bg_mse * 10, 1.0)
    return consistency

def save_comparison(
    src_img: Image.Image,
    edited_img: Image.Image,
    tar_prompt: str,
    save_path: str,
    src_prompt: str = "",
):
    """保存源图像与编辑结果对比图"""
    w, h = src_img.size
    canvas = Image.new("RGB", (w * 2, h + 40), color=(255, 255, 255))
    canvas.paste(src_img, (0, 20))
    canvas.paste(edited_img, (w, 20))
    # 可在上方添加文字标注 (此处略)
    canvas.save(save_path)

def load_image(path: str, size: int = 512) -> torch.Tensor:
    """加载图像并转为张量"""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size))
    arr = np.array(img) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0)
    return tensor

def save_image(tensor: torch.Tensor, path: str):
    """保存张量为图像文件"""
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = tensor.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    arr = (tensor * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)