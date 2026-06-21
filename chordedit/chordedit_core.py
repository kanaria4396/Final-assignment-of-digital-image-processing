"""
ChordEdit 核心算法实现
基于论文 "ChordEdit: One-Step Low-Energy Transport for Image Editing" (CVPR 2026)

核心公式:
1. Chord Control Field (CCF):
   û_t(x_t) = (t * R(x_t, t-δ) + δ * R(x_t, t)) / (t + δ)
   其中 R(x_t, t) = v(x_t, t, c_tar) - v(x_t, t, c_src) 为瞬时漂移差

2. One-Step 编辑:
   x_tar* = x_src + λ * û_t(x_t)

3. Proximal Refinement (可选):
   x_t*   = (1-t) * x_tar* + t * ε,  ε ~ N(0, I)
   x_tar** = x_t* + t * v(x_t*, t, c_tar)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class ChordEditCore(nn.Module):
    """
    ChordEdit 低能量传输图像编辑核心模块

    参数:
        model: 一步式生成模型封装 (SDTurboWrapper / InstaFlowWrapper)
        t: 编辑步长所在时间步 (论文默认 0.9)
        delta: 和弦窗口大小 (论文默认 0.15)
        lambda_scale: 步长缩放因子 (默认 1.0)
        use_proximal: 是否启用 Proximal Refinement
        t_proximal: Proximal Refinement 时间步 (默认与 t 相同)
    """
    def __init__(
        self,
        model,
        t: float = 0.9,
        delta: float = 0.15,
        lambda_scale: float = 1.0,
        use_proximal: bool = True,
        t_proximal: Optional[float] = None,
    ):
        super().__init__()
        self.model = model
        self.t = t
        self.delta = delta
        self.lambda_scale = lambda_scale
        self.use_proximal = use_proximal
        self.t_proximal = t_proximal if t_proximal is not None else t

    def chord_control_field(
        self,
        x_t: torch.Tensor,
        src_embeds: torch.Tensor,
        tar_embeds: torch.Tensor,
        t: float,
        delta: float,
    ) -> torch.Tensor:
        """
        计算和弦控制场 (Chord Control Field, CCF)

        公式:
            û_t = (t * R(x_t, t-delta) + delta * R(x_t, t)) / (t + delta)

        参数:
            x_t: 当前潜空间状态 [B, C, H, W]
            src_embeds: 源文本嵌入 [B, N, D]
            tar_embeds: 目标文本嵌入 [B, N, D]
            t: 当前时间步
            delta: 和弦窗口
        返回:
            û_t: 和弦控制场 [B, C, H, W]
        """
        # 计算 R(x_t, t) = v(x_t, t, c_tar) - v(x_t, t, c_src)
        R_t = self.model.drift_difference(x_t, t, src_embeds, tar_embeds)

        # 计算 R(x_t, t-delta)
        t_prev = max(t - delta, 0.0)
        if abs(t_prev - t) < 1e-6:
            # 若 delta 极小或 t-delta <= 0，退化为单点估计
            R_t_prev = R_t.clone()
        else:
            R_t_prev = self.model.drift_difference(x_t, t_prev, src_embeds, tar_embeds)

        # 加权平均得到低能控制场
        numerator = t * R_t_prev + delta * R_t
        denominator = t + delta
        u_hat = numerator / denominator

        return u_hat

    def one_step_edit(
        self,
        x_src: torch.Tensor,
        src_embeds: torch.Tensor,
        tar_embeds: torch.Tensor,
        t: Optional[float] = None,
        delta: Optional[float] = None,
        lambda_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        单步编辑: x_tar* = x_src + λ * û_t(x_src)

        注意: 对于 flow 模型，x_src 是干净图像在潜空间的表示，
        需要先加噪到时间步 t，或直接以 x_src 作为 x_t。
        论文实现中，SD-Turbo/InstaFlow 直接使用潜空间表示进行编辑。
        """
        t = t if t is not None else self.t
        delta = delta if delta is not None else self.delta
        lambda_scale = lambda_scale if lambda_scale is not None else self.lambda_scale

        # 计算和弦控制场
        u_hat = self.chord_control_field(x_src, src_embeds, tar_embeds, t, delta)

        # 沿和弦场推进
        x_tar_pred = x_src + lambda_scale * u_hat

        return x_tar_pred

    def proximal_refinement(
        self,
        x_tar_pred: torch.Tensor,
        tar_embeds: torch.Tensor,
        t: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Proximal Refinement (近端精修)

        公式:
            x_t*   = (1-t) * x_tar* + t * ε,   ε ~ N(0, I)
            x_tar** = x_t* + t * v(x_t*, t, c_tar)

        作用: 增加高频目标细节，改善语义对齐
        """
        t = t if t is not None else self.t_proximal
        batch_size = x_tar_pred.shape[0]
        device = x_tar_pred.device

        # 重参数化加噪
        epsilon = torch.randn_like(x_tar_pred)
        x_t_star = (1 - t) * x_tar_pred + t * epsilon

        # 目标条件前向步
        with torch.no_grad():
            v_tar = self.model.velocity(x_t_star, t, tar_embeds)
        x_tar_refined = x_t_star + t * v_tar

        return x_tar_refined

    def forward(
        self,
        x_src: torch.Tensor,
        src_embeds: torch.Tensor,
        tar_embeds: torch.Tensor,
        return_intermediate: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        完整前向编辑流程

        参数:
            x_src: 源图像潜空间表示 [B, C, H, W]
            src_embeds: 源文本嵌入 [B, N, D]
            tar_embeds: 目标文本嵌入 [B, N, D]
            return_intermediate: 是否返回中间结果
        返回:
            x_tar: 编辑后图像潜空间表示
            info: 中间结果字典 (若 return_intermediate=True)
        """
        info = {}

        # Step 1: 单步低能量传输
        x_tar_pred = self.one_step_edit(x_src, src_embeds, tar_embeds)
        if return_intermediate:
            info["one_step_pred"] = x_tar_pred.clone()

        # Step 2: 近端精修 (可选)
        if self.use_proximal:
            x_tar = self.proximal_refinement(x_tar_pred, tar_embeds)
            if return_intermediate:
                info["refined_pred"] = x_tar.clone()
        else:
            x_tar = x_tar_pred

        if return_intermediate:
            info["u_hat"] = (x_tar_pred - x_src) / self.lambda_scale
            return x_tar, info
        return x_tar, None

    @torch.no_grad()
    def edit_image(
        self,
        src_image: torch.Tensor,
        src_prompt: str,
        tar_prompt: str,
        return_intermediate: bool = False,
    ):
        """
        端到端图像编辑接口 (输入输出均为像素空间图像)

        参数:
            src_image: [B, 3, H, W], range [0, 1]
            src_prompt: 源描述文本
            tar_prompt: 目标描述文本
        返回:
            edited_image: [B, 3, H, W], range [0, 1]
        """
        # 1. 编码文本
        src_embeds = self.model.encode_prompt(src_prompt)
        tar_embeds = self.model.encode_prompt(tar_prompt)

        # 2. 图像编码到潜空间
        x_src = self.model.encode_image(src_image)

        # 3. ChordEdit 编辑
        x_tar, info = self.forward(
            x_src, src_embeds, tar_embeds,
            return_intermediate=return_intermediate
        )

        # 4. 解码回像素空间
        edited_image = self.model.decode_latents(x_tar)

        if return_intermediate:
            return edited_image, info
        return edited_image
