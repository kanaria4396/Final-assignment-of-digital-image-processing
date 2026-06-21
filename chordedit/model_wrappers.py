"""
一步式生成模型封装 (SD-Turbo / InstaFlow)
提供统一的 velocity / drift 接口
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, Optional
from diffusers import StableDiffusionPipeline, DiffusionPipeline

class BaseOneStepWrapper(nn.Module):
    """一步式模型基类"""
    def __init__(self, model_path: str, device: str = "cuda", dtype: torch.dtype = torch.float16):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        self.pipe = None
        self.vae = None
        self.unet = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None

    def encode_prompt(self, prompt: Union[str, list]) -> torch.Tensor:
        """编码文本提示为嵌入向量"""
        if isinstance(prompt, str):
            prompt = [prompt]
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        with torch.no_grad():
            prompt_embeds = self.text_encoder(text_input_ids)[0]
        return prompt_embeds

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """将图像编码为潜空间表示"""
        # image: [B, 3, H, W], range [-1, 1] or [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        if image.min() >= 0:
            image = image * 2.0 - 1.0
        with torch.no_grad():
            latents = self.vae.encode(image.to(self.dtype)).latent_dist.mode()
            latents = latents * self.vae.config.scaling_factor
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """将潜空间表示解码为图像"""
        latents = latents / self.vae.config.scaling_factor
        with torch.no_grad():
            image = self.vae.decode(latents.to(self.dtype)).sample
        image = (image + 1.0) / 2.0
        image = image.clamp(0, 1)
        return image

    def velocity(self, x_t: torch.Tensor, t: float, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """
        计算向量场 velocity v(x_t, t, c)
        需在子类中根据模型参数化方式实现
        """
        raise NotImplementedError

    def drift_difference(self, x_t: torch.Tensor, t: float,
                         src_embeds: torch.Tensor, tar_embeds: torch.Tensor) -> torch.Tensor:
        """
        计算观测代理场 R(x_t, t) = v(x_t, t, c_tar) - v(x_t, t, c_src)
        对应论文中的 Δv
        """
        v_src = self.velocity(x_t, t, src_embeds)
        v_tar = self.velocity(x_t, t, tar_embeds)
        return v_tar - v_src


class SDTurboWrapper(BaseOneStepWrapper):
    """
    SD-Turbo 模型封装 (Rectified Flow 参数化)
    SD-Turbo 使用 rectified flow，velocity 直接由 UNet 预测
    """
    def __init__(self, model_path: str, device: str = "cuda", dtype: torch.dtype = torch.float16):
        super().__init__(model_path, device, dtype)
        print(f"[SDTurbo] 加载模型: {model_path}")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(device)
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        self.scheduler = self.pipe.scheduler
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()

    def velocity(self, x_t: torch.Tensor, t: float, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """
        Rectified Flow 参数化:
        UNet 直接预测 velocity field v(x_t, t, c)
        timestep 需要缩放为 scheduler 的 sigma 或 timestep 表示
        """
        # SD-Turbo 使用 timesteps，t in [0, 1]，映射到 [0, 999]
        timestep = torch.tensor([int(t * 999)], device=self.device, dtype=torch.long)
        # 扩展 timestep 到 batch size
        timestep = timestep.expand(x_t.shape[0])
        with torch.no_grad():
            velocity = self.unet(
                x_t.to(self.dtype),
                timestep,
                encoder_hidden_states=prompt_embeds.to(self.dtype),
            ).sample
        return velocity


class InstaFlowWrapper(BaseOneStepWrapper):
    """
    InstaFlow 模型封装 (Rectified Flow 参数化)
    InstaFlow 同样基于 rectified flow，接口与 SD-Turbo 类似
    """
    def __init__(self, model_path: str, device: str = "cuda", dtype: torch.dtype = torch.float16):
        super().__init__(model_path, device, dtype)
        print(f"[InstaFlow] 加载模型: {model_path}")
        # InstaFlow 兼容 StableDiffusionPipeline 结构
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(device)
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        self.scheduler = self.pipe.scheduler
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()

    def velocity(self, x_t: torch.Tensor, t: float, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """InstaFlow  velocity field"""
        timestep = torch.tensor([int(t * 999)], device=self.device, dtype=torch.long)
        timestep = timestep.expand(x_t.shape[0])
        with torch.no_grad():
            velocity = self.unet(
                x_t.to(self.dtype),
                timestep,
                encoder_hidden_states=prompt_embeds.to(self.dtype),
            ).sample
        return velocity
