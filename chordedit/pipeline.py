"""
ChordEdit 图像编辑流程封装
支持常见编辑任务: 风格迁移、内容修改、属性编辑等
"""
import torch
import torch.nn as nn
from typing import Union, List, Optional
from PIL import Image
from torchvision import transforms

from .chordedit_core import ChordEditCore
from .model_wrappers import BaseOneStepWrapper


class ChordEditPipeline:
    """
    面向用户的 ChordEdit 图像编辑流程

    用法:
        pipe = ChordEditPipeline(model_path="SD-Turbo", device="cuda")
        result = pipe.edit(src_image, src_prompt="a photo of a dog", tar_prompt="a photo of a dog wearing sunglasses")
    """
    def __init__(
        self,
        model_path: str,
        model_type: str = "sd-turbo",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        t: float = 0.9,
        delta: float = 0.15,
        lambda_scale: float = 1.0,
        use_proximal: bool = True,
    ):
        self.device = device
        self.dtype = dtype

        # 加载模型
        from .model_wrappers import SDTurboWrapper, InstaFlowWrapper
        if model_type.lower() in ("sd-turbo", "sdturbo", "sd_turbo"):
            self.model = SDTurboWrapper(model_path, device=device, dtype=dtype)
        elif model_type.lower() in ("instaflow", "insta-flow"):
            self.model = InstaFlowWrapper(model_path, device=device, dtype=dtype)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        # 初始化 ChordEdit 核心
        self.core = ChordEditCore(
            model=self.model,
            t=t,
            delta=delta,
            lambda_scale=lambda_scale,
            use_proximal=use_proximal,
        )

        # 图像预处理
        self.image_size = 512
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

    def preprocess(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """将输入图像转换为模型可用的张量"""
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = self.transform(image)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
        else:
            raise TypeError("image 必须是 PIL.Image 或 torch.Tensor")
        
        image = image.to(self.device)
        return image

    def postprocess(self, tensor: torch.Tensor) -> Image.Image:
        """将模型输出张量转换为 PIL Image"""
        tensor = tensor.detach().cpu()
        if tensor.dim() == 4:
            tensor = tensor[0]
        tensor = tensor.permute(1, 2, 0).clamp(0, 1)
        array = (tensor.numpy() * 255).astype("uint8")
        return Image.fromarray(array)

    def edit(
        self,
        src_image: Union[Image.Image, torch.Tensor],
        src_prompt: str,
        tar_prompt: str,
        return_intermediate: bool = False,
    ):
        """
        执行图像编辑

        参数:
            src_image: 源图像
            src_prompt: 源文本描述
            tar_prompt: 目标文本描述
            return_intermediate: 是否返回中间结果
        返回:
            edited_image: 编辑后的 PIL Image
            info (optional): 中间结果
        """
        image = self.preprocess(src_image)
        result = self.core.edit_image(
            image,
            src_prompt,
            tar_prompt,
            return_intermediate=return_intermediate,
        )
        if return_intermediate:
            edited_tensor, info = result
            edited_image = self.postprocess(edited_tensor)
            return edited_image, info
        else:
            edited_image = self.postprocess(result)
            return edited_image

    def batch_edit(
        self,
        src_images: List[Union[Image.Image, torch.Tensor]],
        src_prompts: List[str],
        tar_prompts: List[str],
    ) -> List[Image.Image]:
        """批量编辑"""
        results = []
        for img, sp, tp in zip(src_images, src_prompts, tar_prompts):
            results.append(self.edit(img, sp, tp))
        return results

    def style_transfer(
        self,
        src_image: Union[Image.Image, torch.Tensor],
        content_prompt: str,
        style_prompt: str,
    ) -> Image.Image:
        """
        风格迁移任务快捷接口
        例: style_transfer(img, "a photo of a building", "a photo of a building, oil painting style")
        """
        return self.edit(src_image, content_prompt, style_prompt)

    def content_modification(
        self,
        src_image: Union[Image.Image, torch.Tensor],
        src_prompt: str,
        tar_prompt: str,
    ) -> Image.Image:
        """
        内容修改任务快捷接口
        例: content_modification(img, "a red car", "a blue car")
        """
        return self.edit(src_image, src_prompt, tar_prompt)