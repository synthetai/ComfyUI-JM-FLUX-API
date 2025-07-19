import os
import io
import json
import base64
import requests
import time
from PIL import Image
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional

# 尝试导入ComfyUI的folder_paths模块
try:
    import folder_paths
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False

class FluxKontextProNode:
    """
    ComfyUI custom node for FLUX Kontext Pro API
    """
    
    def __init__(self):
        self.api_base_url = "https://api.bfl.ai/v1"
        self.api_key = os.getenv("FLUX_API_KEY")  # 从环境变量获取API密钥
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "ein fantastisches bild"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "FLUX API Key (或设置环境变量FLUX_API_KEY)"
                }),
                "filename_prefix": ("STRING", {
                    "default": "flux_kontext_pro",
                    "tooltip": "生成图片的文件名前缀"
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "输入图片用于图生图"
                }),
                "image_url": ("STRING", {
                    "default": "",
                    "tooltip": "图片URL地址"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "随机种子"
                }),
                "aspect_ratio": (["1:1", "16:9", "21:9", "9:16", "9:21", "4:3", "3:4"], {
                    "default": "1:1",
                    "tooltip": "图片宽高比"
                }),
                "output_format": (["png", "jpeg"], {
                    "default": "png",
                    "tooltip": "输出格式"
                }),
                "prompt_upsampling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否对提示词进行优化"
                }),
                "safety_tolerance": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 6,
                    "tooltip": "安全级别(0最严格，6最宽松)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "image_path")
    FUNCTION = "generate_image"
    CATEGORY = "JM-FLUX-API/FLUX-KONTEXT-PRO"
    
    def encode_image_to_base64(self, image: torch.Tensor) -> str:
        """
        将ComfyUI的图片张量转换为base64编码
        """
        # ComfyUI的图片张量格式: [batch_size, height, width, channels]
        # 取第一个batch
        if len(image.shape) == 4:
            img_array = image[0].cpu().numpy()  # 去掉batch维度: [height, width, channels]
        else:
            img_array = image.cpu().numpy()
        
        # 确保值在0-1范围内，然后转换为0-255
        img_array = np.clip(img_array, 0.0, 1.0)
        img_array = (img_array * 255).astype(np.uint8)
        
        # 转换为PIL图片
        img = Image.fromarray(img_array)
        
        # 转换为base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    def create_task(self, payload: Dict[str, Any], api_key: str) -> Dict[str, Any]:
        """
        创建FLUX生成任务
        """
        headers = {
            'Content-Type': 'application/json',
            'x-key': api_key
        }
        
        url = f"{self.api_base_url}/flux-kontext-pro"
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"创建任务失败: {str(e)}")
    
    def get_result(self, task_id: str) -> Dict[str, Any]:
        """
        获取任务结果
        """
        url = f"{self.api_base_url}/get_result?id={task_id}"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"获取结果失败: {str(e)}")
    
    def wait_for_completion(self, task_id: str, max_wait_time: int = 300, poll_interval: int = 5) -> Dict[str, Any]:
        """
        等待任务完成
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            result = self.get_result(task_id)
            
            if result.get("status") == "Ready":
                return result
            elif result.get("status") in ["Failed", "Error"]:
                raise Exception(f"任务失败: {result.get('details', '未知错误')}")
            
            print(f"任务状态: {result.get('status', 'Unknown')}, 等待中...")
            time.sleep(poll_interval)
        
        raise Exception(f"任务超时，等待时间超过{max_wait_time}秒")
    
    def download_image(self, image_url: str, save_path: str) -> None:
        """
        下载图片到本地
        """
        try:
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图片失败: {str(e)}")
    
    def load_image_as_tensor(self, image_path: str) -> torch.Tensor:
        """
        加载图片为ComfyUI张量格式
        """
        img = Image.open(image_path)
        img = img.convert('RGB')
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        return img_tensor
    
    def get_comfyui_output_dir(self) -> str:
        """
        获取ComfyUI的output目录
        """
        if COMFYUI_AVAILABLE:
            try:
                # 使用ComfyUI的folder_paths获取output目录
                return folder_paths.get_output_directory()
            except:
                pass
        
        # 如果无法获取ComfyUI output目录，尝试推断
        # 查找ComfyUI根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 向上查找，寻找ComfyUI目录
        while current_dir and current_dir != os.path.dirname(current_dir):
            # 检查是否包含ComfyUI的特征文件/目录
            if any(name in os.listdir(current_dir) for name in ['main.py', 'execution.py', 'server.py']):
                comfyui_output = os.path.join(current_dir, "output")
                if os.path.exists(comfyui_output) or current_dir.endswith('ComfyUI'):
                    return comfyui_output
            current_dir = os.path.dirname(current_dir)
        
        # 如果都失败了，使用当前工作目录的output文件夹作为备选
        return os.path.join(os.getcwd(), "output")
    
    def get_next_filename(self, directory: str, prefix: str, extension: str) -> Tuple[str, str]:
        """
        获取下一个可用的文件名
        """
        counter = 1
        while True:
            filename = f"{prefix}_{counter:04d}.{extension}"
            filepath = os.path.join(directory, filename)
            if not os.path.exists(filepath):
                return filename, filepath
            counter += 1
    
    def generate_image(self, prompt: str, api_key: str, filename_prefix: str, 
                      image: Optional[torch.Tensor] = None, image_url: str = "", 
                      seed: int = 42, aspect_ratio: str = "1:1", 
                      output_format: str = "png", prompt_upsampling: bool = False, 
                      safety_tolerance: int = 2) -> Tuple[torch.Tensor, str]:
        """
        主要的图片生成函数
        """
        
        # 使用提供的API密钥或环境变量
        if not api_key and self.api_key:
            api_key = self.api_key
        
        if not api_key:
            raise Exception("请提供FLUX API密钥或设置环境变量FLUX_API_KEY")
        
        # 准备请求payload
        payload = {
            "prompt": prompt,
            "seed": seed,
            "aspect_ratio": aspect_ratio,
            "output_format": output_format,
            "prompt_upsampling": prompt_upsampling,
            "safety_tolerance": safety_tolerance
        }
        
        # 处理输入图片
        if image is not None:
            # 如果有输入的图片张量，转换为base64
            input_image_b64 = self.encode_image_to_base64(image)
            payload["input_image"] = input_image_b64
        elif image_url:
            # 如果有图片URL，直接使用
            payload["input_image"] = image_url
        
        print(f"开始创建FLUX任务...")
        
        # 创建任务
        task_response = self.create_task(payload, api_key)
        task_id = task_response.get("id")
        
        if not task_id:
            raise Exception("创建任务失败，未获取到任务ID")
        
        print(f"任务已创建，ID: {task_id}")
        print(f"等待任务完成...")
        
        # 等待任务完成
        result = self.wait_for_completion(task_id)
        
        # 获取生成的图片URL
        sample_url = result.get("result", {}).get("sample")
        if not sample_url:
            raise Exception("任务完成但未获取到图片链接")
        
        print(f"图片生成完成，开始下载...")
        
        # 获取ComfyUI的output目录
        output_dir = self.get_comfyui_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"图片将保存到目录: {output_dir}")
        
        # 生成文件名
        extension = output_format.lower()
        filename, filepath = self.get_next_filename(output_dir, filename_prefix, extension)
        
        # 下载图片
        self.download_image(sample_url, filepath)
        
        print(f"图片已保存到: {filepath}")
        
        # 加载图片为张量
        image_tensor = self.load_image_as_tensor(filepath)
        
        return (image_tensor, filepath)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "FluxKontextProNode": FluxKontextProNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxKontextProNode": "Flux Kontext Pro"
} 