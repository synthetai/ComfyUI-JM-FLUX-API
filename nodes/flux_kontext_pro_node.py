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
        
        # Debug日志: 记录请求信息
        print("=" * 60)
        print("🚀 [DEBUG] 创建任务 - 请求信息:")
        print(f"   📍 URL: {url}")
        print(f"   🔑 API Key (前8位): {api_key[:8]}..." if api_key else "   🔑 API Key: None")
        print(f"   📝 Headers: {headers}")
        
        # 安全地记录payload（排除可能很大的base64图片数据）
        safe_payload = payload.copy()
        if 'input_image' in safe_payload:
            if safe_payload['input_image'].startswith('data:') or len(safe_payload['input_image']) > 100:
                safe_payload['input_image'] = f"[Base64 Image Data - {len(safe_payload['input_image'])} chars]"
        print(f"   📦 Payload: {json.dumps(safe_payload, indent=2, ensure_ascii=False)}")
        print("=" * 60)
        
        try:
            print("⏳ [DEBUG] 正在发送请求到FLUX API...")
            start_time = time.time()
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            request_time = time.time() - start_time
            print(f"⏱️ [DEBUG] 请求耗时: {request_time:.2f}秒")
            print(f"📊 [DEBUG] HTTP状态码: {response.status_code}")
            print(f"📋 [DEBUG] 响应头: {dict(response.headers)}")
            
            # 记录响应内容
            try:
                response_data = response.json()
                print("✅ [DEBUG] 响应数据:")
                print(json.dumps(response_data, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print("❌ [DEBUG] 响应不是有效的JSON:")
                print(f"响应文本: {response.text}")
            
            response.raise_for_status()
            
            # 验证响应数据
            if 'id' not in response_data:
                print("⚠️ [DEBUG] 警告: 响应中没有找到 'id' 字段")
                print(f"响应键: {list(response_data.keys())}")
            else:
                task_id = response_data['id']
                print(f"🎯 [DEBUG] 成功获取任务ID: {task_id}")
                print(f"🔍 [DEBUG] 任务ID类型: {type(task_id)}")
                print(f"📏 [DEBUG] 任务ID长度: {len(str(task_id))}")
            
            return response_data
            
        except requests.exceptions.RequestException as e:
            print(f"❌ [DEBUG] 请求异常:")
            print(f"   异常类型: {type(e).__name__}")
            print(f"   异常信息: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   HTTP状态码: {e.response.status_code}")
                print(f"   响应文本: {e.response.text}")
            raise Exception(f"创建任务失败: {str(e)}")
    
    def get_result(self, task_id: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        获取任务结果，带重试机制
        """
        url = f"{self.api_base_url}/get_result?id={task_id}"
        
        print("=" * 60)
        print(f"🔍 [DEBUG] 查询任务结果:")
        print(f"   📍 查询URL: {url}")
        print(f"   🎯 任务ID: {task_id}")
        print(f"   🔄 最大重试次数: {max_retries}")
        print("=" * 60)
        
        for attempt in range(max_retries):
            try:
                print(f"📡 [DEBUG] 第 {attempt + 1} 次查询尝试...")
                start_time = time.time()
                
                response = requests.get(url, timeout=30)
                
                request_time = time.time() - start_time
                print(f"⏱️ [DEBUG] 查询耗时: {request_time:.2f}秒")
                print(f"📊 [DEBUG] HTTP状态码: {response.status_code}")
                
                if response.status_code == 404:
                    print("🔍 [DEBUG] 收到404响应:")
                    print(f"   响应头: {dict(response.headers)}")
                    print(f"   响应文本: {response.text}")
                    print(f"   完整查询URL: {url}")
                    
                    # 检查任务ID格式
                    print(f"🔎 [DEBUG] 任务ID详细信息:")
                    print(f"   原始任务ID: '{task_id}'")
                    print(f"   任务ID类型: {type(task_id)}")
                    print(f"   任务ID长度: {len(str(task_id))}")
                    print(f"   是否包含特殊字符: {any(c in task_id for c in [' ', '\n', '\t', '\r'])}")
                    
                response.raise_for_status()
                
                # 记录成功响应
                try:
                    response_data = response.json()
                    print("✅ [DEBUG] 查询成功，响应数据:")
                    print(json.dumps(response_data, indent=2, ensure_ascii=False))
                    return response_data
                except json.JSONDecodeError:
                    print("❌ [DEBUG] 响应不是有效的JSON:")
                    print(f"响应文本: {response.text}")
                    raise Exception("服务器返回的不是有效的JSON响应")
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    # 404错误可能是任务还没有被完全注册，稍等重试
                    if attempt < max_retries - 1:
                        wait_time = 5
                        print(f"⏳ [DEBUG] 任务暂未找到(404)，等待{wait_time}秒后重试 ({attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"❌ [DEBUG] 已重试{max_retries}次仍然404，可能的原因:")
                        print(f"   1. 任务ID无效或格式错误")
                        print(f"   2. 任务已过期或被清理")
                        print(f"   3. API服务器内部问题")
                        print(f"   4. 网络或DNS解析问题")
                        raise Exception(f"任务未找到，已重试{max_retries}次，可能任务ID无效或已过期")
                else:
                    print(f"❌ [DEBUG] HTTP错误 {e.response.status_code}:")
                    print(f"   响应文本: {e.response.text}")
                    raise Exception(f"获取结果失败: {str(e)}")
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 3
                    print(f"🔄 [DEBUG] 网络错误，等待{wait_time}秒后重试 ({attempt + 1}/{max_retries}): {str(e)}")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ [DEBUG] 网络错误，已达到最大重试次数: {str(e)}")
                    raise Exception(f"获取结果失败: {str(e)}")
        
        raise Exception("获取任务结果失败，已达到最大重试次数")
    
    def wait_for_completion(self, task_id: str, max_wait_time: int = 300, poll_interval: int = 10) -> Dict[str, Any]:
        """
        等待任务完成，带初始延迟和智能重试
        """
        start_time = time.time()
        
        print("=" * 60)
        print(f"⏳ [DEBUG] 开始等待任务完成:")
        print(f"   🎯 任务ID: {task_id}")
        print(f"   ⏱️ 最大等待时间: {max_wait_time}秒")
        print(f"   🔄 轮询间隔: {poll_interval}秒")
        print("=" * 60)
        
        # 初始等待10秒，让服务器完全注册任务
        initial_wait = 10
        print(f"⏰ [DEBUG] 初始等待{initial_wait}秒，让服务器完全注册任务...")
        time.sleep(initial_wait)
        
        poll_count = 0
        while time.time() - start_time < max_wait_time:
            poll_count += 1
            elapsed_time = time.time() - start_time
            
            print(f"🔍 [DEBUG] 第{poll_count}次轮询 (已等待 {elapsed_time:.1f}秒)...")
            
            try:
                result = self.get_result(task_id)
                
                status = result.get("status")
                print(f"📊 [DEBUG] 任务状态: {status}")
                
                if status == "Ready":
                    total_time = time.time() - start_time
                    print(f"🎉 [DEBUG] 任务完成! 总耗时: {total_time:.1f}秒")
                    return result
                elif status in ["Failed", "Error"]:
                    error_details = result.get('details', '未知错误')
                    print(f"❌ [DEBUG] 任务失败:")
                    print(f"   状态: {status}")
                    print(f"   详细信息: {error_details}")
                    raise Exception(f"任务失败: {error_details}")
                else:
                    print(f"⏳ [DEBUG] 任务仍在处理中，状态: {status}")
                
                print(f"💤 [DEBUG] 等待{poll_interval}秒后进行下次轮询...")
                time.sleep(poll_interval)
                
            except Exception as e:
                error_msg = str(e)
                # 如果是404错误且还在等待时间内，继续重试
                if "任务未找到" in error_msg and time.time() - start_time < 60:
                    print(f"🔄 [DEBUG] 任务暂未就绪，继续等待... (错误: {error_msg})")
                    time.sleep(poll_interval)
                    continue
                else:
                    print(f"❌ [DEBUG] 轮询过程中发生错误: {error_msg}")
                    raise e
        
        total_wait_time = time.time() - start_time
        print(f"⏰ [DEBUG] 任务超时，总等待时间: {total_wait_time:.1f}秒")
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
        print(f"开始等待任务完成（初始延迟10秒，然后每10秒查询一次状态）...")
        
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