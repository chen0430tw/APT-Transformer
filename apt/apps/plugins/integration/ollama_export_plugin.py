"""
Ollama Export Plugin for APT Model
将APT模型导出为Ollama格式,支持本地部署
"""

import os
import json
import torch
import struct
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List


class OllamaExportPlugin:
    """
    Ollama导出插件
    
    功能:
    1. 导出APT模型为GGUF格式
    2. 创建Modelfile配置
    3. 自动注册到Ollama
    4. 支持多种量化选项
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.name = "ollama-export"
        self.version = "1.0.0"
        self.config = config
        
        self.quantization = config.get('quantization', 'Q4_K_M')  # Q4/Q5/Q8
        self.context_length = config.get('context_length', 2048)
        self.temperature = config.get('temperature', 0.7)
    
    # ==================== GGUF格式转换 ====================
    
    def export_to_gguf(
        self,
        model_path: str,
        output_path: str,
        quantization: Optional[str] = None
    ) -> str:
        """
        将APT模型转换为GGUF格式
        
        Args:
            model_path: APT模型路径
            output_path: 输出GGUF文件路径
            quantization: 量化类型 (Q4_0, Q4_K_M, Q5_K_M, Q8_0等)
            
        Returns:
            GGUF文件路径
        """
        quant = quantization or self.quantization
        print(f"🔄 正在转换APT模型为GGUF格式 (量化: {quant})...")
        
        try:
            # 加载APT模型
            model = torch.load(os.path.join(model_path, "pytorch_model.bin"))
            
            # 创建GGUF文件
            gguf_path = output_path if output_path.endswith('.gguf') else f"{output_path}.gguf"
            
            # 写入GGUF格式
            self._write_gguf(model, gguf_path, quant)
            
            print(f"✅ GGUF文件已创建: {gguf_path}")
            return gguf_path
            
        except Exception as e:
            print(f"❌ GGUF转换失败: {e}")
            raise
    
    def _write_gguf(
        self,
        model: Dict[str, torch.Tensor],
        output_path: str,
        quantization: str
    ):
        """写入GGUF格式文件"""
        with open(output_path, 'wb') as f:
            # GGUF魔数和版本
            f.write(b'GGUF')
            f.write(struct.pack('I', 3))  # 版本3
            
            # 写入元数据
            metadata = self._create_gguf_metadata(model)
            self._write_metadata(f, metadata)
            
            # 写入张量信息
            tensor_count = len(model)
            f.write(struct.pack('Q', tensor_count))
            
            # 量化并写入权重
            for name, tensor in model.items():
                quantized = self._quantize_tensor(tensor, quantization)
                self._write_tensor(f, name, quantized, quantization)
    
    def _create_gguf_metadata(self, model: Dict) -> Dict:
        """创建GGUF元数据"""
        return {
            'general.architecture': 'apt',
            'general.name': 'APT Model',
            'general.file_type': self._get_file_type(self.quantization),
            'apt.context_length': self.context_length,
            'apt.embedding_length': self._infer_embedding_dim(model),
            'apt.block_count': self._infer_layer_count(model),
            'tokenizer.ggml.model': 'gpt2',
        }
    
    def _quantize_tensor(self, tensor: torch.Tensor, quant_type: str) -> bytes:
        """量化张量"""
        if quant_type == 'Q4_0':
            return self._quantize_q4_0(tensor)
        elif quant_type == 'Q4_K_M':
            return self._quantize_q4_k(tensor)
        elif quant_type == 'Q5_K_M':
            return self._quantize_q5_k(tensor)
        elif quant_type == 'Q8_0':
            return self._quantize_q8_0(tensor)
        else:
            # 默认FP16
            return tensor.half().numpy().tobytes()
    
    def _quantize_q4_0(self, tensor: torch.Tensor) -> bytes:
        """4位量化(基础版本)"""
        # 简化实现,实际应该用llama.cpp的量化方法
        flat = tensor.flatten().float()
        
        # 分块量化
        block_size = 32
        n_blocks = (len(flat) + block_size - 1) // block_size
        quantized = bytearray()
        
        for i in range(n_blocks):
            block = flat[i*block_size:(i+1)*block_size]
            
            # 计算scale
            abs_max = block.abs().max().item()
            scale = abs_max / 7.0 if abs_max > 0 else 1.0
            
            # 量化
            quants = torch.clamp(torch.round(block / scale), -8, 7)
            
            # 打包(每个值4位)
            quantized.extend(struct.pack('f', scale))
            for j in range(0, len(quants), 2):
                v1 = int(quants[j].item()) & 0x0F
                v2 = int(quants[j+1].item() if j+1 < len(quants) else 0) & 0x0F
                quantized.append((v2 << 4) | v1)
        
        return bytes(quantized)
    
    def _quantize_q8_0(self, tensor: torch.Tensor) -> bytes:
        """8位量化"""
        flat = tensor.flatten().float()
        
        block_size = 32
        n_blocks = (len(flat) + block_size - 1) // block_size
        quantized = bytearray()
        
        for i in range(n_blocks):
            block = flat[i*block_size:(i+1)*block_size]
            
            abs_max = block.abs().max().item()
            scale = abs_max / 127.0 if abs_max > 0 else 1.0
            
            quants = torch.clamp(torch.round(block / scale), -128, 127)
            
            quantized.extend(struct.pack('f', scale))
            quantized.extend(quants.to(torch.int8).numpy().tobytes())
        
        return bytes(quantized)
    
    def _quantize_q4_k(self, tensor: torch.Tensor) -> bytes:
        """4位K-quants (改进版本)"""
        # 这里应该实现K-quants算法,为简化使用Q4_0
        return self._quantize_q4_0(tensor)
    
    def _quantize_q5_k(self, tensor: torch.Tensor) -> bytes:
        """5位K-quants"""
        # 简化实现
        return self._quantize_q4_0(tensor)
    
    # ==================== Modelfile创建 ====================
    
    def create_modelfile(
        self,
        gguf_path: str,
        output_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        template: Optional[str] = None
    ) -> str:
        """
        创建Ollama Modelfile
        
        Args:
            gguf_path: GGUF模型文件路径
            output_path: Modelfile输出路径
            system_prompt: 系统提示词
            template: 对话模板
            
        Returns:
            Modelfile路径
        """
        print("📝 创建Modelfile...")
        
        if output_path is None:
            output_path = os.path.join(
                os.path.dirname(gguf_path),
                "Modelfile"
            )
        
        # 默认系统提示
        if system_prompt is None:
            system_prompt = "You are a helpful AI assistant powered by APT model."
        
        # 默认模板
        if template is None:
            template = """{{ if .System }}{{ .System }}{{ end }}
{{ if .Prompt }}User: {{ .Prompt }}{{ end }}
Assistant: """
        
        # 创建Modelfile内容
        modelfile_content = f"""# APT Model for Ollama
FROM {gguf_path}

# 模型参数
PARAMETER temperature {self.temperature}
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx {self.context_length}

# 系统提示
SYSTEM \"\"\"{system_prompt}\"\"\"

# 对话模板
TEMPLATE \"\"\"{template}\"\"\"

# 停止词
PARAMETER stop "<|im_end|>"
PARAMETER stop "</s>"
"""
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        
        print(f"✅ Modelfile已创建: {output_path}")
        return output_path
    
    # ==================== Ollama集成 ====================
    
    def register_to_ollama(
        self,
        modelfile_path: str,
        model_name: str
    ) -> bool:
        """
        将模型注册到Ollama
        
        Args:
            modelfile_path: Modelfile路径
            model_name: 模型名称(如: apt-chinese:latest)
            
        Returns:
            是否成功
        """
        print(f"🚀 注册模型到Ollama: {model_name}")
        
        try:
            # 检查Ollama是否安装
            result = subprocess.run(
                ['ollama', '--version'],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print("❌ Ollama未安装,请先安装Ollama")
                print("   访问: https://ollama.ai/download")
                return False
            
            # 创建模型
            result = subprocess.run(
                ['ollama', 'create', model_name, '-f', modelfile_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ 模型已注册到Ollama: {model_name}")
                print(f"\n使用方法:")
                print(f"  ollama run {model_name}")
                return True
            else:
                print(f"❌ 注册失败: {result.stderr}")
                return False
                
        except FileNotFoundError:
            print("❌ Ollama命令未找到,请确保Ollama已正确安装")
            return False
        except Exception as e:
            print(f"❌ 注册过程出错: {e}")
            return False
    
    def test_model(self, model_name: str, prompt: str = "你好！") -> str:
        """
        测试Ollama模型
        
        Args:
            model_name: 模型名称
            prompt: 测试提示词
            
        Returns:
            模型响应
        """
        print(f"🧪 测试模型: {model_name}")
        
        try:
            result = subprocess.run(
                ['ollama', 'run', model_name, prompt],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                response = result.stdout.strip()
                print(f"✅ 模型响应:\n{response}")
                return response
            else:
                print(f"❌ 测试失败: {result.stderr}")
                return ""
                
        except subprocess.TimeoutExpired:
            print("⏱️ 测试超时")
            return ""
        except Exception as e:
            print(f"❌ 测试出错: {e}")
            return ""
    
    # ==================== 完整导出流程 ====================
    
    def export_complete(
        self,
        model_path: str,
        output_dir: str,
        model_name: str = "apt-model",
        register: bool = True
    ) -> Dict[str, str]:
        """
        完整的导出流程
        
        Args:
            model_path: APT模型路径
            output_dir: 输出目录
            model_name: Ollama模型名称
            register: 是否注册到Ollama
            
        Returns:
            导出结果(包含文件路径)
        """
        print(f"🎯 开始完整导出流程...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'gguf_path': None,
            'modelfile_path': None,
            'registered': False
        }
        
        try:
            # 1. 转换为GGUF
            gguf_path = os.path.join(output_dir, f"{model_name}.gguf")
            results['gguf_path'] = self.export_to_gguf(
                model_path, gguf_path
            )
            
            # 2. 创建Modelfile
            results['modelfile_path'] = self.create_modelfile(
                results['gguf_path']
            )
            
            # 3. 注册到Ollama
            if register:
                results['registered'] = self.register_to_ollama(
                    results['modelfile_path'],
                    model_name
                )
            
            print("\n" + "="*50)
            print("✅ 导出完成!")
            print("="*50)
            print(f"GGUF文件: {results['gguf_path']}")
            print(f"Modelfile: {results['modelfile_path']}")
            if results['registered']:
                print(f"Ollama模型: {model_name}")
                print(f"\n使用: ollama run {model_name}")
            
            return results
            
        except Exception as e:
            print(f"❌ 导出失败: {e}")
            raise
    
    # ==================== 工具方法 ====================
    
    def _get_file_type(self, quant: str) -> int:
        """获取GGUF文件类型"""
        file_types = {
            'F32': 0,
            'F16': 1,
            'Q4_0': 2,
            'Q4_1': 3,
            'Q5_0': 6,
            'Q5_1': 7,
            'Q8_0': 8,
            'Q8_1': 9,
            'Q4_K_S': 12,
            'Q4_K_M': 13,
            'Q5_K_S': 14,
            'Q5_K_M': 15,
        }
        return file_types.get(quant, 1)
    
    def _infer_embedding_dim(self, model: Dict) -> int:
        """推断嵌入维度"""
        # 从模型权重中推断
        for name, tensor in model.items():
            if 'embed' in name.lower():
                return tensor.shape[-1]
        return 768  # 默认
    
    def _infer_layer_count(self, model: Dict) -> int:
        """推断层数"""
        layer_nums = set()
        for name in model.keys():
            if 'layer' in name.lower() or 'block' in name.lower():
                # 提取层编号
                parts = name.split('.')
                for part in parts:
                    if part.isdigit():
                        layer_nums.add(int(part))
        return len(layer_nums) if layer_nums else 12  # 默认
    
    def _write_metadata(self, f, metadata: Dict):
        """写入GGUF元数据"""
        # 元数据数量
        f.write(struct.pack('Q', len(metadata)))
        
        for key, value in metadata.items():
            # 键
            key_bytes = key.encode('utf-8')
            f.write(struct.pack('Q', len(key_bytes)))
            f.write(key_bytes)
            
            # 值(根据类型)
            if isinstance(value, str):
                f.write(struct.pack('I', 8))  # 字符串类型
                value_bytes = value.encode('utf-8')
                f.write(struct.pack('Q', len(value_bytes)))
                f.write(value_bytes)
            elif isinstance(value, int):
                f.write(struct.pack('I', 4))  # uint32类型
                f.write(struct.pack('I', value))
            elif isinstance(value, float):
                f.write(struct.pack('I', 6))  # float32类型
                f.write(struct.pack('f', value))
    
    def _write_tensor(self, f, name: str, data: bytes, quant_type: str):
        """写入张量数据"""
        # 张量名称
        name_bytes = name.encode('utf-8')
        f.write(struct.pack('Q', len(name_bytes)))
        f.write(name_bytes)
        
        # 张量类型
        f.write(struct.pack('I', self._get_file_type(quant_type)))
        
        # 张量数据
        f.write(struct.pack('Q', len(data)))
        f.write(data)
    
    # ==================== 插件钩子 ====================
    
    def on_training_end(self, context: Dict[str, Any]):
        """训练结束后自动导出"""
        if self.config.get('auto_export', False):
            model_path = context.get('checkpoint_path')
            output_dir = self.config.get('output_dir', './ollama_export')
            
            self.export_complete(
                model_path,
                output_dir,
                register=self.config.get('auto_register', False)
            )


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 配置
    config = {
        'quantization': 'Q4_K_M',  # Q4_0, Q4_K_M, Q5_K_M, Q8_0
        'context_length': 2048,
        'temperature': 0.7,
        'auto_export': True,
        'auto_register': False,
        'output_dir': './ollama_models'
    }
    
    plugin = OllamaExportPlugin(config)
    
    # 完整导出流程
    results = plugin.export_complete(
        model_path="./trained_model",
        output_dir="./ollama_export",
        model_name="apt-chinese",
        register=True
    )
    
    # 测试模型
    if results['registered']:
        plugin.test_model("apt-chinese", "你好,介绍一下你自己")
    
    print("\n✅ Ollama导出插件示例完成!")
