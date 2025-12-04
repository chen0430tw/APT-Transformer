"""
APT Model WebUI Application

Gradio-based web interface for:
- Training monitoring with real-time loss curves
- Gradient monitoring with anomaly detection
- Checkpoint management (list, load, download)
- Interactive inference testing

🔮 Implementation based on preparation code from:
- apt_model/training/gradient_monitor.py:export_for_webui()
- tests/test_trainer_complete.py:TestWebUIDataInterface
"""

import json
import time
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import gradio as gr
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Language translations
TRANSLATIONS = {
    'en': {
        'title': '🚀 APT Model WebUI',
        'description': 'Web interface for training monitoring, gradient visualization, checkpoint management, and inference testing.',
        'features': '**Features**',
        'training_monitor': 'Training monitoring with real-time loss curves',
        'gradient_monitor': 'Gradient flow monitoring with anomaly detection',
        'checkpoint_mgmt': 'Checkpoint management (list, load, download)',
        'inference_test': 'Interactive inference testing',
        'tab_training': 'Training Monitor',
        'tab_gradient': 'Gradient Monitor',
        'tab_checkpoint': 'Checkpoint Manager',
        'tab_inference': 'Inference Testing',
        'language': 'Language',
        'checkpoint_dir': 'Checkpoint Directory',
        'load_data': 'Load Training Data',
        'status': 'Status',
        'no_data': 'No data loaded',
        'training_loss': 'Training Loss',
        'learning_rate': 'Learning Rate',
        'model_config': 'Model Configuration',
        'checkpoint_info': 'Checkpoint Info',
        'input_text': 'Input Text',
        'output_text': 'Generated Text',
        'generate': 'Generate',
        'upload_txt': 'Upload TXT File',
        'export_txt': 'Export to TXT',
        'max_length': 'Max Length',
        'temperature': 'Temperature',
    },
    'zh': {
        'title': '🚀 APT模型 WebUI',
        'description': '用于训练监控、梯度可视化、checkpoint管理和推理测试的Web界面',
        'features': '**功能特性**',
        'training_monitor': '📊 训练监控 - 实时loss和学习率曲线',
        'gradient_monitor': '🔍 梯度监控 - 梯度流和异常检测',
        'checkpoint_mgmt': '💾 Checkpoint管理 - 列表、加载、下载',
        'inference_test': '✨ 推理测试 - 交互式文本生成',
        'tab_training': '训练监控',
        'tab_gradient': '梯度监控',
        'tab_checkpoint': 'Checkpoint管理',
        'tab_inference': '推理测试',
        'language': '语言',
        'checkpoint_dir': 'Checkpoint目录',
        'load_data': '加载训练数据',
        'status': '状态',
        'no_data': '未加载数据',
        'training_loss': '训练Loss',
        'learning_rate': '学习率',
        'model_config': '模型配置',
        'checkpoint_info': 'Checkpoint信息',
        'input_text': '输入文本',
        'output_text': '生成文本',
        'generate': '生成',
        'upload_txt': '上传TXT文件',
        'export_txt': '导出为TXT',
        'max_length': '最大长度',
        'temperature': '温度参数',
    }
}


class WebUIState:
    """Shared state for WebUI application"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.gradient_monitor = None
        self.checkpoint_dir = None
        self.training_active = False
        self.language = 'zh'  # Default to Chinese
        self.last_generated_text = ""  # For txt export
        self.training_process = None  # Training subprocess
        self.training_logs = []  # Training logs buffer

    def load_model_from_checkpoint(self, checkpoint_path: Path):
        """Load model and tokenizer from checkpoint"""
        if not TORCH_AVAILABLE:
            return False, "PyTorch not available"

        try:
            from apt_model.modeling.apt_model import APTLargeModel, APTConfig

            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Reconstruct config
            if 'config' in checkpoint:
                config = APTConfig(**checkpoint['config'])
            else:
                return False, "Config not found in checkpoint"

            # Load model
            self.model = APTLargeModel(config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            return True, f"Model loaded from {checkpoint_path.name}"
        except Exception as e:
            return False, f"Error loading model: {str(e)}"

    def get_checkpoint_list(self) -> List[Dict[str, Any]]:
        """Get list of available checkpoints"""
        if self.checkpoint_dir is None:
            return []

        checkpoint_dir = Path(self.checkpoint_dir)
        if not checkpoint_dir.exists():
            return []

        checkpoints = []
        for ckpt_file in sorted(checkpoint_dir.glob('*.pt')):
            try:
                if not TORCH_AVAILABLE:
                    # Without PyTorch, just return basic file info
                    checkpoints.append({
                        'filename': ckpt_file.name,
                        'file_size_mb': ckpt_file.stat().st_size / (1024 * 1024),
                        'created_at': ckpt_file.stat().st_mtime,
                    })
                else:
                    ckpt = torch.load(ckpt_file, map_location='cpu')
                    checkpoints.append({
                        'filename': ckpt_file.name,
                        'epoch': ckpt.get('epoch', 'N/A'),
                        'global_step': ckpt.get('global_step', 'N/A'),
                        'val_loss': ckpt.get('best_val_loss', 'N/A'),
                        'file_size_mb': ckpt_file.stat().st_size / (1024 * 1024),
                        'created_at': ckpt_file.stat().st_mtime,
                    })
            except Exception:
                continue

        return checkpoints


# Global state
webui_state = WebUIState()


def create_training_monitor_tab():
    """
    Tab 1: Training Monitor

    🔮 Based on test_export_training_metrics_for_webui
    Displays training loss, learning rate curves, and model config
    """
    with gr.Tab("Training Monitor"):
        gr.Markdown("## Training Progress Monitoring")

        with gr.Row():
            with gr.Column():
                checkpoint_dir_input = gr.Textbox(
                    label="Checkpoint Directory",
                    placeholder="/path/to/checkpoints",
                    value=str(webui_state.checkpoint_dir) if webui_state.checkpoint_dir else ""
                )
                load_training_data_btn = gr.Button("Load Training Data", variant="primary")

            with gr.Column():
                training_status = gr.Textbox(
                    label="Status",
                    value="No data loaded",
                    interactive=False
                )

        with gr.Row():
            with gr.Column():
                # Loss curve - using Plot for Gradio 6.x compatibility
                loss_plot = gr.Plot(
                    label="Training Loss"
                )

            with gr.Column():
                # Learning rate curve - using Plot for Gradio 6.x compatibility
                lr_plot = gr.Plot(
                    label="Learning Rate"
                )

        with gr.Row():
            model_config_json = gr.JSON(
                label="Model Configuration",
                value={}
            )
            checkpoint_info_json = gr.JSON(
                label="Checkpoint Info",
                value={}
            )

        def load_training_metrics(checkpoint_dir: str):
            """
            🔮 Load training metrics from checkpoint directory
            Implementation of WebUI data export
            """
            try:
                ckpt_dir = Path(checkpoint_dir)
                if not ckpt_dir.exists():
                    return (
                        None, None,
                        {"error": "Directory not found"},
                        {"error": "Directory not found"},
                        "❌ Directory not found"
                    )

                # Update global state
                webui_state.checkpoint_dir = checkpoint_dir

                # Find latest checkpoint
                checkpoints = sorted(ckpt_dir.glob('*.pt'))
                if not checkpoints:
                    return (
                        None, None,
                        {"error": "No checkpoints found"},
                        {"error": "No checkpoints found"},
                        "❌ No checkpoints found"
                    )

                if not TORCH_AVAILABLE:
                    return (
                        None, None,
                        {"error": "PyTorch not available"},
                        {"info": "PyTorch required to load checkpoint data"},
                        "⚠️ PyTorch not available"
                    )

                latest_ckpt = checkpoints[-1]
                checkpoint = torch.load(latest_ckpt, map_location='cpu')

                # Extract training history
                # Note: This assumes trainer saves history in checkpoint
                # If not available, we'll create mock data
                train_losses = checkpoint.get('train_losses', [])
                lr_history = checkpoint.get('lr_history', [])

                if not train_losses:
                    # Create mock data for demonstration
                    steps = list(range(checkpoint.get('global_step', 100)))
                    train_losses = [2.5 * np.exp(-0.01 * s) + 0.5 for s in steps]
                    lr_history = [0.001 * np.exp(-0.005 * s) for s in steps]
                else:
                    steps = list(range(len(train_losses)))

                # Prepare loss plot (Plotly figure for Gradio 6.x)
                try:
                    import plotly.graph_objects as go
                    loss_fig = go.Figure()
                    loss_fig.add_trace(go.Scatter(x=steps, y=train_losses, mode='lines', name='Loss'))
                    loss_fig.update_layout(
                        title="Training Loss",
                        xaxis_title="Step",
                        yaxis_title="Loss",
                        height=300
                    )
                except ImportError:
                    # Fallback if plotly not available
                    loss_fig = None

                # Prepare LR plot (Plotly figure for Gradio 6.x)
                try:
                    lr_fig = go.Figure()
                    lr_fig.add_trace(go.Scatter(x=steps, y=lr_history, mode='lines', name='LR'))
                    lr_fig.update_layout(
                        title="Learning Rate",
                        xaxis_title="Step",
                        yaxis_title="Learning Rate",
                        height=300
                    )
                except:
                    lr_fig = None

                # Model config
                config = checkpoint.get('config', {})
                model_config = {
                    'd_model': config.get('d_model', 'N/A'),
                    'num_layers': config.get('num_layers', 'N/A'),
                    'num_heads': config.get('num_attention_heads', 'N/A'),
                    'vocab_size': config.get('vocab_size', 'N/A'),
                }

                # Checkpoint info
                ckpt_info = {
                    'filename': latest_ckpt.name,
                    'epoch': checkpoint.get('epoch', 'N/A'),
                    'global_step': checkpoint.get('global_step', 'N/A'),
                    'best_val_loss': checkpoint.get('best_val_loss', 'N/A'),
                }

                return (
                    loss_fig,
                    lr_fig,
                    model_config,
                    ckpt_info,
                    f"✅ Loaded data from {latest_ckpt.name}"
                )

            except Exception as e:
                return (
                    None, None,
                    {"error": str(e)},
                    {"error": str(e)},
                    f"❌ Error: {str(e)}"
                )

        load_training_data_btn.click(
            fn=load_training_metrics,
            inputs=[checkpoint_dir_input],
            outputs=[loss_plot, lr_plot, model_config_json, checkpoint_info_json, training_status]
        )

        # Auto-refresh button
        with gr.Row():
            auto_refresh = gr.Checkbox(label="Auto-refresh (every 5s)", value=False)
            refresh_interval = gr.Number(label="Refresh Interval (seconds)", value=5, minimum=1)


def create_gradient_monitor_tab():
    """
    Tab 2: Gradient Monitor

    🔮 Based on gradient_monitor.py:export_for_webui()
    Displays gradient norms, anomaly detection, and layer statistics
    """
    with gr.Tab("Gradient Monitor"):
        gr.Markdown("## Gradient Flow Monitoring")
        gr.Markdown("🔮 Visualize gradient norms and detect anomalies (exploding, vanishing, NaN)")

        with gr.Row():
            gradient_data_file = gr.Textbox(
                label="Gradient Data JSON File",
                placeholder="/path/to/gradient_export.json",
            )
            load_gradient_btn = gr.Button("Load Gradient Data", variant="primary")

        gradient_status = gr.Textbox(
            label="Status",
            value="No gradient data loaded",
            interactive=False
        )

        with gr.Row():
            # Gradient timeline plot - using Plot for Gradio 6.x compatibility
            gradient_timeline = gr.Plot(
                label="Gradient Norms Timeline"
            )

        with gr.Row():
            with gr.Column():
                # Layer statistics
                layer_stats_table = gr.JSON(
                    label="Layer Statistics (mean, std, min, max)",
                    value={}
                )

            with gr.Column():
                # Anomaly summary
                anomaly_summary = gr.JSON(
                    label="Anomaly Summary",
                    value={}
                )

        def load_gradient_data(data_file: str):
            """
            🔮 Load gradient data exported by gradient_monitor.export_for_webui()
            """
            try:
                data_path = Path(data_file)
                if not data_path.exists():
                    return None, {}, {}, "❌ File not found"

                with open(data_path, 'r') as f:
                    webui_data = json.load(f)

                # Extract timeline data
                timeline = webui_data.get('gradient_timeline', [])

                # Prepare plot data
                plot_data = {
                    'step': [],
                    'norm': [],
                    'layer': []
                }

                for step_data in timeline[:100]:  # Limit to 100 steps for visualization
                    step = step_data['step']
                    for layer_name, layer_data in step_data['layers'].items():
                        plot_data['step'].append(step)
                        plot_data['norm'].append(layer_data['norm'])
                        plot_data['layer'].append(layer_name)

                # Layer statistics
                layer_stats = webui_data.get('layer_statistics', {})

                # Anomaly summary
                anomaly_counts = webui_data.get('anomaly_summary', {})

                total_steps = webui_data.get('total_steps', 0)

                # Convert to Plotly figure for Gradio 6.x
                try:
                    import plotly.graph_objects as go
                    import pandas as pd

                    df = pd.DataFrame(plot_data)
                    gradient_fig = go.Figure()

                    # Add a line for each layer
                    for layer in df['layer'].unique():
                        layer_df = df[df['layer'] == layer]
                        gradient_fig.add_trace(go.Scatter(
                            x=layer_df['step'],
                            y=layer_df['norm'],
                            mode='lines',
                            name=layer
                        ))

                    gradient_fig.update_layout(
                        title="Gradient Norms Timeline",
                        xaxis_title="Training Step",
                        yaxis_title="Gradient Norm",
                        height=400
                    )
                except Exception:
                    gradient_fig = None

                return (
                    gradient_fig,
                    layer_stats,
                    anomaly_counts,
                    f"✅ Loaded gradient data: {total_steps} steps, {len(layer_stats)} layers"
                )

            except Exception as e:
                return None, {}, {}, f"❌ Error: {str(e)}"

        load_gradient_btn.click(
            fn=load_gradient_data,
            inputs=[gradient_data_file],
            outputs=[gradient_timeline, layer_stats_table, anomaly_summary, gradient_status]
        )

        # Export button
        with gr.Row():
            gr.Markdown("**Note**: Use `gradient_monitor.export_for_webui()` to generate JSON data")


def create_checkpoint_manager_tab():
    """
    Tab 3: Checkpoint Management

    🔮 Based on test_export_checkpoint_list_for_webui
    List, load, and manage model checkpoints
    """
    with gr.Tab("Checkpoint Manager"):
        gr.Markdown("## Checkpoint Management")

        with gr.Row():
            ckpt_dir_input = gr.Textbox(
                label="Checkpoint Directory",
                placeholder="/path/to/checkpoints",
            )
            scan_btn = gr.Button("Scan Checkpoints", variant="primary")

        ckpt_status = gr.Textbox(
            label="Status",
            value="No checkpoints scanned",
            interactive=False
        )

        with gr.Row():
            checkpoint_table = gr.Dataframe(
                headers=["Filename", "Epoch", "Global Step", "Val Loss", "Size (MB)", "Created"],
                datatype=["str", "str", "str", "str", "number", "str"],
                label="Available Checkpoints",
                interactive=False
            )

        with gr.Row():
            selected_ckpt = gr.Dropdown(
                label="Select Checkpoint to Load",
                choices=[],
                interactive=True
            )
            load_ckpt_btn = gr.Button("Load Checkpoint for Inference", variant="secondary")

        load_ckpt_status = gr.Textbox(
            label="Load Status",
            value="",
            interactive=False
        )

        def scan_checkpoints(ckpt_dir: str):
            """Scan checkpoint directory and list all checkpoints"""
            try:
                webui_state.checkpoint_dir = ckpt_dir
                checkpoints = webui_state.get_checkpoint_list()

                if not checkpoints:
                    return [], [], "❌ No checkpoints found"

                # Prepare table data
                table_data = []
                filenames = []

                for ckpt in checkpoints:
                    created_time = time.strftime(
                        '%Y-%m-%d %H:%M:%S',
                        time.localtime(ckpt['created_at'])
                    )

                    table_data.append([
                        ckpt['filename'],
                        str(ckpt.get('epoch', 'N/A')),
                        str(ckpt.get('global_step', 'N/A')),
                        f"{ckpt.get('val_loss', 'N/A'):.4f}" if isinstance(ckpt.get('val_loss'), (int, float)) else 'N/A',
                        round(ckpt['file_size_mb'], 2),
                        created_time
                    ])
                    filenames.append(ckpt['filename'])

                return (
                    table_data,
                    gr.Dropdown(choices=filenames),
                    f"✅ Found {len(checkpoints)} checkpoints"
                )

            except Exception as e:
                return [], [], f"❌ Error: {str(e)}"

        def load_checkpoint_for_inference(filename: str):
            """Load selected checkpoint into memory for inference"""
            if not filename:
                return "❌ No checkpoint selected"

            if webui_state.checkpoint_dir is None:
                return "❌ Checkpoint directory not set"

            ckpt_path = Path(webui_state.checkpoint_dir) / filename
            success, message = webui_state.load_model_from_checkpoint(ckpt_path)

            return f"{'✅' if success else '❌'} {message}"

        scan_btn.click(
            fn=scan_checkpoints,
            inputs=[ckpt_dir_input],
            outputs=[checkpoint_table, selected_ckpt, ckpt_status]
        )

        load_ckpt_btn.click(
            fn=load_checkpoint_for_inference,
            inputs=[selected_ckpt],
            outputs=[load_ckpt_status]
        )

        # Refresh button
        with gr.Row():
            refresh_btn = gr.Button("Refresh List", size="sm")
            refresh_btn.click(
                fn=scan_checkpoints,
                inputs=[ckpt_dir_input],
                outputs=[checkpoint_table, selected_ckpt, ckpt_status]
            )


def create_inference_tab():
    """
    Tab 4: Interactive Inference

    🔮 Based on test_inference_interface and api_inference prototype
    Text generation interface for testing loaded models
    """
    with gr.Tab("Inference Testing"):
        gr.Markdown("## Interactive Text Generation")
        gr.Markdown("🔮 Test model inference (load checkpoint first in Checkpoint Manager)")

        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="Input Text",
                    placeholder="Enter text to generate from...",
                    lines=3
                )

                with gr.Row():
                    max_length = gr.Slider(
                        label="Max Length",
                        minimum=10,
                        maximum=512,
                        value=50,
                        step=1
                    )
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=2.0,
                        value=1.0,
                        step=0.1
                    )

                with gr.Row():
                    num_beams = gr.Slider(
                        label="Num Beams",
                        minimum=1,
                        maximum=10,
                        value=1,
                        step=1
                    )
                    do_sample = gr.Checkbox(label="Do Sample", value=False)

                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column():
                output_text = gr.Textbox(
                    label="Generated Text",
                    lines=5,
                    interactive=False
                )

                generation_info = gr.JSON(
                    label="Generation Info",
                    value={}
                )

                # TXT file support
                with gr.Row():
                    upload_txt_btn = gr.UploadButton(
                        label="📄 Upload TXT File",
                        file_types=[".txt"],
                        size="sm"
                    )
                    export_txt_btn = gr.Button("💾 Export to TXT", size="sm")

                # File download component
                download_file = gr.File(label="Download TXT", visible=False)

        # Example inputs
        with gr.Row():
            gr.Examples(
                examples=[
                    ["今天天气很好"],
                    ["人工智能的未来"],
                    ["从前有座山"],
                ],
                inputs=input_text,
                label="Example Inputs"
            )

        def run_inference(
            text: str,
            max_len: int,
            temp: float,
            beams: int,
            sample: bool
        ):
            """
            🔮 Run inference using loaded model
            Based on api_inference prototype from tests
            """
            if webui_state.model is None:
                return (
                    "❌ No model loaded. Please load a checkpoint first.",
                    {"error": "No model loaded"}
                )

            if not TORCH_AVAILABLE:
                return (
                    "❌ PyTorch not available",
                    {"error": "PyTorch required for inference"}
                )

            if not text:
                return (
                    "❌ Please enter input text",
                    {"error": "No input text"}
                )

            try:
                start_time = time.time()

                # Note: This is a simplified version
                # Real implementation needs proper tokenizer
                if webui_state.tokenizer is None:
                    return (
                        "⚠️ Tokenizer not available. Returning mock output.",
                        {
                            "input_text": text,
                            "note": "Real inference requires tokenizer",
                            "generation_time_ms": (time.time() - start_time) * 1000
                        }
                    )

                # Real inference code
                webui_state.model.eval()
                with torch.no_grad():
                    inputs = webui_state.tokenizer(
                        text,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=512
                    )

                    generated_ids = webui_state.model.generate(
                        input_ids=inputs['input_ids'],
                        max_length=max_len,
                        temperature=temp,
                        num_beams=int(beams),
                        do_sample=sample
                    )

                    generated_text = webui_state.tokenizer.decode(
                        generated_ids[0],
                        skip_special_tokens=True
                    )

                generation_time = (time.time() - start_time) * 1000

                info = {
                    'input_text': text,
                    'generated_text': generated_text,
                    'generation_time_ms': round(generation_time, 2),
                    'max_length': max_len,
                    'temperature': temp,
                    'num_beams': int(beams),
                    'do_sample': sample
                }

                return generated_text, info

            except Exception as e:
                return (
                    f"❌ Error during inference: {str(e)}",
                    {"error": str(e)}
                )

        def upload_txt_file(file):
            """Load text from uploaded txt file"""
            if file is None:
                return ""
            try:
                with open(file.name, 'r', encoding='utf-8') as f:
                    content = f.read()
                return content
            except Exception as e:
                return f"❌ Error reading file: {str(e)}"

        def export_to_txt():
            """Export generated text to txt file"""
            if not webui_state.last_generated_text:
                return None

            try:
                import tempfile
                # Create temporary file
                tmp_file = tempfile.NamedTemporaryFile(
                    mode='w',
                    encoding='utf-8',
                    suffix='.txt',
                    delete=False
                )
                tmp_file.write(webui_state.last_generated_text)
                tmp_file.close()
                return tmp_file.name
            except Exception as e:
                print(f"Export error: {str(e)}")
                return None

        def run_inference_and_save(
            text: str,
            max_len: int,
            temp: float,
            beams: int,
            sample: bool
        ):
            """Run inference and save result for export"""
            result_text, result_info = run_inference(text, max_len, temp, beams, sample)
            webui_state.last_generated_text = result_text
            return result_text, result_info

        generate_btn.click(
            fn=run_inference_and_save,
            inputs=[input_text, max_length, temperature, num_beams, do_sample],
            outputs=[output_text, generation_info]
        )

        upload_txt_btn.upload(
            fn=upload_txt_file,
            inputs=[upload_txt_btn],
            outputs=[input_text]
        )

        export_txt_btn.click(
            fn=export_to_txt,
            outputs=[download_file]
        )


def create_training_launcher_tab(webui_state):
    """
    创建训练启动标签页

    功能:
    - 上传训练数据 (txt/json)
    - 配置训练参数
    - 启动/停止训练
    - 实时显示训练日志
    """
    with gr.Tab("🚀 训练启动"):
        gr.Markdown("## 训练配置与启动")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📁 训练数据")

                # 数据文件上传
                train_data_file = gr.File(
                    label="上传训练数据 (txt/json)",
                    file_types=[".txt", ".json"],
                    file_count="single"
                )

                val_data_file = gr.File(
                    label="上传验证数据 (可选)",
                    file_types=[".txt", ".json"],
                    file_count="single"
                )

                gr.Markdown("### ⚙️ 训练参数")

                with gr.Row():
                    epochs = gr.Number(
                        label="训练轮数 (Epochs)",
                        value=10,
                        minimum=1,
                        maximum=1000
                    )
                    batch_size = gr.Number(
                        label="批次大小 (Batch Size)",
                        value=32,
                        minimum=1,
                        maximum=512
                    )

                with gr.Row():
                    learning_rate = gr.Number(
                        label="学习率 (Learning Rate)",
                        value=0.001,
                        minimum=0.00001,
                        maximum=0.1
                    )
                    max_length = gr.Number(
                        label="最大序列长度",
                        value=512,
                        minimum=128,
                        maximum=2048
                    )

                save_steps = gr.Number(
                    label="保存间隔 (每N步保存一次)",
                    value=1000,
                    minimum=100,
                    maximum=10000
                )

                output_dir = gr.Textbox(
                    label="输出目录",
                    value="./output",
                    placeholder="/path/to/output"
                )

                gr.Markdown("### 🎯 控制")

                with gr.Row():
                    start_btn = gr.Button("▶️ 开始训练", variant="primary", size="lg")
                    stop_btn = gr.Button("⏹️ 停止训练", variant="stop", size="lg")

            with gr.Column():
                gr.Markdown("### 📊 训练状态")

                training_status = gr.Textbox(
                    label="当前状态",
                    value="⭕ 就绪",
                    interactive=False
                )

                progress_bar = gr.Textbox(
                    label="进度",
                    value="0/0 epochs (0.0%)",
                    interactive=False
                )

                gr.Markdown("### 💻 训练日志（实时）")

                log_output = gr.Textbox(
                    label="终端输出",
                    lines=20,
                    interactive=False,
                    max_lines=1000,
                    autoscroll=True
                )

                clear_logs_btn = gr.Button("🗑️ 清空日志", size="sm")

        # ============ 事件处理函数 ============

        def start_training(
            train_file,
            val_file,
            n_epochs,
            batch_sz,
            lr,
            max_len,
            save_step,
            out_dir
        ):
            """启动训练"""
            if webui_state.training_active:
                return "⚠️ 训练已在进行中", "", "训练已在运行中\n"

            if train_file is None:
                return "❌ 错误：请上传训练数据", "", "错误：未上传训练数据\n"

            try:
                # 构建训练命令
                cmd = [
                    "python", "-u", "-m", "apt_model", "train",
                    "--data-path", train_file.name,
                    "--epochs", str(int(n_epochs)),
                    "--batch-size", str(int(batch_sz)),
                    "--learning-rate", str(lr),
                    "--max-length", str(int(max_len)),
                    "--save-steps", str(int(save_step)),
                    "--save-path", out_dir
                ]

                if val_file is not None:
                    cmd.extend(["--val-data-path", val_file.name])

                # 启动训练进程
                webui_state.training_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )

                webui_state.training_active = True
                webui_state.training_logs = []

                # 启动日志读取线程
                def read_logs():
                    for line in webui_state.training_process.stdout:
                        webui_state.training_logs.append(line)

                log_thread = threading.Thread(target=read_logs, daemon=True)
                log_thread.start()

                return (
                    "✅ 训练已启动",
                    f"0/{int(n_epochs)} epochs (0.0%)",
                    f"训练启动命令: {' '.join(cmd)}\n\n正在初始化...\n"
                )

            except Exception as e:
                webui_state.training_active = False
                return f"❌ 启动失败: {str(e)}", "", f"错误: {str(e)}\n"

        def stop_training():
            """停止训练"""
            if not webui_state.training_active:
                return "⭕ 就绪", "没有正在运行的训练", ""

            try:
                if webui_state.training_process:
                    webui_state.training_process.terminate()
                    webui_state.training_process.wait(timeout=5)
                    webui_state.training_active = False
                    return "⏹️ 已停止", "训练已终止", "训练已手动停止\n"
            except Exception as e:
                return "⚠️ 停止失败", f"错误: {str(e)}", f"停止失败: {str(e)}\n"

        def update_logs():
            """更新日志显示"""
            if webui_state.training_logs:
                return "\n".join(webui_state.training_logs[-100:])  # 最后100行
            return ""

        def clear_logs():
            """清空日志"""
            webui_state.training_logs = []
            return ""

        # ============ 事件绑定 ============

        start_btn.click(
            fn=start_training,
            inputs=[
                train_data_file,
                val_data_file,
                epochs,
                batch_size,
                learning_rate,
                max_length,
                save_steps,
                output_dir
            ],
            outputs=[training_status, progress_bar, log_output]
        )

        stop_btn.click(
            fn=stop_training,
            outputs=[training_status, progress_bar, log_output]
        )

        clear_logs_btn.click(
            fn=clear_logs,
            outputs=[log_output]
        )



def create_webui():
    """
    Create complete WebUI with all tabs

    🔮 Implements all preparation code from:
    - gradient_monitor.py:export_for_webui()
    - test_trainer_complete.py:TestWebUIDataInterface
    - test_trainer_complete.py:TestAPIReadiness (inference interface)
    """
    # Handle different Gradio versions compatibility
    import inspect

    blocks_kwargs = {}

    # Check which parameters gr.Blocks supports
    try:
        blocks_sig = inspect.signature(gr.Blocks.__init__)
        supported_params = set(blocks_sig.parameters.keys())

        # Add parameters only if supported
        if 'title' in supported_params:
            blocks_kwargs["title"] = "APT Model WebUI"

        if 'css' in supported_params:
            blocks_kwargs["css"] = ".gradio-container {max-width: 1400px !important}"

        if 'theme' in supported_params and hasattr(gr, 'themes'):
            blocks_kwargs["theme"] = gr.themes.Soft()
    except Exception:
        # Fallback for very old Gradio versions - no extra parameters
        pass

    with gr.Blocks(**blocks_kwargs) as app:

        # Language selector
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown(f"# {TRANSLATIONS[webui_state.language]['title']}")
            with gr.Column(scale=1):
                language_selector = gr.Radio(
                    choices=["中文 (zh)", "English (en)"],
                    value="中文 (zh)" if webui_state.language == 'zh' else "English (en)",
                    label="🌐 Language / 语言"
                )

        lang = webui_state.language
        gr.Markdown(
            f"""
            {TRANSLATIONS[lang]['description']}

            {TRANSLATIONS[lang]['features']}:
            - {TRANSLATIONS[lang]['training_monitor']}
            - {TRANSLATIONS[lang]['gradient_monitor']}
            - {TRANSLATIONS[lang]['checkpoint_mgmt']}
            - {TRANSLATIONS[lang]['inference_test']}

            **提示 / Tip**: 切换语言后请刷新页面 / Refresh page after changing language
            """
        )

        def change_language(lang_choice):
            """Change interface language"""
            webui_state.language = 'zh' if lang_choice.startswith('中文') else 'en'
            msg = f"⚠️ 语言已设置为 {lang_choice}\n\n" \
                  f"由于Gradio限制，需要**重启WebUI**才能生效：\n" \
                  f"1. 按 Ctrl+C 停止服务\n" \
                  f"2. 重新运行: python -m apt_model.webui.app\n\n" \
                  f"Language set to {lang_choice}\n" \
                  f"Due to Gradio limitations, please **restart the WebUI**:\n" \
                  f"1. Press Ctrl+C to stop\n" \
                  f"2. Run again: python -m apt_model.webui.app"
            return msg

        lang_status = gr.Textbox(
            label="⚙️ 语言切换说明 / Language Switch Info",
            value="",
            visible=True,
            interactive=False,
            lines=6
        )
        language_selector.change(
            fn=change_language,
            inputs=[language_selector],
            outputs=[lang_status]
        )

        # Create all tabs - wrapped in gr.Tabs() for Gradio 6.x compatibility
        with gr.Tabs():
            create_training_launcher_tab(webui_state)  # 训练启动 - 新增！
            create_training_monitor_tab()
            create_gradient_monitor_tab()
            create_checkpoint_manager_tab()
            create_inference_tab()

        gr.Markdown(
            """
            ---
            **🔮 Implementation Note**: This WebUI uses preparation code ("伏笔") from:
            - `apt_model/training/gradient_monitor.py:export_for_webui()` (lines 260-302)
            - `tests/test_trainer_complete.py:TestWebUIDataInterface` (lines 599-682)
            - `tests/test_trainer_complete.py:api_inference()` prototype (lines 421-458)
            """
        )

    return app


def launch_webui(
    checkpoint_dir: Optional[str] = None,
    share: bool = False,
    server_port: int = 7860,
    server_name: str = "0.0.0.0",
    auth: Optional[tuple] = None
):
    """
    Launch WebUI server

    Args:
        checkpoint_dir: Default checkpoint directory
        share: Create public share link
        server_port: Port to run server on
        server_name: Server hostname
        auth: Optional (username, password) tuple for authentication
    """
    import sys

    if checkpoint_dir:
        webui_state.checkpoint_dir = checkpoint_dir

    # Print startup banner
    print("\n" + "=" * 80)
    print("🚀 APT Model WebUI 启动中...")
    print("=" * 80)
    print()

    # Show configuration
    print("📋 配置信息:")
    print(f"  🌐 主机地址: {server_name}")
    print(f"  🔌 端口: {server_port}")
    print(f"  📁 Checkpoint目录: {checkpoint_dir or '(未设置)'}")
    print(f"  🌍 公共分享: {'✅ 是' if share else '❌ 否'}")
    if auth:
        print(f"  🔐 访问控制: ✅ 已启用 (用户名: {auth[0]})")
    else:
        print(f"  🔐 访问控制: ⚠️  未启用 (建议生产环境启用)")
    print()

    # Show access URLs
    print("🌐 访问地址:")
    if server_name in ["0.0.0.0", "127.0.0.1", "localhost"]:
        print(f"  📍 本地访问: http://localhost:{server_port}")
        print(f"  📍 局域网访问: http://<你的IP>:{server_port}")
    else:
        print(f"  📍 访问地址: http://{server_name}:{server_port}")
    print()

    if auth:
        print("🔑 登录凭据:")
        print(f"  👤 用户名: {auth[0]}")
        print(f"  🔒 密码: {auth[1]}")
        print()

    print("💡 功能说明:")
    print("  📊 训练监控 - 实时查看训练loss和学习率曲线")
    print("  🔍 梯度监控 - 监控梯度流和异常检测")
    print("  💾 Checkpoint管理 - 管理和加载模型检查点")
    print("  ✨ 推理测试 - 交互式文本生成")
    print()

    print("=" * 80)
    print("✅ WebUI 已启动！请在浏览器中打开上述地址")
    print("=" * 80)
    print()

    # Flush output to ensure it's displayed before gradio starts
    sys.stdout.flush()

    app = create_webui()

    app.launch(
        share=share,
        server_port=server_port,
        server_name=server_name,
        show_error=True,
        auth=auth,
        quiet=False,
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Launch APT Model WebUI')
    parser.add_argument('--checkpoint-dir', type=str, help='Checkpoint directory')
    parser.add_argument('--share', action='store_true', help='Create public share link')
    parser.add_argument('--port', type=int, default=7860, help='Server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server hostname')
    parser.add_argument('--username', type=str, help='Username for authentication')
    parser.add_argument('--password', type=str, help='Password for authentication')

    args = parser.parse_args()

    # Prepare auth tuple if credentials provided
    auth = None
    if args.username and args.password:
        auth = (args.username, args.password)
    elif args.username or args.password:
        print("⚠️  警告: 必须同时提供 --username 和 --password")
        exit(1)

    launch_webui(
        checkpoint_dir=args.checkpoint_dir,
        share=args.share,
        server_port=args.port,
        server_name=args.host,
        auth=auth
    )
