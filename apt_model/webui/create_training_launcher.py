"""
训练启动标签页 - Training Launcher Tab
用于启动和控制模型训练
"""
import gradio as gr
import subprocess
import threading
import queue
from pathlib import Path


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
                    encoding='utf-8',  # 明确指定UTF-8编码，避免Windows cp950编码问题
                    bufsize=1
                )

                webui_state.training_active = True
                webui_state.training_logs = []

                # 启动日志读取线程
                def read_logs():
                    try:
                        # 检查 stdout 是否为 None
                        if webui_state.training_process.stdout is None:
                            webui_state.training_logs.append("错误: 无法读取进程输出流\n")
                            return

                        for line in webui_state.training_process.stdout:
                            webui_state.training_logs.append(line)
                    except (BrokenPipeError, ValueError, AttributeError) as e:
                        webui_state.training_logs.append(f"日志读取错误: {e}\n")
                    except Exception as e:
                        webui_state.training_logs.append(f"未知日志读取错误: {e}\n")

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

        # 自动刷新日志 (Gradio 6.x 不支持 every 参数，改用轮询)
        # 注意: Gradio 6.x 移除了 every 参数，需要使用其他方式实现轮询
        # log_output.change(
        #     fn=update_logs,
        #     outputs=[log_output]
        # )
