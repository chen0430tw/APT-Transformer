#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT模型图形界面启动器
让用户可以通过双击文件启动，无需打开终端
"""

import sys
import os
import subprocess
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading

class APTLauncher:
    """APT模型启动器GUI"""

    def __init__(self, root):
        self.root = root
        self.root.title("APT Transformer 启动器")
        self.root.geometry("800x600")
        self.root.resizable(True, True)

        # 获取项目根目录
        self.project_root = os.path.dirname(os.path.abspath(__file__))

        # 当前运行的进程
        self.current_process = None

        self._create_ui()

    def _create_ui(self):
        """创建用户界面"""

        # 主标题
        title_frame = tk.Frame(self.root, bg="#2C3E50", height=80)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)

        title_label = tk.Label(
            title_frame,
            text="APT Transformer",
            font=("Arial", 24, "bold"),
            bg="#2C3E50",
            fg="white"
        )
        title_label.pack(pady=20)

        # 创建notebook（标签页）
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 各个功能标签页
        self._create_train_tab()
        self._create_finetune_tab()
        self._create_chat_tab()
        self._create_evaluate_tab()
        self._create_tools_tab()

        # 底部状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            bg="#ECF0F1"
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _create_train_tab(self):
        """创建训练标签页"""
        train_frame = ttk.Frame(self.notebook)
        self.notebook.add(train_frame, text="训练模型")

        # 参数配置区域
        param_frame = ttk.LabelFrame(train_frame, text="训练参数", padding=10)
        param_frame.pack(fill=tk.X, padx=10, pady=10)

        # Epochs
        ttk.Label(param_frame, text="训练轮数 (Epochs):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.train_epochs = ttk.Entry(param_frame, width=20)
        self.train_epochs.insert(0, "10")
        self.train_epochs.grid(row=0, column=1, sticky=tk.W, pady=5, padx=5)

        # Batch Size
        ttk.Label(param_frame, text="批次大小 (Batch Size):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.train_batch_size = ttk.Entry(param_frame, width=20)
        self.train_batch_size.insert(0, "8")
        self.train_batch_size.grid(row=1, column=1, sticky=tk.W, pady=5, padx=5)

        # Learning Rate
        ttk.Label(param_frame, text="学习率 (Learning Rate):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.train_lr = ttk.Entry(param_frame, width=20)
        self.train_lr.insert(0, "0.0001")
        self.train_lr.grid(row=2, column=1, sticky=tk.W, pady=5, padx=5)

        # Save Path
        ttk.Label(param_frame, text="保存路径:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.train_save_path = ttk.Entry(param_frame, width=30)
        self.train_save_path.insert(0, "apt_model")
        self.train_save_path.grid(row=3, column=1, sticky=tk.W, pady=5, padx=5)

        # 按钮
        btn_frame = ttk.Frame(train_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(
            btn_frame,
            text="开始训练",
            command=self._run_training
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="停止",
            command=self._stop_process
        ).pack(side=tk.LEFT, padx=5)

        # 输出日志
        log_frame = ttk.LabelFrame(train_frame, text="训练日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.train_log = scrolledtext.ScrolledText(log_frame, height=15, wrap=tk.WORD)
        self.train_log.pack(fill=tk.BOTH, expand=True)

    def _create_finetune_tab(self):
        """创建微调标签页"""
        finetune_frame = ttk.Frame(self.notebook)
        self.notebook.add(finetune_frame, text="微调模型")

        # 参数配置
        param_frame = ttk.LabelFrame(finetune_frame, text="微调参数", padding=10)
        param_frame.pack(fill=tk.X, padx=10, pady=10)

        # Model Path
        ttk.Label(param_frame, text="预训练模型路径:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.ft_model_path = ttk.Entry(param_frame, width=30)
        self.ft_model_path.insert(0, "apt_model")
        self.ft_model_path.grid(row=0, column=1, sticky=tk.W, pady=5, padx=5)
        ttk.Button(param_frame, text="浏览", command=lambda: self._browse_dir(self.ft_model_path)).grid(row=0, column=2, padx=5)

        # Data Path
        ttk.Label(param_frame, text="训练数据路径:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.ft_data_path = ttk.Entry(param_frame, width=30)
        self.ft_data_path.insert(0, "finetune_data.txt")
        self.ft_data_path.grid(row=1, column=1, sticky=tk.W, pady=5, padx=5)
        ttk.Button(param_frame, text="浏览", command=lambda: self._browse_file(self.ft_data_path)).grid(row=1, column=2, padx=5)

        # Epochs
        ttk.Label(param_frame, text="训练轮数:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.ft_epochs = ttk.Entry(param_frame, width=20)
        self.ft_epochs.insert(0, "5")
        self.ft_epochs.grid(row=2, column=1, sticky=tk.W, pady=5, padx=5)

        # Learning Rate
        ttk.Label(param_frame, text="学习率:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.ft_lr = ttk.Entry(param_frame, width=20)
        self.ft_lr.insert(0, "1e-5")
        self.ft_lr.grid(row=3, column=1, sticky=tk.W, pady=5, padx=5)

        # Freeze options
        self.ft_freeze_emb = tk.BooleanVar()
        ttk.Checkbutton(param_frame, text="冻结Embedding层", variable=self.ft_freeze_emb).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=5)

        # Save Path
        ttk.Label(param_frame, text="保存路径:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.ft_save_path = ttk.Entry(param_frame, width=30)
        self.ft_save_path.insert(0, "apt_model_finetuned")
        self.ft_save_path.grid(row=5, column=1, sticky=tk.W, pady=5, padx=5)

        # 按钮
        btn_frame = ttk.Frame(finetune_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(btn_frame, text="开始微调", command=self._run_finetuning).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="停止", command=self._stop_process).pack(side=tk.LEFT, padx=5)

        # 日志
        log_frame = ttk.LabelFrame(finetune_frame, text="微调日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.ft_log = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD)
        self.ft_log.pack(fill=tk.BOTH, expand=True)

    def _create_chat_tab(self):
        """创建聊天标签页"""
        chat_frame = ttk.Frame(self.notebook)
        self.notebook.add(chat_frame, text="对话测试")

        # 模型选择
        model_frame = ttk.LabelFrame(chat_frame, text="模型配置", padding=10)
        model_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(model_frame, text="模型路径:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.chat_model_path = ttk.Entry(model_frame, width=30)
        self.chat_model_path.insert(0, "apt_model")
        self.chat_model_path.grid(row=0, column=1, sticky=tk.W, pady=5, padx=5)
        ttk.Button(model_frame, text="浏览", command=lambda: self._browse_dir(self.chat_model_path)).grid(row=0, column=2, padx=5)

        ttk.Button(model_frame, text="启动聊天界面", command=self._run_chat).grid(row=1, column=0, columnspan=3, pady=10)

        # 说明文本
        info_text = """
聊天模式说明:
1. 点击"启动聊天界面"会在终端中打开交互式对话
2. 在对话中输入文本，模型会生成回复
3. 输入 'quit' 或 'exit' 退出对话
4. 支持多轮对话，模型会记住上下文
        """

        info_label = tk.Label(chat_frame, text=info_text, justify=tk.LEFT, bg="#ECF0F1", padx=10, pady=10)
        info_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _create_evaluate_tab(self):
        """创建评估标签页"""
        eval_frame = ttk.Frame(self.notebook)
        self.notebook.add(eval_frame, text="模型评估")

        # 参数配置
        param_frame = ttk.LabelFrame(eval_frame, text="评估参数", padding=10)
        param_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(param_frame, text="模型路径:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.eval_model_path = ttk.Entry(param_frame, width=30)
        self.eval_model_path.insert(0, "apt_model")
        self.eval_model_path.grid(row=0, column=1, sticky=tk.W, pady=5, padx=5)
        ttk.Button(param_frame, text="浏览", command=lambda: self._browse_dir(self.eval_model_path)).grid(row=0, column=2, padx=5)

        # 按钮
        btn_frame = ttk.Frame(eval_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(btn_frame, text="开始评估", command=self._run_evaluation).pack(side=tk.LEFT, padx=5)

        # 日志
        log_frame = ttk.LabelFrame(eval_frame, text="评估结果", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.eval_log = scrolledtext.ScrolledText(log_frame, height=15, wrap=tk.WORD)
        self.eval_log.pack(fill=tk.BOTH, expand=True)

    def _create_tools_tab(self):
        """创建工具标签页"""
        tools_frame = ttk.Frame(self.notebook)
        self.notebook.add(tools_frame, text="工具")

        # Optuna超参数优化
        optuna_frame = ttk.LabelFrame(tools_frame, text="Optuna超参数优化", padding=10)
        optuna_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(optuna_frame, text="快速测试 (10试验, 3轮):").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Button(optuna_frame, text="运行快速测试", command=self._run_optuna_quick).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(optuna_frame, text="深度优化 (100试验, 10轮):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Button(optuna_frame, text="运行深度优化", command=self._run_optuna_full).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # Debug模式
        debug_frame = ttk.LabelFrame(tools_frame, text="Debug模式", padding=10)
        debug_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(debug_frame, text="开启Debug模式", command=lambda: self._run_command(["python", "-m", "apt_model", "config", "--set-debug", "on"])).pack(side=tk.LEFT, padx=5)
        ttk.Button(debug_frame, text="关闭Debug模式", command=lambda: self._run_command(["python", "-m", "apt_model", "config", "--set-debug", "off"])).pack(side=tk.LEFT, padx=5)

        # 工具日志
        log_frame = ttk.LabelFrame(tools_frame, text="工具日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.tools_log = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD)
        self.tools_log.pack(fill=tk.BOTH, expand=True)

    def _browse_dir(self, entry_widget):
        """浏览选择目录"""
        dirname = filedialog.askdirectory(initialdir=self.project_root)
        if dirname:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, dirname)

    def _browse_file(self, entry_widget):
        """浏览选择文件"""
        filename = filedialog.askopenfilename(initialdir=self.project_root)
        if filename:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, filename)

    def _run_training(self):
        """运行训练"""
        epochs = self.train_epochs.get()
        batch_size = self.train_batch_size.get()
        lr = self.train_lr.get()
        save_path = self.train_save_path.get()

        cmd = [
            sys.executable, "-m", "apt_model", "train",
            "--epochs", epochs,
            "--batch-size", batch_size,
            "--learning-rate", lr,
            "--save-path", save_path
        ]

        self._run_command_async(cmd, self.train_log)

    def _run_finetuning(self):
        """运行微调"""
        model_path = self.ft_model_path.get()
        data_path = self.ft_data_path.get()
        epochs = self.ft_epochs.get()
        lr = self.ft_lr.get()
        save_path = self.ft_save_path.get()

        cmd = [
            sys.executable, "-m", "apt_model", "fine-tune",
            "--model-path", model_path,
            "--data-path", data_path,
            "--epochs", epochs,
            "--learning-rate", lr,
            "--save-path", save_path
        ]

        if self.ft_freeze_emb.get():
            cmd.append("--freeze-embeddings")

        self._run_command_async(cmd, self.ft_log)

    def _run_chat(self):
        """运行聊天"""
        model_path = self.chat_model_path.get()

        cmd = [
            sys.executable, "-m", "apt_model", "chat",
            "--model-path", model_path
        ]

        # 聊天需要在新终端中运行
        self._run_in_terminal(cmd)

    def _run_evaluation(self):
        """运行评估"""
        model_path = self.eval_model_path.get()

        cmd = [
            sys.executable, "-m", "apt_model", "evaluate",
            "--model-path", model_path
        ]

        self._run_command_async(cmd, self.eval_log)

    def _run_optuna_quick(self):
        """运行Optuna快速测试"""
        script_path = os.path.join(self.project_root, "run_optuna_quick_test.sh")

        if os.path.exists(script_path):
            self._run_in_terminal(["bash", script_path])
        else:
            messagebox.showerror("错误", f"脚本不存在: {script_path}")

    def _run_optuna_full(self):
        """运行Optuna完整优化"""
        script_path = os.path.join(self.project_root, "run_optuna_optimization.sh")

        if os.path.exists(script_path):
            result = messagebox.askyesno(
                "确认",
                "完整优化将运行100次试验，预计需要5-10小时。\n是否继续？"
            )
            if result:
                self._run_in_terminal(["bash", script_path])
        else:
            messagebox.showerror("错误", f"脚本不存在: {script_path}")

    def _run_command(self, cmd):
        """同步运行命令"""
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            messagebox.showinfo("完成", result.stdout or "命令执行完成")
        except Exception as e:
            messagebox.showerror("错误", f"执行失败: {str(e)}")

    def _run_command_async(self, cmd, log_widget):
        """异步运行命令"""
        def run():
            try:
                log_widget.delete(1.0, tk.END)
                log_widget.insert(tk.END, f"执行命令: {' '.join(cmd)}\n\n")
                log_widget.see(tk.END)

                self.status_var.set("运行中...")

                self.current_process = subprocess.Popen(
                    cmd,
                    cwd=self.project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )

                for line in self.current_process.stdout:
                    log_widget.insert(tk.END, line)
                    log_widget.see(tk.END)
                    self.root.update()

                self.current_process.wait()

                if self.current_process.returncode == 0:
                    log_widget.insert(tk.END, "\n\n✅ 执行完成\n")
                    self.status_var.set("完成")
                else:
                    log_widget.insert(tk.END, f"\n\n❌ 执行失败 (退出码: {self.current_process.returncode})\n")
                    self.status_var.set("失败")

                log_widget.see(tk.END)

            except Exception as e:
                log_widget.insert(tk.END, f"\n\n❌ 错误: {str(e)}\n")
                log_widget.see(tk.END)
                self.status_var.set("错误")

            finally:
                self.current_process = None

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    def _run_in_terminal(self, cmd):
        """在新终端中运行命令"""
        try:
            if sys.platform == "win32":
                # Windows
                subprocess.Popen(
                    ["cmd", "/c", "start", "cmd", "/k"] + cmd,
                    cwd=self.project_root
                )
            elif sys.platform == "darwin":
                # macOS
                script = f'cd "{self.project_root}" && {" ".join(cmd)}'
                subprocess.Popen(
                    ["osascript", "-e", f'tell app "Terminal" to do script "{script}"']
                )
            else:
                # Linux
                subprocess.Popen(
                    ["x-terminal-emulator", "-e"] + cmd,
                    cwd=self.project_root
                )

            self.status_var.set("已在新终端中启动")

        except Exception as e:
            messagebox.showerror("错误", f"无法启动终端: {str(e)}")

    def _stop_process(self):
        """停止当前进程"""
        if self.current_process:
            self.current_process.terminate()
            self.status_var.set("已停止")
            messagebox.showinfo("提示", "进程已停止")
        else:
            messagebox.showwarning("提示", "没有正在运行的进程")


def main():
    """主函数"""
    root = tk.Tk()
    app = APTLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
