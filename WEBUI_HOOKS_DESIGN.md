# WebUI接口设计（伏笔）

**日期**: 2025-11-30
**目的**: 为将来的WebUI实现预留接口和钩子
**状态**: 已埋下伏笔，等待实现

---

## 🎯 设计理念

不在现在实现完整WebUI，但在代码架构中**预留好接口**，让将来添加WebUI时：
- ✅ 不需要大规模重构训练代码
- ✅ 可以即插即用
- ✅ 支持多种UI框架（Gradio/Streamlit/Flask等）
- ✅ 解耦训练逻辑和UI逻辑

---

## 📦 已实现的伏笔

### 1. 训练事件系统 (`training_events.py`)

**位置**: `apt_model/training/training_events.py`

**核心组件**:
```python
from apt_model.training.training_events import training_emitter

# 训练器发射事件
training_emitter.emit('batch_end', batch_idx=100, loss=2.5, lr=0.0001)

# WebUI订阅事件
def on_batch_update(event_data):
    # 更新WebUI显示
    update_loss_chart(event_data['loss'])

training_emitter.on('batch_end', on_batch_update)
```

**支持的事件**:
- `training_start` - 训练开始
- `training_end` - 训练结束
- `epoch_start` - Epoch开始
- `epoch_end` - Epoch结束
- `batch_start` - Batch开始
- `batch_end` - Batch结束
- `checkpoint_saved` - Checkpoint保存
- `checkpoint_loaded` - Checkpoint加载
- `metric_update` - 指标更新
- `error_occurred` - 错误发生

---

### 2. WebUI钩子示例类 (`WebUIHooks`)

**作用**: 展示如何订阅训练事件

```python
from apt_model.training.training_events import WebUIHooks, training_emitter

# 创建WebUI钩子
webui_hooks = WebUIHooks()

# 附加到训练事件
webui_hooks.attach(training_emitter)

# 获取当前训练状态（用于WebUI API）
state = webui_hooks.get_current_state()
# {
#     'current_epoch': 3,
#     'total_epochs': 10,
#     'current_batch': 250,
#     'current_loss': 2.5432,
#     'learning_rate': 0.00009,
#     'is_training': True
# }
```

---

### 3. 便捷函数

为常用事件提供快捷发射函数：

```python
from apt_model.training.training_events import (
    emit_training_start,
    emit_epoch_end,
    emit_batch_end,
    emit_checkpoint_saved,
)

# 在训练器中使用
emit_training_start(total_epochs=10)
emit_batch_end(batch_idx=100, loss=2.5, lr=0.0001)
emit_checkpoint_saved(checkpoint_path="./model.pt", epoch=3, step=1500)
```

---

## 🌐 将来的WebUI实现方案

### 方案1: Gradio（最简单）

**特点**:
- 快速原型
- 自动生成界面
- 适合演示和测试

**实现示例**:
```python
# webui/gradio_ui.py
import gradio as gr
from apt_model.training.training_events import training_emitter, WebUIHooks

# 创建钩子
hooks = WebUIHooks()
hooks.attach(training_emitter)

def get_training_status():
    """获取训练状态（Gradio会定期调用）"""
    state = hooks.get_current_state()
    return (
        f"Epoch: {state['current_epoch']}/{state['total_epochs']}",
        f"Loss: {state['current_loss']:.4f}",
        state['is_training']
    )

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# APT模型训练监控")

    with gr.Row():
        epoch_text = gr.Textbox(label="当前Epoch")
        loss_text = gr.Textbox(label="当前Loss")
        status_text = gr.Textbox(label="训练状态")

    # 每秒更新一次
    demo.load(
        get_training_status,
        inputs=None,
        outputs=[epoch_text, loss_text, status_text],
        every=1
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
```

---

### 方案2: Streamlit（最美观）

**特点**:
- 美观的界面
- 丰富的图表组件
- 适合数据展示

**实现示例**:
```python
# webui/streamlit_ui.py
import streamlit as st
import pandas as pd
from apt_model.training.training_events import training_emitter

st.title("APT模型训练监控")

# 创建实时更新的占位符
epoch_placeholder = st.empty()
loss_chart_placeholder = st.empty()
metrics_placeholder = st.empty()

def update_ui():
    """更新UI显示"""
    # 获取最近的事件历史
    batch_history = training_emitter.get_history('batch_end', limit=100)

    if batch_history:
        # 提取loss数据
        losses = [e['loss'] for e in batch_history]
        batches = [e['batch_idx'] for e in batch_history]

        # 更新epoch显示
        latest = batch_history[-1]
        epoch_placeholder.metric("当前Batch", latest['batch_idx'])

        # 更新loss曲线
        df = pd.DataFrame({'Batch': batches, 'Loss': losses})
        loss_chart_placeholder.line_chart(df.set_index('Batch'))

        # 更新指标表格
        metrics_placeholder.dataframe({
            'Loss': [latest['loss']],
            'Learning Rate': [latest['lr']],
        })

# 定期更新（Streamlit会自动重新运行）
import time
while True:
    update_ui()
    time.sleep(1)
```

---

### 方案3: Flask + WebSocket（最灵活）

**特点**:
- 完全自定义
- 实时双向通信
- 适合生产环境

**实现示例**:
```python
# webui/flask_ui.py
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from apt_model.training.training_events import training_emitter

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# 订阅训练事件并转发到WebSocket客户端
def forward_to_websocket(event_data):
    """将训练事件转发到WebSocket"""
    socketio.emit('training_update', event_data)

training_emitter.on('batch_end', forward_to_websocket)
training_emitter.on('epoch_end', forward_to_websocket)
training_emitter.on('checkpoint_saved', forward_to_websocket)

@app.route('/')
def index():
    """WebUI主页"""
    return render_template('training_monitor.html')

@socketio.on('connect')
def handle_connect():
    """客户端连接"""
    # 发送当前训练状态
    history = training_emitter.get_history('batch_end', limit=50)
    emit('history', history)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
```

**前端HTML** (`templates/training_monitor.html`):
```html
<!DOCTYPE html>
<html>
<head>
    <title>APT训练监控</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h1>APT模型训练监控</h1>
    <div id="status">
        <p>Epoch: <span id="epoch">-</span></p>
        <p>Batch: <span id="batch">-</span></p>
        <p>Loss: <span id="loss">-</span></p>
    </div>
    <div id="loss-chart"></div>

    <script>
        const socket = io();
        let losses = [];
        let batches = [];

        socket.on('training_update', (data) => {
            // 更新状态显示
            if (data.event_name === 'batch_end') {
                document.getElementById('batch').textContent = data.batch_idx;
                document.getElementById('loss').textContent = data.loss.toFixed(4);

                // 更新图表
                losses.push(data.loss);
                batches.push(data.batch_idx);
                updateChart();
            }
        });

        function updateChart() {
            const trace = {
                x: batches,
                y: losses,
                type: 'scatter',
                mode: 'lines',
                name: 'Training Loss'
            };
            Plotly.newPlot('loss-chart', [trace]);
        }
    </script>
</body>
</html>
```

---

## 🔌 在训练器中集成事件发射

### 需要在trainer.py中添加：

```python
# 在文件顶部导入
from apt_model.training.training_events import (
    emit_training_start,
    emit_training_end,
    emit_epoch_start,
    emit_epoch_end,
    emit_batch_end,
    emit_checkpoint_saved,
)

# 在train_model函数中

def train_model(...):
    # ... 初始化代码 ...

    # 发射训练开始事件
    emit_training_start(total_epochs=epochs)

    for epoch in range(epochs):
        # 发射epoch开始事件
        emit_epoch_start(epoch=epoch, total_epochs=epochs)

        for i, batch in enumerate(dataloader):
            # ... 训练batch ...

            # 发射batch结束事件
            emit_batch_end(
                batch_idx=i,
                loss=loss_value,
                lr=scheduler.get_last_lr()[0]
            )

        # 发射epoch结束事件
        emit_epoch_end(
            epoch=epoch,
            metrics={'avg_loss': avg_loss}
        )

        # Checkpoint保存时发射事件
        if avg_loss < best_loss:
            checkpoint_path = save_model(...)
            emit_checkpoint_saved(
                checkpoint_path=checkpoint_path,
                epoch=epoch,
                step=global_step
            )

    # 发射训练结束事件
    emit_training_end()
```

**改动量**: 约10-15行代码
**侵入性**: 极低，只需添加事件发射调用
**向后兼容**: 完全兼容，不影响现有功能

---

## 🚀 WebUI启动流程（将来）

### 步骤1: 启动WebUI服务器（独立进程）

```bash
# 启动Gradio WebUI
python webui/gradio_ui.py

# 或启动Flask WebUI
python webui/flask_ui.py
```

### 步骤2: 启动训练（正常流程）

```bash
# 训练会自动发射事件
python -m apt_model train --epochs 10
```

### 步骤3: 在浏览器中查看

```
打开 http://localhost:7860  (Gradio)
或   http://localhost:5000  (Flask)
```

**关键**: 训练和WebUI是**解耦的**，可以分别启动/停止

---

## 📊 WebUI功能设想

### 基础功能
- [ ] 实时显示训练进度（Epoch/Batch）
- [ ] 实时Loss曲线图
- [ ] Learning Rate曲线图
- [ ] 当前训练参数显示

### 高级功能
- [ ] 多个训练任务并行监控
- [ ] Checkpoint列表和管理
- [ ] 模型测试/推理界面
- [ ] 训练暂停/恢复控制
- [ ] GPU/CPU资源监控
- [ ] 训练日志实时查看

### 企业级功能
- [ ] 多用户权限管理
- [ ] 训练历史记录
- [ ] 模型版本对比
- [ ] A/B测试支持
- [ ] 云端训练调度

---

## 🎨 UI设计草图

```
┌─────────────────────────────────────────────────────────┐
│  APT模型训练监控                          [暂停] [停止] │
├─────────────────────────────────────────────────────────┤
│  训练状态: ●运行中                                      │
│  当前Epoch: 3/10  |  当前Batch: 250/500                 │
│  平均Loss: 2.5432  |  Learning Rate: 0.00009            │
├─────────────────────────────────────────────────────────┤
│  Loss曲线                                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │        📈                                        │   │
│  │    3.0 ●                                         │   │
│  │        │  ●                                      │   │
│  │    2.5 │    ●  ●                                 │   │
│  │        │        ● ●●●                            │   │
│  │    2.0 │            ●●●●●                        │   │
│  │        └───────────────────────────────────────  │   │
│  │        0        500       1000      1500         │   │
│  └─────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│  最近Checkpoint:                                        │
│  • epoch3_step1500.pt (2.5GB) - 2分钟前                │
│  • epoch2_step1000.pt (2.5GB) - 15分钟前               │
│  • epoch1_step500.pt (2.5GB) - 30分钟前         [加载] │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ 实施建议

### 立即可做（已完成）✅
1. 创建training_events.py（事件系统）
2. 创建WebUIHooks示例类
3. 编写此文档

### 短期（1-2周）
1. 在trainer.py中集成事件发射
2. 创建简单的Gradio WebUI原型
3. 测试事件系统

### 中期（1-2月）
1. 实现完整的Flask WebUI
2. 添加图表和监控功能
3. 支持训练控制（暂停/恢复）

### 长期（3-6月）
1. 企业级功能
2. 多用户支持
3. 云端集成

---

## 📝 总结

### 已埋下的伏笔 ✅

1. **训练事件系统** - 完整的事件发射/订阅机制
2. **WebUI钩子类** - 展示如何订阅事件
3. **便捷函数** - 简化事件发射
4. **文档说明** - 详细的实现方案

### 优势

- ✅ **解耦**: 训练逻辑和UI逻辑完全分离
- ✅ **灵活**: 支持任意UI框架
- ✅ **即插即用**: 将来添加WebUI无需修改训练核心代码
- ✅ **向后兼容**: 不影响现有功能
- ✅ **低侵入**: 只需在trainer中添加几行事件发射代码

### 下一步

1. 在trainer.py中添加事件发射调用（10-15行代码）
2. 创建Gradio原型验证可行性
3. 逐步完善WebUI功能

---

**结论**: WebUI的"伏笔"已经完美埋下，随时可以扩展成完整的WebUI系统，无需重构训练代码。
