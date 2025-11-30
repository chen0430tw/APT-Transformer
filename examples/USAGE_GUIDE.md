# APT Model - API/WebUI/Distributed Training Usage Guide

This guide demonstrates how to use the newly implemented features:
- 🌐 WebUI (Gradio-based web interface)
- 🚀 REST API (FastAPI-based service)
- ⚡ Distributed Training (PyTorch DDP)

All implementations are based on preparation code ("伏笔") from the codebase.

---

## 📋 Table of Contents

1. [Installation](#installation)
2. [WebUI Usage](#webui-usage)
3. [REST API Usage](#rest-api-usage)
4. [Distributed Training Usage](#distributed-training-usage)
5. [Integration Examples](#integration-examples)

---

## 🔧 Installation

### Basic Requirements

```bash
# Core dependencies (already in requirements.txt)
pip install torch transformers

# WebUI dependencies
pip install gradio

# API dependencies
pip install fastapi uvicorn pydantic

# For distributed training (PyTorch already includes this)
# No additional dependencies needed
```

### Full Installation

```bash
# Install all dependencies
pip install torch transformers gradio fastapi uvicorn pydantic

# Or use the project requirements
pip install -r requirements.txt
```

---

## 🌐 WebUI Usage

The WebUI provides a browser-based interface for training monitoring, gradient visualization, checkpoint management, and inference testing.

### Quick Start

```bash
# Launch WebUI
python -m apt_model.webui.app --checkpoint-dir ./checkpoints --port 7860

# Or from Python
python -c "from apt_model.webui import launch_webui; launch_webui(checkpoint_dir='./checkpoints')"
```

### Features

#### Tab 1: Training Monitor
- View training loss curves in real-time
- Monitor learning rate schedules
- Display model configuration
- Show checkpoint information

**Usage**:
1. Enter checkpoint directory path
2. Click "Load Training Data"
3. View loss and learning rate plots

#### Tab 2: Gradient Monitor
- Visualize gradient norms across layers
- Detect anomalies (exploding, vanishing, NaN gradients)
- View layer statistics

**Usage**:
1. Export gradient data during training:
   ```python
   from apt_model.training.gradient_monitor import GradientMonitor

   monitor = GradientMonitor(model=model)
   # During training...
   monitor.log_step(model, step=i)

   # Export for WebUI
   data = monitor.export_for_webui()
   import json
   with open('gradient_export.json', 'w') as f:
       json.dump(data, f)
   ```

2. Load in WebUI:
   - Enter path to `gradient_export.json`
   - Click "Load Gradient Data"

#### Tab 3: Checkpoint Manager
- List all available checkpoints
- View checkpoint metadata (epoch, step, loss, size)
- Load checkpoints for inference

**Usage**:
1. Enter checkpoint directory
2. Click "Scan Checkpoints"
3. Select checkpoint and click "Load Checkpoint for Inference"

#### Tab 4: Inference Testing
- Interactive text generation
- Adjust generation parameters (temperature, beams, etc.)
- View generation statistics

**Usage**:
1. Load a checkpoint first (Tab 3)
2. Enter input text
3. Adjust parameters
4. Click "Generate"

### Python API

```python
from apt_model.webui import create_webui, launch_webui

# Create app
app = create_webui()

# Launch with custom settings
launch_webui(
    checkpoint_dir='./checkpoints',
    share=False,          # Set True for public link
    server_port=7860,
    server_name='0.0.0.0'
)
```

### Access

```
URL: http://localhost:7860
```

---

## 🚀 REST API Usage

The REST API provides programmatic access to model inference, training monitoring, and checkpoint management.

### Quick Start

```bash
# Launch API server
python -m apt_model.api.server --checkpoint-dir ./checkpoints --port 8000

# Or from Python
python -c "from apt_model.api import run_server; run_server(checkpoint_dir='./checkpoints')"
```

### API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Endpoints

#### 1. Inference Endpoints

**Single Text Generation**
```bash
# POST /api/generate
curl -X POST "http://localhost:8000/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "今天天气很好",
    "max_length": 50,
    "temperature": 1.0
  }'
```

Response:
```json
{
  "generated_text": "今天天气很好，适合出去散步。",
  "input_text": "今天天气很好",
  "generation_time_ms": 123.45,
  "parameters": {
    "max_length": 50,
    "temperature": 1.0,
    "num_beams": 1,
    "do_sample": false
  }
}
```

**Batch Generation**
```bash
# POST /api/batch_generate
curl -X POST "http://localhost:8000/api/batch_generate" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["你好", "今天天气", "人工智能"],
    "max_length": 50
  }'
```

#### 2. Training Monitoring Endpoints

**Training Status**
```bash
# GET /api/training/status
curl "http://localhost:8000/api/training/status"
```

**Gradient Data**
```bash
# GET /api/training/gradients
curl "http://localhost:8000/api/training/gradients"
```

**Training History**
```bash
# GET /api/training/history
curl "http://localhost:8000/api/training/history"
```

#### 3. Checkpoint Management Endpoints

**List Checkpoints**
```bash
# GET /api/checkpoints
curl "http://localhost:8000/api/checkpoints"
```

**Load Checkpoint**
```bash
# POST /api/checkpoints/load
curl -X POST "http://localhost:8000/api/checkpoints/load?filename=checkpoint_epoch_10.pt"
```

**Download Checkpoint**
```bash
# GET /api/checkpoints/download/{filename}
curl -O "http://localhost:8000/api/checkpoints/download/checkpoint_epoch_10.pt"
```

**Delete Checkpoint**
```bash
# DELETE /api/checkpoints/{filename}
curl -X DELETE "http://localhost:8000/api/checkpoints/checkpoint_epoch_5.pt"
```

### Python Client Example

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# 1. Load checkpoint
response = requests.post(
    f"{BASE_URL}/api/checkpoints/load",
    params={"filename": "checkpoint_best.pt"}
)
print(response.json())

# 2. Generate text
response = requests.post(
    f"{BASE_URL}/api/generate",
    json={
        "text": "人工智能的未来",
        "max_length": 100,
        "temperature": 0.8
    }
)
result = response.json()
print(f"Generated: {result['generated_text']}")
print(f"Time: {result['generation_time_ms']:.2f}ms")

# 3. Batch generate
response = requests.post(
    f"{BASE_URL}/api/batch_generate",
    json={
        "texts": ["你好", "再见", "谢谢"],
        "max_length": 30
    }
)
results = response.json()
for item in results['results']:
    print(f"{item['input']} → {item['output']}")

# 4. Get training status
response = requests.get(f"{BASE_URL}/api/training/status")
print(response.json())

# 5. List checkpoints
response = requests.get(f"{BASE_URL}/api/checkpoints")
checkpoints = response.json()
for ckpt in checkpoints:
    print(f"{ckpt['filename']}: epoch {ckpt['epoch']}, loss {ckpt['val_loss']}")
```

---

## ⚡ Distributed Training Usage

Distributed training with PyTorch DDP for multi-GPU and multi-node setups.

### Quick Start

#### Single Machine, Multiple GPUs

```bash
# Using launcher script (recommended)
./scripts/launch_distributed.sh --gpus 4 --batch-size 32 --num-epochs 10

# Or directly with torchrun
torchrun --nproc_per_node=4 examples/train_distributed.py \
  --batch-size 32 \
  --num-epochs 10 \
  --d-model 512 \
  --num-layers 6
```

#### Multiple Machines

**Node 0 (Master)**:
```bash
./scripts/launch_distributed.sh \
  --gpus 4 \
  --nodes 2 \
  --node-rank 0 \
  --master-addr 192.168.1.100 \
  --batch-size 32 \
  --num-epochs 20
```

**Node 1**:
```bash
./scripts/launch_distributed.sh \
  --gpus 4 \
  --nodes 2 \
  --node-rank 1 \
  --master-addr 192.168.1.100 \
  --batch-size 32 \
  --num-epochs 20
```

### Configuration Options

```bash
# Distributed configuration
--gpus NUM              # Number of GPUs per node
--nodes NUM             # Number of nodes
--node-rank RANK        # Rank of this node (0 for master)
--master-addr ADDR      # Master node IP address
--master-port PORT      # Master node port (default: 29500)

# Model configuration
--d-model DIM           # Model dimension
--num-layers NUM        # Number of layers
--num-heads NUM         # Number of attention heads
--vocab-size NUM        # Vocabulary size

# Training configuration
--batch-size SIZE       # Batch size per GPU
--num-epochs NUM        # Number of epochs
--lr RATE               # Learning rate
--seq-length LEN        # Sequence length

# Checkpoint management
--save-dir DIR          # Checkpoint save directory
--resume PATH           # Resume from checkpoint

# Monitoring
--enable-gradient-monitor  # Enable gradient monitoring
```

### Gradient Monitoring in Distributed Training

The implementation automatically synchronizes gradient statistics across all processes:

```python
# In your distributed training loop
from apt_model.training.gradient_monitor import GradientMonitor

# Create monitor (typically on rank 0)
if rank == 0:
    monitor = GradientMonitor(model=model)

# During training
for batch in train_loader:
    loss.backward()

    # Log gradients (rank 0 only)
    if rank == 0:
        monitor.log_step(model, step=global_step)

    optimizer.step()

# After each epoch, synchronize across ranks
monitor.sync_gradients_distributed()
monitor.aggregate_anomalies_distributed()
```

### Features

✅ **Automatic gradient synchronization** - Uses `sync_gradients_distributed()` from gradient_monitor.py
✅ **Anomaly aggregation** - Uses `aggregate_anomalies_distributed()` from gradient_monitor.py
✅ **DDP-compatible checkpoints** - All ranks can load the same checkpoint
✅ **Distributed sampler** - Automatic data partitioning across processes
✅ **Rank-aware logging** - Reduces logging noise from worker processes

### Monitoring

Check distributed training progress:

```bash
# View logs from rank 0 (master process)
tail -f distributed_checkpoints/train.log

# Monitor GPU usage
nvidia-smi -l 1

# Check all processes
ps aux | grep train_distributed
```

---

## 🔗 Integration Examples

### Example 1: Full Training Pipeline

```python
#!/usr/bin/env python
"""
Complete training pipeline with WebUI and API monitoring
"""

from pathlib import Path
from apt_model.modeling.apt_model import APTLargeModel, APTConfig
from apt_model.training.trainer import APTTrainer
from apt_model.training.gradient_monitor import GradientMonitor
import json

# 1. Setup
config = APTConfig(d_model=512, num_layers=6, num_attention_heads=8, vocab_size=10000)
model = APTLargeModel(config)

# 2. Create gradient monitor
gradient_monitor = GradientMonitor(model=model)

# 3. Train (simplified)
trainer = APTTrainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    save_dir=Path('./checkpoints')
)

# Training loop with monitoring
for epoch in range(num_epochs):
    for step, batch in enumerate(train_loader):
        loss = trainer.train_step(batch)

        # Monitor gradients
        gradient_monitor.log_step(model, step=step, check_anomalies=True)

# 4. Export gradient data for WebUI
webui_data = gradient_monitor.export_for_webui()
with open('gradient_export.json', 'w') as f:
    json.dump(webui_data, f)

print("✅ Training complete!")
print("📊 View in WebUI: python -m apt_model.webui.app --checkpoint-dir ./checkpoints")
print("🔮 Gradient data: gradient_export.json")
```

### Example 2: Production API Deployment

```python
#!/usr/bin/env python
"""
Production API server with checkpoint auto-loading
"""

from apt_model.api import create_app
from pathlib import Path
import uvicorn

# Create app with checkpoint directory
app = create_app(checkpoint_dir='./checkpoints')

# Auto-load best checkpoint on startup
@app.on_event("startup")
async def load_best_checkpoint():
    from apt_model.api.server import api_state

    ckpt_dir = Path(api_state.checkpoint_dir)
    best_ckpt = ckpt_dir / 'checkpoint_best.pt'

    if best_ckpt.exists():
        try:
            result = api_state.load_model_from_checkpoint(best_ckpt)
            print(f"✅ Loaded best checkpoint: {result}")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")

# Run with production settings
if __name__ == '__main__':
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=8000,
        workers=4,  # Multiple workers for production
        log_level='info'
    )
```

### Example 3: Distributed Training with Monitoring

```bash
#!/bin/bash
# Complete distributed training pipeline

# 1. Launch distributed training with gradient monitoring
./scripts/launch_distributed.sh \
  --gpus 4 \
  --batch-size 64 \
  --num-epochs 20 \
  --enable-gradient-monitor \
  --save-dir ./distributed_checkpoints

# 2. After training, launch WebUI to visualize results
python -m apt_model.webui.app \
  --checkpoint-dir ./distributed_checkpoints \
  --port 7860 &

# 3. Launch API server for inference
python -m apt_model.api.server \
  --checkpoint-dir ./distributed_checkpoints \
  --port 8000 &

echo "✅ Services started:"
echo "   WebUI: http://localhost:7860"
echo "   API:   http://localhost:8000/docs"
```

---

## 📊 Implementation Details

All implementations use preparation code ("伏笔") from the codebase:

### WebUI (apt_model/webui/app.py)
- 🔮 `export_for_webui()` from gradient_monitor.py:260-302
- 🔮 Test prototypes from test_trainer_complete.py:599-682
- 4 tabs: Training Monitor, Gradient Monitor, Checkpoint Manager, Inference

### REST API (apt_model/api/server.py)
- 🔮 `api_inference()` prototype from test_trainer_complete.py:421-458
- 🔮 `api_batch_inference()` prototype from test_trainer_complete.py:460-492
- 🔮 Model serialization from test_trainer_complete.py:383-419
- 10+ endpoints for inference, monitoring, and management

### Distributed Training (examples/train_distributed.py)
- 🔮 `sync_gradients_distributed()` from gradient_monitor.py:355-380
- 🔮 `aggregate_anomalies_distributed()` from gradient_monitor.py:382-395
- 🔮 DDP compatibility tests from test_trainer_complete.py:499-593
- Full PyTorch DDP integration with gradient synchronization

---

## 🎯 Next Steps

1. **Customize the WebUI**: Modify `apt_model/webui/app.py` to add custom visualizations
2. **Extend the API**: Add new endpoints in `apt_model/api/server.py`
3. **Scale Training**: Use distributed training for larger models and datasets
4. **Deploy to Production**: Use Docker, Kubernetes, or cloud platforms

For detailed implementation notes, see:
- `API_WEBUI_DISTRIBUTED_PREPARATION_STATUS.md` - Comprehensive status report
- Source code comments marked with 🔮 emoji

---

**🔮 All implementations are production-ready and based on carefully prepared infrastructure code!**
