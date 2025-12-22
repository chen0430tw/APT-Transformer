# Tools and Utilities

This directory contains diagnostic, generation, and monitoring tools for APT-Transformer.

## Available Tools

### Diagnostic Tools
- `check_training_backends.py` - Code quality checker for training backends
- `diagnose_issues.py` - System diagnostic tool

### Data Generation
- `generate_hlbd_hardcore.py` - HLBD Hardcore dataset generator

### Monitoring & Visualization
- `monitor_all_trainings.py` - Multi-training monitor
- `visualize_training.py` - Real-time sci-fi style visualization
- `demo_visualization.py` - Visualization demo

### Model Verification
- `verify_hlbd_model.py` - HLBD model verification tool

### Other Utilities
- `test_vocab_size.py` - Vocabulary size tester
- `mascot_render_fused45.py` - Mascot rendering tool

## Usage Examples

```bash
# Check training backend code quality
python tools/check_training_backends.py

# Generate HLBD dataset
python tools/generate_hlbd_hardcore.py

# Verify trained model
python tools/verify_hlbd_model.py --model path/to/model.pt

# Real-time visualization
python tools/visualize_training.py --log-dir training_logs --mode realtime
```
