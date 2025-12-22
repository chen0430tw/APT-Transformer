# Data Directory

This directory contains datasets for APT-Transformer training.

## Available Datasets

- `HLBD_Hardcore_Full.json` - HLBD Hardcore dataset (575 samples)
  - 5 modules: geometry, arithmetic, zodiac, physics, reverse_english
  - Designed to prevent shortcut learning

## Dataset Format

All datasets use JSON format with the following structure:

```json
{
  "metadata": {
    "total_samples": 575,
    "modules": ["geometry", "arithmetic", "zodiac", "physics", "reverse_english"]
  },
  "data": {
    "module_name": [
      {"input": "...", "output": "..."},
      ...
    ]
  }
}
```

## Generating New Datasets

Use the dataset generation tool:

```bash
python tools/generate_hlbd_hardcore.py
```
