# AI Photo Enhancer

Local desktop application for removing reflections from photos taken through glass, powered by [WindowSeat](https://github.com/huawei-bayerlab/windowseat-reflection-removal) (Huawei BayerLab, 2025).

## Requirements

- **Windows 10/11** (or Linux)
- **NVIDIA GPU** with 12GB+ VRAM (16GB+ recommended)
- **Python 3.10+**
- **HuggingFace account** (for model download)

## Quick Start (Windows)

1. **Setup** — run `setup.bat` (creates venv, installs dependencies)
2. **Set HF token** — `set HF_TOKEN=hf_your_token_here`
3. **Run** — run `run.bat` (opens browser UI at http://127.0.0.1:7860)

## Features

- **Single image** — upload/drag an image, get reflection-free result
- **Batch processing** — point to a folder, process all images
- **Adjustable parameters:**
  - Tiling mode (short-edge vs fixed resolution)
  - 4-bit quantization (reduces VRAM from ~24GB to ~12GB)
  - Output format (PNG/JPEG/WebP)

## Architecture

```
app.py              — Main entry point (Gradio UI + inference)
run_server.py       — REST API server (FastAPI, for programmatic use)
backend/
  config.py         — Configuration and parameter defaults
  engine.py         — Reflection removal inference engine (sequential offload)
  api.py            — FastAPI endpoints
setup.bat           — Windows setup script
run.bat             — Windows launch script
requirements.txt    — Python dependencies
```

## API Usage

For programmatic access, run the FastAPI server:

```bash
python run_server.py
```

Then call `POST /process` with:
```json
{
  "input_path": "C:\\Photos\\input",
  "output_dir": "C:\\Photos\\output",
  "params": {"use_4bit": true, "output_format": "png"}
}
```

## Model

Uses **Qwen-Image-Edit-2509** as base model with the **reflection removal LoRA** adapter. Model weights are downloaded automatically from HuggingFace on first run (~10GB).
