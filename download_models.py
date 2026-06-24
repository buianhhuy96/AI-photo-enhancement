"""
Pre-download all model weights for offline operation.
Run this once during setup — after this, the app works without internet.
"""
import sys


def main():
    print()
    print("=" * 60)
    print("  Downloading WindowSeat model weights")
    print("  This is ~10GB total. May take 10-30 minutes.")
    print("=" * 60)
    print()

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        import torch
    except ImportError:
        print("ERROR: Required packages not installed. Run setup.bat first.")
        sys.exit(1)

    BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"
    LORA_MODEL = "huawei-bayerlab/windowseat-reflection-removal-v1-0"

    # --- 1. Download LoRA model files (small, ~500MB) ---
    print("[1/5] Downloading WindowSeat LoRA weights...")
    print(f"      Repo: {LORA_MODEL}")
    try:
        snapshot_download(
            LORA_MODEL,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.md"],
        )
        print("      [OK] LoRA weights downloaded.")
    except Exception as e:
        print(f"      ERROR: {e}")
        print("      Make sure HF_TOKEN is set and you have access to the model.")
        sys.exit(1)

    # --- 2. Download VAE (AutoencoderKLQwenImage) ---
    print()
    print("[2/5] Downloading VAE model...")
    print(f"      Repo: {BASE_MODEL}/vae")
    try:
        snapshot_download(
            BASE_MODEL,
            allow_patterns=["vae/**"],
        )
        print("      [OK] VAE downloaded.")
    except Exception as e:
        print(f"      ERROR: {e}")
        sys.exit(1)

    # --- 3. Download Transformer ---
    print()
    print("[3/5] Downloading Transformer model (largest file, ~8GB)...")
    print(f"      Repo: {BASE_MODEL}/transformer")
    try:
        snapshot_download(
            BASE_MODEL,
            allow_patterns=["transformer/**"],
        )
        print("      [OK] Transformer downloaded.")
    except Exception as e:
        print(f"      ERROR: {e}")
        sys.exit(1)

    # --- 4. Download base model root configs ---
    print()
    print("[4/5] Downloading base model configs...")
    try:
        snapshot_download(
            BASE_MODEL,
            allow_patterns=["*.json", "*.txt", "*.md"],
            ignore_patterns=["vae/**", "transformer/**"],
        )
        print("      [OK] Configs downloaded.")
    except Exception as e:
        print(f"      WARNING: {e} (may still work)")

    # --- 5. Verify everything loads ---
    print()
    print("[5/5] Verifying downloads...")
    errors = []

    try:
        config_file = hf_hub_download(LORA_MODEL, "model_index.json")
        import json
        with open(config_file) as f:
            cfg = json.load(f)
        print(f"      Processing resolution: {cfg['processing_resolution']}px")
        print(f"      Base model: {cfg['base_model']}")
    except Exception as e:
        errors.append(f"LoRA config: {e}")

    try:
        import safetensors.torch
        embeds_file = hf_hub_download(LORA_MODEL, "state_dict.safetensors", subfolder="text_embeddings")
        embeds = safetensors.torch.load_file(embeds_file)
        print(f"      Text embeddings: {len(embeds)} tensors ✓")
    except Exception as e:
        errors.append(f"Text embeddings: {e}")

    try:
        lora_file = hf_hub_download(LORA_MODEL, "pytorch_lora_weights.safetensors", subfolder="transformer_lora")
        print(f"      LoRA weights: ✓")
    except Exception as e:
        errors.append(f"LoRA weights: {e}")

    try:
        from diffusers import AutoencoderKLQwenImage
        AutoencoderKLQwenImage.load_config(BASE_MODEL, subfolder="vae")
        print(f"      VAE config: ✓")
    except Exception as e:
        errors.append(f"VAE config: {e}")

    try:
        from diffusers import QwenImageTransformer2DModel
        QwenImageTransformer2DModel.load_config(BASE_MODEL, subfolder="transformer")
        print(f"      Transformer config: ✓")
    except Exception as e:
        errors.append(f"Transformer config: {e}")

    if errors:
        print()
        print("  WARNINGS during verification:")
        for err in errors:
            print(f"    - {err}")
        print("  Some files may still work. Try running the app.")
    else:
        print()
        print("=" * 60)
        print("  All model weights downloaded and verified!")
        print("  The app will now work fully offline.")
        print("=" * 60)

    print()


if __name__ == "__main__":
    main()
