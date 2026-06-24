"""
WindowSeat Reflection Removal Engine.
Sequential offload: loads VAE and Transformer separately to fit in ~12-16GB VRAM.
Based on Qwen-Image-Edit-2509 + WindowSeat LoRA.
"""
import gc
import json
import math
import os
from pathlib import Path
from typing import Callable, Optional

# Set CUDA memory config before any torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import imageio.v2 as imageio
import numpy as np
import safetensors
import torch
import torchvision
from diffusers import (
    AutoencoderKLQwenImage,
    BitsAndBytesConfig,
    QwenImageEditPipeline,
    QwenImageTransformer2DModel,
)
from huggingface_hub import hf_hub_download
from peft import LoraConfig
from PIL import Image

from .config import BASE_MODEL_URI, LORA_MODEL_URI, InferenceParams, SUPPORTED_EXTENSIONS


def flush_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def fetch_state_dict(repo_id: str, weight_name: str, subfolder: str = None):
    file_path = hf_hub_download(repo_id, weight_name, subfolder=subfolder)
    return safetensors.torch.load_file(file_path)


def _match_batch(t: torch.Tensor, B: int) -> torch.Tensor:
    """Match tensor batch dimension to B (expand, repeat, or slice)."""
    if t.size(0) == B:
        return t
    if t.size(0) == 1 and B > 1:
        return t.expand(B, *t.shape[1:])
    if t.size(0) > B:
        return t[:B]
    reps = (B + t.size(0) - 1) // t.size(0)
    return t.repeat((reps,) + (1,) * (t.ndim - 1))[:B]


# ---------- Encoding / Decoding ----------

def encode_single(image_tensor: torch.Tensor, vae) -> torch.Tensor:
    """Encode a single image [3,H,W] normalized to [-1,1]."""
    dev = next(vae.parameters()).device
    dt = next(vae.parameters()).dtype
    image = image_tensor.unsqueeze(0).to(device=dev, dtype=dt)
    out = vae.encode(image.unsqueeze(2)).latent_dist.sample()
    latents_mean = torch.tensor(vae.config.latents_mean, device=out.device, dtype=out.dtype)
    latents_mean = latents_mean.view(1, vae.config.z_dim, 1, 1, 1)
    latents_std_inv = 1.0 / torch.tensor(vae.config.latents_std, device=out.device, dtype=out.dtype)
    latents_std_inv = latents_std_inv.view(1, vae.config.z_dim, 1, 1, 1)
    out = (out - latents_mean) * latents_std_inv
    return out.cpu()


def decode_single(latents: torch.Tensor, vae) -> torch.Tensor:
    """Decode latents back to image."""
    dev = next(vae.parameters()).device
    dt = next(vae.parameters()).dtype
    latents = latents.to(device=dev, dtype=dt)
    latents_mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype)
    latents_mean = latents_mean.view(1, vae.config.z_dim, 1, 1, 1)
    latents_std_inv = 1.0 / torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype)
    latents_std_inv = latents_std_inv.view(1, vae.config.z_dim, 1, 1, 1)
    latents = latents / latents_std_inv + latents_mean
    out = vae.decode(latents)
    out = out.sample[:, :, 0]
    return out.cpu()


# ---------- Transformer step ----------

def flow_step_single(model_input: torch.Tensor, transformer, vae_config: dict,
                     vae_dtype, embeds_dict: dict) -> torch.Tensor:
    """Run single flow matching step with transformer."""
    model_input = model_input.to(next(transformer.parameters()).device)

    prompt_embeds = embeds_dict["prompt_embeds"]
    prompt_mask = embeds_dict["prompt_mask"]
    if prompt_mask.dtype != torch.bool:
        prompt_mask = prompt_mask > 0

    if model_input.ndim == 5 and model_input.shape[2] == 1:
        model_input_4d = model_input[:, :, 0]
    elif model_input.ndim == 4:
        model_input_4d = model_input
    else:
        raise ValueError(f"Unexpected latent shape: {model_input.shape}")

    B, C, H, W = model_input_4d.shape
    device = next(transformer.parameters()).device

    prompt_embeds = prompt_embeds[:B].to(device=device, dtype=torch.bfloat16)
    prompt_mask = prompt_mask[:B].to(device=device, dtype=torch.bool)

    packed_model_input = QwenImageEditPipeline._pack_latents(
        model_input_4d, batch_size=B, num_channels_latents=C, height=H, width=W,
    )
    packed_model_input = packed_model_input.to(torch.bfloat16)

    timestep = torch.full((B,), 499.0 / 1000.0, device=device, dtype=torch.bfloat16)

    h_img = H // 2
    w_img = W // 2
    img_shapes = [[(1, h_img, w_img)]] * B
    txt_seq_lens = prompt_mask.sum(dim=1).tolist()

    attention_kwargs = getattr(transformer, "attention_kwargs", None) or {}

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model_pred = transformer(
            hidden_states=packed_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_mask,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            guidance=None,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]

    temperal_downsample = vae_config.get("temperal_downsample", None)
    vae_scale_factor = 2 ** len(temperal_downsample) if temperal_downsample else 8

    model_pred = QwenImageEditPipeline._unpack_latents(
        model_pred, height=H * vae_scale_factor, width=W * vae_scale_factor,
        vae_scale_factor=vae_scale_factor,
    )

    return (model_input.to(vae_dtype) - model_pred.to(vae_dtype)).cpu()


# ---------- Image utilities ----------

def lanczos_resize_chw(x, out_hw: tuple) -> np.ndarray:
    """Resize [C,H,W] array using Lanczos interpolation."""
    H_out, W_out = int(out_hw[0]), int(out_hw[1])
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().float().numpy()
    else:
        arr = np.asarray(x, dtype=np.float32)
    C = arr.shape[0]
    out = np.empty((C, H_out, W_out), dtype=np.float32)
    for c in range(C):
        img = Image.fromarray(arr[c].astype(np.float32), mode="F")
        img = img.resize((W_out, H_out), resample=Image.LANCZOS)
        out[c] = np.asarray(img, dtype=np.float32)
    return out


# ---------- Tiling ----------

def _starts(size: int, T: int, min_overlap: int) -> list:
    if size <= T:
        return [0]
    stride = max(1, T - min_overlap)
    xs = list(range(0, size - T + 1, stride))
    last = size - T
    if xs[-1] != last:
        xs.append(last)
    return sorted(set(xs))


def _required_side(size: int, nmax: int, min_overlap: int) -> int:
    nmax = max(1, int(nmax))
    if nmax == 1:
        return size
    return math.ceil((size + (nmax - 1) * min_overlap) / nmax)


def compute_tiles(img_w: int, img_h: int, processing_resolution: int,
                  params: InferenceParams) -> list:
    """Compute tile coordinates [(x0, y0, x1, y1), ...] for an image."""
    W, H = img_w, img_h

    if params.use_short_edge_tile:
        tile_size = max(min(W, H), processing_resolution)
    else:
        tile_size = processing_resolution

    # Upscale dimensions if smaller than tile
    if W < tile_size or H < tile_size:
        scale = tile_size / min(W, H)
        W = round(scale * W)
        H = round(scale * H)

    ow, oh = params.min_overlap_w, params.min_overlap_h
    T_low = max(
        _required_side(W, params.max_num_tiles_w, ow),
        _required_side(H, params.max_num_tiles_h, oh),
        ow + 1, oh + 1,
    )
    T_high = min(W, H)
    if T_low > T_high:
        T_low = T_high
    T = max(T_low, min(tile_size, T_high))

    xs = _starts(W, T, ow)
    ys = _starts(H, T, oh)
    return [(x0, y0, x0 + T, y0 + T) for y0 in ys for x0 in xs]


def prepare_tile(img_path: str, tile_info: tuple, processing_resolution: int) -> torch.Tensor:
    """Load image, crop tile, resize to processing resolution. Returns [3,H,W] in [-1,1]."""
    x0, y0, x1, y1 = tile_info
    img = Image.open(img_path).convert("RGB")
    W_orig, H_orig = img.size
    arr = np.array(img, dtype=np.uint8).transpose(2, 0, 1)  # [3, H, W]

    tW, tH = x1 - x0, y1 - y0
    H, W = arr.shape[1], arr.shape[2]

    # Upscale if needed
    if W < tW or H < tH:
        scale = tW / min(W, H)
        arr = lanczos_resize_chw(arr.astype(np.float32), (round(scale * H), round(scale * W)))

    # Normalize to [-1, 1]
    if arr.dtype == np.uint8:
        norm = arr.astype(np.float32) / 255.0 * 2.0 - 1.0
    else:
        norm = arr / 255.0 * 2.0 - 1.0 if arr.max() > 1.0 else arr * 2.0 - 1.0

    # Crop
    norm = norm[:, y0:y1, x0:x1]

    # Resize to processing_resolution
    norm = lanczos_resize_chw(norm, (processing_resolution, processing_resolution))
    return torch.from_numpy(norm).float()


# ---------- Main Engine ----------

class WindowSeatEngine:
    """Manages model loading and inference with sequential offloading."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processing_resolution: Optional[int] = None
        self.embeds_dict: Optional[dict] = None
        self._initialized = False

    def initialize(self):
        """Download configs and text embeddings (lightweight, kept in memory)."""
        if self._initialized:
            return

        # Get processing resolution from LoRA config
        config_file = hf_hub_download(LORA_MODEL_URI, "model_index.json")
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        self.processing_resolution = config_dict["processing_resolution"]

        # Load text embeddings (small tensor, stays in memory)
        self.embeds_dict = fetch_state_dict(
            LORA_MODEL_URI, "state_dict.safetensors", subfolder="text_embeddings"
        )
        self.embeds_dict = {k: v.to(self.device) for k, v in self.embeds_dict.items()}
        self._initialized = True

    def get_processing_resolution(self, params: InferenceParams) -> int:
        """Return effective processing resolution."""
        self.initialize()
        return params.processing_resolution or self.processing_resolution

    def process_image(
        self,
        img_path: str,
        output_path: str,
        params: InferenceParams,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> str:
        """
        Process a single image through the full pipeline.
        Returns the output file path.

        progress_callback(status_message, fraction_0_to_1)
        """
        self.initialize()
        proc_res = self.get_processing_resolution(params)

        def report(msg: str, frac: float):
            if progress_callback:
                progress_callback(msg, frac)

        report("Computing tiles...", 0.0)

        with Image.open(img_path) as im:
            orig_W, orig_H = im.size

        tiles = compute_tiles(orig_W, orig_H, proc_res, params)
        report(f"Tiles: {len(tiles)}", 0.05)

        vae = None
        transformer = None
        try:
            # --- Phase 1: Encode tiles with VAE ---
            report("Loading VAE for encoding...", 0.05)
            print(f"[engine] Phase 1: Loading VAE (float32, no device_map)")
            flush_cuda()
            vae = AutoencoderKLQwenImage.from_pretrained(
                BASE_MODEL_URI, subfolder="vae", torch_dtype=torch.float32,
                low_cpu_mem_usage=True, use_safetensors=True,
            )
            vae = vae.to(self.device)
            vae.eval()
            vae_config = dict(vae.config)
            vae_dtype = vae.dtype
            print(f"[engine] Phase 1: VAE loaded, dtype={vae_dtype}, device={next(vae.parameters()).device}")

            encoded_tiles = []
            for i, tile_info in enumerate(tiles):
                tile_tensor = prepare_tile(img_path, tile_info, proc_res)
                with torch.no_grad():
                    latent = encode_single(tile_tensor, vae)
                encoded_tiles.append(latent)
                flush_cuda()
                report(f"Encoding tile {i+1}/{len(tiles)}", 0.05 + 0.25 * (i + 1) / len(tiles))

            del vae
            vae = None
            flush_cuda()
            print(f"[engine] Phase 1 DONE: {len(encoded_tiles)} tiles encoded")

            # --- Phase 2: Transformer (flow step) ---
            report("Loading transformer...", 0.30)
            flush_cuda()
            if params.use_4bit:
                nf4 = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
                )
                transformer = QwenImageTransformer2DModel.from_pretrained(
                    BASE_MODEL_URI, subfolder="transformer", torch_dtype=torch.bfloat16,
                    quantization_config=nf4, device_map=self.device,
                )
            else:
                transformer = QwenImageTransformer2DModel.from_pretrained(
                    BASE_MODEL_URI, subfolder="transformer", torch_dtype=torch.bfloat16,
                    device_map=self.device,
                )

            # Load LoRA
            lora_config = LoraConfig.from_pretrained(LORA_MODEL_URI, subfolder="transformer_lora")
            transformer.add_adapter(lora_config)
            state_dict = fetch_state_dict(LORA_MODEL_URI, "pytorch_lora_weights.safetensors", subfolder="transformer_lora")
            missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
            if unexpected:
                raise ValueError(f"Unexpected keys in LoRA state dict: {unexpected}")
            transformer.eval()

            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                report(f"Transformer loaded: {alloc:.2f}/{total:.2f} GiB VRAM", 0.33)

            processed_tiles = []
            for i, latent in enumerate(encoded_tiles):
                print(f"[engine] Phase 2: Processing tile {i+1}/{len(encoded_tiles)}")
                with torch.no_grad():
                    output_latent = flow_step_single(latent, transformer, vae_config, vae_dtype, self.embeds_dict)
                processed_tiles.append(output_latent)
                flush_cuda()
                report(f"Transformer tile {i+1}/{len(tiles)}", 0.35 + 0.35 * (i + 1) / len(tiles))

            del transformer
            transformer = None
            flush_cuda()
            print(f"[engine] Phase 2 DONE: {len(processed_tiles)} tiles processed")

            # --- Phase 3: Decode tiles with VAE ---
            report("Loading VAE for decoding...", 0.70)
            print("[engine] Phase 3: Loading VAE for decode (float32, no device_map)")
            flush_cuda()
            vae = AutoencoderKLQwenImage.from_pretrained(
                BASE_MODEL_URI, subfolder="vae", torch_dtype=torch.float32,
                low_cpu_mem_usage=True, use_safetensors=True,
            )
            vae = vae.to(self.device)
            vae.eval()

            decoded_tiles = []
            for i, latent in enumerate(processed_tiles):
                with torch.no_grad():
                    decoded = decode_single(latent, vae)
                decoded_tiles.append(decoded)
                flush_cuda()
                report(f"Decoding tile {i+1}/{len(tiles)}", 0.70 + 0.20 * (i + 1) / len(tiles))

            del vae
            vae = None
            flush_cuda()

        finally:
            # Guarantee GPU cleanup even on exception
            if vae is not None:
                del vae
            if transformer is not None:
                del transformer
            flush_cuda()

        # --- Phase 4: Stitch tiles ---
        report("Stitching tiles...", 0.90)
        W_full = max(t[2] for t in tiles)
        H_full = max(t[3] for t in tiles)

        acc = torch.zeros(3, H_full, W_full, dtype=torch.float32)
        wsum = torch.zeros(H_full, W_full, dtype=torch.float32)

        for tile_info, decoded in zip(tiles, decoded_tiles):
            x0, y0, x1, y1 = tile_info
            tile_img = decoded.squeeze(0)
            h, w = tile_img.shape[-2:]
            tH, tW = y1 - y0, x1 - x0

            if h != tH or w != tW:
                tile_img = torch.from_numpy(lanczos_resize_chw(tile_img.numpy(), (tH, tW)))

            # Triangular blending window
            wx = 1 - (2 * torch.arange(tW, dtype=torch.float32) / max(tW - 1, 1) - 1).abs()
            wy = 1 - (2 * torch.arange(tH, dtype=torch.float32) / max(tH - 1, 1) - 1).abs()
            w2 = (wy[:, None] * wx[None, :]).clamp_min(1e-3)

            acc[:, y0:y1, x0:x1] += tile_img * w2
            wsum[y0:y1, x0:x1] += w2

        stitched = acc / wsum.clamp_min(1e-6)

        # Convert to PIL and resize to original dimensions
        x01 = ((stitched + 1.0) / 2.0).clamp(0.0, 1.0)
        pil = torchvision.transforms.functional.to_pil_image(x01)
        pil_resized = pil.resize((orig_W, orig_H), resample=Image.LANCZOS)

        # Save output
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if params.output_format == "jpg":
            pil_resized.save(output_path, quality=params.jpg_quality)
        elif params.output_format == "webp":
            pil_resized.save(output_path, quality=params.jpg_quality)
        else:
            pil_resized.save(output_path)

        report("Done", 1.0)
        return output_path

    def process_batch(
        self,
        input_paths: list,
        output_dir: str,
        params: InferenceParams,
        progress_callback: Optional[Callable[[str, float, int, int], None]] = None,
    ) -> list:
        """
        Process multiple images.
        progress_callback(status, image_fraction, current_index, total)
        Returns list of output paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []

        for idx, img_path in enumerate(input_paths):
            name = Path(img_path).stem
            ext = f".{params.output_format}"
            out_path = os.path.join(output_dir, f"{name}_clean{ext}")

            if os.path.exists(out_path):
                results.append(out_path)
                if progress_callback:
                    progress_callback(f"Skipped (exists): {name}", 1.0, idx, len(input_paths))
                continue

            def img_progress(msg, frac):
                if progress_callback:
                    progress_callback(msg, frac, idx, len(input_paths))

            try:
                result = self.process_image(img_path, out_path, params, img_progress)
                results.append(result)
            except Exception as e:
                results.append(None)
                if progress_callback:
                    progress_callback(f"Error: {e}", 0.0, idx, len(input_paths))

        return results


def collect_images(path: str) -> list:
    """Collect image paths from a file or directory."""
    p = Path(path)
    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
        return [str(p)]
    elif p.is_dir():
        return sorted(str(f) for f in p.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS)
    return []
