"""Business logic services — called by any frontend (Gradio, React, CLI)."""
import time
import traceback
from pathlib import Path
from PIL import Image

from backend.config import params_from_quality


def process_single_image(engine, img, quality, strength, use_4bit, output_format, jpg_quality, mock_mode=False, progress_cb=None):
    """Process a single PIL image. Returns (preview, original, processed, status_msg).

    Args:
        engine: The AI engine instance (or None in mock mode).
        img: PIL Image to process.
        quality: int 1-5.
        strength: float 0.0-1.0 blending factor.
        use_4bit: bool.
        output_format: str "png"/"jpg"/"webp".
        jpg_quality: int 50-100.
        mock_mode: bool.
        progress_cb: callable(fraction, description) or None.
    """
    if img is None:
        return None, None, None, "No image provided."

    if mock_mode:
        for i in range(10):
            time.sleep(0.3)
            if progress_cb:
                progress_cb(i / 10, f"[MOCK] Step {i+1}/10")
        processed = img.copy()
        preview = Image.blend(img, processed, strength)
        return preview, img, processed, "Done — adjust Strength for live preview."

    if not engine._initialized:
        if progress_cb:
            progress_cb(0.0, "Loading model (~30-60s)...")
        engine.initialize()

    params = params_from_quality(int(quality), use_4bit, output_format, int(jpg_quality))
    tmp_dir = Path("temp_processing")
    tmp_dir.mkdir(exist_ok=True)
    input_path = str(tmp_dir / "input.png")
    img.save(input_path)
    output_path = str(tmp_dir / f"output.{output_format}")

    try:
        engine.process_image(input_path, output_path, params, progress_cb)
        processed = Image.open(output_path).convert("RGB")
        preview = Image.blend(img, processed, strength)
        return preview, img, processed, "Done — adjust Strength for live preview."
    except Exception as e:
        traceback.print_exc()
        return None, None, None, f"Error: {e}"


def blend_preview(strength, original, processed):
    """Instant strength blending between original and processed images."""
    if original is None or processed is None:
        return None
    return Image.blend(original, processed, strength)


def resize_image(img, width, height, upscale_engine=None, denoise_strength=0.5):
    """Resize a PIL Image to the given dimensions.

    Uses Real-ESRGAN AI upscaling when scaling up (if engine available),
    LANCZOS for downscaling.

    Returns resized PIL Image.
    """
    if img is None:
        return None

    orig_w, orig_h = img.size
    is_upscale = (width * height) > (orig_w * orig_h)

    if is_upscale and upscale_engine is not None:
        return upscale_engine.upscale(img, width, height, denoise_strength)

    return img.resize((width, height), Image.LANCZOS)


def summarize_upload(file_list):
    """Return (first_image, info_string) for uploaded files."""
    if not file_list:
        return None, "No images."
    first_img = Image.open(file_list[0]).convert("RGB")
    count = len(file_list)
    names = [Path(f).name for f in file_list[:5]]
    info = f"{count} image{'s' if count > 1 else ''} ready"
    if count > 5:
        info += f" — {', '.join(names)}... +{count - 5} more"
    else:
        info += f" — {', '.join(names)}"
    return first_img, info
