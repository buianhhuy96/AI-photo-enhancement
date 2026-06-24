"""Business logic services — called by any frontend (Gradio, React, CLI)."""
import time
import traceback
from pathlib import Path
from PIL import Image

from backend.config import params_from_quality

# Thumbnail size for filmstrip (width, height)
THUMBNAIL_SIZE = (120, 80)


def generate_thumbnails(file_list):
    """Create compressed thumbnails for the filmstrip bar.
    Returns list of (PIL.Image, label) tuples for gr.Gallery.
    """
    if not file_list:
        return []
    thumbs = []
    for f in file_list:
        img = Image.open(f).convert("RGB")
        img.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
        thumbs.append((img, Path(f).stem))
    return thumbs


def load_image_at_index(file_list, index):
    """Load full-resolution image at the given index. Returns PIL Image or None."""
    if not file_list or index is None or index < 0 or index >= len(file_list):
        return None
    return Image.open(file_list[index]).convert("RGB")


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


def export_all_images(engine, file_list, quality, strength, use_4bit, output_format, jpg_quality, mock_mode=False, progress_cb=None):
    """Process and export all images in file_list. Returns status string.

    Args:
        progress_cb: callable(fraction, description) or None.
    """
    if not file_list:
        return "No images imported."

    export_dir = Path("exports")
    export_dir.mkdir(exist_ok=True)
    params = params_from_quality(int(quality), use_4bit, output_format, int(jpg_quality))
    total = len(file_list)
    lines = []

    for idx, img_data in enumerate(file_list):
        img = Image.open(img_data).convert("RGB")
        name = Path(img_data).stem

        if mock_mode:
            time.sleep(0.3)
            if progress_cb:
                progress_cb((idx + 1) / total, f"[MOCK] {name}")
            out_path = export_dir / f"{name}_clean.{output_format}"
            img.save(str(out_path))
            lines.append(f"[{idx+1}/{total}] {name}")
            continue

        if not engine._initialized:
            if progress_cb:
                progress_cb(0.0, "Loading model...")
            engine.initialize()

        tmp_dir = Path("temp_processing")
        tmp_dir.mkdir(exist_ok=True)
        input_path = str(tmp_dir / "batch_input.png")
        img.save(input_path)
        out_path = export_dir / f"{name}_clean.{output_format}"

        def _pcb(msg, frac, _i=idx, _t=total):
            if progress_cb:
                progress_cb((_i + frac) / _t, f"[{_i+1}/{_t}] {msg}")

        try:
            engine.process_image(input_path, str(out_path), params, _pcb)
            if strength < 1.0:
                proc = Image.open(str(out_path)).convert("RGB")
                Image.blend(img, proc, strength).save(str(out_path))
            lines.append(f"[{idx+1}/{total}] Done: {name}")
        except Exception as e:
            lines.append(f"[{idx+1}/{total}] Error: {name} — {e}")

    return f"Exported {total} images → {export_dir}\n\n" + "\n".join(lines)


def load_first_image(file_list):
    """Load and return the first image from a file list as RGB PIL Image."""
    if not file_list:
        return None
    return Image.open(file_list[0]).convert("RGB")


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
