"""Configuration and parameter defaults for reflection removal inference."""
from dataclasses import dataclass
from typing import Optional


BASE_MODEL_URI = "Qwen/Qwen-Image-Edit-2509"
LORA_MODEL_URI = "huawei-bayerlab/windowseat-reflection-removal-v1-0"

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


@dataclass
class InferenceParams:
    """User-adjustable inference parameters."""
    # Tiling
    use_short_edge_tile: bool = True
    max_num_tiles_w: int = 4
    max_num_tiles_h: int = 4
    min_overlap_w: int = 64
    min_overlap_h: int = 64

    # Quantization
    use_4bit: bool = True  # NF4 quantization for transformer (saves VRAM)

    # Output
    output_format: str = "png"  # png, jpg, webp
    jpg_quality: int = 95

    # Processing resolution override (None = use model default from config)
    processing_resolution: Optional[int] = None


# Quality presets: maps a 1-5 slider to tiling parameters
# Higher quality = more tiles, more overlap = slower but better on high-res images
QUALITY_PRESETS = {
    1: {  # Fast — minimal tiling, best for quick previews
        "use_short_edge_tile": True,
        "max_num_tiles_w": 2,
        "max_num_tiles_h": 2,
        "min_overlap_w": 32,
        "min_overlap_h": 32,
    },
    2: {  # Balanced — default, good for most photos
        "use_short_edge_tile": True,
        "max_num_tiles_w": 4,
        "max_num_tiles_h": 4,
        "min_overlap_w": 64,
        "min_overlap_h": 64,
    },
    3: {  # High — more tiles for large/detailed images
        "use_short_edge_tile": True,
        "max_num_tiles_w": 6,
        "max_num_tiles_h": 6,
        "min_overlap_w": 96,
        "min_overlap_h": 96,
    },
    4: {  # Very High — fixed tiling, preserves fine details
        "use_short_edge_tile": False,
        "max_num_tiles_w": 6,
        "max_num_tiles_h": 6,
        "min_overlap_w": 128,
        "min_overlap_h": 128,
    },
    5: {  # Maximum — most tiles, maximum quality, slowest
        "use_short_edge_tile": False,
        "max_num_tiles_w": 8,
        "max_num_tiles_h": 8,
        "min_overlap_w": 192,
        "min_overlap_h": 192,
    },
}


def params_from_quality(quality: int, use_4bit: bool = True,
                        output_format: str = "png", jpg_quality: int = 95) -> InferenceParams:
    """Create InferenceParams from a quality level (-1 to 3, default 0).
    Maps: -1→1, 0→2, 1→3, 2→4, 3→5 internally."""
    internal = max(1, min(5, quality + 2))
    preset = QUALITY_PRESETS[internal]
    return InferenceParams(
        use_short_edge_tile=preset["use_short_edge_tile"],
        max_num_tiles_w=preset["max_num_tiles_w"],
        max_num_tiles_h=preset["max_num_tiles_h"],
        min_overlap_w=preset["min_overlap_w"],
        min_overlap_h=preset["min_overlap_h"],
        use_4bit=use_4bit,
        output_format=output_format,
        jpg_quality=jpg_quality,
    )
