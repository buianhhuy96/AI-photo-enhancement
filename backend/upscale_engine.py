"""Real-ESRGAN upscaling engine with denoise strength control."""
import numpy as np
from PIL import Image


class UpscaleEngine:
    """Wraps Real-ESRGAN for AI upscaling with configurable denoise strength."""

    def __init__(self):
        self._model = None
        self._initialized = False
        self._current_denoise = None

    def initialize(self, denoise_strength=0.5):
        """Load the realesr-general-x4v3 model with given denoise strength."""
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        import torch

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        half = device.type == 'cuda'

        # realesr-general-x4v3 supports denoise_strength interpolation
        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_conv=32, upscale=4, act_type='prelu',
        )

        model_path_base = 'realesr-general-x4v3'

        self._model = RealESRGANer(
            scale=4,
            model_path=None,  # Will use auto-download
            dni_weight=[denoise_strength, 1 - denoise_strength] if denoise_strength < 1 else None,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=half,
            device=device,
        )
        self._initialized = True
        self._current_denoise = denoise_strength

    def upscale(self, img, target_width, target_height, denoise_strength=0.5):
        """Upscale a PIL Image to target dimensions.

        Uses Real-ESRGAN for 4x upscale, then resizes to exact target.
        Falls back to LANCZOS if Real-ESRGAN fails.

        Args:
            img: PIL Image
            target_width: desired output width
            target_height: desired output height
            denoise_strength: 0.0 (no denoise) to 1.0 (full denoise)

        Returns:
            PIL Image at target dimensions
        """
        if not self._initialized or self._current_denoise != denoise_strength:
            self.initialize(denoise_strength)

        # Convert PIL to numpy BGR (OpenCV format expected by Real-ESRGAN)
        img_np = np.array(img)[:, :, ::-1]  # RGB to BGR

        try:
            output, _ = self._model.enhance(img_np, outscale=4)
            # Convert back to PIL RGB
            result = Image.fromarray(output[:, :, ::-1])  # BGR to RGB
            # Resize to exact target dimensions
            result = result.resize((target_width, target_height), Image.LANCZOS)
            return result
        except Exception:
            # Fallback to LANCZOS
            return img.resize((target_width, target_height), Image.LANCZOS)
