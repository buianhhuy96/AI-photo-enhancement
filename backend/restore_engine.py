"""NAFNet-based image restoration engine for denoise and deblur."""
import os
import numpy as np
from pathlib import Path
from PIL import Image


class RestoreEngine:
    """Wraps NAFNet for denoising and deblurring."""

    def __init__(self):
        self._denoise_model = None
        self._deblur_model = None
        self._device = None

    def _get_device(self):
        if self._device is None:
            import torch
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return self._device

    def _ensure_denoise_model(self):
        if self._denoise_model is not None:
            return
        import torch
        from basicsr.archs.nafnet_arch import NAFNet
        from huggingface_hub import hf_hub_download

        device = self._get_device()

        # NAFNet-SIDD-width64 for denoising
        model = NAFNet(
            img_channel=3, width=64, middle_blk_num=1,
            enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1],
        )

        # Download weights
        weight_path = hf_hub_download(
            "jasonzhou2/NAFNet",
            "NAFNet-SIDD-width64.pth",
        )
        checkpoint = torch.load(weight_path, map_location=device, weights_only=True)
        if 'params' in checkpoint:
            model.load_state_dict(checkpoint['params'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device).eval()
        self._denoise_model = model

    def _ensure_deblur_model(self):
        if self._deblur_model is not None:
            return
        import torch
        from basicsr.archs.nafnet_arch import NAFNet
        from huggingface_hub import hf_hub_download

        device = self._get_device()

        # NAFNet-GoPro-width64 for deblurring
        model = NAFNet(
            img_channel=3, width=64, middle_blk_num=1,
            enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1],
        )

        weight_path = hf_hub_download(
            "jasonzhou2/NAFNet",
            "NAFNet-GoPro-width64.pth",
        )
        checkpoint = torch.load(weight_path, map_location=device, weights_only=True)
        if 'params' in checkpoint:
            model.load_state_dict(checkpoint['params'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device).eval()
        self._deblur_model = model

    def _process(self, model, img, tile_size=256):
        """Process a PIL image through a NAFNet model with tiling for large images."""
        import torch

        device = self._get_device()

        # PIL -> numpy -> tensor
        np_img = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0).to(device)

        _, _, h, w = tensor.shape

        # Pad to multiple of 64
        pad_h = (64 - h % 64) % 64
        pad_w = (64 - w % 64) % 64
        if pad_h > 0 or pad_w > 0:
            tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')

        with torch.no_grad():
            # Use tiling for large images to avoid OOM
            if h * w > tile_size * tile_size * 4:
                output = self._tiled_inference(model, tensor, tile_size)
            else:
                output = model(tensor)

        # Remove padding
        output = output[:, :, :h, :w]

        # tensor -> numpy -> PIL
        output = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        result = (output * 255).astype(np.uint8)
        return Image.fromarray(result)

    def _tiled_inference(self, model, tensor, tile_size=256, overlap=32):
        """Process large images in tiles to avoid OOM."""
        import torch

        _, _, h, w = tensor.shape
        output = torch.zeros_like(tensor)
        weight = torch.zeros_like(tensor)

        stride = tile_size - overlap

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                y_start = max(0, y_end - tile_size)
                x_start = max(0, x_end - tile_size)

                tile = tensor[:, :, y_start:y_end, x_start:x_end]
                with torch.no_grad():
                    tile_out = model(tile)

                output[:, :, y_start:y_end, x_start:x_end] += tile_out
                weight[:, :, y_start:y_end, x_start:x_end] += 1

        return output / weight

    def denoise(self, img, strength=1.0):
        """Denoise a PIL Image.

        Args:
            img: PIL Image
            strength: 0.0 = original, 1.0 = full denoise
        Returns:
            PIL Image
        """
        self._ensure_denoise_model()
        result = self._process(self._denoise_model, img)
        if strength < 1.0:
            # Blend with original
            result = Image.blend(img, result, strength)
        return result

    def deblur(self, img, strength=1.0):
        """Deblur a PIL Image.

        Args:
            img: PIL Image
            strength: 0.0 = original, 1.0 = full deblur
        Returns:
            PIL Image
        """
        self._ensure_deblur_model()
        result = self._process(self._deblur_model, img)
        if strength < 1.0:
            result = Image.blend(img, result, strength)
        return result
