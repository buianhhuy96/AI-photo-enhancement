"""Skin retouching engine using frequency separation with automatic skin detection."""
import numpy as np
from PIL import Image, ImageFilter


class SkinRetouchEngine:
    """Smooth skin texture while preserving color and non-skin areas."""

    def __init__(self):
        self._seg_model = None
        self._seg_processor = None
        self._device = None

    def _get_device(self):
        if self._device is None:
            import torch
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return self._device

    def _detect_skin_color(self, img_array):
        """Detect skin using YCrCb + HSV color space thresholding.

        Works on face, arms, legs — any exposed skin regardless of body part.
        Handles a wide range of skin tones.
        """
        import cv2

        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # YCrCb detection (most robust for skin across ethnicities)
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, np.array([50, 133, 77]), np.array([255, 173, 127]))

        # HSV detection (catches additional skin tones)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([30, 180, 255]))

        # Union of both detections for better coverage
        mask = cv2.bitwise_or(mask_ycrcb, mask_hsv)

        # Morphological cleanup: remove noise, fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Smooth edges for natural blending
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        return mask.astype(np.float32) / 255.0

    def _guided_filter(self, guide, src, radius, eps):
        """Edge-preserving guided filter (O(n) box filter implementation).

        Smooths src while respecting edges in guide.
        """
        import cv2

        guide = guide.astype(np.float32)
        src = src.astype(np.float32)

        mean_g = cv2.boxFilter(guide, -1, (radius, radius))
        mean_s = cv2.boxFilter(src, -1, (radius, radius))
        mean_gs = cv2.boxFilter(guide * src, -1, (radius, radius))
        mean_gg = cv2.boxFilter(guide * guide, -1, (radius, radius))

        cov_gs = mean_gs - mean_g * mean_s
        var_g = mean_gg - mean_g * mean_g

        a = cov_gs / (var_g + eps)
        b = mean_s - a * mean_g

        mean_a = cv2.boxFilter(a, -1, (radius, radius))
        mean_b = cv2.boxFilter(b, -1, (radius, radius))

        return mean_a * guide + mean_b

    def _frequency_separation(self, img_array, radius=12, eps=0.02):
        """Split image into low-frequency (color/shape) and high-frequency (texture).

        Uses guided filter for edge-preserving decomposition.
        """
        img_float = img_array.astype(np.float32) / 255.0

        # Process each channel
        low_freq = np.zeros_like(img_float)
        for c in range(3):
            low_freq[:, :, c] = self._guided_filter(
                img_float[:, :, c], img_float[:, :, c], radius, eps
            )

        high_freq = img_float - low_freq
        return low_freq, high_freq

    def retouch(self, img, strength=0.5, detail_size=0.01):
        """Retouch skin in a PIL Image.

        Args:
            img: Input PIL Image (RGB).
            strength: 0.0 = no change, 1.0 = maximum smoothing.
            detail_size: Radius as fraction of image short edge (0.001–0.02).
                         Smaller = fine pores only, larger = broader smoothing.

        Returns:
            Retouched PIL Image (RGB). Color is preserved exactly.
        """
        if strength <= 0:
            return img.copy()

        strength = min(strength, 1.0)
        img_array = np.array(img)

        # Adaptive radius based on image resolution
        short_edge = min(img_array.shape[0], img_array.shape[1])
        radius = max(3, int(short_edge * detail_size))

        # Step 1: Detect skin regions
        skin_mask = self._detect_skin_color(img_array)

        # Step 2: Frequency separation
        eps = 0.01 + (1.0 - strength) * 0.04  # Lower eps = more smoothing
        low_freq, high_freq = self._frequency_separation(img_array, radius=radius, eps=eps)

        # Step 3: Suppress high-frequency detail in skin areas
        # mask shape: (H, W) -> (H, W, 1) for broadcasting
        mask_3d = skin_mask[:, :, np.newaxis]

        # Blend: keep full high-freq outside skin, reduce inside skin
        suppression = strength * mask_3d
        result = low_freq + high_freq * (1.0 - suppression)

        # Convert back to uint8
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(result)
