"""Skin retouching engine — blemish removal via chrominance smoothing in LAB space."""
import numpy as np
from PIL import Image


class SkinRetouchEngine:
    """Remove blemishes and even skin tone while preserving texture."""

    def _detect_skin_color(self, img_array):
        """Detect skin using YCrCb + HSV color space thresholding.

        Works on face, arms, legs — any exposed skin regardless of body part.
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
        """Edge-preserving guided filter (O(n) box filter implementation)."""
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

    def retouch(self, img, strength=0.5, detail_size=0.01):
        """Retouch skin: remove blemishes (color anomalies) and optionally smooth texture.

        Works in LAB color space:
        - Smooths A/B channels (chrominance) to remove colored blemishes (red marks,
          brown spots, uneven tone) while keeping ALL luminance texture (pores).
        - At higher strength, lightly smooths L channel for subtle texture softening.

        Args:
            img: Input PIL Image (RGB).
            strength: 0.0 = no change, 1.0 = maximum blemish removal + light texture smoothing.
            detail_size: Radius as fraction of image short edge (0.001–0.02).
                         Controls the scale of color anomalies to target.

        Returns:
            Retouched PIL Image (RGB). Skin texture preserved.
        """
        import cv2

        if strength <= 0:
            return img.copy()

        strength = min(strength, 1.0)
        img_array = np.array(img)

        # Adaptive radius based on image resolution
        short_edge = min(img_array.shape[0], img_array.shape[1])
        radius = max(3, int(short_edge * detail_size))
        # Ensure odd kernel size for boxFilter
        if radius % 2 == 0:
            radius += 1

        # Step 1: Detect skin regions
        skin_mask = self._detect_skin_color(img_array)
        mask = skin_mask  # (H, W) float 0-1

        # Step 2: Convert to LAB
        img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB).astype(np.float32)
        L = img_lab[:, :, 0]
        A = img_lab[:, :, 1]
        B = img_lab[:, :, 2]

        # Step 3: Smooth chrominance channels (A and B) — removes colored blemishes
        # Use guided filter with L as guide to preserve edges at skin/non-skin boundaries
        eps_color = 0.5  # Low eps = stronger smoothing of color
        A_smooth = self._guided_filter(L / 255.0, A, radius, eps_color)
        B_smooth = self._guided_filter(L / 255.0, B, radius, eps_color)

        # Blend smoothed chrominance into skin regions based on strength
        A_result = A * (1 - strength * mask) + A_smooth * (strength * mask)
        B_result = B * (1 - strength * mask) + B_smooth * (strength * mask)

        # Step 4: Optional light luminance smoothing at high strength (>0.5)
        # This gives subtle texture softening without destroying pores
        texture_strength = max(0, (strength - 0.5) * 0.4)  # 0 at strength=0.5, 0.2 at strength=1.0
        if texture_strength > 0:
            # Use smaller radius for L to preserve structure
            l_radius = max(3, radius // 2)
            if l_radius % 2 == 0:
                l_radius += 1
            L_smooth = self._guided_filter(L / 255.0, L, l_radius, 2.0)
            L_result = L * (1 - texture_strength * mask) + L_smooth * (texture_strength * mask)
        else:
            L_result = L

        # Step 5: Recombine and convert back
        result_lab = np.stack([L_result, A_result, B_result], axis=-1)
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        result_rgb = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)

        return Image.fromarray(result_rgb)
