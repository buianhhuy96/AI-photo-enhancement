"""Skin retouching engine — AI face parsing + LAB chrominance smoothing."""
import numpy as np
from PIL import Image


class SkinRetouchEngine:
    """Remove blemishes and even skin tone using AI segmentation + LAB processing."""

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = None

    def _get_device(self):
        if self._device is None:
            import torch
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return self._device

    def _ensure_model(self):
        """Load the face parsing model (SegFormer, ~84M params)."""
        if self._model is not None:
            return
        import torch
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

        self._processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        self._model = SegformerForSemanticSegmentation.from_pretrained(
            "jonathandinu/face-parsing",
        )
        self._model.to(self._get_device())
        self._model.eval()

    def _detect_skin_ai(self, img):
        """Detect skin using AI face parsing model.

        Returns a float mask (H, W) where 1.0 = skin area.
        Includes: skin, nose, neck (labels 1, 2, 17).
        Excludes: eyes, eyebrows, lips, hair, glasses, ears.
        """
        import torch
        from torch import nn

        self._ensure_model()
        device = self._get_device()

        inputs = self._processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        logits = outputs.logits  # (1, num_labels, H/4, W/4)

        # Resize to original image dimensions
        upsampled = nn.functional.interpolate(
            logits, size=(img.height, img.width),
            mode='bilinear', align_corners=False,
        )
        labels = upsampled.argmax(dim=1)[0].cpu().numpy()  # (H, W)

        # Skin-related labels: 1=skin, 2=nose, 17=neck
        skin_labels = {1, 2, 17}
        mask = np.isin(labels, list(skin_labels)).astype(np.uint8) * 255

        return mask

    def _detect_skin_color(self, img_array):
        """Fallback: detect skin using color thresholding for body areas (arms/legs)."""
        import cv2

        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # YCrCb detection — wide range for diverse skin tones
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, np.array([40, 125, 75]), np.array([255, 180, 135]))

        # HSV detection
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, np.array([0, 10, 60]), np.array([35, 200, 255]))

        mask = cv2.bitwise_or(mask_ycrcb, mask_hsv)

        # Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return mask

    def _build_skin_mask(self, img, img_array):
        """Build combined skin mask: AI face parsing + color-based body detection."""
        import cv2

        # AI face parsing (precise for face/neck)
        ai_mask = self._detect_skin_ai(img)

        # Color-based detection (catches body skin: arms, legs, chest)
        color_mask = self._detect_skin_color(img_array)

        # Union: AI mask covers face precisely, color mask adds body areas
        combined = cv2.bitwise_or(ai_mask, color_mask)

        # Dilate to cover blemishes that were rejected by both detectors
        short_edge = min(img_array.shape[0], img_array.shape[1])
        dilate_size = max(7, int(short_edge * 0.025))
        if dilate_size % 2 == 0:
            dilate_size += 1
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        combined = cv2.dilate(combined, dilate_kernel, iterations=1)

        # Smooth edges for natural blending
        combined = cv2.GaussianBlur(combined, (7, 7), 0)

        return combined.astype(np.float32) / 255.0

    def _guided_filter(self, guide, src, radius, eps):
        """Edge-preserving guided filter for optional texture smoothing."""
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

    def retouch(self, img, strength=0.5, detail_size=0.05):
        """Retouch skin: remove blemishes and even tone while preserving texture.

        Uses AI face parsing for precise skin detection, then smooths chrominance
        (color) in LAB space to remove colored blemishes without affecting texture.

        Args:
            img: Input PIL Image (RGB).
            strength: 0.0 = no change, 1.0 = maximum blemish removal.
            detail_size: Blur radius as fraction of image short edge (0.02–0.15).
                         Larger = smooths bigger blemishes but may look unnatural.

        Returns:
            Retouched PIL Image (RGB).
        """
        import cv2

        if strength <= 0:
            return img.copy()

        strength = min(strength, 1.0)
        img_array = np.array(img)

        # Adaptive radius based on image resolution
        # Blemishes are typically 1-5% of face width; need blur much larger than
        # blemish size to average them against surrounding healthy skin.
        # Since L-channel preserves all texture, large A/B blur is safe.
        short_edge = min(img_array.shape[0], img_array.shape[1])
        radius = max(11, int(short_edge * detail_size))
        if radius % 2 == 0:
            radius += 1

        # Step 1: AI skin detection
        mask = self._build_skin_mask(img, img_array)
        print(f"[skin_retouch] mask stats: min={mask.min():.3f}, max={mask.max():.3f}, "
              f"mean={mask.mean():.3f}, nonzero={np.count_nonzero(mask > 0.1)}/{mask.size}")
        print(f"[skin_retouch] strength={strength}, detail_size={detail_size}, "
              f"radius={radius}, blur_size={radius*2+1}, sigma={radius*0.8:.1f}")

        # Step 2: Convert to LAB
        img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB).astype(np.float32)
        L = img_lab[:, :, 0]
        A = img_lab[:, :, 1]
        B = img_lab[:, :, 2]

        # Step 3: Smooth chrominance (removes colored blemishes)
        # Use masked blur: only average chrominance from skin pixels.
        # This prevents non-skin colors (hair, background) bleeding in.
        # Formula: smooth = Blur(channel * mask) / Blur(mask)
        blur_size = radius * 2 + 1
        sigma = radius * 0.5

        mask_blur = cv2.GaussianBlur(mask, (blur_size, blur_size), sigma)
        mask_blur = np.maximum(mask_blur, 1e-6)  # avoid division by zero

        A_masked = A * mask
        B_masked = B * mask
        A_smooth = cv2.GaussianBlur(A_masked, (blur_size, blur_size), sigma) / mask_blur
        B_smooth = cv2.GaussianBlur(B_masked, (blur_size, blur_size), sigma) / mask_blur

        # Second pass at high strength for more aggressive smoothing
        if strength > 0.5:
            A_smooth2 = A_smooth * mask
            B_smooth2 = B_smooth * mask
            A_smooth = cv2.GaussianBlur(A_smooth2, (blur_size, blur_size), sigma) / mask_blur
            B_smooth = cv2.GaussianBlur(B_smooth2, (blur_size, blur_size), sigma) / mask_blur

        # Blend into skin regions
        A_result = A * (1 - strength * mask) + A_smooth * (strength * mask)
        B_result = B * (1 - strength * mask) + B_smooth * (strength * mask)
        print(f"[skin_retouch] A diff: mean={np.abs(A_result - A).mean():.2f}, max={np.abs(A_result - A).max():.2f}")
        print(f"[skin_retouch] B diff: mean={np.abs(B_result - B).mean():.2f}, max={np.abs(B_result - B).max():.2f}")

        # Step 4: Optional light luminance smoothing at high strength
        texture_strength = max(0, (strength - 0.5) * 0.4)
        if texture_strength > 0:
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
