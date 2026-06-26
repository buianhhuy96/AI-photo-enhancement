"""Skin retouching engine — AI face parsing + spot blemish detection + inpainting."""
import numpy as np
from PIL import Image


class SkinRetouchEngine:
    """Remove blemishes by detecting individual spots and inpainting them."""

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

        Returns:
            skin_mask: Binary mask (H, W) uint8 where 255 = skin area.
            protect_mask: Binary mask (H, W) uint8 where 255 = protected area
                          (eyes, eyebrows, lips, hair, ears) that must NOT be inpainted.
        """
        import torch
        from torch import nn

        self._ensure_model()
        device = self._get_device()

        inputs = self._processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        logits = outputs.logits

        upsampled = nn.functional.interpolate(
            logits, size=(img.height, img.width),
            mode='bilinear', align_corners=False,
        )
        labels = upsampled.argmax(dim=1)[0].cpu().numpy()

        # jonathandinu/face-parsing labels (CelebAMask-HQ):
        # 0=background, 1=skin, 2=nose, 3=eye_glasses, 4=l_eye, 5=r_eye,
        # 6=l_brow, 7=r_brow, 8=l_ear, 9=r_ear, 10=mouth, 11=u_lip,
        # 12=l_lip, 13=hair, 14=hat, 15=earring, 16=necklace, 17=neck, 18=cloth
        skin_labels = {1, 2, 17}
        mask = np.isin(labels, list(skin_labels)).astype(np.uint8) * 255

        # Protected zones: eyes, eyebrows, lips, mouth, hair, ears, glasses
        protect_labels = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
        protect_mask = np.isin(labels, list(protect_labels)).astype(np.uint8) * 255

        return mask, protect_mask

    def _detect_skin_color(self, img_array):
        """Fallback: detect skin using color thresholding for body areas."""
        import cv2

        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, np.array([40, 125, 75]), np.array([255, 180, 135]))

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, np.array([0, 10, 60]), np.array([35, 200, 255]))

        mask = cv2.bitwise_or(mask_ycrcb, mask_hsv)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return mask

    def _build_skin_mask(self, img, img_array):
        """Build combined skin mask: AI face parsing + color-based body detection.

        Returns:
            skin_mask: Where blemishes can exist (uint8, 255=skin).
            protect_mask: Where inpainting must NOT touch (uint8, 255=protected).
        """
        import cv2

        ai_mask, protect_mask = self._detect_skin_ai(img)
        color_mask = self._detect_skin_color(img_array)
        combined = cv2.bitwise_or(ai_mask, color_mask)

        # Small dilation to include blemishes at skin boundary
        short_edge = min(img_array.shape[0], img_array.shape[1])
        dilate_size = max(5, int(short_edge * 0.01))
        if dilate_size % 2 == 0:
            dilate_size += 1
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        combined = cv2.dilate(combined, dilate_kernel, iterations=1)

        # Dilate protection zone to create safety buffer around eyes/lips/hair
        protect_dilate = max(5, int(short_edge * 0.015))
        if protect_dilate % 2 == 0:
            protect_dilate += 1
        protect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (protect_dilate, protect_dilate))
        protect_mask = cv2.dilate(protect_mask, protect_kernel, iterations=1)

        # Remove protected areas from skin mask
        combined = cv2.bitwise_and(combined, cv2.bitwise_not(protect_mask))

        return combined, protect_mask

    def _detect_blemishes(self, img_array, skin_mask, sensitivity=0.5):
        """Detect individual blemish spots within the skin mask.

        Finds pixels that deviate significantly from their local neighborhood
        in luminance or chrominance -- these are the actual blemishes.

        Args:
            img_array: RGB image as numpy array.
            skin_mask: Binary skin mask (uint8, 255=skin).
            sensitivity: 0.0 = only obvious blemishes, 1.0 = very sensitive.

        Returns:
            Binary blemish mask (uint8, 255=blemish pixel).
        """
        import cv2

        h, w = img_array.shape[:2]
        short_edge = min(h, w)

        # Convert to LAB for perceptual analysis
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Local neighborhood size -- must be larger than blemishes
        # Typical blemish: 0.5-3% of face width; neighborhood: 5-10%
        local_radius = max(15, int(short_edge * 0.06))
        if local_radius % 2 == 0:
            local_radius += 1

        # Compute local mean within skin-only regions (masked blur)
        skin_f = (skin_mask / 255.0).astype(np.float32)
        skin_blur = cv2.GaussianBlur(skin_f, (local_radius, local_radius), local_radius * 0.4)
        skin_blur = np.maximum(skin_blur, 1e-6)

        local_means = []
        for ch in range(3):
            ch_masked = img_lab[:, :, ch] * skin_f
            ch_blur = cv2.GaussianBlur(ch_masked, (local_radius, local_radius), local_radius * 0.4)
            local_means.append(ch_blur / skin_blur)

        # Compute deviation from local mean for each pixel
        L_dev = img_lab[:, :, 0] - local_means[0]  # negative = darker than surroundings
        A_dev = img_lab[:, :, 1] - local_means[1]  # positive = more red
        B_dev = img_lab[:, :, 2] - local_means[2]  # negative = more blue/purple

        # Blemishes are: darker + redder/purpler than surroundings
        dark_score = np.maximum(-L_dev, 0)   # how much darker than neighborhood
        red_score = np.maximum(A_dev, 0)     # how much redder
        blue_score = np.maximum(-B_dev, 0)   # how much more purple/blue

        # Combined perceptual blemish score
        blemish_score = dark_score + red_score * 0.7 + blue_score * 0.5

        # Adaptive threshold based on sensitivity
        skin_pixels = blemish_score[skin_mask > 0]
        if len(skin_pixels) == 0:
            return np.zeros((h, w), dtype=np.uint8)

        # Threshold: spots that deviate significantly from mean score
        score_mean = np.mean(skin_pixels)
        score_std = np.std(skin_pixels)
        # sensitivity=0 -> 3.0 std (few spots), sensitivity=1 -> 1.0 std (many spots)
        threshold = score_mean + score_std * (3.0 - sensitivity * 2.0)
        threshold = max(threshold, 3.0)  # minimum to avoid noise

        # Create blemish mask
        blemish_mask = ((blemish_score > threshold) & (skin_mask > 0)).astype(np.uint8) * 255

        # Morphological cleanup: remove tiny noise dots
        min_spot = max(2, int(short_edge * 0.003))
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_spot, min_spot))
        blemish_mask = cv2.morphologyEx(blemish_mask, cv2.MORPH_OPEN, clean_kernel)

        # Dilate to cover the full extent of each blemish
        dilate_size = max(3, int(short_edge * 0.005))
        if dilate_size % 2 == 0:
            dilate_size += 1
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        blemish_mask = cv2.dilate(blemish_mask, dilate_kernel, iterations=1)

        # Constrain to skin mask
        blemish_mask = cv2.bitwise_and(blemish_mask, skin_mask)

        print(f"[skin_retouch] threshold={threshold:.1f}, score_mean={score_mean:.1f}, score_std={score_std:.1f}")

        return blemish_mask

    def retouch(self, img, strength=0.5, detail_size=0.05):
        """Retouch skin: detect individual blemishes and inpaint them.

        Like Photoshop's Spot Healing Brush applied automatically to all blemishes.

        Args:
            img: Input PIL Image (RGB).
            strength: 0.0 = no change, 1.0 = full inpainting replacement.
            detail_size: Controls blemish detection sensitivity (0.02-0.15).
                         Higher = detects more/subtler blemishes.

        Returns:
            Retouched PIL Image (RGB).
        """
        import cv2

        if strength <= 0:
            return img.copy()

        strength = min(strength, 1.0)
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        short_edge = min(h, w)

        # Step 1: AI skin detection (with protected zones)
        skin_mask, protect_mask = self._build_skin_mask(img, img_array)

        # Step 2: Detect individual blemishes within skin
        sensitivity = min(detail_size / 0.15, 1.0)  # normalize to 0-1
        blemish_mask = self._detect_blemishes(img_array, skin_mask, sensitivity)

        n_blemish = np.count_nonzero(blemish_mask)
        n_skin = np.count_nonzero(skin_mask)
        print(f"[skin_retouch] skin_pixels={n_skin}, blemish_pixels={n_blemish} "
              f"({100*n_blemish/max(n_skin,1):.1f}% of skin), sensitivity={sensitivity:.2f}")

        if n_blemish == 0:
            print("[skin_retouch] No blemishes detected")
            return img.copy()

        # Step 3: Inpaint blemishes (like Photoshop healing brush)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Inpainting radius -- how far to propagate texture from boundary
        inpaint_radius = max(3, int(short_edge * 0.01))

        inpainted = cv2.inpaint(img_bgr, blemish_mask, inpaint_radius, cv2.INPAINT_TELEA)

        # Step 4: Blend based on strength with soft edges
        blend_mask = cv2.GaussianBlur(
            blemish_mask.astype(np.float32) / 255.0,
            (5, 5), 0
        )
        blend_mask = (blend_mask * strength)[:, :, np.newaxis]

        result_bgr = (img_bgr.astype(np.float32) * (1 - blend_mask) +
                      inpainted.astype(np.float32) * blend_mask)
        result_bgr = np.clip(result_bgr, 0, 255).astype(np.uint8)

        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
