"""Skin retouching engine — GFPGAN face restoration + face parsing mask + LAB blending."""
import numpy as np
from PIL import Image


class SkinRetouchEngine:
    """Remove blemishes using GFPGAN to regenerate skin, masked by face parsing."""

    def __init__(self):
        self._parsing_model = None
        self._parsing_processor = None
        self._gfpgan = None
        self._device = None

    def _get_device(self):
        if self._device is None:
            import torch
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return self._device

    def _ensure_parsing_model(self):
        """Load the face parsing model (SegFormer) for skin masking."""
        if self._parsing_model is not None:
            return
        import torch
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

        self._parsing_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        self._parsing_model = SegformerForSemanticSegmentation.from_pretrained(
            "jonathandinu/face-parsing",
        )
        self._parsing_model.to(self._get_device())
        self._parsing_model.eval()

    def _ensure_gfpgan(self):
        """Load GFPGAN face restoration model."""
        if self._gfpgan is not None:
            return
        from gfpgan import GFPGANer

        # Model auto-downloads from GitHub releases if not present
        model_path = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"

        self._gfpgan = GFPGANer(
            model_path=model_path,
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
        )

    def _get_skin_mask(self, img):
        """Get skin-only mask from face parsing.

        Returns float mask (H, W) in [0, 1] where 1.0 = skin area.
        Skin labels: 1=skin, 10=nose, 14=neck.
        Protected: eyes, eyebrows, lips, mouth, hair, ears, glasses.
        """
        import torch
        import cv2
        from torch import nn

        self._ensure_parsing_model()
        device = self._get_device()

        inputs = self._parsing_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self._parsing_model(**inputs)

        logits = outputs.logits
        upsampled = nn.functional.interpolate(
            logits, size=(img.height, img.width),
            mode='bilinear', align_corners=False,
        )
        labels = upsampled.argmax(dim=1)[0].cpu().numpy()

        # CelebAMask-HQ labels for jonathandinu/face-parsing:
        # 0=background, 1=skin, 2=nose, 3=glasses, 4=l_eye, 5=r_eye,
        # 6=l_brow, 7=r_brow, 8=l_ear, 9=r_ear, 10=earring, 11=mouth,
        # 12=u_lip, 13=l_lip, 14=neck, 15=necklace, 16=cloth, 17=hair, 18=hat
        skin_labels = {1, 2, 14}
        mask = np.isin(labels, list(skin_labels)).astype(np.uint8) * 255

        # Smooth edges for natural blending
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        return mask.astype(np.float32) / 255.0

    def _restore_face(self, img_array):
        """Run GFPGAN on the image to get a restored face version.

        Args:
            img_array: RGB numpy array (H, W, 3).

        Returns:
            Restored RGB numpy array (H, W, 3), or None if no face detected.
        """
        import cv2

        self._ensure_gfpgan()

        # GFPGAN expects BGR input
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        _, _, restored_img = self._gfpgan.enhance(
            img_bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.5,
        )

        if restored_img is None:
            return None

        # Convert back to RGB
        return cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)

    def retouch(self, img, strength=0.5, detail_size=0.05):
        """Remove blemishes: GFPGAN regenerates skin, blended via face parsing mask.

        Takes L-channel (luminance/texture) from GFPGAN output and blends it
        onto original skin. This removes dark spots while preserving skin tone.

        Args:
            img: Input PIL Image (RGB).
            strength: 0.0 = no change, 1.0 = full GFPGAN skin replacement.
            detail_size: Not used in GFPGAN mode (kept for API compat).

        Returns:
            Retouched PIL Image (RGB).
        """
        import cv2

        if strength <= 0:
            return img.copy()

        strength = min(strength, 1.0)
        img_array = np.array(img)

        # Step 1: Get GFPGAN restored face
        restored = self._restore_face(img_array)
        if restored is None:
            print("[skin_retouch] No face detected by GFPGAN")
            return img.copy()

        # Step 2: Get skin mask from face parsing
        skin_mask = self._get_skin_mask(img)
        mask = skin_mask[:, :, np.newaxis]  # (H, W, 1) for broadcasting

        # Step 3: LAB blending - take L from GFPGAN (clean texture), keep A/B from original (skin tone)
        img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB).astype(np.float32)
        restored_lab = cv2.cvtColor(restored, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Blend L-channel (removes blemishes = dark spots in luminance)
        L_orig = img_lab[:, :, 0]
        L_restored = restored_lab[:, :, 0]
        L_result = L_orig * (1 - strength * skin_mask) + L_restored * (strength * skin_mask)

        # Keep original chrominance (preserves exact skin tone)
        result_lab = np.stack([L_result, img_lab[:, :, 1], img_lab[:, :, 2]], axis=-1)
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        result_rgb = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)

        print(f"[skin_retouch] L diff in skin: mean={np.abs(L_result - L_orig).mean():.2f}, "
              f"max={np.abs(L_result - L_orig).max():.2f}, strength={strength}")

        return Image.fromarray(result_rgb)

    def even_tone(self, img, strength=0.5):
        """Even out skin tone using GFPGAN's chrominance.

        Blends A/B channels from GFPGAN (evenly-toned) with original,
        while keeping L-channel from original (preserves texture).

        Args:
            img: Input PIL Image (RGB).
            strength: 0.0 = no change, 1.0 = full GFPGAN tone.

        Returns:
            Tone-evened PIL Image (RGB).
        """
        import cv2

        if strength <= 0:
            return img.copy()

        strength = min(strength, 1.0)
        img_array = np.array(img)

        # Get GFPGAN restored face
        restored = self._restore_face(img_array)
        if restored is None:
            print("[skin_tone] No face detected by GFPGAN")
            return img.copy()

        # Get skin mask
        skin_mask = self._get_skin_mask(img)

        # LAB blending - take A/B from GFPGAN (even tone), keep L from original (texture)
        img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB).astype(np.float32)
        restored_lab = cv2.cvtColor(restored, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Keep original L (all texture preserved)
        L = img_lab[:, :, 0]

        # Blend A/B toward GFPGAN (evens redness/patches)
        A_orig = img_lab[:, :, 1]
        B_orig = img_lab[:, :, 2]
        A_restored = restored_lab[:, :, 1]
        B_restored = restored_lab[:, :, 2]

        A_result = A_orig * (1 - strength * skin_mask) + A_restored * (strength * skin_mask)
        B_result = B_orig * (1 - strength * skin_mask) + B_restored * (strength * skin_mask)

        result_lab = np.stack([L, A_result, B_result], axis=-1)
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        result_rgb = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)

        print(f"[skin_tone] A diff: mean={np.abs(A_result - A_orig).mean():.2f}, "
              f"B diff: mean={np.abs(B_result - B_orig).mean():.2f}")

        return Image.fromarray(result_rgb)
