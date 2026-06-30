"""Skin retouching engine — BiSeNet face parsing + frequency separation + LAB blending."""
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Fix compatibility: basicsr imports removed torchvision.transforms.functional_tensor
import torchvision.transforms.functional as _F
import sys
if "torchvision.transforms.functional_tensor" not in sys.modules:
    sys.modules["torchvision.transforms.functional_tensor"] = _F

# BiSeNet (yakhyo/face-parsing) label mapping — CelebAMask-HQ order:
# 0=background, 1=skin, 2=l_brow, 3=r_brow, 4=l_eye, 5=r_eye,
# 6=eye_g (glasses), 7=l_ear, 8=r_ear, 9=ear_r (earring), 10=nose,
# 11=mouth, 12=u_lip, 13=l_lip, 14=neck, 15=neck_l (necklace),
# 16=cloth, 17=hair, 18=hat
_BISENET_WEIGHT_URL = "https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.pt"


class SkinRetouchEngine:
    """Remove blemishes using frequency separation, masked by BiSeNet face parsing."""

    def __init__(self):
        self._parsing_model = None
        self._gfpgan = None
        self._device = None
        self._mask_cache_key = None
        self._mask_cache = None
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def _get_device(self):
        if self._device is None:
            import torch
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return self._device

    def _ensure_parsing_model(self):
        """Load the BiSeNet face parsing model (ResNet18 backbone)."""
        if self._parsing_model is not None:
            return
        import torch
        from backend.models.bisenet import BiSeNet

        device = self._get_device()

        # Download weights if needed
        weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
        os.makedirs(weights_dir, exist_ok=True)
        weight_path = os.path.join(weights_dir, 'bisenet_resnet18.pt')

        if not os.path.exists(weight_path):
            print(f"[skin_retouch] Downloading BiSeNet face parsing model (~43MB)...")
            torch.hub.download_url_to_file(_BISENET_WEIGHT_URL, weight_path)
            print(f"[skin_retouch] BiSeNet model downloaded to {weight_path}")

        model = BiSeNet(num_classes=19, backbone_name='resnet18')
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device)
        model.eval()
        self._parsing_model = model

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

    def _get_skin_mask(self, img, feather=0.5):
        """Get skin-only mask from BiSeNet face parsing (cached per image).

        Returns float mask (H, W) in [0, 1] where 1.0 = skin area.
        BiSeNet labels: 1=skin, 7/8=ears, 10=nose, 14=neck.
        Protected: eyes(4,5), brows(2,3), lips(12,13), mouth(11).
        feather: 0.0 = hard edge, 1.0 = very soft/wide feather.
        """
        import hashlib
        import torch
        import cv2

        # Cache key: hash of raw image bytes + dimensions + feather
        img_bytes = img.tobytes()
        cache_key = hashlib.md5(img_bytes[:4096] + img_bytes[-4096:]).hexdigest() + f"_{img.size}_{feather:.2f}"
        if self._mask_cache_key == cache_key and self._mask_cache is not None:
            return self._mask_cache

        self._ensure_parsing_model()
        device = self._get_device()

        # BiSeNet preprocessing: resize to 512x512, normalize with ImageNet stats
        resized = img.resize((512, 512), resample=Image.BILINEAR)
        input_tensor = self._transform(resized).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = self._parsing_model(input_tensor)
        # Use feat_out (index 0) for inference
        logits = outputs[0].squeeze(0).cpu().numpy()  # (19, 512, 512)
        labels_512 = logits.argmax(0)  # (512, 512)

        # Resize labels back to original image size
        labels_pil = Image.fromarray(labels_512.astype(np.uint8))
        labels = np.array(labels_pil.resize((img.width, img.height), resample=Image.NEAREST))

        # BiSeNet (yakhyo/face-parsing) labels:
        # 0=bg, 1=skin, 2=l_brow, 3=r_brow, 4=l_eye, 5=r_eye, 6=glasses,
        # 7=l_ear, 8=r_ear, 9=earring, 10=nose, 11=mouth, 12=u_lip,
        # 13=l_lip, 14=neck, 15=necklace, 16=cloth, 17=hair, 18=hat
        skin_labels = {1, 7, 8, 10, 14}  # skin + ears + nose + neck
        mask = np.isin(labels, list(skin_labels)).astype(np.uint8) * 255

        # Morphological closing: fill small holes/gaps inside skin regions
        # This fixes patchy detection on profiles without extending beyond face
        # Must happen BEFORE nostril/mouth exclusion so closing doesn't refill them
        short_edge = min(img.width, img.height)
        close_k = max(15, int(short_edge * 0.04))
        if close_k % 2 == 0:
            close_k += 1
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

        # Don't let closing bleed into non-face areas
        # Non-face: brows(2,3), eyes(4,5), glasses(6), earring(9),
        # mouth(11), lips(12,13), necklace(15), cloth(16), hair(17), hat(18)
        non_face_hard = {2, 3, 4, 5, 6, 9, 11, 12, 13, 15, 16, 17, 18}
        hard_block = np.isin(labels, list(non_face_hard)).astype(np.uint8) * 255
        mask = np.where(hard_block > 0, 0, mask).astype(np.uint8)

        # Exclude nostrils: only the actual dark nostril holes
        nose_region = (labels == 10)
        if nose_region.any():
            gray = np.array(img.convert('L'))
            nose_pixels = gray[nose_region]
            # Only exclude very dark pixels (actual holes, not nose sides)
            nostril_thresh = np.percentile(nose_pixels, 10)  # darkest 10% only
            nostrils = nose_region & (gray <= nostril_thresh)
            nostril_u8 = nostrils.astype(np.uint8) * 255
            # Minimal dilation — just the hole itself
            nk = max(3, int(short_edge * 0.005))
            if nk % 2 == 0:
                nk += 1
            nostril_u8 = cv2.dilate(nostril_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nk, nk)))
            mask = np.where(nostril_u8 > 0, 0, mask).astype(np.uint8)

        # Protect eye socket / eyelid area: dilate eye+brow regions and subtract
        eye_brow_labels = {2, 3, 4, 5}  # l_brow, r_brow, l_eye, r_eye
        eye_brow_mask = np.isin(labels, list(eye_brow_labels)).astype(np.uint8) * 255
        # Dilate to cover eyelid skin around eyes/brows
        protect_k = max(5, int(short_edge * 0.025))
        if protect_k % 2 == 0:
            protect_k += 1
        protect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (protect_k, protect_k))
        eye_protection = cv2.dilate(eye_brow_mask, protect_kernel, iterations=1)
        # Subtract protection zone from skin mask
        mask = np.where(eye_protection > 0, 0, mask).astype(np.uint8)

        # Protect mouth/mustache area: dilate mouth+lip regions and subtract
        mouth_labels = {11, 12, 13}  # mouth, upper lip, lower lip
        mouth_mask = np.isin(labels, list(mouth_labels)).astype(np.uint8) * 255
        mouth_k = max(5, int(short_edge * 0.02))
        if mouth_k % 2 == 0:
            mouth_k += 1
        mouth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mouth_k, mouth_k))
        mouth_protection = cv2.dilate(mouth_mask, mouth_kernel, iterations=1)
        mask = np.where(mouth_protection > 0, 0, mask).astype(np.uint8)

        # Feather edges: controlled by feather param (0=hard, 1=very soft)
        # Range: 0.8% to 5% of short edge
        feather_frac = 0.008 + feather * 0.042
        blur_k = max(7, int(short_edge * feather_frac))
        if blur_k % 2 == 0:
            blur_k += 1
        mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)

        result = mask.astype(np.float32) / 255.0
        self._mask_cache_key = cache_key
        self._mask_cache = result
        return result

    def get_mask_overlay(self, img, opacity=0.5, feather=0.5):
        """Render a red overlay on detected skin areas for visualization.

        Args:
            img: Input PIL Image (RGB).
            opacity: 0.0 = no overlay, 1.0 = solid red.

        Returns:
            PIL Image with red overlay on skin mask.
        """
        img_array = np.array(img)
        skin_mask = self._get_skin_mask(img, feather=feather)

        # Red overlay: [255, 0, 0]
        overlay = img_array.astype(np.float32)
        mask = skin_mask[:, :, np.newaxis] * opacity
        red = np.array([255.0, 50.0, 50.0])
        overlay = overlay * (1 - mask) + (overlay * (1 - 0.6) + red * 0.6) * mask
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return Image.fromarray(overlay)

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

    def retouch(self, img, strength=0.5, detail_size=0.05, texture_amount=0.5, texture_scale=0.5, feather=0.5):
        """Remove blemishes via two-radius frequency separation (no AI generation).

        Uses two Gaussian blur radii to separate the image into three bands:
        - Fine pores (very high freq): original - blur(original, r_small)
        - Medium features/blemishes: between r_small and r_big — smoothed away
        - Face structure (very low freq): blur(original, r_big)

        After smoothing, synthetic noise texture is added to avoid plastic look.

        Args:
            img: Input PIL Image (RGB).
            strength: 0.0 = no change, 1.0 = full blemish removal.
            detail_size: Controls the large blur radius (fraction of short edge).
                         Larger = smoother base = more aggressive blemish removal.
            texture_amount: 0.0 = no texture (plastic), 1.0 = strong pore texture.
            texture_scale: 0.0 = fine grain, 1.0 = coarse pores.

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

        # Get skin mask from face parsing
        skin_mask = self._get_skin_mask(img, feather=feather)

        # Convert to LAB — work on L channel (luminance/texture)
        img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB).astype(np.float32)
        L_orig = img_lab[:, :, 0]

        # r_big: creates smooth base without blemishes (controlled by detail_size)
        r_big = max(5, int(short_edge * detail_size))
        if r_big % 2 == 0:
            r_big += 1
        sigma_big = r_big * 0.4
        ksize_big = r_big * 2 + 1

        # Create smooth base (low frequency: features larger than r_big)
        L_smooth_base = cv2.GaussianBlur(L_orig, (ksize_big, ksize_big), sigma_big)

        # Add synthetic skin texture (Gaussian noise + blur to mimic pore scale)
        L_textured = L_smooth_base
        if texture_amount > 0:
            # Target amplitude after blur: 2-9 L units (visible range on 0-255 scale)
            target_amplitude = 2.0 + texture_amount * 7.0

            # Pore blur radius: texture_scale controls grain size
            # 0.0 = fine grain (~1px), 1.0 = coarse pores (~1% of short edge)
            pore_radius = max(1, int(1 + texture_scale * short_edge * 0.01))
            if pore_radius % 2 == 0:
                pore_radius += 1
            pore_sigma = pore_radius * 0.6

            # Generate deterministic noise (seeded by image dimensions for consistency)
            rng = np.random.RandomState(seed=h * 10000 + w)
            noise = rng.normal(0, 1.0, (h, w)).astype(np.float32)

            # Blur noise to pore scale (this reduces amplitude)
            noise = cv2.GaussianBlur(noise, (pore_radius * 2 + 1, pore_radius * 2 + 1), pore_sigma)

            # Rescale to target amplitude (compensate for blur reduction)
            noise_std = noise.std()
            if noise_std > 0:
                noise = noise * (target_amplitude / noise_std)

            L_textured = L_smooth_base + noise

        # Blend onto original with strength and skin mask
        L_result = L_orig * (1 - strength * skin_mask) + L_textured * (strength * skin_mask)

        # Keep original chrominance (preserves exact skin tone)
        result_lab = np.stack([L_result, img_lab[:, :, 1], img_lab[:, :, 2]], axis=-1)
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        result_rgb = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)

        print(f"[skin_retouch] r_big={r_big}, texture={texture_amount:.2f}, "
              f"scale={texture_scale:.2f}, strength={strength}")

        return Image.fromarray(result_rgb)

    def even_tone(self, img, strength=0.5, feather=0.5):
        """Even out skin tone by smoothing chrominance variations within skin.

        Uses masked Gaussian blur on A/B channels to average out redness,
        blotchiness, and uneven pigmentation. L-channel stays untouched
        so all texture is preserved.

        Args:
            img: Input PIL Image (RGB).
            strength: 0.0 = no change, 1.0 = fully evened tone.

        Returns:
            Tone-evened PIL Image (RGB).
        """
        import cv2

        if strength <= 0:
            return img.copy()

        strength = min(strength, 1.0)
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        short_edge = min(h, w)

        # Get skin mask from face parsing
        skin_mask = self._get_skin_mask(img, feather=feather)

        # Convert to LAB
        img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB).astype(np.float32)
        L = img_lab[:, :, 0]
        A = img_lab[:, :, 1]
        B = img_lab[:, :, 2]

        # Large blur for tone evening (10% of short edge)
        radius = max(15, int(short_edge * 0.10))
        if radius % 2 == 0:
            radius += 1
        blur_size = radius * 2 + 1
        sigma = radius * 0.4

        # Masked blur: only average skin chrominance (excludes hair/background)
        mask_blur = cv2.GaussianBlur(skin_mask, (blur_size, blur_size), sigma)
        mask_blur = np.maximum(mask_blur, 1e-6)

        A_smooth = cv2.GaussianBlur(A * skin_mask, (blur_size, blur_size), sigma) / mask_blur
        B_smooth = cv2.GaussianBlur(B * skin_mask, (blur_size, blur_size), sigma) / mask_blur

        # Blend chrominance toward local skin average
        A_result = A * (1 - strength * skin_mask) + A_smooth * (strength * skin_mask)
        B_result = B * (1 - strength * skin_mask) + B_smooth * (strength * skin_mask)

        # L untouched — all texture preserved
        result_lab = np.stack([L, A_result, B_result], axis=-1)
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        result_rgb = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)

        print(f"[skin_tone] A diff: mean={np.abs(A_result - A).mean():.2f}, max={np.abs(A_result - A).max():.2f}")
        print(f"[skin_tone] B diff: mean={np.abs(B_result - B).mean():.2f}, max={np.abs(B_result - B).max():.2f}")

        return Image.fromarray(result_rgb)
