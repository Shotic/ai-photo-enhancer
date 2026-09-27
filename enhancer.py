import os
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import onnxruntime as ort
from huggingface_hub import hf_hub_download

class ImageEnhancer:
    def __init__(self, model_repo="Heliosoph/realesrgan-onnx", model_filename="realesr-general-x4v3.onnx"):
        self.model_path = self._ensure_model(model_repo, model_filename)
        
        # Configure ONNX Runtime execution providers
        available_providers = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        if "DmlExecutionProvider" in available_providers:
            providers.append("DmlExecutionProvider")
        providers.append("CPUExecutionProvider")
        
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def _ensure_model(self, repo_id, filename):
        local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, filename)
        
        if os.path.exists(local_path):
            return local_path
            
        print(f"Downloading model {filename} from {repo_id}...")
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
        return downloaded_path

    def _infer_patch(self, patch_np):
        # Convert HWC [0, 1] float32 to BCHW
        inp = patch_np.transpose(2, 0, 1)[None, ...].astype(np.float32)
        out = self.session.run(None, {self.input_name: inp})[0][0]
        # Convert BCHW back to HWC [0, 255] uint8
        out = out.transpose(1, 2, 0)
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)

    def _tiled_super_resolution(self, img_np, tile_size=384, overlap=32, scale=4):
        h, w, c = img_np.shape
        if h <= tile_size and w <= tile_size:
            return self._infer_patch(img_np)
        
        out_h, out_w = h * scale, w * scale
        output = np.zeros((out_h, out_w, c), dtype=np.float32)
        weights = np.zeros((out_h, out_w, 1), dtype=np.float32)
        
        stride = tile_size - overlap
        
        y_starts = list(range(0, h - tile_size, stride))
        if not y_starts or y_starts[-1] + tile_size < h:
            y_starts.append(max(0, h - tile_size))
            
        x_starts = list(range(0, w - tile_size, stride))
        if not x_starts or x_starts[-1] + tile_size < w:
            x_starts.append(max(0, w - tile_size))
            
        y_starts = sorted(list(set(y_starts)))
        x_starts = sorted(list(set(x_starts)))
        
        pad_out = overlap * scale // 2
        
        for y in y_starts:
            for x in x_starts:
                tile = img_np[y:y+tile_size, x:x+tile_size]
                th, tw, _ = tile.shape
                enhanced_tile = self._infer_patch(tile)
                eth, etw, _ = enhanced_tile.shape
                
                tile_weight = np.ones((eth, etw, 1), dtype=np.float32)
                if pad_out > 0:
                    ramp_y = np.linspace(0.01, 1.0, pad_out, dtype=np.float32)
                    ramp_x = np.linspace(0.01, 1.0, pad_out, dtype=np.float32)
                    if y > 0:
                        tile_weight[:pad_out, :, 0] *= ramp_y[:, None]
                    if y + tile_size < h:
                        tile_weight[-pad_out:, :, 0] *= ramp_y[::-1, None]
                    if x > 0:
                        tile_weight[:, :pad_out, 0] *= ramp_x[None, :]
                    if x + tile_size < w:
                        tile_weight[:, -pad_out:, 0] *= ramp_x[None, ::-1]
                
                oy, ox = y * scale, x * scale
                output[oy:oy+eth, ox:ox+etw] += enhanced_tile.astype(np.float32) * tile_weight
                weights[oy:oy+eth, ox:ox+etw] += tile_weight
                
        output = np.clip(output / np.maximum(weights, 1e-4), 0, 255).astype(np.uint8)
        return output

    def enhance(
        self,
        image: Image.Image,
        target_scale: str = "4x",
        sharpness_boost: float = 1.1,
        vibrancy_boost: float = 1.05,
        contrast_boost: float = 1.03,
        skin_smoothing: float = 0.0,
        max_input_dimension: int = 1920
    ) -> Image.Image:
        if image is None:
            raise ValueError("No image provided.")

        # Normalize EXIF orientation (critical for iOS Safari uploads)
        image = ImageOps.exif_transpose(image)

        # Handle alpha channel (RGBA/PNG)
        alpha = None
        if image.mode == "RGBA":
            alpha = image.split()[-1]
            image = image.convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        # Guard rail against runaway OOM on huge camera files
        orig_w, orig_h = image.size
        max_dim = max(orig_w, orig_h)
        if max_dim > max_input_dimension:
            scale_down = max_input_dimension / float(max_dim)
            new_w = int(orig_w * scale_down)
            new_h = int(orig_h * scale_down)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            if alpha:
                alpha = alpha.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Prepare normalized float array [0, 1] for neural network
        img_np = np.array(image, dtype=np.float32) / 255.0

        # Run 4x deep learning super-resolution
        enhanced_np = self._tiled_super_resolution(img_np, tile_size=384, overlap=32, scale=4)
        enhanced_img = Image.fromarray(enhanced_np)

        # Downscale if user chose 1x or 2x
        if target_scale == "1x (Enhanced Original Size)":
            enhanced_img = enhanced_img.resize(image.size, Image.Resampling.LANCZOS)
            if alpha:
                enhanced_img.putalpha(alpha)
        elif target_scale == "2x":
            target_w = image.size[0] * 2
            target_h = image.size[1] * 2
            enhanced_img = enhanced_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            if alpha:
                alpha_2x = alpha.resize((target_w, target_h), Image.Resampling.LANCZOS)
                enhanced_img.putalpha(alpha_2x)
        else: # 4x
            if alpha:
                alpha_4x = alpha.resize(enhanced_img.size, Image.Resampling.LANCZOS)
                enhanced_img.putalpha(alpha_4x)

        # Apply portrait retouching adjustments
        if skin_smoothing > 0.0:
            # Gentle bilateral-style smoothing blend to soften blotches while keeping edges
            blurred = enhanced_img.filter(ImageFilter.GaussianBlur(radius=1.5 * skin_smoothing))
            enhanced_img = Image.blend(enhanced_img, blurred, alpha=min(skin_smoothing * 0.4, 0.5))

        if sharpness_boost != 1.0:
            enhancer = ImageEnhance.Sharpness(enhanced_img)
            enhanced_img = enhancer.enhance(sharpness_boost)

        if vibrancy_boost != 1.0:
            enhancer = ImageEnhance.Color(enhanced_img)
            enhanced_img = enhancer.enhance(vibrancy_boost)

        if contrast_boost != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced_img)
            enhanced_img = enhancer.enhance(contrast_boost)

        return enhanced_img
