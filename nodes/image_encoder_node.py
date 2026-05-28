import os
import tempfile
import numpy as np
from PIL import Image

try:
    import folder_paths
except ImportError:
    folder_paths = None


class LlamaImageEncoderNode:
    """Converts a ComfyUI IMAGE tensor to a temp image file for llama-cli --image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "ComfyUI image tensor",
                }),
            },
            "optional": {
                "format": (["png", "jpg"], {
                    "default": "png",
                    "tooltip": "Output image format",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("image_path",)
    FUNCTION = "encode"
    CATEGORY = "AI/LlamaCpp"

    def encode(self, image, format="png"):
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        temp_dir = None
        if folder_paths:
            try:
                temp_dir = folder_paths.get_temp_directory()
            except Exception:
                pass

        suffix = ".png" if format == "png" else ".jpg"
        fd, path = tempfile.mkstemp(suffix=suffix, prefix="llama_img_", dir=temp_dir)
        os.close(fd)

        if format == "jpg":
            if pil_img.mode == "RGBA":
                pil_img = pil_img.convert("RGB")
            pil_img.save(path, "JPEG", quality=95)
        else:
            pil_img.save(path, "PNG")

        return (path,)
