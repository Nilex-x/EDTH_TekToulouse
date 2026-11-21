"""Axelera Voyager SDK preprocessing transform for RF-DETR (``axcompile --transform``).

Matches ``export_rfdetr_axelera.py`` and ``predict_onnx_baseline.py``:
  RGB -> bilinear resize to 384x384 -> float32 [0, 1] -> ImageNet normalize -> CHW.

Usage on an Axelera host:
    axcompile -i exports/rfdetr/backbone/inference_model.onnx \\
        --imageset ../calibration --transform preprocess.py \\
        --input-shape 1,3,384,384 \\
        -o exports/rfdetr/metis_axm/
"""

from __future__ import annotations

import numpy as np
from PIL import Image

IMG_SIZE = 384
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _to_pil_rgb(image: object) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def get_preprocess_transform(image: object, size: int = IMG_SIZE) -> np.ndarray:
    """Voyager SDK calibration / inference callback returning float32 CHW tensor."""
    rgb = _to_pil_rgb(image).resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    return ((chw - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)
