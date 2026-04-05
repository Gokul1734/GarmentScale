"""Decode uploads with EXIF orientation so width/height match how the photo was taken."""
from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps


def decode_image_bytes_to_bgr(data: bytes) -> np.ndarray | None:
    """Return BGR uint8 image or None. Applies EXIF transpose (phones often need this)."""
    if not data:
        return None
    try:
        pil = Image.open(io.BytesIO(data))
        pil = ImageOps.exif_transpose(pil)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        rgb = np.asarray(pil, dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        pass
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
