from __future__ import annotations
from typing import Any
import base64
from io import BytesIO
from PIL import Image


def image_to_base64(im: Image.Image) -> str:
    buf = BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def truncate(x: list[Any], head: int = 100, tail: int = 1000) -> list[Any]:
    if head <= 0 or tail <= 0:
        raise ValueError("head and tail must be positive integers")
    if len(x) <= head + tail:
        return x
    return x[:head] + x[-tail:]
