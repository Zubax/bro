from __future__ import annotations
from typing import Any
import base64
import traceback
from io import BytesIO
from datetime import datetime
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


def format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def get_local_time_llm() -> dict[str, Any]:
    """
    LLM-friendly current time representation.
    """
    now = datetime.now()
    return {
        "iso": now.isoformat(),
        "posix": now.timestamp(),
        "tz": (str(now.astimezone().tzinfo) if now.astimezone().tzinfo else "UTC"),
        "year": now.year,
        "month": now.month,
        "month_name": now.strftime("%B"),
        "day": now.day,
        "hour": now.hour,
        "minute": now.minute,
        "second": now.second,
        "weekday": now.isoweekday(),
        "weekday_name": now.strftime("%A"),
    }
