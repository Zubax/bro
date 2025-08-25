from __future__ import annotations
from typing import Any
import json
import base64
import traceback
import re
from io import BytesIO
from datetime import datetime
from pathlib import Path
import logging
import subprocess

from PIL import Image

from openai import OpenAI
from openai.types import FileObject
from openai.types.file_create_params import ExpiresAfter


_logger = logging.getLogger(__name__)


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


def openai_upload_files(
    client: OpenAI,
    files: list[Path],
    *,
    expiration_time: int = 3600 * 24 * 30,
) -> list[FileObject]:
    _logger.info(f"📤 Uploading {len(files)} files: {[str(f) for f in files]}")
    file_objects: list[FileObject] = []
    for file in files:
        fobj = client.files.create(
            file=(file.name, file.read_bytes()),  # File name is required! Otherwise the model will refuse to use it!
            purpose="user_data",
            expires_after=ExpiresAfter(anchor="created_at", seconds=int(expiration_time)),
        )
        file_objects.append(fobj)
    _logger.debug(f"Uploaded {len(file_objects)} files: {file_objects}")
    return file_objects


def locate_file(filename: str | Path) -> Path | None:
    """
    If the filename is absolute, return it if it exists.
    If it's relative, search predefined locations for a matching file name.
    Returns the resolved Path if found, otherwise None.
    """
    fn = Path(filename).expanduser()
    if fn.exists() and not fn.is_dir():
        return fn.resolve()
    if fn.is_absolute():
        return None
    for path in Path.home().rglob(fn.name):
        if path.is_file() and path.name == fn.name:
            return path.resolve()
    return None


def run_shell_command(cmd: str) -> tuple[int, str, str]:
    _logger.debug(f"🖥️ Running shell command: {cmd!r}")
    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = proc.communicate()
    _logger.debug(f"Command exited with status {proc.returncode}")
    return proc.returncode, stdout, stderr


def split_trailing_json(text: str) -> tuple[str, Any]:
    """
    Extract parsed JSON from the end of the message. None if not found.
    Sometimes, simple LLMs forget to generate proper Markdown code blocks, so this function attempts to be forgiving.
    Returns the parsed JSON and the other text before it.
    """
    if (js := _RE_JSON_BACKTICKS.search(text)) is not None:
        try:
            return text[: js.start()].rstrip(), json.loads(js[1])
        except Exception as ex:
            _logger.debug(f"Failed to parse JSON from backticks: {ex}", exc_info=True)
            return text, None
    js = text.splitlines()[-1]
    try:
        return text[: text.rfind(js)].rstrip(), json.loads(js)
    except Exception as ex:
        _logger.debug(f"Failed to parse JSON from last line: {ex}", exc_info=True)
        return text, None


_RE_JSON_BACKTICKS = re.compile(r"(?ims)^```(?:json)?\n(.+)\n```$")
