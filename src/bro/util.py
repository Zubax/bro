from __future__ import annotations
from typing import Any
import os
import sys
import tempfile
import json
import base64
import traceback
import re
from io import BytesIO
from datetime import datetime
from pathlib import Path
import logging
import socket
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


def prune_context_text_only(
    context: list[dict[str, Any]],
    keep_roles: frozenset[str] = frozenset(("user", "system", "assistant")),
) -> list[dict[str, Any]]:
    """
    Constructs a new LLM context that only contains text messages from the specified roles.
    This is intended for use for summarization and long context truncation.
    Everything else (images, files, tool calls) is removed.
    Consecutive duplicate messages are also removed.
    """
    out_ctx: list[dict[str, Any]] = []
    for src in context:
        if src.get("role") not in keep_roles:
            continue
        content: list[dict[str, Any]] = []
        out_item = {"role": src["role"], "content": content}
        for in_item in src.get("content", []) or []:
            match in_item:
                case {"type": ty, "text": str(text), **_rest} if ty in {"text", "input_text", "output_text"}:
                    content.append({"type": ty, "text": text})
                case _:
                    pass
        if not content:
            continue  # No content left after pruning, skip this message
        if out_ctx and out_ctx[-1] == out_item:  # Skip duplicates (such as repeated screenshots)
            continue
        out_ctx.append(out_item)
    return out_ctx


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


def get_upstream_ip() -> str | None:
    """
    Get a system's upstream IP address; returns None on failure.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("1.1.1.1", 80))  # datagram connect does not send packets
            return str(s.getsockname()[0])
    except OSError:
        return None


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


def run_python_code(code: str) -> tuple[int, str, str]:
    """
    Execute Python source in a child process using the current interpreter.
    Returns (exit_code, stdout, stderr).
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        path = f.name
    try:
        proc = subprocess.run([sys.executable, path], capture_output=True, text=True)
        return proc.returncode, proc.stdout, proc.stderr
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def split_trailing_json(text: str) -> tuple[str, Any]:
    """
    Extract parsed JSON from the end of the message. None if not found.
    Sometimes, simple LLMs forget to generate proper Markdown code blocks, so this function attempts to be forgiving.
    VERY FORGIVING! Garbage in, garbage out; be sure to validate the output JSON schema.
    Returns the parsed JSON and the other text before it.
    """
    for regexp in _RE_TRAILING_JSON_WONKY:
        if (js := regexp.search(text)) is not None and js[1]:
            try:
                return text[: js.start()].rstrip(), json.loads(js[1])
            except ValueError:
                pass
    return text, None


_RE_TRAILING_JSON_WONKY = [
    re.compile(r"(?is)`+(?:json)?\n(.+?)\n*`*\s*$"),
    re.compile(r"(?i)(.+?)\s*$"),
]
