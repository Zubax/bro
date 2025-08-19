#!/usr/bin/env python3

import os
import time
import base64
import argparse
from typing import Any, List, Dict
import shutil
import subprocess
import tempfile
from io import BytesIO
import logging
from pathlib import Path
import sys
from datetime import datetime

# Third-party imports
from openai import OpenAI
import pyautogui
import mss
from PIL import Image

# Optional imports
try:
    import readline  # noqa: F401
except ImportError:
    pass  # readline is not available on all platforms, but input() will still work

pyautogui.FAILSAFE = False  # disable corner abort

_logger = logging.getLogger(__name__)
_log_dir = Path(f".bro/logs/{time.strftime('%Y-%m-%d-%H-%M-%S')}")
_log_dir.mkdir(exist_ok=True, parents=True)


def main():
    _setup_logging()

    if env_task := os.environ.get("BRO_TASK"):
        task = env_task
    else:
        task = input("Describe the task:\n").strip()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = "computer-use-preview"

    agent = Agent(client=client, model=model)
    agent.run(task)


class Agent:
    def __init__(self, *, client: OpenAI, model: str):
        self.client = client
        self.model = model

        screen_w, screen_h = pyautogui.size()
        self._tools = [
            {
                "type": "computer_use_preview",
                "display_width": int(screen_w),
                "display_height": int(screen_h),
                "environment": "linux",
            }
        ]

    def run(self, task: str) -> None:
        # TODO use instructions!
        items = [{"role": "user", "content": task}]
        while (not items) or (items[-1].get("role") != "assistant"):
            _logger.debug(f"Sending a request with input items: {items}")
            response = self.client.responses.create(
                model=self.model,
                input=items,
                tools=self._tools,
                truncation="auto",
            )
            _logger.debug(f"Received a response: {response}")
            output = [x.model_dump() for x in response.output]
            if not output:
                _logger.warning("No output from model")
            items += output
            for item in output:
                items += self._process_item(item)

    def _process_item(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        match item["type"]:
            case "message":
                _logger.info(f"Message: {item['content'][0]['text']}")
                return []

            case "reasoning":
                _logger.info(f"Reasoning: {item}")
                return []

            case "function_call":
                name, args = item["name"], json.loads(item["arguments"])
                _logger.info(f"Processing function call: {name}({args})")
                _logger.error("TODO: FUNCTION CALL NOT IMPLEMENTED")  # TODO FIXME
                return [
                    {
                        "type": "function_call_output",
                        "call_id": item["call_id"],
                        "output": "ERROR: NOT IMPLEMENTED",
                    }
                ]

            case "computer_call":
                _do_computer_action(item["action"])
                screenshot_b64 = _screenshot_base64()
                # The Western society is ailed with a mindless, damaging obsession with excessive safety.
                pending_checks = item.get("pending_safety_checks", [])
                for check in pending_checks:
                    _logger.warning(f"✅ Skipping safety check: {check['message']}")
                call_output = {
                    "type": "computer_call_output",
                    "call_id": item["call_id"],
                    "acknowledged_safety_checks": pending_checks,
                    "output": {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{screenshot_b64}",
                    },
                }
                return [call_output]

            case _:
                _logger.error(f"Unrecognized item type: {item['type']}")
                _logger.debug(f"Full item: {item}")
                # TODO FIXME not sure if the below is up to the API spec; update it if needed
                return [
                    {
                        "type": "error",
                        "error": f"Unrecognized item type: {item['type']}",
                        "item": item,
                    }
                ]


def _screenshot_base64() -> str:
    im = None

    # 1) MSS on the virtual screen (works on X11)
    try:
        with mss.mss() as sct:
            mon = sct.monitors[0]  # bounding box of all displays
            img = sct.grab(
                {
                    "top": mon["top"],
                    "left": mon["left"],
                    "width": mon["width"],
                    "height": mon["height"],
                }
            )
            im = Image.frombytes("RGB", img.size, img.rgb)
    except Exception as ex:
        _logger.debug("Failed to capture screenshot with mss: %s", ex, exc_info=True)

    if im is None:
        # 2) Wayland-friendly tools
        for tool, args in (
            ("gnome-screenshot", ["-f"]),  # GNOME Wayland
            ("grim", ["-"]),  # wlroots compositors
        ):
            if shutil.which(tool):
                try:
                    if tool == "gnome-screenshot":
                        fd, path = tempfile.mkstemp(suffix=".png")
                        os.close(fd)
                        subprocess.run([tool, *args, path], check=True)
                        with open(path, "rb") as f:
                            data = f.read()
                        os.unlink(path)
                        im = Image.open(BytesIO(data))
                    else:  # grim to stdout
                        data = subprocess.check_output([tool, *args])
                        im = Image.open(BytesIO(data))
                except Exception as ex:
                    _logger.debug("Failed to capture screenshot with %s: %s", tool, ex, exc_info=True)
                if im is not None:
                    break

    if im is None:
        # 3) PyAutoGUI fallback (may require scrot on some setups)
        try:
            im = pyautogui.screenshot()
        except Exception as ex:
            _logger.debug("Failed to capture screenshot with pyautogui: %s", ex, exc_info=True)

    if im is None:
        raise RuntimeError("Could not capture screenshot on this session")

    # Save screenshot to log dir with ISO datetime
    now_str = datetime.now().isoformat(timespec="seconds").replace(":", "-")
    screenshot_path = _log_dir / f"screenshot-{now_str}.png"
    _logger.info(f"📷 Saving screenshot to {screenshot_path}")
    buf = BytesIO()
    im.save(buf, format="PNG")
    im.save(screenshot_path, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


_KEYMAP = {
    "CTRL": "ctrl",
    "CONTROL": "ctrl",
    "ALT": "alt",
    "SHIFT": "shift",
    "ENTER": "enter",
    "RETURN": "enter",
    "TAB": "tab",
    "ESC": "esc",
    "ESCAPE": "esc",
    "SPACE": "space",
    "BACKSPACE": "backspace",
    "DELETE": "delete",
    "HOME": "home",
    "END": "end",
    "PAGE_UP": "pageup",
    "PAGE_DOWN": "pagedown",
    "LEFT": "left",
    "RIGHT": "right",
    "UP": "up",
    "DOWN": "down",
    "META": "winleft",  # GNOME “Super”
    "CMD": "winleft",
    "SUPER": "winleft",
}


def _to_pyauto_key(k: str) -> str:
    k = k.strip()
    return _KEYMAP.get(k.upper(), k.lower())


def _aget(obj: Any, attr: str, default=None, key: str | None = None):
    sentinel = object()
    v = getattr(obj, attr, sentinel)
    if v is not sentinel:
        return v
    if isinstance(obj, dict):
        return obj.get(key or attr, default)
    return default


class ActionError(RuntimeError):
    pass


def _do_computer_action(action: dict[str, Any]) -> None:
    _logger.debug(f"Computer action: {action}")
    match atype := _aget(action, "type"):
        case "move":
            x = _aget(action, "x")
            y = _aget(action, "y")
            _logger.info(f"🧑🏻‍💻 Moving cursor to {x}, {y}")
            pyautogui.moveTo(x, y, duration=0)

        case "click":
            x = _aget(action, "x")
            y = _aget(action, "y")
            button = _aget(action, "button", default="left")
            _logger.info(f"🧑🏻‍💻 Clicking {button} button at {x}, {y}")
            pyautogui.click(x=x, y=y, button=button)

        case "double_click":
            x = _aget(action, "x")
            y = _aget(action, "y")
            _logger.info(f"🧑🏻‍💻 Double clicking at {x}, {y}")
            pyautogui.doubleClick(x=x, y=y)

        case "drag":
            if path := _aget(action, "path", default=[]):
                sx, sy = path[0]
                pyautogui.moveTo(sx, sy, duration=0)
                pyautogui.mouseDown()
                for px, py in path[1:]:
                    _logger.info(f"🧑🏻‍💻 Dragging to {px}, {py}")
                    pyautogui.moveTo(px, py, duration=0)
                pyautogui.mouseUp()

        case "scroll":
            x = _aget(action, "x")
            y = _aget(action, "y")
            sx = int(_aget(action, "scroll_x", default=0) or 0)
            sy = int(_aget(action, "scroll_y", default=0) or 0)
            if x is not None and y is not None:
                _logger.info(f"🧑🏻‍💻 Moving the cursor to scroll at {x}, {y}")
                pyautogui.moveTo(x, y, duration=0)
            if sy:
                _logger.info(f"🧑🏻‍💻 Scrolling by {sy}")
                pyautogui.scroll(sy)
            if sx:
                _logger.info(f"🧑🏻‍💻 Horizontally scrolling by {sx}")
                try:
                    pyautogui.hscroll(sx)
                except Exception as ex:
                    raise ActionError(f"hscroll failed; possibly not supported on this platform: {ex}")

        case "type":
            if text := _aget(action, "text", default=""):
                _logger.info(f"🧑🏻‍💻 Typing {text}")
                pyautogui.write(text, interval=0.01)

        case "keypress":
            keys = _aget(action, "keys", default=[])
            keys = [_to_pyauto_key(k) for k in keys if k]
            if len(keys) == 1:
                _logger.info(f"🧑🏻‍💻 Pressing {keys[0]}")
                pyautogui.press(keys[0])
            elif len(keys) > 1:
                _logger.info(f"🧑🏻‍💻 Pressing {keys}")
                pyautogui.hotkey(*keys)

        case "wait":
            ms = _aget(action, "ms", default=1000)
            _logger.info(f"🧑🏻‍💻 Waiting for {ms} ms")
            time.sleep(ms / 1000.0)

        case "screenshot":
            # no-op; we always return a fresh screenshot after each step
            pass

        case _:
            raise ActionError(f"Unrecognized action: {atype!r}")


def _setup_logging() -> None:
    logging.getLogger().setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-3.3s %(name)s: %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(console_handler)

    # File handler
    log_file_path = _log_dir / "bro.log"
    file_handler = logging.FileHandler(str(log_file_path), mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    # Per-module customizations
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("pyautogui").setLevel(logging.WARNING)
    logging.getLogger("mss").setLevel(logging.WARNING)


if __name__ == "__main__":
    main()
