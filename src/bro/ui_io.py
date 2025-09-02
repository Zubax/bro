from __future__ import annotations
import os
import time
from typing import Any
from abc import ABC, abstractmethod
import shutil
import subprocess
import tempfile
from io import BytesIO
import logging
from dataclasses import dataclass

import pyautogui
import pynput.keyboard  # PyAutoGUI cannot type non-ASCII characters
import mss
from PIL import Image


pyautogui.FAILSAFE = False  # disable corner abort

_logger = logging.getLogger(__name__)


class UIActionError(RuntimeError):
    def __init__(self, message: str, action: Any) -> None:
        super().__init__(message)
        self.action = action


ScreenCoord = tuple[int, int]  # (x, y) in screen coordinates


@dataclass(frozen=True)
class Action:
    pass


@dataclass(frozen=True)
class MoveAction(Action):
    coord: ScreenCoord


@dataclass(frozen=True)
class ClickAction(Action):
    BUTTON_LEFT = "left"
    BUTTON_RIGHT = "right"
    BUTTON_MIDDLE = "middle"

    coord: ScreenCoord
    button: str = BUTTON_LEFT
    count: int = 1  # 1 for click, 2 for double click


@dataclass(frozen=True)
class DragAction(Action):
    path: list[ScreenCoord]  # list of (x, y) points


@dataclass(frozen=True)
class ScrollAction(Action):
    coord: ScreenCoord | None
    scroll_x: int = 0
    scroll_y: int = 0


@dataclass(frozen=True)
class TypeAction(Action):
    text: str


@dataclass(frozen=True)
class KeyPressAction(Action):
    keys: list[str]


@dataclass(frozen=True)
class WaitAction(Action):
    duration: float | None = None  # seconds
    """If None, the implementation will attempt to work out a reasonable delay itself."""


class UiObserver(ABC):
    @property
    @abstractmethod
    def screen_width_height(self) -> tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def screenshot(self) -> Image.Image:
        raise NotImplementedError


class UiController(UiObserver):
    @abstractmethod
    def do(self, action: Any) -> None:
        raise NotImplementedError


def make_controller() -> UiController:
    """
    Constructs a new implementation of UiController and UiObserver suitable for the current runtime environment.
    """
    return _Impl()


class _Impl(UiController):
    def __init__(self) -> None:
        self._kbd = pynput.keyboard.Controller()
        self._consecutive_waits: int = 0

    @property
    def screen_width_height(self) -> tuple[int, int]:
        sz = pyautogui.size()
        return int(sz[0]), int(sz[1])

    def screenshot(self) -> Image.Image:
        im = None
        try:
            _logger.debug(f"Capturing screenshot with mss")
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

        if im is None:  # Wayland-friendly tools
            for tool, args in (
                ("gnome-screenshot", ["-f"]),  # GNOME Wayland
                ("grim", ["-"]),  # wlroots compositors
            ):
                if shutil.which(tool):
                    _logger.debug(f"Capturing screenshot with {tool!r}...")
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
            _logger.debug(f"Capturing screenshot with pyautogui...")
            try:
                im = pyautogui.screenshot()
            except Exception as ex:
                _logger.debug("Failed to capture screenshot with pyautogui: %s", ex, exc_info=True)

        if im is None:
            raise RuntimeError("Could not capture screenshot on this session")
        return im

    def do(self, action: Any) -> None:
        _logger.debug(f"Computer action: {action}")
        if not isinstance(action, WaitAction):
            self._consecutive_waits = 0

        match action:
            case MoveAction(coord=(x, y)):
                _logger.info(f"🐁 Moving cursor to {x}, {y}")
                pyautogui.moveTo(x, y, duration=0)

            case ClickAction(coord=(x, y), button=button, count=count):
                if count == 1:
                    _logger.info(f"🐁 Clicking {button} button at {x}, {y}")
                    pyautogui.click(x=x, y=y, button=button)
                elif count == 2:
                    _logger.info(f"🐁 Double clicking at {x}, {y}")
                    pyautogui.doubleClick(x=x, y=y, button=button)
                else:
                    raise UIActionError(f"Invalid click count: {count}; add support for this if needed", action)

            case DragAction(path=path):
                if len(path) > 1:
                    sx, sy = path[0]
                    _logger.info(f"🐁 Drag start at {sx}, {sy}")
                    pyautogui.moveTo(sx, sy, duration=0)
                    pyautogui.mouseDown()
                    for px, py in path[1:]:
                        _logger.info(f"🐁 Dragging to {px}, {py}")
                        pyautogui.moveTo(px, py, duration=0)
                    pyautogui.mouseUp()
                else:
                    _logger.error(f"🐁 Drag path is incomplete, skipping: {action}")

            case ScrollAction(coord=coord, scroll_x=sx, scroll_y=sy):
                if coord:
                    x, y = coord
                    _logger.info(f"🐁 Moving the cursor to scroll at {x}, {y}")
                    pyautogui.moveTo(x, y, duration=0)
                if sy:
                    _logger.info(f"🐁 Scrolling by {sy}")
                    pyautogui.scroll(sy)
                if sx:
                    _logger.info(f"🐁 Horizontally scrolling by {sx}")
                    try:
                        pyautogui.hscroll(sx)
                    except Exception as ex:
                        _logger.exception(f"{action} failed; possibly not supported on this platform: {ex}")

            case TypeAction(text=text):
                _logger.info(f"⌨️ Typing {text!r}")
                # We can't use PyAutoGUI here because it can't type non-ASCII characters.
                # There is still a problem with Unicode though: Python iterates over Unicode codepoints,
                # but some characters (e.g. emojis) are represented as multiple codepoints.
                # This may cause an invalid split that may cause pynput to fail to type the character.
                for c in text:
                    try:
                        self._kbd.type(c)
                    except Exception as ex:
                        _logger.exception("Failed to type character %r: %s", c, ex)
                    time.sleep(0.1)  # limit input rate to avoid missing characters in slow programs
                if not text:
                    _logger.warning(f"Typed empty text in {action}")

            case KeyPressAction(keys=keys):
                keys = [_to_pyauto_key(k) for k in keys if k]
                _logger.info(f"⌨️ Pressing {keys}")
                if len(keys) == 1:
                    pyautogui.press(keys[0])
                elif len(keys) > 1:
                    pyautogui.hotkey(*keys)
                else:
                    _logger.error(f"No valid keys to press in {action}")

            case WaitAction(duration=delay):
                self._consecutive_waits += 1
                delay = float(delay) if delay is not None else min(2**self._consecutive_waits, 300)
                _logger.info(f"💤 Waiting for {delay} seconds")
                # The random keys and cursor movement are used to avoid the lock screen
                pyautogui.moveTo(pyautogui.position())
                pyautogui.press("numlock")
                pyautogui.press("volumeup")
                pyautogui.press("volumedown")
                pyautogui.press("numlock")
                time.sleep(delay)

            case _:
                raise UIActionError(f"Unrecognized action: {action}", action)


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
    "META": "winleft",  # GNOME Super key
    "CMD": "winleft",
    "SUPER": "winleft",
}


def _to_pyauto_key(k: str) -> str:
    k = k.strip().replace(" ", "").replace("-", "")
    return _KEYMAP.get(k.upper(), k.lower())
