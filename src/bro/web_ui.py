from __future__ import annotations
import os
import platform
import sqlite3
from typing import Any
import datetime
from abc import ABC, abstractmethod
import threading
import logging

from PIL import Image

from nicegui import ui

from bro.brofiles import DB_FILE
from bro.ui_io import UiObserver
from bro import __version__
from bro.util import get_upstream_ip, image_to_base64, format_exception


__all__ = ["View", "Controller"]

HOST = "0.0.0.0"
PORT = 1488

_logger = logging.getLogger(__name__)


class Controller(ABC):
    @abstractmethod
    def get_screenshot(self) -> Image.Image:
        raise NotImplementedError

    @abstractmethod
    def get_reflection(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_db(self) -> sqlite3.Connection:
        raise NotImplementedError


class View:
    def __init__(
        self,
        ctrl: Controller,
        *,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        self._ctrl = ctrl
        host = host or HOST
        port = port or PORT
        self._endpoint = f"http://{host}:{port}/"
        self._thread = threading.Thread(
            target=lambda: ui.run(
                host=host,
                port=port,
                reload=False,
                show=False,
                title=f"Bro v{__version__} @ {platform.node()} ({get_upstream_ip() or 'unknown IP'})",
                favicon="🤖",
                log_config=None,  # An empty log config is essential! Otherwise, Uvicorn will break our logging setup!
            ),
            daemon=True,
        )
        self._setup()

    def start(self) -> None:
        self._thread.start()

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def _setup(self) -> None:
        # Large screenshot lightbox overlay
        with ui.element("div").classes(
            "fixed inset-0 bg-black bg-opacity-90 hidden flex items-center justify-center z-50 cursor-pointer"
        ) as screenshot_overlay:
            screenshot_big = ui.image().classes("max-w-[95vw] max-h-[95vh] object-contain")
            screenshot_overlay.on("click", lambda _: screenshot_overlay.classes(remove="flex", add="hidden"))

        # Main layout
        with ui.row().classes("w-full flex-nowrap"):
            with ui.column().classes("w-1/2"):
                ui.label("Log")
                ui.button("Action A")
                ui.input("Input A")

            with ui.column().classes("w-1/2"):
                # Screenshot area
                def update_screenshot():
                    im = self._ctrl.get_screenshot()
                    src = "data:image/png;base64," + image_to_base64(im)
                    screenshot_thumb.source = src
                    screenshot_big.source = src

                screenshot_thumb = (
                    ui.image().style("max-width: 100%; height: auto; border: none;").classes("cursor-pointer")
                )
                screenshot_thumb.on("click", lambda _: screenshot_overlay.classes(remove="hidden", add="flex"))
                update_screenshot()
                ui.timer(10.0, update_screenshot)

                # Reflection area
                def do_reflect() -> None:
                    try:
                        _logger.info("🪞 Reflecting...")
                        r = self._ctrl.get_reflection()
                        _logger.info(f"🪞 Reflection:\n{r}")
                        reflection.set_content(r)
                        reflection_timestamp.set_text(f"Reflected {_now()}")
                        ui.update(reflection, reflection_timestamp)
                    except Exception as e:
                        _logger.exception("Reflection failed")
                        reflection.set_content(f"**Reflection failed:** {e}\n\n```python\n{format_exception(e)}\n```")
                        ui.update(reflection)

                def update_reflection_begin():
                    reflection.set_content("*🪞 Reflecting...*")
                    reflection_timestamp.set_text(f"Started {_now()}")
                    threading.Thread(target=do_reflect, daemon=True).start()

                with ui.row():
                    ui.button("Reflect").on("click", lambda _: update_reflection_begin())
                    reflection_timestamp = ui.label().style("color: gray; font-style: italic;")
                    reflection_timestamp.set_text("Press the button to reflect")
                reflection = ui.markdown()


def _now() -> str:
    return datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
