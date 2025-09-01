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
    _LOG_ROWS_PER_PAGE = 100
    _LOG_COLUMNS = [
        {"name": "id", "label": "#", "field": "id", "align": "right"},
        {"name": "ts", "label": "Time", "field": "ts", "align": "left"},
        {"name": "pid", "label": "PID", "field": "pid", "align": "right"},
        {"name": "tid", "label": "TID", "field": "tid", "align": "right"},
        {"name": "path", "label": "File", "field": "path", "align": "left"},
        {"name": "line", "label": "Line", "field": "line", "align": "right"},
        {"name": "func", "label": "Fun", "field": "func", "align": "left"},
        {"name": "level_name", "label": "Lvl", "field": "level_name", "align": "left"},
        {"name": "name", "label": "Name", "field": "name", "align": "left"},
        {
            "name": "message",
            "label": "Message",
            "field": "message",
            "align": "left",
            "classes": "truncate max-w-xs",
            "style": "",
        },
        {"name": "exception", "label": "Exception", "field": "exception", "align": "left"},
    ]
    _LOG_COLUMN_DEFAULTS = {
        "align": "left",
        "headerClasses": "uppercase text-primary",
        "style": "width:1%; white-space:nowrap;",
        "required": True,
    }
    for col in {"id", "pid", "tid", "path", "line", "func", "exception"}:
        for c in _LOG_COLUMNS:
            if c["name"] == col:
                c["classes"] = "hidden"
                c["headerClasses"] = "hidden"

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

    def _get_log_count(self) -> int:
        cur = self._ctrl.get_db().execute("SELECT MAX(id) FROM logs;")
        (count,) = cur.fetchone() or (0,)
        return count

    def _get_logs(self, *, older_than: datetime.datetime) -> list[dict[str, Any]]:
        query = """
        SELECT
            id,
            timestamp AS ts,
            process_id AS pid,
            thread_id AS tid,
            path,
            line,
            func,
            level_name,
            name,
            message,
            exception
        FROM logs
        WHERE timestamp < ?
        ORDER BY id ASC
        LIMIT ?;
        """
        cur = self._ctrl.get_db().execute(query, (older_than.isoformat(timespec="seconds"), self._LOG_ROWS_PER_PAGE))
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in rows]

    def _setup_log_table(self) -> None:
        rows = self._get_logs(older_than=datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=1))
        with ui.scroll_area().classes("h-full"):
            log_table = (
                ui.table(
                    rows=rows,
                    columns=self._LOG_COLUMNS,
                    column_defaults=self._LOG_COLUMN_DEFAULTS,
                )
                .classes("w-full w-full text-xs leading-tight")
                .props('dense flat square separator="none" virtual-scroll table-style="table-layout: auto"')
            )
        with ui.row():
            log_nav_newest = ui.button("⏮️ Newest").props("flat")
            log_nav_newer = ui.button("◀️ Newer").props("flat")
            log_nav_older = ui.button("Older ▶️").props("flat")

    def _setup(self) -> None:
        ui.add_head_html(  # language=HTML
            """\
<link href="https://fonts.googleapis.com/css2?family=Ubuntu+Mono&display=swap" rel="stylesheet">
<style>
    html, body, #app { height: 100%; }
    body {
        font-family: "Ubuntu Mono", monospace;
        overflow: hidden;
    }
    .dense-table .q-table td, .dense-table .q-table th { padding: 2px 6px !important; }
    .dense-table .q-table thead tr { height: 22px; }
    .dense-table .q-table tbody td { height: 20px; }
</style>
        """
        )

        # Large screenshot lightbox overlay
        with ui.element("div").classes(
            "fixed inset-0 bg-black bg-opacity-90 hidden flex items-center justify-center z-50 cursor-pointer"
        ) as screenshot_overlay:
            screenshot_big = ui.image().classes("max-w-[95vw] max-h-[95vh] object-contain")
            screenshot_overlay.on("click", lambda _: screenshot_overlay.classes(remove="flex", add="hidden"))

        # Main layout
        with ui.element("div").classes("fixed inset-0"):
            with ui.row().classes("w-full h-full flex-nowrap"):
                with ui.column().classes("w-2/3 h-full"):
                    self._setup_log_table()

                with ui.column().classes("w-1/3"):
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
                            reflection.set_content(
                                f"**Reflection failed:** {e}\n\n```python\n{format_exception(e)}\n```"
                            )
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
