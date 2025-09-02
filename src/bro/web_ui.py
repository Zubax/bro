from __future__ import annotations
import platform
import sqlite3
from typing import Any
from datetime import datetime, timezone, timedelta
from abc import ABC, abstractmethod
import threading
import logging

from PIL import Image

from nicegui import ui

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


_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class View:
    # language=html
    _HEAD = """\
<link href="https://fonts.googleapis.com/css2?family=Ubuntu+Mono&display=swap" rel="stylesheet">
<style>
    html, body, #app { height: 100%; }
    
    body {
        font-family: "Ubuntu Mono", monospace;
        overflow: hidden;
    }
    
    .q-scrollarea__content {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .dense-table .q-table td, .dense-table .q-table th { padding: 2px 6px !important; }
    .dense-table .q-table thead tr { height: 22px; }
    .dense-table .q-table tbody td { height: 20px; }
    
    .log-level    { display:inline-block; padding:0 6px; border-radius:4px; font-weight:600; }
    .log-debug    { background:#eee; color:#000; }
    .log-info     { background:#040; color:#8f8; }
    .log-warning  { background:#440; color:#ff8; }
    .log-error    { background:#400; color:#f88; }
    .log-critical { background:#000; color:#f88; }
</style>
        """

    _LOG_ROW_LIMIT = 10_000
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

    def _get_newest_log_time(self) -> datetime:
        cur = self._ctrl.get_db().execute("SELECT timestamp FROM logs ORDER BY id DESC LIMIT 1;")
        row = cur.fetchone()
        if row is None:
            return datetime.now(timezone.utc)
        (ts_str,) = row
        return datetime.fromisoformat(ts_str)

    def _get_logs(
        self,
        *,
        time_window: tuple[datetime, datetime],
        min_level: int = _LOG_LEVELS["DEBUG"],
        name_wildcard: str | None = None,
    ) -> list[dict[str, Any]]:
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
        WHERE timestamp <= ? AND timestamp >= ? AND level >= ? AND name LIKE ?
        ORDER BY id ASC
        LIMIT ?;
        """
        pat = name_wildcard.replace("*", "%").replace("?", "_") if name_wildcard else "%"
        cur = self._ctrl.get_db().execute(
            query,
            (
                time_window[1].isoformat(),
                time_window[0].isoformat(),
                min_level,
                pat,
                self._LOG_ROW_LIMIT,
            ),
        )
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in rows]

    def _setup_log_table(self) -> None:
        with ui.scroll_area().classes("h-full"):
            table = ui.table(rows=[], columns=self._LOG_COLUMNS, column_defaults=self._LOG_COLUMN_DEFAULTS)
            table.classes("w-full w-full text-xs leading-tight")
            table.props('dense flat square separator="none" virtual-scroll table-style="table-layout: auto"')
        table.add_slot(
            "body-cell-ts",  # language=html
            """\
        <q-td :props="props">
          <span :title="props.value">
            {{
              ((d)=>{
                const p2=n=>String(n).padStart(2,'0'), p3=n=>String(n).padStart(3,'0');
                return `${p2(d.getMonth()+1)}-${p2(d.getDate())} ${p2(d.getHours())}:${p2(d.getMinutes())}:${p2(d.getSeconds())}.${p3(d.getMilliseconds())}`;
              })(new Date(props.value))
            }}
          </span>
        </q-td>
        """,
        )
        table.add_slot(
            "body-cell-level_name",  # language=html
            """\
        <q-td :props="props">
          <q-badge class="log-level" :class="'log-' + (props.value || '').toLowerCase()">
            {{
              ({DEBUG:'DBG', INFO:'INF', WARNING:'WRN', ERROR:'ERR', CRITICAL:'CRT'}
              [(props.value || '').toUpperCase()] ||
              (props.value || '').toString().slice(0,3).toUpperCase())
            }}
          </q-badge>
        </q-td>
        """,
        )

        def load() -> None:
            v = date_picker.value or {}
            date_from = datetime.fromisoformat(v["from"])
            date_to = datetime.fromisoformat(v["to"]) + timedelta(days=1)
            table.rows = self._get_logs(
                time_window=(date_from, date_to),
                min_level=_LOG_LEVELS[level_select.value],
                name_wildcard=wildcard.value,
            )
            date_menu.close()
            table.run_method("scrollTo", len(table.rows) - 1)  # This doesn't work but idk why

        def load_recent() -> None:
            newest = self._get_newest_log_time()
            date_from, date_to = newest - timedelta(days=1), newest + timedelta(minutes=1)
            date_picker.value = {"from": date_from.date().isoformat(), "to": date_to.date().isoformat()}
            load()

        with ui.row().classes("items-center"):
            ui.button("Load recent").on("click", load_recent)
            date_input = ui.input("Show logs from date range").props("readonly").classes("w-56")
            date_input.on("click", lambda: date_menu.open())
            with date_input:
                with ui.menu() as date_menu:  # hidden until input is clicked
                    date_picker = ui.date().props("range")
                    date_picker.on_value_change(load)
            date_picker.bind_value(
                date_input,
                forward=lambda x: f'{x["from"]} - {x["to"]}' if x else "",
                backward=lambda s: (
                    {"from": s.split(" - ")[0], "to": s.split(" - ")[1]} if " - " in (s or "") else None
                ),
            )
            level_select = ui.select(options=list(_LOG_LEVELS.keys()), value="INFO").classes("w-32")
            level_select.on("update:model-value", lambda _: load())
            wildcard = ui.input(
                label="Name wildcard",
                value="bro.*",
                placeholder="* matches any, ? matches one",
                validation={
                    "Invalid characters": lambda v: all((c.isalnum() or c in "*?._") for c in v),
                    "Must begin with a letter or _": lambda v: not v or (v[0].isalpha() or v[0] == "_"),
                    "Cannot end with a dot": lambda v: not v.endswith("."),
                    "Cannot contain consecutive dots": lambda v: ".." not in v,
                    "Too long": lambda v: len(v) <= 100,
                },
            ).classes("w-48")
            wildcard.on("keyup.enter", load)

        load_recent()

    def _setup(self) -> None:
        ui.add_head_html(self._HEAD)

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

                with ui.column().classes("w-1/3 h-full"):
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

                    with ui.scroll_area().classes("h-full"):
                        with ui.row():
                            ui.button("Reflect").on("click", lambda _: update_reflection_begin())
                            reflection_timestamp = ui.label().style("color: gray; font-style: italic;")
                            reflection_timestamp.set_text("Press the button to reflect")
                        reflection = ui.markdown()


def _now() -> str:
    return datetime.now().isoformat(sep=" ", timespec="seconds")
