from __future__ import annotations
import platform
import sqlite3
from typing import Any
from datetime import datetime, timezone, timedelta
from abc import ABC, abstractmethod
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, Future

from PIL import Image

from nicegui import ui

from bro import __version__
from bro.util import get_upstream_ip, image_to_base64, format_exception


__all__ = ["View", "Controller"]

HOST = "0.0.0.0"
PORT = 8814

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
        """
        TODO: do not expose sqlite directly because it leaks abstraction all over the place; instead, provide DB
        connectors with methods to query and write logs, etc. The same connector can be used to notify consumers when
        new log entries are added, so that the UI can auto-refresh.
        """
        raise NotImplementedError


_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


# noinspection HtmlUnknownAttribute
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
    .log-level    { display:inline-block; padding:0 3px; border-radius:4px; font-weight:600; }
    .log-debug    { background:#eee; color:#000; }
    .log-info     { background:#040; color:#8f8; }
    .log-warning  { background:#440; color:#ff8; }
    .log-error    { background:#400; color:#f88; }
    .log-critical { background:#000; color:#f88; }
</style>
        """

    # language=html
    _LOG_TABLE_BODY_SLOT = """\
<q-tr :id="'logrow-' + props.row.id" :props="props" @click="props.expand = !props.expand" class="cursor-pointer">
  <q-td v-for="col in props.cols" :key="col.name" :props="props">
    <template v-if="col.name==='ts'">
      <span :title="props.row.ts">
        {{
          ((d)=>{
            const p2 = n => String(n).padStart(2,'0'), p3 = n => String(n).padStart(3,'0');
            return `${p2(d.getMonth()+1)}-${p2(d.getDate())} ${p2(d.getHours())}:${p2(d.getMinutes())}:${p2(d.getSeconds())}.${p3(d.getMilliseconds())}`;
          })(new Date(props.row.ts))
        }}
      </span>
    </template>
    <template v-else-if="col.name==='level_name'">
      <q-badge class="log-level" :class="'log-' + (props.row.level_name || '').toLowerCase()">
        {{
          ({DEBUG:'DBG', INFO:'INF', WARNING:'WRN', ERROR:'ERR', CRITICAL:'CRT'}
          [(props.row.level_name || '').toUpperCase()] ||
          (props.row.level_name || '').toString().slice(0,3).toUpperCase())
        }}
      </q-badge>
    </template>
    <template v-else-if="col.name==='message'">
      <span class="truncate max-w-xs" :title="props.row.message">{{ props.row.message }}</span>
    </template>
    <template v-else>
      {{ col.value }}
    </template>
  </q-td>
</q-tr>

<q-tr v-show="props.expand" :props="props" class="bg-gray-100">
  <q-td colspan="100%">
    <div class="space-y-2">
      <div class="text-xs text-gray-600 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-1 items-center">
        <div><span class="font-semibold">PID:</span> {{ props.row.pid }}</div>
        <div><span class="font-semibold">TID:</span> {{ props.row.tid }}</div>
        <div><span class="font-semibold">Record</span> #{{ props.row.id }} {{ props.row.level_name }}</div>
        <div class="col-span-1 sm:col-span-2 lg:col-span-1">
          <span class="font-semibold">File:</span> {{ props.row.path }}:{{ props.row.line }}
        </div>
        <div><span class="font-semibold">Module:</span> {{ props.row.name }}</div>
        <div><span class="font-semibold">Func:</span> {{ props.row.func }}</div>
      </div>
      <hr/>
      <pre class="whitespace-pre-wrap break-words m-0">{{ props.row.message }}</pre>
      <template v-if="props.row.exception">
        <div class="font-semibold mt-2">Exception</div>
        <pre class="whitespace-pre-wrap break-words overflow-x-auto m-0">{{ props.row.exception }}</pre>
      </template>
    </div>
  </q-td>
</q-tr>
"""

    _LOG_ROW_LIMIT = 1000
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
        ORDER BY id DESC
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
        return [dict(zip(columns, row)) for row in rows[::-1]]

    def _setup_log_table(self) -> None:
        """
        TODO FIXME: this UI is bad; the table needs to be infinitely scrollable with auto-loading more rows when
        scrolling to the bottom. For now I'm keeping this but it needs to be redone.
        """

        def scroll_bottom():
            try:
                # language=js
                ui.run_javascript(
                    """
                    (()=>{
                      const go = ()=>{  // The row ids are defined in the slot template
                        const rows = document.querySelectorAll('tr[id^="logrow-"]');  
                        if (!rows.length) return;
                        const last = rows[rows.length-1];
                        const sc = last.closest('.q-table__middle');
                        if (sc) sc.scrollTop = last.offsetTop;
                        else last.scrollIntoView({block:'end'});
                      };
                      requestAnimationFrame(()=>requestAnimationFrame(go));
                    })();
                    """
                )
            except Exception as ex:
                _logger.exception(f"Failed to scroll the log table: {ex}")

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
            ui.timer(0.0, scroll_bottom, once=True)
            ui.timer(0.1, scroll_bottom, once=True)  # just in case

        def load_recent() -> None:
            newest = self._get_newest_log_time()
            date_from, date_to = newest - timedelta(days=1), newest + timedelta(minutes=1)
            date_picker.value = {"from": date_from.date().isoformat(), "to": date_to.date().isoformat()}
            load()

        def on_reload_timer() -> None:
            if auto_reload_recent.value:
                load_recent()

        with ui.column().classes("w-full h-full"):
            # Main element -- the log table
            with ui.element("div").classes("flex-1 min-h-0 w-full"):
                table = ui.table(rows=[], columns=self._LOG_COLUMNS, column_defaults=self._LOG_COLUMN_DEFAULTS)
            table.classes("h-full w-full text-xs leading-tight")
            # virtual-scroll does not work correctly with expandable rows, so we don't use it.
            table.props('dense flat square separator="none" row-key="id" table-style="table-layout: auto"')
            table.add_slot("body", self._LOG_TABLE_BODY_SLOT)

            # Log table controls underneath
            with ui.row().classes("items-center shrink-0"):
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
                ui.button("Scroll⬇️").on("click", scroll_bottom).props("flat")
                auto_reload_recent = ui.checkbox("Auto reload recent", value=False)
                ui.timer(10.0, on_reload_timer)

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
                        screenshot_thumb.props(f"src={src!r}")
                        screenshot_big.source = src

                    screenshot_thumb = ui.element("img").classes("cursor-pointer block max-w-full h-auto border-0")
                    screenshot_thumb.on("click", lambda _: screenshot_overlay.classes(remove="hidden", add="flex"))
                    update_screenshot()
                    ui.timer(10.0, update_screenshot)

                    # Reflection area
                    fut_reflection: Future[None] | None = None

                    def do_reflect() -> None:
                        try:
                            _logger.debug("🪞 Reflecting...")
                            r = self._ctrl.get_reflection()
                            _logger.debug(f"🪞 Reflection:\n{r}")
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
                        nonlocal fut_reflection
                        if fut_reflection is None or fut_reflection.done():
                            fut_reflection = ThreadPoolExecutor(1).submit(do_reflect)
                        else:
                            _logger.warning("Reflection already in progress, no point clicking again")

                    with ui.scroll_area().classes("h-full"):
                        with ui.row():
                            ui.button("Reflect").on("click", lambda _: update_reflection_begin())
                            reflection_timestamp = ui.label().style("color: gray; font-style: italic;")
                            reflection_timestamp.set_text("Press the button to reflect")
                        reflection = ui.markdown()


def _now() -> str:
    return datetime.now().isoformat(sep=" ", timespec="seconds")
