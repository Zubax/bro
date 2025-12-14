from __future__ import annotations
import sys
import logging
from pathlib import Path
import datetime
import sqlite3
import colorlog


def setup(*, log_file: Path, db_file: Path) -> None:
    logging.getLogger().setLevel(logging.DEBUG)

    # Console handler
    console_handler = colorlog.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(asctime)s %(log_color)s%(levelname)-3.3s %(name)s%(reset)s: %(message)s", "%H:%M:%S"
        )
    )
    logging.getLogger().addHandler(console_handler)

    # File handler
    log_file_path = log_file
    file_handler = logging.FileHandler(str(log_file_path), mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    # SQLite handler
    sqlite_log_path = db_file
    sqlite_handler = _SqliteHandler(sqlite_log_path)
    sqlite_handler.setLevel(logging.DEBUG)
    sqlite_handler.setFormatter(logging.Formatter())
    logging.getLogger().addHandler(sqlite_handler)

    # Per-module customizations
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("pyautogui").setLevel(logging.WARNING)
    logging.getLogger("mss").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


class _SqliteHandler(logging.Handler):
    _SCHEMA = """\
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    process_id INTEGER,
    thread_id INTEGER,
    level INTEGER,
    level_name TEXT,
    path TEXT,
    line INTEGER,
    func TEXT,
    name TEXT,
    message TEXT,
    exception TEXT
)
"""

    def __init__(self, db_path: Path) -> None:
        super().__init__()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        with self._conn:
            self._conn.execute(self._SCHEMA)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            fmt = self.formatter or logging.Formatter()
            ts = datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc).isoformat(
                timespec="milliseconds"
            )
            msg = record.getMessage()
            exc_text: str | None = None
            if record.exc_info:
                exc_text = fmt.formatException(record.exc_info)
            elif record.stack_info:  # stack_info is already a string
                exc_text = record.stack_info
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO logs
                    (timestamp, process_id, thread_id, level, level_name, path, line, func, name, message, exception)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts,
                        record.process,
                        record.thread,
                        record.levelno,
                        record.levelname,
                        record.pathname,
                        record.lineno,
                        record.funcName,
                        record.name,
                        msg,
                        exc_text,
                    ),
                )
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        self._conn.close()
        super().close()
