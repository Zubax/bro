from __future__ import annotations
import os
import sqlite3
from typing import Any
from abc import ABC, abstractmethod
import threading
from urllib.parse import urlencode

from flask import Flask, g, request, jsonify, render_template_string, abort

from PIL import Image

from bro.brofiles import DB_FILE
from bro.ui_io import UiObserver
from bro import __version__


__all__ = ["View", "Controller"]

DEFAULT_PER_PAGE = 200
MAX_PER_PAGE = 10_000
HOST = "0.0.0.0"
PORT = 1488


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
    def __init__(self, ctrl: Controller) -> None:
        self._ctrl = ctrl
        self._app = Flask(__name__)
        self._thread = threading.Thread(
            target=lambda: self._app.run(host=HOST, port=PORT, threaded=True, use_reloader=False),
            daemon=True,
        )

        @self._app.route("/")
        def index():
            return render_template_string(
                HTML_INDEX,
                version=__version__,
            )

    def start(self) -> None:
        self._thread.start()

    @property
    def endpoint(self) -> str:
        return f"http://{HOST}:{PORT}/"

    def _run_sql(self, query: str, params: tuple = ()) -> list[sqlite3.Row]:
        db = self._ctrl.get_db()
        try:
            cur = db.execute(query, params)
            rows = cur.fetchall()
            cur.close()
            return rows
        except sqlite3.DatabaseError as e:
            abort(500, f"DB error: {e}")


def _request_parse_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        v = int(request.args.get(name, default))
    except ValueError:
        v = default
    return max(lo, min(hi, v))


# language=HTML
HTML_INDEX = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    {% if refresh %}
    <meta http-equiv="refresh" content="{{ refresh|int }}">
    {% endif %}
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Bro monitor</title>
    <style>
    </style>
  </head>
  <body>
    <header>
      <h1>Bro monitor</h1>
    </header>

    <footer>
      <span>
          <a href="https://github.com/Zubax/bro">Bro v.{{version}}</a> &copy;
          <a href="https://zubax.com/">Zubax Robotics</a></span>
    </footer>
  </body>
</html>
"""
