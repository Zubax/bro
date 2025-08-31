from __future__ import annotations
import os
import json
import logging
import shutil
import sys
import argparse

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

from openai import OpenAI

from bro import ui_io, logs, web_ui
from bro.reasoner import Context
from bro.executive.hierarchical import HierarchicalExecutive
from bro.executive.ui_tars_7b import UiTars7bExecutive
from bro.reasoner.openai_generic import OpenAiGenericReasoner
from bro.brofiles import PROMPT_HISTORY_FILE, USER_SYSTEM_PROMPT_FILE, SNAPSHOT_FILE, LOG_FILE, DB_FILE

_logger = logging.getLogger(__name__)


def main() -> None:
    logs.setup(log_file=LOG_FILE, db_file=DB_FILE)
    _logger.debug("Session started")

    parser = argparse.ArgumentParser(description="Run Bro")
    parser.add_argument("--resume", action="store_true", help="Resume from existing state file if available.")
    args = parser.parse_args()

    ps = PromptSession(history=FileHistory(PROMPT_HISTORY_FILE))

    user_system_prompt = USER_SYSTEM_PROMPT_FILE.read_text() if USER_SYSTEM_PROMPT_FILE.is_file() else None
    _logger.info(f"User system prompt {USER_SYSTEM_PROMPT_FILE} contains {len(user_system_prompt or '')} characters")

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))

    # Construct the system
    ui = ui_io.make_controller()
    exe = HierarchicalExecutive(
        inferior=UiTars7bExecutive(ui=ui, client=openrouter_client),
        ui=ui,
        client=openai_client,
        model="gpt-5",
    )
    rsn = OpenAiGenericReasoner(
        executive=exe,
        ui=ui,
        client=openai_client,
        user_system_prompt=user_system_prompt,
    )

    # Start the web UI
    web_ctrl = WebController(ui=ui, rsn=rsn)
    web_view = web_ui.View(ctrl=web_ctrl)
    web_view.start()
    _logger.info(f"🌐 Web UI at {web_view.endpoint}")

    try:
        # Restore from snapshot or start fresh
        snapshot_file = SNAPSHOT_FILE.resolve()
        if args.resume:
            if not snapshot_file.is_file():
                _logger.error(f"Cannot resume because file not found: {snapshot_file}")
                sys.exit(1)
            _logger.warning(f"Resuming {snapshot_file}")
            bak = snapshot_file.with_name(snapshot_file.name + ".bak")
            bak.unlink(missing_ok=True)
            shutil.copy(snapshot_file, bak)
            rsn.restore(json.loads(snapshot_file.read_text(encoding="utf-8")))
            _logger.info("Optionally, enter a new prompt to change the task, or submit an empty prompt to resume as-is")
            if (ctx := _prompt(ps)).prompt:
                rsn.task(ctx)
        else:
            _logger.info(f"🍃 Starting fresh; use --resume to resume from a state snapshot if available")
            _logger.info("💡 Protip: the prompt can reference local files and URLs")
            rsn.task(_prompt(ps))

        # Main loop
        _logger.info("🚀 START")
        while True:
            try:
                result = rsn.step()
                snap = rsn.snapshot()
                snapshot_file.write_text(json.dumps(snap, indent=2), encoding="utf-8")
            except KeyboardInterrupt:
                _logger.info(
                    "🚫 Step aborted by user. Please do either:\n"
                    "1. Enter nothing to resume the current task unchanged.\n"
                    "2. Type a new message to the agent (possibly a new task).\n"
                    "3. Ctrl-C again to quit."
                )
                if (ctx := _prompt(ps)).prompt:
                    rsn.task(ctx)
            else:
                if result is not None:
                    _logger.info("🏁 " * 40 + "\n" + result)
                    rsn.task(_prompt(ps))
    except KeyboardInterrupt:
        _logger.info("🚫 Task aborted by user")


class WebController(web_ui.Controller):
    def __init__(self, ui: ui_io.UiObserver, rsn: OpenAiGenericReasoner) -> None:
        self._ui = ui
        self._rsn = rsn
        self._db = logs.sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True, check_same_thread=False)

    def get_screenshot(self) -> ui_io.Image.Image:
        return self._ui.screenshot()

    def get_reflection(self) -> str:
        return self._rsn.legilimens()

    def get_db(self) -> logs.sqlite3.Connection:
        return self._db


def _prompt(ps: PromptSession) -> Context:
    try:
        txt = ps.prompt("🛑[Alt+Enter]>>> ", multiline=True).strip()
    except EOFError:
        raise KeyboardInterrupt
    return Context(prompt=txt, files=[])
