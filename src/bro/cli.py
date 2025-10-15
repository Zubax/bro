from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
import sys
from time import sleep

try:
    import readline  # noqa: F401
except ImportError:
    pass

from openai import OpenAI

from bro import ui_io, logs, web_ui
from bro.reasoner.openai_generic import OpenAiGenericReasoner
from bro.executive import Executive
from bro.executive.hierarchical import HierarchicalExecutive
from bro.executive.ui_tars_7b import UiTars7bExecutive
from bro.executive.openai_cua import OpenAiCuaExecutive
from bro.brofiles import USER_SYSTEM_PROMPT_FILE, SNAPSHOT_FILE, LOG_FILE, DB_FILE
from bro.connector.slack import SlackConnector
from bro.conversation import ConversationHandler

_logger = logging.getLogger(__name__)


def main() -> None:
    logs.setup(log_file=LOG_FILE, db_file=DB_FILE)
    _logger.debug("Session started")

    parser = argparse.ArgumentParser(description="Run Bro")
    parser.add_argument("--resume", action="store_true", help="Resume from existing state file if available.")
    parser.add_argument(
        "--exe",
        "-E",
        type=str,
        required=True,
        choices=["gpt-5+ui-tars-7b", "gpt-5+openai-cua", "openai-cua"],
        help="The executive stack to use",
    )
    args = parser.parse_args()

    user_system_prompt = USER_SYSTEM_PROMPT_FILE.read_text() if USER_SYSTEM_PROMPT_FILE.is_file() else None
    _logger.info(f"User system prompt {USER_SYSTEM_PROMPT_FILE} contains {len(user_system_prompt or '')} characters")

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))

    # Construct the system
    ui = ui_io.make_controller()
    exe: Executive | None = None
    match (args.exe or "").lower():
        case "gpt-5+ui-tars-7b":
            exe = HierarchicalExecutive(
                inferior=UiTars7bExecutive(ui=ui, client=openrouter_client),
                ui=ui,
                client=openai_client,
                model="gpt-5",
            )
        case "gpt-5+openai-cua":
            exe = HierarchicalExecutive(
                inferior=OpenAiCuaExecutive(ui=ui, client=openai_client),
                ui=ui,
                client=openai_client,
                model="gpt-5",
            )
        case "openai-cua":
            exe = OpenAiCuaExecutive(ui=ui, client=openai_client)
        case _:
            _logger.error(f"Unknown executive specification: {args.exe!r}")
            sys.exit(1)

    rsn = OpenAiGenericReasoner(
        executive=exe,
        ui=ui,
        client=openai_client,
        user_system_prompt=user_system_prompt,
    )

    connector = SlackConnector(bot_token=os.environ["SLACK_BOT_TOKEN"],
                               app_token=os.environ["SLACK_APP_TOKEN"])
    conversation = ConversationHandler(connector, user_system_prompt, openai_client, reasoner=rsn)

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

        # Main loop
        _logger.info("🚀 START")
        while True:
            # If we handled a message, proceed to the next spin immediately
            # Otherwise sleep.
            conversation.spin()
            snap = rsn.snapshot()
            snapshot_file.write_text(json.dumps(snap, indent=2), encoding="utf-8")
            sleep(10)

    except KeyboardInterrupt:
        _logger.info("🚫 Task aborted by user")


class WebController(web_ui.Controller):
    def __init__(self, ui: ui_io.UiObserver, rsn: OpenAiGenericReasoner) -> None:
        self._ui = ui
        self._rsn = rsn
        self._db = sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True, check_same_thread=False)

    def get_screenshot(self) -> ui_io.Image.Image:
        return self._ui.screenshot()

    def get_reflection(self) -> str:
        return self._rsn.legilimens()

    def get_db(self) -> sqlite3.Connection:
        return self._db
