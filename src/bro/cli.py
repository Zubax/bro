from __future__ import annotations
import os
import json
import logging
import shutil
from pathlib import Path
import sys
import argparse

import colorlog

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

from openai import OpenAI

from bro import ui_io
from bro.reasoner import Context
from bro.executive.hierarchical import HierarchicalExecutive
from bro.executive.ui_tars_7b import UiTars7bExecutive
from bro.reasoner.openai_generic import OpenAiGenericReasoner


BRODIR = Path().home() / ".bro"
BRODIR.mkdir(parents=True, exist_ok=True)

PROMPT_HISTORY_FILE = BRODIR / "prompt_history.txt"
USER_SYSTEM_PROMPT_FILE = BRODIR / "system_prompt.txt"
SNAPSHOT_FILE = Path("state.bro.json")

_logger = logging.getLogger(__name__)


def main() -> None:
    log_dir = Path(f".bro/")
    if log_dir.exists():
        shutil.rmtree(log_dir, ignore_errors=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    _setup_logging(log_dir)

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
        inferior=UiTars7bExecutive(ui=ui, state_dir=log_dir, client=openrouter_client),
        ui=ui,
        state_dir=log_dir,
        client=openai_client,
        model="gpt-5",
    )
    rsn = OpenAiGenericReasoner(
        executive=exe,
        ui=ui,
        state_dir=log_dir,
        client=openai_client,
        user_system_prompt=user_system_prompt,
    )

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
    try:
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


def _prompt(ps: PromptSession) -> Context:
    try:
        txt = ps.prompt("🛑[Alt+Enter]>>> ", multiline=True).strip()
    except EOFError:
        raise KeyboardInterrupt
    return Context(prompt=txt, files=[])


def _setup_logging(log_dir: Path) -> None:
    logging.getLogger().setLevel(logging.DEBUG)

    # Console handler
    console_handler = colorlog.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(asctime)s %(log_color)s%(levelname)-3.3s %(name)s%(reset)s: %(message)s", "%H:%M:%S"
        )
    )
    logging.getLogger().addHandler(console_handler)

    # File handler
    log_file_path = log_dir / "bro.log"
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
