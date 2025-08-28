from __future__ import annotations
import os
import time
import json
import logging
from pathlib import Path
import sys

try:
    import libreadline
except ImportError:
    pass

from openai import OpenAI

from bro import ui_io
from bro.reasoner import Context
from bro.executive.hierarchical import HierarchicalExecutive
from bro.executive.ui_tars_7b import UiTars7bExecutive
from bro.reasoner.openai_generic import OpenAiGenericReasoner


SNAPSHOT_NAME = "bro_snapshot.json"
PROMPT_NAME = "prompt.txt"

_logger = logging.getLogger(__name__)
_dir = Path(f".bro/{time.strftime('%Y-%m-%d-%H-%M-%S')}")
_dir.mkdir(exist_ok=True, parents=True)


def main() -> None:
    _setup_logging()
    context, snap_file = _build_context(sys.argv[1:])

    system_prompt_path = Path.home() / ".bro.txt"
    user_system_prompt = system_prompt_path.read_text() if system_prompt_path.is_file() else None

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))

    ui = ui_io.make_controller()
    # Low-capability models require a low max step limit because they tend to go off the rails.
    exe = HierarchicalExecutive(
        inferior=UiTars7bExecutive(ui=ui, state_dir=_dir, client=openrouter_client),
        ui=ui,
        state_dir=_dir,
        client=openai_client,
        model="gpt-5",
    )
    rsn = OpenAiGenericReasoner(
        executive=exe,
        ui=ui,
        state_dir=_dir,
        client=openai_client,
        user_system_prompt=user_system_prompt,
    )
    if snap_file.is_file():
        _logger.warning(f"♻️ Restoring snapshot from {snap_file}")
        snap_file.with_name(snap_file.name + ".bak").write_text(snap_file.read_text(encoding="utf-8"), encoding="utf-8")
        rsn.restore(json.loads(snap_file.read_text(encoding="utf-8")))
    else:
        _logger.info(f"🔴 Starting fresh: no snapshot at {snap_file}")
        rsn.task(context)

    _logger.info("🚀 START")
    try:
        while True:
            try:
                result = rsn.step()
                snap = rsn.snapshot()
                snap_file.write_text(json.dumps(snap, indent=2), encoding="utf-8")
                if result is not None:
                    _logger.info("🏁 " * 40 + "\n" + result)
                    _logger.info("🛑 Awaiting user response; enter next task or Ctrl-C to quit.")
                    next_task = input("> ").strip()
                    rsn.task(Context(prompt=next_task, files=[]))
            except KeyboardInterrupt:
                _logger.info(
                    "🚫 Step aborted by user. Please do either:\n"
                    "1. Press Enter to resume the current task unchanged.\n"
                    "2. Type a new message to the agent (possibly a new task).\n"
                    "3. Ctrl-C again to quit."
                )
                if msg := input("> ").strip():
                    rsn.task(Context(prompt=msg, files=[]))
    except KeyboardInterrupt:
        _logger.info("🚫 Task aborted by user")


def _build_context(paths: list[str]) -> tuple[Context, Path]:
    _logger.debug(f"Building context from paths: {paths}")
    all_files: list[Path] = []
    for path_str in paths:
        _logger.debug(f"Processing path: {path_str}")
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        if path.is_dir():
            for file_path in path.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith("."):
                    all_files.append(file_path)
        else:
            all_files.append(path)
    _logger.info(f"🗂️ Context files:\n" + "\n".join(f"{i+1:02d}. {f}" for i, f in enumerate(all_files)))

    # Read the prompt and exclude it from the context
    prompt_files = [f for f in all_files if f.name == PROMPT_NAME]
    if len(prompt_files) > 1:
        raise ValueError(f"Multiple prompt.txt files found: {prompt_files}")
    if len(prompt_files) == 1:
        try:
            prompt = prompt_files[0].read_text(encoding="utf-8").strip()
        except Exception as e:
            raise RuntimeError(f"Failed to read {prompt_files[0]}: {e}")
    else:
        raise ValueError("No prompt.txt file found in the provided paths.")
    _logger.debug(f"Prompt:\n{prompt}")

    # Find existing snapshot or default location
    snapshot_files = [f for f in all_files if f.name == SNAPSHOT_NAME]
    if len(snapshot_files) > 1:
        raise ValueError(f"Multiple snapshots found: {snapshot_files}")
    snapshot = snapshot_files[0] if len(snapshot_files) == 1 else (prompt_files[0].parent / SNAPSHOT_NAME)

    unwanted = {f for f in all_files if f.suffix == ".bak"} | set(prompt_files) | {snapshot}
    return Context(prompt=prompt, files=[f for f in all_files if f not in unwanted]), snapshot


def _setup_logging() -> None:
    logging.getLogger().setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-3.3s %(name)s: %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(console_handler)

    # File handler
    log_file_path = _dir / "bro.log"
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
