from __future__ import annotations
import os
import time
import logging
from pathlib import Path
import sys

from bro import reasoner, ui_io, executive
from bro.reasoner import Context


_logger = logging.getLogger(__name__)
_dir = Path(f".bro/{time.strftime('%Y-%m-%d-%H-%M-%S')}")
_dir.mkdir(exist_ok=True, parents=True)


def main() -> None:
    _setup_logging()
    context = _build_context(sys.argv[1:])

    openai_api_key = os.getenv("OPENAI_API_KEY")
    ui = ui_io.make_controller()
    exe = executive.OpenAiCuaExecutive(
        ui=ui,
        state_dir=_dir,
        openai_api_key=openai_api_key,
    )
    rsn = reasoner.OpenAiReasoner(
        executive=exe,
        ui=ui,
        state_dir=_dir,
        openai_api_key=openai_api_key,
    )

    try:
        result = rsn.run(context)
    except KeyboardInterrupt:
        _logger.info("🚫 Task aborted by user")
        sys.exit(1)
    else:
        print(result)


def _build_context(paths: list[str]) -> Context:
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
    prompt_files = [f for f in all_files if f.name == "prompt.txt"]
    if len(prompt_files) > 1:
        raise ValueError(f"Multiple prompt.txt files found: {prompt_files}")
    if len(prompt_files) == 1:
        try:
            prompt = prompt_files[0].read_text(encoding="utf-8").strip()
        except Exception as e:
            raise RuntimeError(f"Failed to read {prompt_files[0]}: {e}")
    else:
        prompt = "Summarize the files."
        _logger.warning("No prompt.txt file found; using a default prompt, which is:\n%r", prompt)
    _logger.debug(f"Prompt:\n{prompt}")

    return Context(
        prompt=prompt,
        files=[f for f in all_files if f not in prompt_files],
    )


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
