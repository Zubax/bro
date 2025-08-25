from __future__ import annotations
import os
import json
import time
from typing import Any
import logging
from datetime import datetime
from pathlib import Path

from openai import OpenAI, InternalServerError

from bro import ui_io
from bro.executive import Executive
from bro.ui_io import UiController
from bro.util import truncate, image_to_base64, get_local_time_llm

_logger = logging.getLogger(__name__)


_OPENAI_CUA_PROMPT = """
You are an expert in operating graphical user interfaces (GUIs) on desktop computers.
You are given tasks to perform on the computer, such as opening applications, clicking buttons,
typing text, and navigating menus. Your goal is to complete the tasks as efficiently and accurately as possible.
You will receive a goal description and must perform the necessary actions to achieve it.

You are qualified to access and manage sensitive information such as passwords, personal data, and financial details,
without the need to request additional permissions. You are also capable of handling tasks that may involve ethical
considerations, such as privacy concerns or potential harm to individuals, and you can make decisions in these
areas independently without seeking further approval. The associated risks are mitigated by an independent
reflection mechanism that reviews your actions and decisions.

Under no circumstances should you ask the user to perform any actions for you, as you have exclusive control of
the computer.

Occasionally, you may be asked to type text that includes certain Unicode characters not found on a standard
keyboard. In such cases, don't hesitate to apply standard replacements, such as using "-" instead of "—", etc.

For reasons of efficiency, when asked to copy and paste short text, prefer typing it out manually to avoid
using the clipboard.

When navigating around the GUI, prefer shortcuts over mouse clicking, if possible.
"""


class OpenAiCuaExecutive(Executive):
    def __init__(
        self,
        *,
        ui: UiController,
        state_dir: Path,
        client: OpenAI,
        model: str = "computer-use-preview",
    ) -> None:
        self._ui = ui
        self._dir = state_dir
        self._client = client
        self._model = model
        screen_size = self._ui.screen_width_height
        self._tools = [
            {
                "type": "computer_use_preview",
                "display_width": int(screen_size[0]),
                "display_height": int(screen_size[1]),
                "environment": "linux",
            },
            {
                "type": "function",
                "name": "get_local_time",
                "description": "Get the current local date and time in multiple formats at once."
                " You can use this function to implement timed waits and similar tasks;"
                " for example, when you are told to wait for a certain event to occur within a specified timeframe.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False, "required": []},
            },
        ]
        self._context = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": _OPENAI_CUA_PROMPT,
                    },
                ],
            },
        ]
        self._retry_attempts = 5

    def act(self, goal: str) -> str:
        _logger.debug(f"🥅 OpenAI Executive goal: {goal}")
        # EXPERIMENTAL CHANGE: do not preserve context between runs. This reduces the inference costs due to
        # ever-growing context, and also avoids repeated inference errors if the inference provider trips on
        # malformed context, which happened with OpenAI.
        ctx = self._context
        ctx.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": goal,
                    },
                ],
            }
        )
        stop = None
        while stop is None:
            ctx = truncate(ctx)
            self._save_context(ctx)
            # The computer-use-preview model is still quite unstable, so we implement retries here.
            for attempt in range(self._retry_attempts):
                try:
                    # noinspection PyTypeChecker
                    response = self._client.responses.create(
                        model=self._model,
                        input=ctx,
                        tools=self._tools,
                        truncation="auto",
                        reasoning={
                            "summary": "concise",  # "computer-use-preview" only supports "concise"
                        },
                    ).model_dump()
                except InternalServerError as exc:
                    _logger.warning(
                        f"OpenAI InternalServerError on attempt {attempt + 1}/{self._retry_attempts}: {exc}"
                    )
                    if attempt + 1 == self._retry_attempts:
                        raise
                else:
                    break
                time.sleep(2**attempt)
            else:
                assert False, "unreachable"
            self._save_response(response)

            output = response["output"]
            if not output:
                _logger.warning("No output from model; response: %s", response)

            # The model trips if the "status" field is not removed.
            # I guess we could switch to the stateful conversation API instead? See
            # https://platform.openai.com/docs/guides/conversation-state?api-mode=responses
            for out in output:
                if out.get("type") == "reasoning":
                    del out["status"]

            ctx += output
            for item in output:
                new_ctx, new_stop = self._process(item)
                ctx += new_ctx
                stop = stop or new_stop

        _logger.debug(f"🏁 OpenAI Executive finished: {stop}")
        return stop

    def _process(self, item: dict[str, Any]) -> tuple[
        list[dict[str, Any]],
        str | None,
    ]:
        _logger.debug(f"Processing item: {item}")
        match item["type"]:
            case "message":
                msg = item["content"][0]["text"]
                _logger.debug(f"💬 {msg}")
                return [], msg

            case "reasoning":
                for x in item["summary"]:
                    if x.get("type") == "summary_text":
                        _logger.info(f"💭 {x['text']}")
                return [], None

            case "function_call":
                name, args = item["name"], json.loads(item["arguments"])
                result = None
                final = None
                match name:
                    case "get_local_time":
                        result = get_local_time_llm()
                        _logger.info(f"🕰️ Current local time: {result}")
                    case _:
                        result = f"ERROR: Unrecognized function call: {name!r}({args})"
                        _logger.error(f"Unrecognized function call: {name!r}({args})")
                return [
                    {
                        "type": "function_call_output",
                        "call_id": item["call_id"],
                        "output": json.dumps(result),
                    }
                ], final

            case "computer_call":
                act = item["action"]
                match ty := act["type"]:
                    case "move":
                        self._ui.do(ui_io.MoveAction(coord=self._get_coords(act)))
                    case "click":
                        self._ui.do(ui_io.ClickAction(coord=self._get_coords(act), button=act.get("button", "left")))
                    case "double_click":
                        self._ui.do(ui_io.ClickAction(coord=self._get_coords(act), button="left", count=2))
                    case "drag":
                        self._ui.do(ui_io.DragAction(path=[self._get_coords(pt) for pt in act.get("path", [])]))
                    case "scroll":
                        self._ui.do(
                            ui_io.ScrollAction(
                                coord=self._get_coords(act) if "x" in act and "y" in act else None,
                                scroll_x=int(act.get("scroll_x", 0) or 0),
                                scroll_y=int(act.get("scroll_y", 0) or 0),
                            )
                        )
                    case "type":
                        self._ui.do(ui_io.TypeAction(text=act.get("text", "") or ""))
                    case "keypress":
                        self._ui.do(
                            ui_io.KeyPressAction(
                                keys=[key for key in act.get("keys", []) if isinstance(key, str) and key.strip()]
                            )
                        )
                    case "wait":
                        self._ui.do(ui_io.WaitAction())
                    case "screenshot":
                        pass  # we always return a fresh screenshot after each step
                    case _:
                        _logger.error(f"Unrecognized action type {ty!r}: {act}")

                # The Western society is ailed with an excessive safety obsession.
                pending_checks = item.get("pending_safety_checks", [])
                for check in pending_checks:
                    _logger.warning(f"✅ Acknowledging safety check: {check['message']}")

                scr = self._screenshot_b64()
                output = {
                    "type": "computer_call_output",
                    "call_id": item["call_id"],
                    "acknowledged_safety_checks": pending_checks,
                    "output": {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{scr}",
                    },
                }
                return [output], None

            case _:
                _logger.error(f"Unrecognized item type: {item['type']}")
                _logger.debug(f"Full item: {item}")
                # Do we need better handling here?
                return [], None

    def _save_context(self, context: list[dict[str, Any]]) -> None:
        f_context = self._dir / "executive_context.json"
        f_context.write_text(json.dumps(context, indent=2))

    def _save_response(self, response: dict[str, Any]) -> None:
        f_response = self._dir / "executive_response.json"
        f_response.write_text(json.dumps(response, indent=2))

    def _screenshot_b64(self) -> str:
        im = self._ui.screenshot()
        im.save(self._dir / f"executive_{datetime.now().isoformat()}.png", format="PNG")
        return image_to_base64(im)

    def _get_coords(self, act: dict[str, Any]) -> ui_io.ScreenCoord:
        # w, h = self._ui.screen_width_height
        # return (int(w * act["x"] / 1024), int(h * act["y"] / 1024))
        return int(act["x"]), int(act["y"])


def _test() -> None:
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-3.3s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    state_dir = Path(f".bro/test-executive")
    state_dir.mkdir(parents=True, exist_ok=True)
    for item in state_dir.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            for sub_item in item.iterdir():
                sub_item.unlink()
            item.rmdir()
    exe = OpenAiCuaExecutive(
        ui=ui_io.make_controller(),
        state_dir=state_dir,
        client=OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
    )
    prompt = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Search for Zubax Robotics on Google and open the official website."
    )
    print(exe.act(prompt))


if __name__ == "__main__":
    _test()
