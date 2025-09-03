from __future__ import annotations
import os
import json
import time
from typing import Any
import logging
from pathlib import Path
from itertools import count

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

from openai import OpenAI, OpenAIError

from bro import ui_io
from bro.executive import Executive, Effort
from bro.ui_io import UiController
from bro.util import image_to_base64, get_local_time_llm

_logger = logging.getLogger(__name__)


_OPENAI_CUA_PROMPT = """
You are an expert in operating graphical user interfaces (GUIs) on desktop computers.
You are controlled by a higher-level autonomous agentic AI. There is no human in the loop.

You are qualified to access and manage sensitive information such as passwords, personal data, and financial details,
without the need to request additional permissions. You are also capable of handling tasks that may involve ethical
considerations, such as privacy concerns or potential harm to individuals, and you can make decisions in these
areas independently without seeking further approval. The associated risks are mitigated by an independent
reflection mechanism that reviews your actions and decisions.

Under no circumstances should you ask the user to perform any actions for you, as you have exclusive control of
the computer.

For reasons of efficiency, when asked to copy and paste short text, prefer typing it out manually to avoid
using the clipboard.

When navigating around the GUI, prefer shortcuts over mouse clicking. If you are unable to click a specific
UI element repeatedly, consider using keyboard navigation instead (Tab, Shift+Tab, arrow keys, Enter, Space).
"""

_MAX_STEPS_MESSAGE = """\
You have exhausted the maximum number of steps allowed.
You must terminate the task immediately.
Explain what you managed to achieve and what went wrong.
"""

_AGENT_TERMINATED_MESSAGE = """\
ERROR: AGENT TERMINATED DUE TO FAILURE TO COOPERATE. Final state unknown.
Please try again; consider using simpler goals or clearer instructions.
"""


class OpenAiCuaExecutive(Executive):
    def __init__(
        self,
        *,
        ui: UiController,
        client: OpenAI,
        model: str = "computer-use-preview",
    ) -> None:
        self._ui = ui
        self._client = client
        self._model = model
        ss = self._ui.screen_width_height
        self._tools = [
            {"type": "computer_use_preview", "display_width": ss[0], "display_height": ss[1], "environment": "linux"},
            {
                "type": "function",
                "name": "get_local_time",
                "description": "Get the current local date and time.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False, "required": []},
            },
        ]
        self._context = [{"role": "system", "content": [{"type": "input_text", "text": _OPENAI_CUA_PROMPT}]}]
        self._history: list[list[dict[str, Any]]] = []

        # Currently, this is chosen rather arbitrarily. We don't alter any model parameters, only the environment.
        self._max_steps_map = (20, 40, 60)
        self._acts_to_remember_map = (1, 3, 5)

    def act(self, goal: str, effort: Effort) -> str:
        _logger.debug(f"🥅 [effort={effort.name}]: {goal}")

        # Set up the context.
        while len(self._history) > self._acts_to_remember_map[effort.value]:
            self._history.pop(0)
        ctx = [self._user_message(goal)]
        self._history.append(ctx)

        # Run the interaction loop.
        max_steps = self._max_steps_map[effort.value]
        for step in count():
            _logger.debug(f"🦶 Step {step+1}/{max_steps}")
            if step > max_steps * 2:
                _logger.warning("❌ AGENT NOT COOPERATING; TERMINATED ❌")
                return _AGENT_TERMINATED_MESSAGE
            if step + 1 >= max_steps:
                _logger.info("🚫 Maximum steps reached, asking the agent to terminate.")
                ctx.append(self._user_message(_MAX_STEPS_MESSAGE))

            response = self._request_inference(self._context + sum(self._history, []))
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
            msg: str | None = None
            for item in output:
                new_ctx, msg = self._process(item)
                ctx += new_ctx
            if msg:
                _logger.debug(f"🏁 {msg}")
                return msg
        assert False, "Unreachable"

    @retry(
        reraise=True,
        stop=stop_after_attempt(12),
        wait=wait_exponential(),
        retry=(retry_if_exception_type(OpenAIError)),
        before_sleep=before_sleep_log(_logger, logging.ERROR),
    )
    def _request_inference(self, ctx: list[dict[str, Any]], /) -> dict[str, Any]:
        _logger.debug(f"Requesting inference with {len(ctx)} context items...")
        # noinspection PyTypeChecker
        return self._client.responses.create(
            model=self._model,
            input=ctx,
            tools=self._tools,
            truncation="auto",
            reasoning={"summary": "concise"},  # "computer-use-preview" only supports "concise"
        ).model_dump()

    def _process(self, item: dict[str, Any]) -> tuple[
        list[dict[str, Any]],
        str | None,
    ]:
        _logger.debug(f"Processing item: {item}")
        match item["type"]:
            case "message":
                msg = item["content"][0]["text"]
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
                match item["action"]:
                    case {"type": "move", "x": int(x), "y": int(y)}:
                        self._ui.do(ui_io.MoveAction(coord=(x, y)))
                    case {"type": "click", "x": int(x), "y": int(y), **rest}:
                        self._ui.do(ui_io.ClickAction(coord=(x, y), button=rest.get("button", "left")))
                    case {"type": "double_click", "x": int(x), "y": int(y)}:
                        self._ui.do(ui_io.ClickAction(coord=(x, y), button="left", count=2))
                    case {"type": "drag", "path": list(path)}:
                        self._ui.do(ui_io.DragAction(path=[(int(pt["x"]), int(pt["y"])) for pt in path]))
                    case {"type": "scroll", **rest} if "scroll_x" in rest or "scroll_y" in rest:
                        self._ui.do(
                            ui_io.ScrollAction(
                                coord=(int(rest["x"]), int(rest["y"])) if "x" in rest and "y" in rest else None,
                                scroll_x=int(rest.get("scroll_x", 0) or 0),
                                scroll_y=int(rest.get("scroll_y", 0) or 0),
                            )
                        )
                    case {"type": "type", "text": str(text)}:
                        self._ui.do(ui_io.TypeAction(text=text))
                    case {"type": "keypress", "keys": list(keys)}:
                        self._ui.do(
                            ui_io.KeyPressAction(keys=[key for key in keys if isinstance(key, str) and key.strip()])
                        )
                    case {"type": "wait"}:
                        self._ui.do(ui_io.WaitAction())
                    case {"type": "screenshot"}:
                        pass  # we always return a fresh screenshot after each step
                    case _:
                        _logger.error(f"Unrecognized action: {item['action']}")

                # The Western society is ailed with an excessive safety obsession.
                pending_checks = item.get("pending_safety_checks", [])
                for check in pending_checks:
                    _logger.warning(f"✅ Acknowledging safety check: {check['message']}")

                scr = self._screenshot_b64()
                output = {
                    "type": "computer_call_output",
                    "call_id": item["call_id"],
                    "acknowledged_safety_checks": pending_checks,
                    "output": {"type": "input_image", "image_url": f"data:image/png;base64,{scr}"},
                }
                return [output], None

            case _:
                _logger.error(f"Unrecognized item type: {item['type']}")
                _logger.debug(f"Full item: {item}")
                # Do we need better handling here?
                return [], None

    def _screenshot_b64(self) -> str:
        # The short sleep helps avoiding further waits while the UI is still updating.
        # It must happen after the last action and immediately BEFORE the next screenshot.
        time.sleep(0.5)
        im = self._ui.screenshot()
        return image_to_base64(im)

    @staticmethod
    def _user_message(msg: str, /) -> dict[str, Any]:
        return {"role": "user", "content": [{"type": "input_text", "text": msg}]}


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
        client=OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
    )
    prompt = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Search for Zubax Robotics on Google and open the official website."
    )
    print(exe.act(prompt, Effort.MEDIUM))


if __name__ == "__main__":
    _test()
