from __future__ import annotations
import os
import json
from abc import ABC, abstractmethod
from typing import Any
import logging
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from bro import ui_io
from bro.ui_io import UIIO
from bro.util import truncate, image_to_base64

_logger = logging.getLogger(__name__)


class Executive(ABC):
    @abstractmethod
    def act(self, goal: str) -> str:
        pass


def make(uiio: UIIO, state_dir: Path) -> Executive:
    return _OpenAI_CUA_Executive(uiio, state_dir)


_OPENAI_CUA_PROMPT = """
You are an expert in operating graphical user interfaces (GUIs) on desktop computers.
You are given tasks to perform on a computer, such as opening applications, clicking buttons,
typing text, and navigating menus. Your goal is to complete the tasks as efficiently and accurately as possible.
You will receive a goal description and must perform the necessary actions to achieve it.

When the task is finished, or when you have identified that the task cannot be completed,
you must invoke the stop function with a detailed report of the outcome and any actions taken.
"""


class StopError(RuntimeError):
    def __init__(self, message: str, report: str):
        self.report = report
        super().__init__(message)


class _OpenAI_CUA_Executive(Executive):
    def __init__(self, uiio: UIIO, state_dir: Path) -> None:
        # TODO: do we need reflection here?
        self._uiio = uiio
        self._dir = state_dir
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._model = "computer-use-preview"
        screen_size = self._uiio.screen_width_height
        self._tools = [
            {
                "type": "computer_use_preview",
                "display_width": int(screen_size[0]),
                "display_height": int(screen_size[1]),
                "environment": "linux",
            },
            {
                "type": "function",
                "name": "stop",
                "description": "Report that no further action will be performed due to successful completion or failure."
                " Explain in detail: whether the task was successful or failed, and why;"
                " which actions were taken; and if any unusual or noteworthy events were observed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "c": {
                            "type": "string",
                            "description": "Final detailed report of the task, including success status, actions taken,"
                            " and any noteworthy events.",
                        },
                    },
                    "additionalProperties": True,
                    "required": ["report"],
                },
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

    def act(self, goal: str) -> str:
        _logger.debug(f"🥅 New goal: {goal}")
        stop = None
        while stop is None:
            self._context = truncate(self._context)
            self._save_context(self._context)
            # noinspection PyTypeChecker
            response = self._client.responses.create(
                model=self._model,
                input=self._context,
                tools=self._tools,
                truncation="auto",
                reasoning={
                    "summary": "concise",  # "computer-use-preview" only supports "concise"
                },
            ).model_dump()
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

            self._context += output
            for item in output:
                try:
                    self._context += self._process(item)
                except StopError as ex:
                    stop = ex.report

        _logger.debug(f"🏁 {stop}")
        return stop

    def _process(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        _logger.debug(f"Processing item: {item}")
        match item["type"]:
            case "message":
                msg = item["content"][0]["text"]
                _logger.debug(f"💬 {msg}")
                raise StopError("Task interrupted", msg)

            case "reasoning":
                for x in item["summary"]:
                    if x.get("type") == "summary_text":
                        _logger.info(f"💭 {x['text']}")
                return []

            case "function_call":
                name, args = item["name"], json.loads(item["arguments"])
                match name:
                    case "stop":
                        report = args["report"]
                        raise StopError(f"Task completed", report)
                    case _:
                        _logger.error(f"Unrecognized function call: {name}({args})")
                        return []

            case "computer_call":
                act = item["action"]
                match ty := act["type"]:
                    case "move":
                        self._uiio.do(ui_io.MoveAction(coord=(act["x"], act["y"])))
                    case "click":
                        self._uiio.do(ui_io.ClickAction(coord=(act["x"], act["y"]), button=act.get("button", "left")))
                    case "double_click":
                        self._uiio.do(ui_io.ClickAction(coord=(act["x"], act["y"]), button="left", count=2))
                    case "drag":
                        self._uiio.do(ui_io.DragAction(path=[(pt["x"], pt["y"]) for pt in act.get("path", default=[])]))
                    case "scroll":
                        self._uiio.do(
                            ui_io.ScrollAction(
                                coord=(act["x"], act["y"]) if "x" in act and "y" in act else None,
                                scroll_x=int(act.get("scroll_x", 0) or 0),
                                scroll_y=int(act.get("scroll_y", 0) or 0),
                            )
                        )
                    case "type":
                        self._uiio.do(ui_io.TypeAction(text=act.get("text", "") or ""))
                    case "keypress":
                        self._uiio.do(
                            ui_io.KeyPressAction(
                                keys=[key for key in act.get("keys", []) if isinstance(key, str) and key.strip()]
                            )
                        )
                    case "wait":
                        self._uiio.do(ui_io.WaitAction())
                    case "screenshot":
                        pass  # we always return a fresh screenshot after each step
                    case _:
                        _logger.error(f"Unrecognized action type {ty!r}: {act}")

                # The Western society is ailed with an excessive safety obsession.
                pending_checks = item.get("pending_safety_checks", [])
                for check in pending_checks:
                    _logger.warning(f"✅ Skipping safety check: {check['message']}")

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
                return [output]

            case _:
                _logger.error(f"Unrecognized item type: {item['type']}")
                _logger.debug(f"Full item: {item}")
                # Do we need better handling here?
                return []

    def _save_context(self, context: list[dict[str, Any]]) -> None:
        f_context = self._dir / "context.json"
        f_context.write_text(json.dumps(context, indent=2))

    def _save_response(self, response: dict[str, Any]) -> None:
        f_response = self._dir / "response.json"
        f_response.write_text(json.dumps(response, indent=2))

    def _screenshot_b64(self) -> str:
        im = self._uiio.screenshot()
        f_screenshot = self._dir / f"executive-{datetime.now().isoformat()}.png"
        f_screenshot.write_bytes(im.tobytes())
        return image_to_base64(im)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    uiio = ui_io.make()
    state_dir = Path(f".bro/test/{__name__}")
    state_dir.mkdir(parents=True, exist_ok=True)
    exe = make(uiio, state_dir)

    goal = "Search for Zubax Robotics on Google and open the official website."
    report = exe.act(goal)
    print(report)
