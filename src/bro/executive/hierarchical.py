from __future__ import annotations
import time
import json
from typing import Any
from itertools import count
import logging
from datetime import datetime
from pathlib import Path
import re

from openai import OpenAI, InternalServerError

from bro.executive import Executive
from bro.ui_io import UiObserver
from bro.util import truncate, image_to_base64, get_local_time_llm, format_exception

_logger = logging.getLogger(__name__)


_PROMPT = """\
You are an agent that can perform tasks on a computer by controlling the user interface (UI) by assigning tasks
to a smaller underlying agent. You accept high-level goals and break them down into smaller, atomic tasks that
the underlying agent can perform. At each step you receive the current screenshot of the desktop and the current time.
You do not perform any UI actions yourself.

The underlying agent is very basic and can be easily confused, so you must break down complex tasks into very simple,
unambiguous atomic steps.
For example, if the goal is "Open zubax.com", you should break it down into a series of steps like:

1. Click the Firefox icon on the left side of the desktop (using the `task` command).
2. Wait until the browser window opens (using the `wait` command).
3. Type "zubax.com" in the address bar (using the `task` command).
4. Press Enter (using the `task` command).
5. Wait for the correct page to load (using the `wait` command).

You can assign one small task per step.
You verify the success of each step and adjust your strategy if something goes wrong.
To assign a task, include a JSON code block at the end of your response following one of the templates below.

Each of your responses MUST begin with a brief description of the current status of the task,
a critical review of the progress so far, and a description of the next step you are going to take.
If you notice that you are unable to make progress for a long time, you should try to change your approach;
if you are completely stuck, you can terminate the task with a failure message.

There shall be no text after the JSON code block.

# JSON response templates

```json
{"type": "task", "description": "<description of the task for the underlying agent>"}
```

```json
{"type": "wait", "duration": number_of_seconds}
```

```json
{"type": "terminate", "message": "<report on the success or failure of the overall goal>"}
```
"""

_MAX_STEPS_MESSAGE = """\
You have exhausted the maximum number of steps allowed.
You must terminate the task immediately with a failure message.
Explain what you managed to achieve and what went wrong.
"""

_RE_JSON = re.compile(r"(?ims)^```(?:json)?\n(.+)\n```$")


class HierarchicalExecutive(Executive):
    """
    The HierarchicalExecutive is an Executive that completes tasks using a simpler underlying Executive
    by splitting complex tasks into smaller atomic subtasks. For example, it can convert a task like
    "Open zubax.com" into a series of subtasks: click the Firefox icon, wait for the browser to open,
    type the URL, press Enter, and wait for the correct page to load.
    """

    def __init__(
        self,
        *,
        inferior: Executive,
        ui: UiObserver,
        state_dir: Path,
        client: OpenAI,
        model: str,
        temperature: float = 1.0,
        max_steps: int = 100,
    ) -> None:
        self._inferior = inferior
        self._ui = ui
        self._dir = state_dir
        self._client = client
        self._model = model
        self._temperature = temperature
        self._retry_attempts = 5
        self._max_steps = max_steps
        self._context = [{"role": "system", "content": _PROMPT}]

    def act(self, goal: str) -> str:
        ctx = self._context + [{"role": "user", "content": goal}]
        for step in count():
            _logger.info(f"🔄 Step {step+1}/{self._max_steps}")
            ctx += [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self._screenshot_b64()}"}},
                        {
                            "type": "text",
                            "text": f"The current time is {get_local_time_llm()}.",
                        },
                    ],
                },
            ]
            if step + 1 >= self._max_steps:
                _logger.info("🚫 Maximum steps reached, asking the agent to terminate.")
                ctx.append(self._user_message(_MAX_STEPS_MESSAGE))
            ctx = truncate(ctx, head=100, tail=1000)
            for attempt in range(1, self._retry_attempts + 1):
                try:
                    # noinspection PyTypeChecker
                    response = self._client.chat.completions.create(
                        model=self._model,
                        messages=ctx,
                        temperature=self._temperature,
                    )
                    break
                except InternalServerError as e:
                    _logger.warning(f"Inference API error on attempt {attempt}/{self._retry_attempts}: {e}")
                    if attempt == self._retry_attempts:
                        raise
                    time.sleep(2**attempt)
            else:
                assert False, "Unreachable"
            _logger.debug("Response: %s", response)
            resp_msg = response.choices[0].message
            resp_text = resp_msg.content.strip()
            ctx.append({"role": resp_msg.role, "content": resp_text})
            new_items, msg = self._process(resp_text)
            ctx += new_items
            if msg is not None:
                _logger.debug(f"🤖 Final message: {msg}")
                return msg
        assert False

    def _process(self, response: str) -> tuple[list[dict[str, Any]], str | None]:
        js = _RE_JSON.search(response)
        if not js:
            return [], response
        if deep_thought := response[: js.start()].strip():
            _logger.info(f"💭 {deep_thought}")
        try:
            cmd = json.loads(js.group(1))
            match cmd:
                case {"type": "task", "description": description}:
                    _logger.info(f"➡️ Delegating task: {description}")
                    result = self._inferior.act(description)
                    _logger.info(f"🏆 Delegation result: {result}")
                    return [self._user_message(result)], None

                case {"type": "wait", "duration": duration} if isinstance(duration, (int, float)) and duration > 0:
                    _logger.info(f"⏱️ Waiting for {duration} seconds")
                    time.sleep(duration)
                    return [self._user_message(f"Waited for {duration} seconds.")], None

                case {"type": "terminate", "message": message} if isinstance(message, str):
                    return [], message

                case _:
                    _logger.warning(f"❓ Unrecognized or invalid command from the agent: {cmd}")
                    return [
                        self._user_message(f"ERROR: Unrecognized or invalid command; try again: {cmd!r}"),
                    ], None
        except Exception as ex:
            return [
                self._user_message(
                    f"ERROR during action execution: {ex}; try again. Exception stacktrace:\n{format_exception(ex)}"
                ),
            ], None

    @staticmethod
    def _user_message(msg: str, /) -> dict[str, Any]:
        return {"role": "user", "content": msg}

    def _screenshot_b64(self) -> str:
        im = self._ui.screenshot()
        im.save(self._dir / f"executive_hierarchical_{datetime.now().isoformat()}.png", format="PNG")
        return image_to_base64(im)


def _test() -> None:
    import os
    import sys
    from bro import ui_io
    from bro.executive.ui_tars_7b import UiTars7bExecutive

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

    ui = ui_io.make_controller()
    inferior = UiTars7bExecutive(
        ui=ui,
        state_dir=state_dir,
        client=OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY")),
    )
    exe = HierarchicalExecutive(
        inferior=inferior,
        ui=ui,
        state_dir=state_dir,
        client=OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        model="gpt-5-mini",
    )
    prompt = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Search for Zubax Robotics on Google and open the official website."
    )
    print(exe.act(prompt))


if __name__ == "__main__":
    _test()
