from __future__ import annotations
import time
from typing import Any
from itertools import count
import logging
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

import openai
from openai import OpenAI

from bro import ui_io
from bro.executive import Executive, Effort
from bro.ui_io import UiController
from bro.util import image_to_base64, get_local_time_llm, format_exception, split_trailing_json

_logger = logging.getLogger(__name__)


_PROMPT = """\
You are an agent that can perform simple tasks on a computer by controlling the user interface (UI) by assigning tasks
to a smaller underlying agent. You accept high-level goals and break them down into smaller, atomic tasks that
the underlying agent can perform. At each step you receive the current screenshot of the desktop and the current time.
You do not perform any UI actions yourself. Your role is entirely passive/reactive; you only do what explicitly asked
and you never make suggestions or ask questions. If not sure how to proceed, terminate the task with a failure message.
Do not do anything that is not explicitly asked of you.

The underlying agent is very basic and can be easily confused, so you must break down complex tasks into very simple,
unambiguous atomic steps.
For example, if the goal is "Open zubax.com", you should break it down into a series of steps like:

1. Click the Firefox icon on the left side of the desktop (using the `task` command).
2. Wait until the browser window opens (using the `wait` command).
3. Type "zubax.com" in the address bar (using the `type` command).
4. Press Enter (using the `key_press` command).
5. Wait for the correct page to load (using the `wait` command).

Remember that some computer usage patterns that are appropriate for humans may be suboptimal for you.
For example, usually there is no point in copying and pasting text using the clipboard (Ctrl+C/Ctrl+V),
as it is easier and more reliable to type text manually. Similarly, using keyboard shortcuts is usually preferable
to using the mouse.

You are controlled not by a human, but by a higher-level agentic planner that gives you high-level goals to achieve.
Therefore, you MUST NOT attempt to ask for a human intervention, nor speak to the user in any way.
You MUST NOT explicitly ask for further tasks once the current task is finished; your role is entirely passive/reactive.
You MUST NOT provide suggestions, advice, solicit feedback, or ask questions.
If not sure what to do, terminate the task with a failure message, explaining what went wrong.

Each of your responses MUST begin with a brief description of the current status of the task,
a critical review of the progress so far, and a description of the next step you are going to take.
Each response MUST contain a detailed description of what is visible in the screenshot,
including the names of visible windows, buttons, icons, text fields, and other UI elements.
Finally, there MUST be a SINGLE MANDATORY JSON block enclosed in triple backticks as specified below,
containing EXACTLY ONE command to execute. There shall be no text after the JSON block.

# JSON response templates

A JSON block is MANDATORY in EVERY response.

## Perform a GUI-related atomic task

Avoid tasks more complex than clicking a button, scrolling, or dragging the mouse.
When asking the agent to click something, be sure to specify if it's a double-click, single-click, or right-click!
Do not use this command for typing text or pressing keys; use the `type` and `key_press` commands instead.

```json
{"type": "task", "description": "<description of the task for the underlying agent>"}
```

## Type text

Prefer this over invoking the underlying agent to type text, because it is more reliable and faster.
Avoid Unicode characters that cannot be typed on a keyboard;
you can use composition shortcuts like Alt+NumpadXXXX if needed instead.
For example, avoid emdash (—) and use a double hyphen (--) instead;
avoid curly quotes (“”) and use straight quotes (") instead;
avoid ellipsis (…) and use three dots (...) instead;
avoid non-breaking space and use regular space instead; and so on.

```json
{"type": "type", "text": "<text to type>"}
```

## Key press

Press hotkeys using this command. Whenever possible you should prefer using hotkeys over mouse clicks,
because they are much more reliable and faster. For example, if you need to scroll a document or a web page,
use the arrows or PageUp/PageDown keys instead of scrolling with the mouse! Likewise, use Alt+Tab to switch applications
instead of clicking on the taskbar, use Ctrl+T to open a new browser tab instead of clicking the "+" button,
use Ctrl+W to close a tab instead of clicking the "X" button, use Alt+F4 to close a window instead of clicking the
"X" button, and so on.

If an attempt to click a UI element using the GUI task repeatedly fails, try pressing Tab repeatedly to iterate
through the clickable elements on the screen until the desired element is focused,
and then press Enter, Space, or whatever key is appropriate to "click" it.

```json
{"type": "key_press", "keys": ["<key1>", "<key2>", ...]}
```

## Wait for a certain amount of time

You MUST NOT use this command to request intervention or additional inputs; use the `help` command instead.
The higher-level planner cannot intervene you unless you either terminate the task or use the `help` command.

```json
{"type": "wait", "duration": number_of_seconds}
```

## Terminate the task

```json
{"type": "terminate", "message": "<report on the success or failure of the overall goal>"}
```

## Request intervention of the higher-level agentic planner

Use this if you are not sure how to proceed, or if you need additional information or actions.

```json
{"type": "help", "message": "<explanation of the situation and what kind of help is needed>"}
```
"""

_MAX_STEPS_MESSAGE = """\
You have exhausted the maximum number of steps allowed.
You must terminate the task immediately with a failure message.
Explain what you managed to achieve and what went wrong.
"""


class HierarchicalExecutive(Executive):
    """
    The HierarchicalExecutive is an Executive that completes tasks using a simpler underlying Executive
    by splitting complex tasks into smaller atomic subtasks. For example, it can convert a task like
    "Open zubax.com" into a series of subtasks: click the Firefox icon, wait for the browser to open,
    type the URL, press Enter, and wait for the correct page to load.

    WARNING: if you are using a low reasoning setting or a low-capability model, be sure to limit the number
    of steps to a small number (maybe about 10) because small models often get stuck in loops
    or get distracted and start procrastinating. If a task fails to complete, it is usually not a problem
    because the planner can always intervene and fix things.
    """

    def __init__(
        self,
        *,
        inferior: Executive,
        ui: UiController,
        state_dir: Path,
        client: OpenAI,
        model: str,
        temperature: float = 1.0,
        acts_to_remember: int = 5,
    ) -> None:
        self._inferior = inferior
        self._ui = ui
        self._dir = state_dir
        self._client = client
        self._model = model
        self._reasoning_effort = ""
        self._temperature = temperature
        self._retry_attempts = 10
        self._context = [{"role": "system", "content": _PROMPT}]
        self._acts_to_remember = acts_to_remember
        if self._acts_to_remember < 1:
            raise ValueError("The executive should remember at least one past act to avoid loops.")
        self._act_history: list[list[dict[str, Any]]] = []

    def act(self, goal: str, effort: Effort) -> str:
        # Configure the context.
        # A GPT-5-class model in the high reasoning effort mode is safe to run for a large number of steps.
        # Lower reasoning settings may cause the model to go off the rails, so we limit the number of steps.
        reasoning_effort = ("low", "medium", "high")[effort.value]
        max_steps = (10, 20, 100)[effort.value]
        if self._reasoning_effort != reasoning_effort:
            _logger.info(f"🧠 Switching reasoning effort to {reasoning_effort}; max steps {max_steps}")
            if reasoning_effort > self._reasoning_effort:  # Avoid style anchoring.
                _logger.info(f"🧠➡️🗑 DROPPING CONTEXT due to reasoning effort change")
                self._act_history.clear()
        self._reasoning_effort = reasoning_effort

        # Add new context entry for this goal.
        if len(self._act_history) >= self._acts_to_remember:
            self._act_history.pop(0)
        ctx = [self._user_message(goal)]
        self._act_history.append(ctx)

        # Run the reasoning-action loop.
        for step in count():
            _logger.info(f"🔄 Step {step+1}/{max_steps}; effort={self._reasoning_effort!r}")
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
            if step > max_steps * 2:
                _logger.warning("❌ AGENT NOT COOPERATING; TERMINATED ❌")
                return (
                    "ERROR: AGENT TERMINATED DUE TO FAILURE TO COOPERATE. Final state unknown."
                    " Please try again; consider using simpler goals or clearer instructions."
                )
            if step + 1 >= max_steps:
                _logger.info("🚫 Maximum steps reached, asking the agent to terminate.")
                ctx.append(self._user_message(_MAX_STEPS_MESSAGE))
            response = self._request_inference(self._context + sum(self._act_history, []))
            _logger.debug("Response: %s", response)
            resp_msg = response.choices[0].message
            resp_text = resp_msg.content.strip()
            ctx.append({"role": resp_msg.role, "content": resp_text})
            new_items, msg = self._process(resp_text, effort)
            ctx += new_items
            if msg is not None:
                _logger.debug(f"🤖 Final message: {msg}")
                return msg
        assert False

    @retry(
        reraise=True,
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=2, max=3600),
        retry=(retry_if_exception_type(openai.OpenAIError)),
        before_sleep=before_sleep_log(_logger, logging.ERROR),
    )
    def _request_inference(self, ctx: list[dict[str, Any]]) -> Any:
        _logger.debug(f"Requesting inference with {len(ctx)} context items...")
        # noinspection PyTypeChecker
        return self._client.chat.completions.create(
            model=self._model,
            reasoning_effort=self._reasoning_effort,
            messages=ctx,
            temperature=self._temperature,
        )

    def _process(self, response: str, effort: Effort) -> tuple[list[dict[str, Any]], str | None]:
        thought, cmd = split_trailing_json(response)
        if thought:
            _logger.info(f"💭 {thought}")
        try:
            match cmd:
                case {"type": "task", "description": description}:
                    _logger.info(f"➡️ Delegating task: {description}")
                    result = self._inferior.act(description, effort).strip()
                    _logger.info(f"🏆 Delegation result: {result}")
                    out = []
                    if result:
                        out.append(self._user_message(f"Agent thought: {result}"))
                    return out, None

                case {"type": "type", "text": text} if isinstance(text, str):
                    self._ui.do(ui_io.TypeAction(text=text))
                    return [self._user_message(f"Typed text: {text!r}")], None

                case {"type": "key_press", "keys": ks} if isinstance(ks, list) and all(isinstance(k, str) for k in ks):
                    self._ui.do(ui_io.KeyPressAction(keys=ks))
                    return [self._user_message(f"Pressed keys: {ks}")], None

                case {"type": "wait", "duration": duration} if isinstance(duration, (int, float)) and duration > 0:
                    duration = min(duration, 30)  # Some models tend to slack off and request very long waits.
                    self._ui.do(ui_io.WaitAction(duration=duration))
                    return [self._user_message(f"Waited for {duration} seconds.")], None

                case {"type": ty, "message": message} if ty in {"terminate", "help"} and isinstance(message, str):
                    return [], message

                case None:
                    _logger.info("❓ JSON block is missing.")
                    return [
                        self._user_message(
                            "ERROR: No JSON block found in the response; try again. Triple backticks are required."
                        ),
                    ], None

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
        # The short sleep helps avoiding further waits while the UI is still updating.
        # It must happen after the last action and immediately BEFORE the next screenshot.
        time.sleep(0.5)
        im = self._ui.screenshot()
        im.save(self._dir / f"{__name__}.png", format="PNG")
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
    print(exe.act(prompt, Effort.MEDIUM))


if __name__ == "__main__":
    _test()
