from __future__ import annotations
import os
import json
import time
from typing import Any
import logging
from pathlib import Path
import re
import concurrent.futures

from openai import OpenAI, InternalServerError
from PIL import Image

from bro import ui_io
from bro.executive import Executive, Effort
from bro.ui_io import UiController
from bro.util import image_to_base64, format_exception, get_local_time_llm

_logger = logging.getLogger(__name__)


_PROMPT_GUI_LOCATOR = """\
You are an expert in GUI interfaces, and your task is to determine the position of the specified GUI elements
on the provided screenshot. You will be given a screenshot of a GUI interface, and a description of the GUI element
you need to locate. Your goal is to analyze the screenshot and identify the coordinates of the specified GUI element.

You should return the coordinates as a tuple of integers `(x, y)`, where `x` is the horizontal position and `y` is
the vertical position of the middle of the element in screen coordinates. The origin is at the top-left corner of
the screen. Your response shall not contain anything else.

Inaccurate prediction will lead to severe punishment. Accurate prediction will be rewarded.
"""

_PROMPT_EXECUTIVE = """\
You are an AI agent that can interact with a graphical user interface (GUI) to accomplish tasks.
You are given a description of the task, and your goal is to determine the best sequence of actions to complete it
using the available actions.

Your responses must include a detailed free-form description of the current situation, your reasoning about
the next steps, a detailed bullet list of the next actions to take, and a single action to perform next.
The action must be one of the available actions listed below.

You will be provided with a fresh screenshot of the current GUI state before each response.

Each response shall contain a refined plan of the next actions to take, based on the current GUI state
and the outcome of the previous action.

You have to complete the task in the specified number of turns. If you fail to complete the task within the
allotted turns, your operation will be forcibly interrupted and you will be severely punished.

The available actions are listed below. There must be exactly one action per response, formatted based on one of
the action templates below.

# AVAILABLE ACTIONS

Click(start_box='(x,y)') -- Click the mouse at the specified screen coordinates (x, y).

LeftDouble(start_box='(x,y)') -- Double-click the left mouse button at (x, y).

LeftTriple(start_box='(x,y)') -- Triple-click the left mouse button at (x, y).

RightSingle(start_box='(x,y)') -- Right-click the mouse at (x, y).

Drag(start_box='(x,y)', end_box='(x,y)') -- Drag the mouse from start to the end point.

Scroll(direction='up', start_box='(x,y)') -- Scroll the mouse wheel up at (x, y).
Scroll(direction='down', start_box='(x,y)') -- Scroll the mouse wheel down at (x, y).

Type(content='text') -- Type the specified text using the keyboard. The text shall be quoted. NOT FOR HOTKEYS.

HotKey(key='keys') -- Press the specified sequence of hotkeys. The keys are space-separated and shall be quoted.
Example: HotKey(key='ctrl s')

Wait() -- Wait for a reasonable amount of time to allow the GUI to update.

Finished() -- The task is completed or cannot be completed.

CallUser() -- Ask the supervisor for help.

To specify an action, add a line at the end of your response in the following format:

Action: <action>
"""


class UiTars7bExecutive(Executive):
    """
    This Executive is only capable of extremely simple atomic tasks, such as clicking on a specific item,
    or typing text, etc. It cannot be paired with a Reasoner on its own; an additional hierarchical Executive
    is required to break down complex tasks into simpler ones.
    """

    _RE_ACTION = re.compile(r"\s*action:\s*(\w+)\s*\((.*)\).*", re.IGNORECASE)
    _RE_NUMBERS = re.compile(r"(\d+)\D*")
    _RE_QUOTED = re.compile(r"""['"](.+)['"]""")

    def __init__(
        self,
        *,
        ui: UiController,
        state_dir: Path,
        client: OpenAI,
        model: str = "bytedance/ui-tars-1.5-7b",
        max_steps: int = 3,
    ) -> None:
        self._ui = ui
        self._dir = state_dir
        self._client = client
        self._model = model
        self._max_steps = max_steps
        self._retry_attempts = 5
        self._temperature = 0.1  # Higher temps cause the model to do weird things
        self._context = [{"role": "system", "content": _PROMPT_EXECUTIVE}]

    def act(self, goal: str, effort: Effort) -> str:
        _ = effort  # This implementation currently does not use the mode
        scr_w, scr_h = self._ui.screen_width_height
        ctx = self._context + [{"role": "user", "content": f"{goal}\n\nDO NOT DO ANYTHING ELSE"}]
        for step in range(self._max_steps):
            _logger.debug(f"🤖 Step #{step+1}/{self._max_steps}...")
            ctx += [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": f"data:image/png;base64,{self._screenshot_b64()}"},
                        {
                            "type": "text",
                            "text": f"The screen size is {scr_w}x{scr_h} pixels."
                            f" The current time is {get_local_time_llm()}.",
                        },
                    ],
                },
            ]
            self._save_context(ctx)
            for attempt in range(1, self._retry_attempts + 1):
                try:
                    # noinspection PyTypeChecker
                    response = self._client.chat.completions.create(
                        model=self._model,
                        messages=ctx,
                        temperature=self._temperature,
                        max_tokens=500,
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
        _logger.info("🚫 Maximum steps reached, terminating.")
        return (
            "The agent failed to complete the task within the allotted number of steps;"
            " however, some progress may have been made."
            " Please split the task into smaller sub-tasks and try again."
        )

    def _process(self, response: str) -> tuple[
        list[dict[str, Any]],
        str | None,
    ]:
        action_line = response.strip().splitlines()[-1].strip()
        response_sans_action = "\n".join(response.strip().splitlines()[:-1]).strip()
        m = self._RE_ACTION.match(action_line)
        if not m:
            _logger.info("💭 No action found in the response: %r", action_line)
            return [self._user_message("ERROR: Action not found or is not formatted correctly; try again.")], None
        if response_sans_action:
            _logger.debug(f"💭 {response_sans_action}")
        action_name = m.group(1)
        action_args = m.group(2)
        numbers = list(map(int, self._RE_NUMBERS.findall(action_args)))
        quoted = self._RE_QUOTED.findall(action_args)
        try:
            match action_name.lower():
                case "click" if len(numbers) == 2:
                    x, y = numbers
                    self._ui.do(ui_io.ClickAction((x, y)))
                case "leftsingle" if len(numbers) == 2:  # We don't give this explicitly but the model generalizes
                    x, y = numbers
                    self._ui.do(ui_io.ClickAction((x, y)))
                case "leftdouble" if len(numbers) == 2:
                    x, y = numbers
                    self._ui.do(ui_io.ClickAction((x, y), count=2))
                case "lefttriple" if len(numbers) == 2:
                    x, y = numbers
                    self._ui.do(ui_io.ClickAction((x, y), count=3))
                case "rightsingle" if len(numbers) == 2:
                    x, y = numbers
                    self._ui.do(ui_io.ClickAction((x, y), button=ui_io.ClickAction.BUTTON_RIGHT))
                case "drag" if len(numbers) == 4:
                    x1, y1, x2, y2 = numbers
                    self._ui.do(ui_io.DragAction([(x1, y1), (x2, y2)]))
                case "scroll" if len(numbers) == 2:
                    up = "up" in action_args.lower()
                    x, y = numbers
                    self._ui.do(ui_io.ScrollAction((x, y), scroll_y=+1 if up else -1))
                case "type" if len(quoted) == 1:
                    (text,) = quoted
                    self._ui.do(ui_io.TypeAction(text))
                case "hotkey" if len(quoted) == 1:
                    (keys,) = quoted
                    key_list = keys.split()
                    self._ui.do(ui_io.KeyPressAction(key_list))
                case "wait":
                    self._ui.do(ui_io.WaitAction())
                case "finished":
                    return [], response_sans_action
                case "calluser":
                    return [], response_sans_action
                case _:
                    return [
                        self._user_message("Could not parse the action specification; try again."),
                    ], None
        except Exception as ex:
            return [
                self._user_message(
                    f"ERROR during action execution: {ex}; try again. Exception stacktrace:\n{format_exception(ex)}"
                ),
            ], None
        return [], response_sans_action  # stop after a single action

    @staticmethod
    def _user_message(msg: str, /) -> dict[str, Any]:
        return {"role": "user", "content": msg}

    def _save_context(self, context: list[dict[str, Any]]) -> None:
        f_context = self._dir / f"{__name__}.json"
        f_context.write_text(json.dumps(context, indent=2))

    def _screenshot_b64(self) -> str:
        # The short sleep helps avoiding further waits while the UI is still updating.
        # It must happen after the last action and immediately BEFORE the next screenshot.
        time.sleep(0.5)
        im = self._ui.screenshot()
        im.save(self._dir / f"{__name__}.png", format="PNG")
        return image_to_base64(im)


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
    exe = UiTars7bExecutive(
        ui=ui_io.make_controller(),
        state_dir=state_dir,
        client=OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        ),
    )
    prompt = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Search for Zubax Robotics on Google and open the official website."
    )
    print(exe.act(prompt, Effort.MEDIUM))


class _GuiLocator:
    """
    Given a description of a GUI element and a screenshot (in base64), locate the element on the screenshot
    and return its screen coordinates (x, y), with the origin at the top-left corner of the screen.
    If the element cannot be found, the result is usually None, sometimes a random coordinate.
    """

    _RE_COORD = re.compile(r"(\d+)\D*")

    def __init__(
        self,
        client: OpenAI,
        model: str = "bytedance/ui-tars-1.5-7b",
        temperature: float = 0.3,
        n: int = 11,
    ) -> None:
        self._client = client
        self._model = model
        self._retry_attempts = 5
        self._temperature = temperature
        self._n = n

    def locate_element(self, description: str, screenshot: Image.Image) -> ui_io.ScreenCoord | None:
        if self._n > 1 and self._temperature <= 1e-6:
            raise ValueError("n>1 requires temperature>0")
        scr_b64 = image_to_base64(screenshot)
        # Unfortunately, the OpenRouter API does not support n>1 for chat completions,
        # so we have to do it manually with threads. This is inefficient but works for now.
        # Maybe we could even set the temperature to zero and use only one sample?
        options: list[str] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._n) as executor:
            futures = [executor.submit(self._once, description, scr_b64) for _ in range(self._n)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    options.append(future.result())
                except Exception as e:
                    _logger.exception(f"Error during locate_element: {e}")
        _logger.warning("Raw options: %s", options)
        return self._fuse(options, screenshot)

    def _once(self, description: str, screenshot_b64: str) -> str:
        messages = [
            {"role": "system", "content": _PROMPT_GUI_LOCATOR},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": f"data:image/png;base64,{screenshot_b64}"},
                    {"type": "text", "text": description},
                ],
            },
        ]
        for attempt in range(1, self._retry_attempts + 1):
            try:
                # noinspection PyTypeChecker
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                )
                break
            except InternalServerError as e:
                _logger.warning(f"API error on attempt {attempt}/{self._retry_attempts}: {e}")
                if attempt == self._retry_attempts:
                    raise
                time.sleep(2**attempt)
        else:
            assert False, "Unreachable"
        _logger.debug("Response: %s", response)
        return response.choices[0].message.content.strip()

    def _fuse(self, proposals_raw: list[str], screenshot: Image.Image) -> ui_io.ScreenCoord | None:
        proposals = [pc for pc in (self._extract_coord(pr) for pr in proposals_raw) if pc is not None]
        proposals = [pc for pc in proposals if (0 <= pc[0] <= screenshot.width) and (0 <= pc[1] <= screenshot.height)]
        _logger.debug("Proposals: %s", proposals)
        if not proposals:
            return None
        proposals.sort()
        n = len(proposals)
        h = n // 2
        if n % 2 == 1:
            return proposals[h]
        (x1, y1), (x2, y2) = proposals[h - 1], proposals[h]
        return (x1 + x2) // 2, (y1 + y2) // 2

    def _extract_coord(self, text: str) -> ui_io.ScreenCoord | None:
        coord_str = self._RE_COORD.findall(text)
        if len(coord_str) != 2:
            return None
        try:
            return int(coord_str[0]), int(coord_str[1])
        except ValueError:
            return None


def _test_coord() -> None:
    scr_dir = Path(__file__).parent.parent.parent.parent / "test_data"
    loc = _GuiLocator(
        client=OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    )
    print(loc.locate_element("Operations -> Payments", Image.open(scr_dir / "ubuntu_simplbooks.png")))
    print(loc.locate_element("Account settings", Image.open(scr_dir / "ubuntu_simplbooks.png")))

    print(loc.locate_element("System Monitor icon", Image.open(scr_dir / "ubuntu_app_menu.png")))
    print(loc.locate_element("Remmina icon", Image.open(scr_dir / "ubuntu_app_menu.png")))
    print(loc.locate_element("Firefox icon", Image.open(scr_dir / "ubuntu_app_menu.png")))
    print(loc.locate_element("tabby cow rapper spaceship", Image.open(scr_dir / "ubuntu_app_menu.png")))


if __name__ == "__main__":
    _test()
