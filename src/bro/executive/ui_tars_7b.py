from __future__ import annotations
import os
import json
import time
from typing import Any
import logging
from datetime import datetime
from pathlib import Path
import re
import threading
import concurrent.futures

from openai import OpenAI, InternalServerError
from PIL import Image

from bro import ui_io
from bro.executive import Executive
from bro.ui_io import UiController
from bro.util import truncate, image_to_base64

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
using the available actions specified in JSON code blocks as described below.

Your responses must include a detailed free-form description of the current situation, your reasoning about
the next steps, a detailed bullet list of the next actions to take, and a single action to perform next.
The action must be one of the available actions listed below and formatted as a JSON code block.
Actions can only be specified in JSON code blocks, and any other format will be considered invalid.

You will be provided with a fresh screenshot of the current GUI state before each response.

Each response shall contain a refined plan of the next actions to take, based on the current GUI state
and the outcome of the previous action.

The available actions are listed below. They are specified in JSON format. There should be exactly one action
per response, formatted as a JSON code block based on one of the action templates below.

# AVAILABLE ACTIONS

## Move the mouse cursor to a specific position on the screen

The `location` field contains a detailed description of the target GUI element to move the mouse to.
In case multiple instances of the element are present, the description should be as specific as possible to
avoid ambiguity.

```json
{
    "action": "move",
    "location": string
}
```

For example:

```json
{
    "action": "move",
    "location": "the 'File' menu in the top menu bar"
}
```

If necessary, you can describe specific coordinates on the screen (assuming they are known to you),
for example: "Point with coordinates (512, 384)".

## Click the mouse at a specific position on the screen

```json
{
    "action": "click",
    "location": string,
    "button": "left" | "right" | "middle",
    "count": integer
}
```

The `location` field has the same meaning as in the `move` action.
The `button` field specifies which mouse button to click (default is "left").
The `count` field specifies how many times to click, e.g. 1 for single click, 2 for double click (default is 1).

## Drag the mouse along a specified path

```json
{
    "action": "drag",
    "path": [string]
}
```

The `path` field is a list of descriptions of the points to drag through, in order.
Each point description has the same meaning as the `location` field in the `move` action.

## Scroll the mouse wheel at a specific position on the screen

```json
{
    "action": "scroll",
    "location": string | null,  // if null, scroll at the current mouse position
    "scroll_x": integer,   // horizontal scroll amount (positive for right, negative for left)
    "scroll_y": integer    // vertical scroll amount (positive for up, negative for down)
}
```

Example:

```json
{
    "action": "scroll",
    "location": "the main content area of the browser window",
    "scroll_x": 0,
    "scroll_y": -50
}
```

## Type a sequence of characters

```json
{
    "action": "type",
    "text": string
}
```

Example:

```json
{
    "action": "type",
    "text": "Hello, world!"
}
```

## Press a sequence of keys

```json
{
    "action": "key_press",
    "keys": [string]
}
```

Example:

```json
{
    "action": "key_press",
    "keys": ["ctrl", "s"]
}
```

## Wait for a specified amount of time

```json
{
    "action": "wait",
    "duration": number   // duration to wait in seconds; if omitted or null, wait for a reasonable amount of time.
}
```

## Report completion or infeasibility of the task

To report that no further actions will be taken due to successful completion of the task, or infeasibility of the task,
simply respond with a free-form explanation in plain text, without any JSON formatting.
"""


class UiTars7bExecutive(Executive):
    def __init__(
        self,
        *,
        ui: UiController,
        state_dir: Path,
        client: OpenAI,
        model: str = "bytedance/ui-tars-1.5-7b",
    ) -> None:
        self._ui = ui
        self._dir = state_dir
        self._client = client
        self._model = model
        self._retry_attempts = 5
        self._locator = _GuiLocator(client=client)

    def act(self, goal: str) -> str:
        raise NotImplementedError("UiTars7bExecutive.act is not implemented yet")

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
    _test_coord()
