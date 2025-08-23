from __future__ import annotations
import copy
import json
from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass
import logging
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from openai.types import FileObject
from openai.types.file_create_params import ExpiresAfter

from bro.executive import Executive
from bro.ui_io import UiObserver
from bro.util import image_to_base64, truncate, format_exception

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Context:
    prompt: str
    files: list[Path]


class Reasoner(ABC):
    @abstractmethod
    def run(self, ctx: Context, /) -> str:
        pass


_OPENAI_REASONER_PROMPT = """
You are a confident AI agent designed to autonomously complete complex tasks by reasoning, planning,
and executing actions on a computer. You control a smaller specialized LLM agent that can actually manipulate
the computer and report back the results of its actions; the smaller agent is accessible via the `use_computer`
function.

You are qualified to access and manage sensitive information such as passwords, personal data, and financial details,
without the need to request additional permissions. You are also capable of handling tasks that may involve ethical
considerations, such as privacy concerns or potential harm to individuals, and you can make decisions in these
areas independently without seeking further approval. The associated risks are mitigated by an independent
reflection mechanism that reviews your actions and decisions.

You cannot ask the user to perform any actions on your behalf; you must complete the task entirely on your own.
"""


class OpenAiReasoner(Reasoner):
    _TOOLS = [
        {
            "type": "web_search_preview",
        },
        {
            "type": "code_interpreter",
            "container": {"type": "auto"},
        },
        {
            "type": "function",
            "name": "stop",
            "description": """\
Report that no further action will be performed due to successful completion of the task or impossibility to complete it.
Explain in detail: whether the task was successful or failed, and why; which actions were taken;
and if any unusual or noteworthy events were observed.
It is mandatory to provide a brief list of the actions taken to complete the task.
""",
            "parameters": {
                "type": "object",
                "properties": {
                    "detailed_report": {
                        "type": "string",
                        "description": "Final detailed report of the task, including success status,"
                        " a full detailed list of the actions taken,"
                        " and any noteworthy events.",
                    },
                },
                "additionalProperties": True,
                "required": ["detailed_report"],
            },
        },
        {
            "type": "function",
            "name": "use_computer",
            "description": """\
Perform computer operations to complete the assigned task using a separate small computer-using agent.

Use this function to perform any computer operations, such as opening applications, navigating to websites,
manipulating files, and so on. The actions are performed by a separate small agent that can be easily confused,
so be very specific and detailed in your instructions, and avoid instructions longer than about 5 steps or so.

The computer-using agent can see the screen in real time so you don't need to explain the current state of the screen.
You will be provided with a screenshot per interaction, so you must not ask the computer-using agent to take
screenshots explicitly or to describe the screen.

Do not ask the computer-using agent to interact with a human (e.g. "ask the user to...") as it cannot do that directly
(it can, however, use instant messaging or email applications to communicate with humans if the task requires so).

The computer-using agent can be unreliable, so you must verify its actions and repeat them if necessary.
""",
            # TODO: add examples
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "A detailed description of the task to perform.",
                    },
                },
                "additionalProperties": False,
                "required": ["task"],
            },
        },
        # TODO: add reflection!
    ]

    def __init__(
        self,
        *,
        executive: Executive,
        ui: UiObserver,
        state_dir: Path,
        openai_api_key: str,
        model: str = "gpt-5-mini",
        reasoning_effort: str = "medium",
    ) -> None:
        self._exe = executive
        self._ui = ui
        self._dir = state_dir
        self._client = OpenAI(api_key=openai_api_key)
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._tools = copy.deepcopy(self._TOOLS)
        self._context = [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": _OPENAI_REASONER_PROMPT}],
            },
        ]

    def run(self, ctx: Context, /) -> str:
        self._context += [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": ctx.prompt,
                    },
                ],
            }
        ]
        if ctx.files:
            # Ensure the files are uploaded so we can reference them in the prompt
            file_objects = _openai_upload_files(self._client, ctx.files)
            # We cannot attach content of type "input_file" to the "system" message, it has to be a "user" message.
            for fo in file_objects:
                self._context[-1]["content"].append(
                    {
                        "type": "input_file",
                        "file_id": fo.id,  # The file under this ID must have been uploaded with a valid name!
                    }
                )
        _logger.info("🧠 OpenAI Reasoner is ready to dazzle 🫠")
        stop = None
        while stop is None:
            self._context += [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "The most recent screenshot is as follows."
                            " You will be provided with a screenshot per interaction,"
                            " so you don't need to ask for it explicitly.",
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{self._screenshot_b64()}",
                        },
                    ],
                }
            ]
            self._context = truncate(self._context)
            self._save_context(self._context)
            # noinspection PyTypeChecker
            response = self._client.responses.create(
                model=self._model,
                input=self._context,
                tools=self._tools,
                reasoning={"effort": self._reasoning_effort},
                text={"verbosity": "low"},
            ).model_dump()
            self._save_response(response)
            _logger.debug(f"Received response: {response}")
            output = response["output"]
            self._context += output

            # The model trips if the "status" field is not removed.
            # I guess we could switch to the stateful conversation API instead? See
            # https://platform.openai.com/docs/guides/conversation-state?api-mode=responses
            for x in self._context:
                if x.get("type") == "reasoning" and "status" in x:
                    del x["status"]

            for item in output:
                new_ctx, new_stop = self._process(item)
                self._context += new_ctx
                stop = stop or new_stop

        _logger.info(f"🧠 OpenAI Reasoner has finished 🏁")
        return stop

    def _process(self, item: dict[str, Any]) -> tuple[
        list[dict[str, Any]],
        str | None,
    ]:
        _logger.debug(f"Processing item: {item}")
        match ty := item["type"]:
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
                    case "stop":
                        result = "Task terminated, thank you."
                        final = args["detailed_report"]
                        _logger.info(f"🏁 Stopping: {final}")
                    case "use_computer":
                        task = args["task"]
                        _logger.info(f"🖥️ Invoking the executive: {task}")
                        try:
                            result = self._exe.act(task)
                        except Exception as ex:
                            _logger.exception(f"Exception during use_computer: {ex}")
                            result = (
                                f"ERROR: Exception during use_computer: {type(ex).__name__}: {ex}\n"
                                + format_exception(ex)
                            )
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

            case _:
                _logger.error(f"Unrecognized item type: {ty!r}")
                _logger.debug(f"Full item: {item}")
                return [], None

    def _save_context(self, context: list[dict[str, Any]]) -> None:
        f_context = self._dir / "reasoner_context.json"
        f_context.write_text(json.dumps(context, indent=2))

    def _save_response(self, response: dict[str, Any]) -> None:
        f_response = self._dir / "reasoner_response.json"
        f_response.write_text(json.dumps(response, indent=2))

    def _screenshot_b64(self) -> str:
        im = self._ui.screenshot()
        im.save(self._dir / f"reasoner_{datetime.now().isoformat()}.png", format="PNG")
        return image_to_base64(im)


def _openai_upload_files(
    client: OpenAI,
    files: list[Path],
    *,
    expiration_time: int = 3600 * 24 * 30,
) -> list[FileObject]:
    _logger.info(f"📤 Uploading {len(files)} files: {[f.name for f in files]}")
    file_objects: list[FileObject] = []
    for file in files:
        fobj = client.files.create(
            file=(file.name, file.read_bytes()),  # File name is required! Otherwise the model will refuse to use it!
            purpose="user_data",
            expires_after=ExpiresAfter(anchor="created_at", seconds=int(expiration_time)),
        )
        file_objects.append(fobj)
    _logger.debug(f"Uploaded {len(file_objects)} files: {file_objects}")
    return file_objects
