from __future__ import annotations
import os
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
from bro.util import image_to_base64, truncate

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Context:
    prompt: str
    files: list[Path]


class Reasoner(ABC):
    def run(self, ctx: Context) -> str:
        pass


_OPENAI_REASONER_PROMPT = """
Say mew
"""


class OpenAiReasoner(Reasoner):
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
        self._tools = [
            {"type": "web_search_preview"},
            {"type": "code_interpreter", "container": {"type": "auto"}},
            {
                "type": "function",
                "name": "stop",
                "description": "Report that no further action will be performed due to successful completion of the task"
                " or impossibility to complete it."
                " Explain in detail: whether the task was successful or failed, and why;"
                " which actions were taken; and if any unusual or noteworthy events were observed.",
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
        ]
        self._context = [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": _OPENAI_REASONER_PROMPT}],
            },
        ]

    def run(self, ctx: Context) -> str:
        self._context += [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": ctx.prompt,
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{self._screenshot_b64()}",
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

            # TODO: loop --- feed next screenshot etc.
            for item in output:
                try:
                    self._context += self._process(item)
                except _StopError as ex:
                    stop = ex.report

        _logger.info(f"🧠 OpenAI Reasoner has finished 🏁: {stop}")
        return stop

    def _process(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        _logger.debug(f"Processing item: {item}")
        match ty := item["type"]:
            case "message":
                msg = item["content"][0]["text"]
                _logger.debug(f"💬 {msg}")
                raise _StopError("Task interrupted", msg)

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
                        raise _StopError(f"Task completed", report)
                    case _:
                        _logger.error(f"Unrecognized function call: {name}({args})")
                        return []

            case _:
                _logger.error(f"Unrecognized item type: {ty!r}")
                _logger.debug(f"Full item: {item}")
                return []

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


class _StopError(RuntimeError):
    def __init__(self, message: str, report: str):
        self.report = report
        super().__init__(message)


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
