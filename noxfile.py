# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>
# type: ignore

import shlex
import shutil
from pathlib import Path

# noinspection PyPackageRequirements
import nox

ROOT = Path(__file__).parent.resolve()

BYPRODUCTS = [
    "*.egg-info",
    "src/*.egg-info",
    ".coverage*",
    "html*",
    ".*cache",
    "__pycache__",
    ".*compiled",
    "*.log",
    "*.tmp",
    "*.jafit.png",
    "*.jafit.tab",
]

nox.options.error_on_external_run = True


@nox.session(python=False)
def clean(session: nox.Session) -> None:
    for w in BYPRODUCTS:
        for f in Path.cwd().glob(w):
            try:
                session.log(f"Removing: {f}")
                if f.is_dir():
                    shutil.rmtree(f, ignore_errors=True)
                else:
                    f.unlink(missing_ok=True)
            except Exception as ex:
                session.error(f"Failed to remove {f}: {ex}")


@nox.session(reuse_venv=True)
def mypy(session: nox.Session) -> None:
    session.install("-e", ".")
    session.install("mypy ~= 1.18")
    session.run("mypy", "src/bro")


@nox.session(reuse_venv=True)
def black(session: nox.Session) -> None:
    session.install("black ~= 25.11")
    session.run("black", "--check", ".")
