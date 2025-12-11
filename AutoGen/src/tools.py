"""Tools for the AutoGen Planner/Executor example.

These tools are simple Python functions that can be registered with AutoGen
agents so they can be invoked as part of the conversation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def save_plan_to_file(steps: Iterable[str], filename: str = "autogen_plan.txt") -> str:
    """Save a list of plan steps to a text file under the ``logs`` folder.

    Parameters
    ----------
    steps:
        Iterable of human-readable plan step strings.
    filename:
        Name of the file to create under the repository-level ``logs``
        directory. Defaults to ``autogen_plan.txt``.

    Returns
    -------
    str
        Path to the written file as a string.
    """

    # Resolve repository root as two levels up from this file: AutoGen/src/..
    root = Path(__file__).resolve().parents[2]
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    path = logs_dir / filename
    lines: List[str] = [f"{i+1}. {step}\n" for i, step in enumerate(steps)]
    path.write_text("".join(lines), encoding="utf-8")

    return str(path)
