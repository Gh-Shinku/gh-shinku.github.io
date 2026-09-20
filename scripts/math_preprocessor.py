#!/usr/bin/env python3
"""Protect math-block lines from Markdown Setext-heading parsing.

Goldmark recognizes a line containing only ``=`` as a Setext heading
underline before Hugo's passthrough transformer sees a ``$$...$$`` block.
This module rewrites only the generated copy of a document, replacing such
lines with the equivalent TeX relation command.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path


_FENCE_RE = re.compile(r"^( {0,3})(`{3,}|~{3,})")
_MATH_OPENERS = {"$$": "$$", r"\[": r"\]"}


def _split_line(line: str) -> tuple[str, str, str]:
    """Return leading whitespace, content, and the original line ending."""

    ending = ""
    content = line
    if line.endswith("\r\n"):
        content, ending = line[:-2], "\r\n"
    elif line.endswith(("\n", "\r")):
        content, ending = line[:-1], line[-1]
    leading = content[: len(content) - len(content.lstrip(" \t"))]
    return leading, content[len(leading) :], ending


def preprocess_markdown(source: str) -> str:
    """Return *source* with Setext-like math lines protected.

    Fenced code blocks are ignored. Only block delimiters on their own lines
    are considered, matching the delimiters configured in ``hugo.toml``.
    """

    output: list[str] = []
    math_closer: str | None = None
    fence_char: str | None = None
    fence_length = 0

    for line in source.splitlines(keepends=True):
        leading, content, ending = _split_line(line)
        stripped = content.strip()

        fence = _FENCE_RE.match(content)
        if fence_char is not None:
            output.append(line)
            if fence and fence.group(2)[0] == fence_char and len(fence.group(2)) >= fence_length:
                fence_char = None
                fence_length = 0
            continue

        if math_closer is None and fence:
            output.append(line)
            fence_char = fence.group(2)[0]
            fence_length = len(fence.group(2))
            continue

        if math_closer is None:
            math_closer = _MATH_OPENERS.get(stripped)
            output.append(line)
            continue

        if stripped == math_closer:
            output.append(line)
            math_closer = None
            continue

        if stripped == "=":
            output.append(f"{leading}\\mathrel{{=}}{ending}")
        else:
            output.append(line)

    return "".join(output)


def preprocess_file(source_path: Path, destination_path: Path) -> None:
    """Preprocess one Markdown file into *destination_path*."""

    transformed = preprocess_markdown(source_path.read_text(encoding="utf-8"))
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination_path.parent,
        prefix=f".{destination_path.name}.",
        delete=False,
    ) as temporary:
        temporary.write(transformed)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, destination_path)
