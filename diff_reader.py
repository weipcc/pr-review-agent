"""
diff_reader.py
Parses the raw text of a git diff into structured data.

Example of the output format:
[
    {
        "filename": "app.py",
        "hunks": [
            {
                "header": "@@ -10,6 +10,8 @@",
                "old_start": 10,
                "new_start": 10,
                "lines": [
                    {"type": "context", "content": "def foo():"},
                    {"type": "add", "content": "    x = 1", "new_line_no": 11},
                    {"type": "del", "content": "    y = 2", "old_line_no": 11},
                ],
            }
        ],
    }
]
"""

import re
from dataclasses import dataclass, field


@dataclass
class DiffLine:
    type: str  # "add" | "del" | "context"
    content: str
    old_line_no: int | None = None
    new_line_no: int | None = None


@dataclass
class DiffHunk:
    header: str
    old_start: int
    new_start: int
    lines: list[DiffLine] = field(default_factory=list)


@dataclass
class FileDiff:
    filename: str
    hunks: list[DiffHunk] = field(default_factory=list)


HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
FILE_HEADER_RE = re.compile(r"^diff --git a/(.+) b/(.+)$")


def parse_diff(raw_diff: str) -> list[FileDiff]:
    """Parse the raw text of a git diff into a structured list of FileDiff objects."""
    files: list[FileDiff] = []
    current_file: FileDiff | None = None
    current_hunk: DiffHunk | None = None
    old_line_no = 0
    new_line_no = 0

    for line in raw_diff.splitlines():
        file_match = FILE_HEADER_RE.match(line)
        if file_match:
            current_file = FileDiff(filename=file_match.group(2))
            files.append(current_file)
            current_hunk = None
            continue

        hunk_match = HUNK_HEADER_RE.match(line)
        if hunk_match and current_file is not None:
            old_start = int(hunk_match.group(1))
            new_start = int(hunk_match.group(2))
            current_hunk = DiffHunk(header=line, old_start=old_start, new_start=new_start)
            current_file.hunks.append(current_hunk)
            old_line_no = old_start
            new_line_no = new_start
            continue

        if current_hunk is None:
            continue

        if line.startswith("+") and not line.startswith("+++"):
            current_hunk.lines.append(
                DiffLine(type="add", content=line[1:], new_line_no=new_line_no)
            )
            new_line_no += 1
        elif line.startswith("-") and not line.startswith("---"):
            current_hunk.lines.append(
                DiffLine(type="del", content=line[1:], old_line_no=old_line_no)
            )
            old_line_no += 1
        elif line.startswith(" "):
            current_hunk.lines.append(
                DiffLine(
                    type="context",
                    content=line[1:],
                    old_line_no=old_line_no,
                    new_line_no=new_line_no,
                )
            )
            old_line_no += 1
            new_line_no += 1
        # Anything else, such as "\ No newline at end of file", is ignored

    return files
