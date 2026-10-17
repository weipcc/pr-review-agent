"""
diff_reader.py
負責從本地 git repo 讀取 diff，並解析成結構化資料。

輸出格式範例:
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
import subprocess
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


def get_raw_diff(repo_path: str, base: str = "HEAD") -> str:
    """
    取得本地 git diff 的原始文字輸出。

    repo_path: 目標 repo 的路徑
    base: 比較基準，例如 "main"、"HEAD"（預設 HEAD 代表比對 working directory 的未 commit 修改）
    """
    if base == "HEAD":
        cmd = ["git", "-C", repo_path, "diff", "HEAD"]
    else:
        cmd = ["git", "-C", repo_path, "diff", f"{base}...HEAD"]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"git diff 執行失敗: {result.stderr}")
    return result.stdout


def parse_diff(raw_diff: str) -> list[FileDiff]:
    """把 git diff 的原始文字解析成結構化的 FileDiff 清單。"""
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
        # 其他如 "\ No newline at end of file" 直接忽略

    return files


def get_parsed_diff(repo_path: str, base: str = "HEAD") -> list[FileDiff]:
    """整合函式：直接回傳解析好的 diff 結構。"""
    raw = get_raw_diff(repo_path, base)
    return parse_diff(raw)


if __name__ == "__main__":
    # 簡單測試：對目前資料夾跑一次
    diffs = get_parsed_diff(".")
    for f in diffs:
        print(f"檔案: {f.filename}, hunks 數量: {len(f.hunks)}")