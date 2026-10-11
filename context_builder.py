"""
context_builder.py
把每個被改動檔案的 diff 組成要餵給 LLM 的文字，並提供專案背景資訊（README / PR 描述）的截斷工具。
檔案內容本身不再放進 prompt，由 agent 在 sandbox 裡自行探索。
"""

from diff_reader import FileDiff

# 專案背景資訊（README / PR 描述）的長度上限。這些內容會被放進每一次 LLM 呼叫，
# 太長會白白增加 token，所以只取開頭一段（README 的開頭通常就是專案簡介與架構概述）。
MAX_README_CHARS = 6000
MAX_PR_DESCRIPTION_CHARS = 3000


def truncate_text(text: str, limit: int) -> str:
    """超過 limit 個字元就截斷並標註，避免專案背景資訊撐大 prompt。"""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"


def build_diff_text(file_diff: FileDiff) -> str:
    """把單一檔案的 diff hunks 轉成人類/LLM 可讀的文字格式。"""
    parts = [f"File: {file_diff.filename}"]
    for hunk in file_diff.hunks:
        parts.append(hunk.header)
        for line in hunk.lines:
            prefix = {"add": "+", "del": "-", "context": " "}[line.type]
            parts.append(f"{prefix}{line.content}")
    return "\n".join(parts)


def build_diffs_for_files(file_diffs: list[FileDiff]) -> dict[str, dict]:
    """{filename: {"diff": ...}}: the structure reviewer.py expects (file contents are explored in the sandbox)."""
    return {fd.filename: {"diff": build_diff_text(fd)} for fd in file_diffs}
