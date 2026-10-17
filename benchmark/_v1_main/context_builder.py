"""
context_builder.py
負責為每個被改動的檔案組合出要餵給 LLM 的上下文。

策略：
- 檔案不大（預設 < 400 行）時，直接讀取完整內容
- 檔案太大時，只取每個 hunk 前後 N 行，避免 prompt 過長
"""

from pathlib import Path

from diff_reader import FileDiff

MAX_FULL_FILE_LINES = 400
CONTEXT_WINDOW = 30  # 檔案太大時，hunk 前後各取幾行


def read_file_lines(repo_path: str, filename: str) -> list[str]:
    """讀取本地檔案內容，回傳每一行的清單（不含換行符號）。"""
    file_path = Path(repo_path) / filename
    if not file_path.exists():
        # 檔案可能已被刪除，回傳空清單
        return []
    return file_path.read_text(encoding="utf-8", errors="replace").splitlines()


def build_file_context(repo_path: str, file_diff: FileDiff) -> str:
    """
    針對單一檔案，組合出要放進 prompt 的上下文文字。
    回傳格式是一段可讀的文字，包含行號，方便 LLM 對應到正確位置。
    """
    lines = read_file_lines(repo_path, file_diff.filename)

    if not lines:
        return "(檔案已被刪除或無法讀取)"

    if len(lines) <= MAX_FULL_FILE_LINES:
        numbered = [f"{i + 1}: {line}" for i, line in enumerate(lines)]
        return "\n".join(numbered)

    # 檔案太大，只取每個 hunk 附近的內容
    snippets: list[str] = []
    seen_ranges: set[tuple[int, int]] = set()

    for hunk in file_diff.hunks:
        start = max(0, hunk.new_start - 1 - CONTEXT_WINDOW)
        end = min(len(lines), hunk.new_start - 1 + CONTEXT_WINDOW)
        if (start, end) in seen_ranges:
            continue
        seen_ranges.add((start, end))

        snippet_lines = [f"{i + 1}: {lines[i]}" for i in range(start, end)]
        snippets.append(f"...(第 {start + 1} 行到第 {end} 行)...\n" + "\n".join(snippet_lines))

    return "\n\n".join(snippets)


def build_diff_text(file_diff: FileDiff) -> str:
    """把單一檔案的 diff hunks 轉成人類/LLM 可讀的文字格式。"""
    parts = [f"檔案: {file_diff.filename}"]
    for hunk in file_diff.hunks:
        parts.append(hunk.header)
        for line in hunk.lines:
            prefix = {"add": "+", "del": "-", "context": " "}[line.type]
            parts.append(f"{prefix}{line.content}")
    return "\n".join(parts)


def build_context_for_files(repo_path: str, file_diffs: list[FileDiff]) -> dict[str, dict]:
    """
    對所有變更檔案，組合出 {filename: {"diff": ..., "context": ...}} 的結構，
    方便 reviewer.py 直接拿去組 prompt。
    """
    result: dict[str, dict] = {}
    for file_diff in file_diffs:
        result[file_diff.filename] = {
            "diff": build_diff_text(file_diff),
            "context": build_file_context(repo_path, file_diff),
        }
    return result


if __name__ == "__main__":
    from diff_reader import get_parsed_diff

    diffs = get_parsed_diff(".")
    contexts = build_context_for_files(".", diffs)
    for filename, data in contexts.items():
        print(f"=== {filename} ===")
        print(data["diff"][:200])
        print("---")
