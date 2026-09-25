"""
github_context_builder.py
負責透過 GitHub API 讀取 PR head commit 版本的檔案內容，
取代原本從本機檔案系統讀取的方式。
"""

import base64

import requests

from context_builder import MAX_README_CHARS, build_diff_text, truncate_text
from diff_reader import FileDiff

GITHUB_API_BASE = "https://api.github.com"
MAX_FULL_FILE_LINES = 400
CONTEXT_WINDOW = 30


def get_file_content(owner: str, repo: str, path: str, ref: str) -> str:
    """
    透過 GitHub Contents API 取得指定 commit（ref）版本的檔案內容。
    若檔案不存在（例如被刪除），回傳空字串。
    """
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{path}"
    response = requests.get(url, params={"ref": ref}, timeout=30)

    if response.status_code == 404:
        return ""
    response.raise_for_status()

    data = response.json()
    if data.get("encoding") == "base64":
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    return data.get("content", "")


def get_readme(owner: str, repo: str, ref: str) -> str:
    """
    透過 GitHub README API 取得指定 commit（ref）版本的 README，並截斷到長度上限。
    README 只是補充背景，抓不到（沒有 README、網路或速率限制問題）就回傳空字串，
    不要讓整個審查因此失敗。
    """
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/readme"
    try:
        response = requests.get(url, params={"ref": ref}, timeout=30)
        if response.status_code == 404:
            return ""
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"Warning: could not fetch README ({e}); continuing without it.")
        return ""

    if data.get("encoding") == "base64":
        text = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    else:
        text = data.get("content", "")
    return truncate_text(text, MAX_README_CHARS)


def build_file_context(owner: str, repo: str, ref: str, file_diff: FileDiff) -> str:
    """針對單一檔案，組合出要放進 prompt 的上下文文字（含行號）。"""
    content = get_file_content(owner, repo, file_diff.filename, ref)
    if not content:
        return "(File was deleted or could not be read)"

    lines = content.splitlines()

    if len(lines) <= MAX_FULL_FILE_LINES:
        numbered = [f"{i + 1}: {line}" for i, line in enumerate(lines)]
        return "\n".join(numbered)

    snippets: list[str] = []
    seen_ranges: set[tuple[int, int]] = set()

    for hunk in file_diff.hunks:
        start = max(0, hunk.new_start - 1 - CONTEXT_WINDOW)
        end = min(len(lines), hunk.new_start - 1 + CONTEXT_WINDOW)
        if (start, end) in seen_ranges:
            continue
        seen_ranges.add((start, end))

        snippet_lines = [f"{i + 1}: {lines[i]}" for i in range(start, end)]
        snippets.append(f"...(lines {start + 1} to {end})...\n" + "\n".join(snippet_lines))

    return "\n\n".join(snippets)


def build_context_for_files(
    owner: str, repo: str, ref: str, file_diffs: list[FileDiff]
) -> dict[str, dict]:
    """
    對所有變更檔案，組合出 {filename: {"diff": ..., "context": ...}} 的結構，
    方便 reviewer.py 直接拿去組 prompt。
    """
    result: dict[str, dict] = {}
    for file_diff in file_diffs:
        result[file_diff.filename] = {
            "diff": build_diff_text(file_diff),
            "context": build_file_context(owner, repo, ref, file_diff),
        }
    return result
