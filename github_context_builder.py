"""
github_context_builder.py
Reads file contents at the PR head commit through the GitHub API,
replacing the original approach of reading from the local filesystem.
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
    Get a file's content at the given commit (ref) through the GitHub Contents API.
    If the file does not exist (e.g. it was deleted), return an empty string.
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
    Get the README at the given commit (ref) through the GitHub README API and truncate it to the length cap.
    The README is only supplementary background; if it cannot be fetched (no README, network or rate-limit problems), return an empty string,
    so the whole review does not fail because of it.
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
    """Build the context text (with line numbers) to put into the prompt for a single file."""
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
    For all changed files, build a {filename: {"diff": ..., "context": ...}} structure,
    which reviewer.py can use directly to build prompts.
    """
    result: dict[str, dict] = {}
    for file_diff in file_diffs:
        result[file_diff.filename] = {
            "diff": build_diff_text(file_diff),
            "context": build_file_context(owner, repo, ref, file_diff),
        }
    return result
