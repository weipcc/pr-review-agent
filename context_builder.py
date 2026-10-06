"""
context_builder.py
Builds the context fed to the LLM for each changed file.

Strategy:
- If the file is small (default < 400 lines), read its full content
- If the file is too large, take only N lines before and after each hunk to keep the prompt from growing too long
"""

from pathlib import Path

from diff_reader import FileDiff

MAX_FULL_FILE_LINES = 400
CONTEXT_WINDOW = 30  # Lines to take before and after each hunk when the file is too large

# Length cap for project context (README / PR description). This content goes into every LLM call,
# and being too long just wastes tokens, so only the opening section is kept (a README's opening is usually the project intro and architecture overview).
MAX_README_CHARS = 6000
MAX_PR_DESCRIPTION_CHARS = 3000

README_CANDIDATES = ("README.md", "README.rst", "README.txt", "README")


def truncate_text(text: str, limit: int) -> str:
    """Truncate and mark text longer than limit characters, so project context does not bloat the prompt."""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"


def read_local_readme(repo_path: str) -> str:
    """Read the README at the local repo root (return an empty string if not found) and truncate it to the length cap."""
    for name in README_CANDIDATES:
        path = Path(repo_path) / name
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            return truncate_text(text, MAX_README_CHARS)
    return ""


def read_file_lines(repo_path: str, filename: str) -> list[str]:
    """Read a local file and return a list of its lines (without newline characters)."""
    file_path = Path(repo_path) / filename
    if not file_path.exists():
        # The file may have been deleted; return an empty list
        return []
    return file_path.read_text(encoding="utf-8", errors="replace").splitlines()


def build_file_context(repo_path: str, file_diff: FileDiff) -> str:
    """
    Build the context text to put into the prompt for a single file.
    The result is a readable block of text with line numbers, so the LLM can map comments to the right locations.
    """
    lines = read_file_lines(repo_path, file_diff.filename)

    if not lines:
        return "(File was deleted or could not be read)"

    if len(lines) <= MAX_FULL_FILE_LINES:
        numbered = [f"{i + 1}: {line}" for i, line in enumerate(lines)]
        return "\n".join(numbered)

    # The file is too large; take only the content near each hunk
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


def build_diff_text(file_diff: FileDiff) -> str:
    """Convert a single file's diff hunks into a text format readable by humans and the LLM."""
    parts = [f"File: {file_diff.filename}"]
    for hunk in file_diff.hunks:
        parts.append(hunk.header)
        for line in hunk.lines:
            prefix = {"add": "+", "del": "-", "context": " "}[line.type]
            parts.append(f"{prefix}{line.content}")
    return "\n".join(parts)


def build_context_for_files(repo_path: str, file_diffs: list[FileDiff]) -> dict[str, dict]:
    """
    For all changed files, build a {filename: {"diff": ..., "context": ...}} structure,
    which reviewer.py can use directly to build prompts.
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
