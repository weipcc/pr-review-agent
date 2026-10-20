"""
context_builder.py
Builds the text fed to the LLM from the diff of every changed file, and provides a truncation helper for the project background information (README / PR description).
The file contents themselves are no longer put in the prompt; the agent explores them on its own in the sandbox.
"""

from diff_reader import FileDiff

# Length limit for the project background information (README / PR description). This content goes into every LLM call,
# and too much of it only adds tokens, so only the beginning is kept (the start of a README is usually the project introduction and architecture overview).
MAX_README_CHARS = 6000
MAX_PR_DESCRIPTION_CHARS = 3000


def truncate_text(text: str, limit: int) -> str:
    """Truncate and mark the text if it is longer than limit characters, so the project background does not bloat the prompt."""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"


def build_diff_text(file_diff: FileDiff) -> str:
    """Convert the diff hunks of a single file into a text format readable by humans and the LLM."""
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
