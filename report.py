"""
report.py
Turns the aggregated result produced by aggregator.py into a human-readable Markdown report.

Changes in this version (local experimental version, not yet uploaded):
- The report output language is unified to English (previously the LLM content sometimes drifted into Chinese following the language of the files under review;
  besides translating the template text to English here, a rule forcing English output was also added to the system prompt in reviewer.py)
- Layout improvements: an overview table is added at the top of the file list so readers can see at a glance "which files, each file's verdict,
  and how many issues each has" without scrolling through everything; each issue is also split into a clear label line + quote + explanation block,
  replacing the long sentence previously crammed into a single bullet
- Shorter reports: only "files with issues" get the full explanation (change_intent + review + each issue's
  evidence/suggestion/code); "files without issues" go into a one-line-per-file summary list at the end of the report instead of repeating a full
  explanation. This keeps report length from growing linearly with the number of files in a PR and diluting what actually needs attention.
"""

CATEGORY_EMOJI = {
    "bug": "🐛",
    "security": "🔒",
    "performance": "⚡",
    "style": "🎨",
    "refactor": "♻️",
    "nitpick": "🧹",
    "question": "❓",
}

CONFIDENCE_LABEL = {
    "high": "High",
    "medium": "Medium",
    "low": "Low",
}

RECOMMENDATION_BADGE = {
    "needs_changes": "🔴 Needs Changes",
    "minor_comments": "🟡 Minor Comments",
    "looks_good": "🟢 Looks Good",
}


def _format_line_range(issue: dict) -> str:
    start = issue.get("start_line")
    end = issue.get("end_line", start)
    if start is None:
        return ""
    if end and end != start:
        return f"Line {start}-{end}"
    return f"Line {start}"


def _render_summary_table(files: list[dict]) -> list[str]:
    """File overview table, so readers can see the whole picture without scrolling."""
    lines = ["| File | Recommendation | Issues |", "|---|---|---|"]
    for file_result in files:
        badge = RECOMMENDATION_BADGE.get(file_result.get("recommendation"), "")
        issue_count = len(file_result["issues"])
        issues_cell = str(issue_count) if issue_count else "–"
        lines.append(f"| `{file_result['filename']}` | {badge} | {issues_cell} |")
    lines.append("")
    return lines


def _render_compact_file(file_result: dict) -> str:
    """For files without issues: a one-line summary of what the file changed, without repeating the full explanation."""
    badge = RECOMMENDATION_BADGE.get(file_result.get("recommendation"), "")
    emoji = badge.split(" ", 1)[0] if badge else ""
    filename = file_result["filename"]
    change_intent = file_result.get("change_intent", "")
    if change_intent:
        return f"- {emoji} `{filename}` — {change_intent}"
    return f"- {emoji} `{filename}`"


def _render_issue(index: int, issue: dict) -> list[str]:
    emoji = CATEGORY_EMOJI.get(issue["category"], "ℹ️")
    category_label = issue["category"].capitalize()
    confidence_label = CONFIDENCE_LABEL.get(issue.get("confidence"), issue.get("confidence", ""))
    line_range = _format_line_range(issue)

    lines = [f"**{index}. {emoji} {category_label}** · {confidence_label} confidence · {line_range}", ""]

    evidence = issue.get("evidence")
    if evidence:
        for evidence_line in evidence.splitlines():
            lines.append(f"> `{evidence_line}`" if evidence_line.strip() else ">")
        lines.append("")

    lines.append(issue["comment"])
    lines.append("")

    if issue.get("suggestion"):
        lines.append(f"**Suggestion:** {issue['suggestion']}")
        lines.append("")

    if issue.get("suggested_code"):
        lines.append("**Suggested fix:**")
        lines.append("```")
        lines.extend(issue["suggested_code"].splitlines())
        lines.append("```")
        lines.append("")

    return lines


def render_markdown(aggregated: dict) -> str:
    """Convert the aggregated result into a complete Markdown string."""
    lines = ["# PR Review Report", ""]

    pr_title = aggregated.get("pr_title")
    if pr_title:
        pr_url = aggregated.get("pr_url")
        lines.append(f"**PR:** {pr_title}" + (f" ({pr_url})" if pr_url else ""))

    overall_recommendation = aggregated.get("overall_recommendation")
    if overall_recommendation:
        lines.append(f"**Overall recommendation:** {RECOMMENDATION_BADGE.get(overall_recommendation, overall_recommendation)}")
    lines.append(f"**Summary:** {aggregated['overall_summary']}")
    lines.append("")

    files = aggregated["files"]
    if not files:
        return "\n".join(lines).rstrip() + "\n"

    lines.extend(_render_summary_table(files))
    lines.append("---")
    lines.append("")

    # Only "files with issues" get the full explanation; "files without issues" go into the one-line summary list at the end,
    # so every clean file does not repeat change_intent/summary and the report does not grow linearly with the number of files.
    detailed_files = [f for f in files if f["issues"]]
    clean_files = [f for f in files if not f["issues"]]

    for index, file_result in enumerate(detailed_files):
        filename = file_result["filename"]
        change_intent = file_result.get("change_intent", "")
        summary = file_result["summary"]
        recommendation = file_result.get("recommendation")
        issues = file_result["issues"]

        badge = RECOMMENDATION_BADGE.get(recommendation, "")
        lines.append(f"## `{filename}` — {badge}")
        lines.append("")
        if change_intent:
            lines.append(f"**What changed:** {change_intent}")
        if summary:
            lines.append(f"**Review:** {summary}")
        lines.append("")

        for i, issue in enumerate(issues, 1):
            lines.extend(_render_issue(i, issue))

        # Unless this is the last block (no more files with issues follow, and there is no clean-files list),
        # add a separator line; this avoids a dangling "---" at the end.
        is_last_detailed = index == len(detailed_files) - 1
        if not is_last_detailed or clean_files:
            lines.append("---")
            lines.append("")

    if clean_files:
        lines.append("### Files with no issues")
        lines.append("")
        for file_result in clean_files:
            lines.append(_render_compact_file(file_result))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def save_report(content: str, output_path: str) -> None:
    """Save the report content to a file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
