"""
report.py
負責把 aggregator.py 產生的彙整結果，轉成人類可讀的 Markdown 報告。
"""

SEVERITY_EMOJI = {
    "bug": "🐛",
    "security": "🔒",
    "performance": "⚡",
    "style": "🎨",
}


def render_markdown(aggregated: dict) -> str:
    """把彙整結果轉成完整的 Markdown 字串。"""
    lines = ["# PR Review 報告", ""]
    lines.append(f"**總評**：{aggregated['overall_summary']}")
    lines.append("")

    if aggregated["total_issues"] == 0:
        lines.append("沒有發現需要修改的問題 ✅")
        return "\n".join(lines)

    for file_result in aggregated["files"]:
        filename = file_result["filename"]
        summary = file_result["summary"]
        issues = file_result["issues"]

        lines.append(f"## `{filename}`")
        if summary:
            lines.append(f"_{summary}_")
        lines.append("")

        if not issues:
            lines.append("沒有發現問題。")
            lines.append("")
            continue

        for issue in issues:
            emoji = SEVERITY_EMOJI.get(issue["severity"], "ℹ️")
            lines.append(f"- {emoji} **Line {issue['line']}** [{issue['severity']}]: {issue['comment']}")
            if issue.get("suggestion"):
                lines.append(f"  - 建議：{issue['suggestion']}")
        lines.append("")

    return "\n".join(lines)


def save_report(content: str, output_path: str) -> None:
    """把報告內容存成檔案。"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
