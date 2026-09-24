"""
report.py
負責把 aggregator.py 產生的彙整結果，轉成人類可讀的 Markdown 報告。

本版改動（本機實驗版，尚未上傳）：
- 報告輸出語言統一改成英文（之前 LLM 內容有時會跟著被審查檔案的語言混入中文，
  這裡除了模板文字翻成英文，也在 reviewer.py 的 system prompt 加了強制英文輸出的規則）
- 排版改進：檔案清單最上面加一個總覽表格，讓讀者一眼看到「有哪些檔案、各自的結論、
  各有幾個問題」，不用逐一往下捲；每個問題也拆成清楚的標籤行 + 引用 + 說明區塊，
  取代原本擠在同一個項目符號裡的長句
- 精簡篇幅：只有「有問題的檔案」才展開完整說明（change_intent + review + 每條 issue 的
  證據/建議/代碼）；「沒有問題的檔案」收進報告最後一個一行摘要的清單，不再重複展開整段
  說明。避免 PR 檔案數一多，報告長度跟著線性膨脹，稀釋掉真正需要注意的內容。
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
    """檔案總覽表格，讓讀者不用往下捲就能看到全貌。"""
    lines = ["| File | Recommendation | Issues |", "|---|---|---|"]
    for file_result in files:
        badge = RECOMMENDATION_BADGE.get(file_result.get("recommendation"), "")
        issue_count = len(file_result["issues"])
        issues_cell = str(issue_count) if issue_count else "–"
        lines.append(f"| `{file_result['filename']}` | {badge} | {issues_cell} |")
    lines.append("")
    return lines


def _render_compact_file(file_result: dict) -> str:
    """沒有問題的檔案，只用一行摘要交代這個檔案改了什麼，不重複展開整段說明。"""
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
    """把彙整結果轉成完整的 Markdown 字串。"""
    lines = ["# PR Review Report", ""]

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

    # 只有「有問題的檔案」才展開完整說明；「沒有問題的檔案」放進最後的一行摘要清單，
    # 避免每個 clean 檔案都重複一遍 change_intent/summary，讓報告長度跟著檔案數線性膨脹。
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

        # 除非這是最後一段內容（後面既沒有其他有問題的檔案，也沒有 clean 檔案清單），
        # 否則加分隔線；避免結尾出現孤立的 "---"。
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
    """把報告內容存成檔案。"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
