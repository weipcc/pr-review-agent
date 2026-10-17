"""
aggregator.py
負責把每個檔案的 review 結果彙整起來，並依嚴重程度排序，
以及產生整份 PR 的總體摘要。
"""

# 嚴重程度排序權重，數字越小越優先顯示
SEVERITY_ORDER = {
    "bug": 0,
    "security": 1,
    "performance": 2,
    "style": 3,
}


def sort_issues(issues: list[dict]) -> list[dict]:
    """依 severity 權重排序單一檔案內的 issues。"""
    return sorted(issues, key=lambda x: SEVERITY_ORDER.get(x.get("severity", "style"), 99))


def aggregate_results(review_results: list[dict]) -> dict:
    """
    輸入: review_file/review_all_files 產生的結果清單
    輸出:
    {
        "files": [ {filename, summary, issues(排序後)} ... ],
        "total_issues": int,
        "severity_counts": {"bug": n, "performance": n, ...},
        "overall_summary": str,
    }
    """
    files_output = []
    severity_counts = {"bug": 0, "security": 0, "performance": 0, "style": 0}
    total_issues = 0

    for result in review_results:
        sorted_issues = sort_issues(result.get("issues", []))
        for issue in sorted_issues:
            severity = issue.get("severity", "style")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            total_issues += 1

        files_output.append(
            {
                "filename": result["filename"],
                "summary": result.get("summary", ""),
                "issues": sorted_issues,
            }
        )

    # 檔案本身也依「是否有 bug/security」優先排序，問題嚴重的檔案排前面
    def file_priority(f):
        if not f["issues"]:
            return 99
        return SEVERITY_ORDER.get(f["issues"][0]["severity"], 99)

    files_output.sort(key=file_priority)

    overall_summary = _build_overall_summary(total_issues, severity_counts, len(files_output))

    return {
        "files": files_output,
        "total_issues": total_issues,
        "severity_counts": severity_counts,
        "overall_summary": overall_summary,
    }


def _build_overall_summary(total_issues: int, severity_counts: dict, file_count: int) -> str:
    if total_issues == 0:
        return f"審查了 {file_count} 個檔案，未發現明顯問題。"

    parts = []
    if severity_counts.get("bug"):
        parts.append(f"{severity_counts['bug']} 個 bug")
    if severity_counts.get("security"):
        parts.append(f"{severity_counts['security']} 個安全性疑慮")
    if severity_counts.get("performance"):
        parts.append(f"{severity_counts['performance']} 個效能疑慮")
    if severity_counts.get("style"):
        parts.append(f"{severity_counts['style']} 個風格建議")

    detail = "、".join(parts)
    return f"審查了 {file_count} 個檔案，共發現 {total_issues} 個問題：{detail}。"
