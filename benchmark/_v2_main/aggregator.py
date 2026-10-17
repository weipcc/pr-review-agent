"""
aggregator.py
負責把每個檔案的 review 結果彙整起來，並依嚴重程度排序，
以及產生整份 PR 的總體摘要。

本版改動（配合 reviewer.py 的 schema 擴充，本機實驗版，尚未上傳）：
- severity -> category（新增 refactor/nitpick/question）
- 加入 recommendation 的彙整（looks_good/minor_comments/needs_changes 計數），
  作為整份 PR 的結論性摘要依據
- overall_summary 的模板文字改成英文，並拿掉重複的結論字句
  （結論已經由 overall_recommendation 單獨表達，report.py 會另外顯示，不用在這裡重複一次）
- 修掉 file_priority 裡直接用 ["category"] 索引可能觸發 KeyError 的問題，改用 .get()
"""

# 分類排序權重，數字越小越優先顯示
CATEGORY_ORDER = {
    "bug": 0,
    "security": 1,
    "performance": 2,
    "refactor": 3,
    "style": 4,
    "nitpick": 5,
    "question": 6,
}

RECOMMENDATION_ORDER = {
    "needs_changes": 0,
    "minor_comments": 1,
    "looks_good": 2,
}

CATEGORY_LABEL = {
    "bug": "bug",
    "security": "security issue",
    "performance": "performance issue",
    "refactor": "refactor suggestion",
    "style": "style issue",
    "nitpick": "nitpick",
    "question": "open question",
}


def sort_issues(issues: list[dict]) -> list[dict]:
    """依 category 權重排序單一檔案內的 issues。"""
    return sorted(issues, key=lambda x: CATEGORY_ORDER.get(x.get("category", "style"), 99))


def aggregate_results(review_results: list[dict]) -> dict:
    """
    輸入: review_file/review_all_files 產生的結果清單
    輸出:
    {
        "files": [ {filename, change_intent, summary, recommendation, issues(排序後)} ... ],
        "total_issues": int,
        "category_counts": {"bug": n, "security": n, ...},
        "recommendation_counts": {"needs_changes": n, "minor_comments": n, "looks_good": n},
        "overall_recommendation": str,
        "overall_summary": str,
    }
    """
    files_output = []
    category_counts = {k: 0 for k in CATEGORY_ORDER}
    recommendation_counts = {k: 0 for k in RECOMMENDATION_ORDER}
    total_issues = 0

    for result in review_results:
        sorted_issues = sort_issues(result.get("issues", []))
        for issue in sorted_issues:
            category = issue.get("category", "style")
            category_counts[category] = category_counts.get(category, 0) + 1
            total_issues += 1

        recommendation = result.get("recommendation", "looks_good")
        recommendation_counts[recommendation] = recommendation_counts.get(recommendation, 0) + 1

        files_output.append(
            {
                "filename": result["filename"],
                "change_intent": result.get("change_intent", ""),
                "summary": result.get("summary", ""),
                "recommendation": recommendation,
                "issues": sorted_issues,
            }
        )

    # 檔案本身依 recommendation 優先排序，需要修改的檔案排前面；
    # recommendation 相同時再依最嚴重的 issue category 排序
    def file_priority(f):
        rec_rank = RECOMMENDATION_ORDER.get(f["recommendation"], 99)
        issue_rank = CATEGORY_ORDER.get(f["issues"][0].get("category", "style"), 99) if f["issues"] else 99
        return (rec_rank, issue_rank)

    files_output.sort(key=file_priority)

    overall_recommendation = _overall_recommendation(recommendation_counts, len(files_output))
    overall_summary = _build_overall_summary(total_issues, category_counts, len(files_output))

    return {
        "files": files_output,
        "total_issues": total_issues,
        "category_counts": category_counts,
        "recommendation_counts": recommendation_counts,
        "overall_recommendation": overall_recommendation,
        "overall_summary": overall_summary,
    }


def _overall_recommendation(recommendation_counts: dict, file_count: int) -> str:
    """整份 PR 的結論：任一檔案 needs_changes 就整體 needs_changes；
    否則任一檔案 minor_comments 就整體 minor_comments；全部 looks_good 才是 looks_good。"""
    if file_count == 0:
        return "looks_good"
    if recommendation_counts.get("needs_changes"):
        return "needs_changes"
    if recommendation_counts.get("minor_comments"):
        return "minor_comments"
    return "looks_good"


def _pluralize(count: int, singular: str) -> str:
    return f"{count} {singular}" if count == 1 else f"{count} {singular}s"


def _build_overall_summary(total_issues: int, category_counts: dict, file_count: int) -> str:
    if total_issues == 0:
        return f"Reviewed {file_count} file(s); no issues found."

    parts = [
        _pluralize(category_counts[category], CATEGORY_LABEL[category])
        for category in CATEGORY_ORDER
        if category_counts.get(category)
    ]
    detail = ", ".join(parts)
    return f"Reviewed {file_count} file(s) and found {total_issues} issue(s): {detail}."
