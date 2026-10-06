"""
aggregator.py
Aggregates each file's review result and sorts by severity,
and produces the overall summary for the whole PR.

Changes in this version (matching the schema expansion in reviewer.py; local experimental version, not yet uploaded):
- severity -> category (adds refactor/nitpick/question)
- Adds aggregation of recommendation (counts of looks_good/minor_comments/needs_changes),
  as the basis for the PR-level conclusion
- The overall_summary template text is now in English, with the redundant conclusion wording removed
  (the conclusion is already expressed separately by overall_recommendation and shown by report.py, so it need not be repeated here)
- Fixes file_priority indexing ["category"] directly, which could raise a KeyError; it now uses .get()
"""

# Category sort weights; the smaller the number, the earlier it is shown
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
    """Sort the issues within a single file by category weight."""
    return sorted(issues, key=lambda x: CATEGORY_ORDER.get(x.get("category", "style"), 99))


def aggregate_results(review_results: list[dict]) -> dict:
    """
    Input: the list of results produced by review_file/review_all_files
    Output:
    {
        "files": [ {filename, change_intent, summary, recommendation, issues (sorted)} ... ],
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

    # Files themselves are sorted by recommendation first, with files that need changes at the front;
    # when recommendations are equal, sort by the most severe issue category
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
    """PR-level conclusion: if any file is needs_changes, the whole PR is needs_changes;
    otherwise, if any file is minor_comments, the whole PR is minor_comments; only if all files are looks_good is it looks_good."""
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
