"""
main.py
CLI 入口。串接 diff_reader -> context_builder -> reviewer -> aggregator -> report。

使用範例:
    python main.py --repo-path /path/to/your/repo --base main
    python main.py --repo-path . --output review.md
"""

import argparse
import sys

from aggregator import aggregate_results
from context_builder import build_context_for_files
from diff_reader import get_parsed_diff
from report import render_markdown, save_report
from reviewer import review_all_files


def parse_args():
    parser = argparse.ArgumentParser(description="本地端 PR Review Agent")
    parser.add_argument(
        "--repo-path",
        default=".",
        help="要審查的 git repo 路徑（預設為目前資料夾）",
    )
    parser.add_argument(
        "--base",
        default="HEAD",
        help="比較基準分支，例如 'main'。預設 HEAD，代表比對尚未 commit 的變更",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="把報告存成檔案的路徑，例如 review.md。不指定則直接印在終端機",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"讀取 diff 中... (repo: {args.repo_path}, base: {args.base})")
    file_diffs = get_parsed_diff(args.repo_path, args.base)

    if not file_diffs:
        print("沒有偵測到任何變更，結束。")
        sys.exit(0)

    print(f"偵測到 {len(file_diffs)} 個變更檔案，組合上下文中...")
    file_contexts = build_context_for_files(args.repo_path, file_diffs)

    print("呼叫 LLM 進行審查中...")
    review_results = review_all_files(file_contexts)

    print("彙整結果中...")
    aggregated = aggregate_results(review_results)

    report_text = render_markdown(aggregated)

    if args.output:
        save_report(report_text, args.output)
        print(f"報告已存成: {args.output}")
    else:
        print("\n" + report_text)


if __name__ == "__main__":
    main()
