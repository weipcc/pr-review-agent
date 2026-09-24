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
    parser = argparse.ArgumentParser(description="Local PR Review Agent")
    parser.add_argument(
        "--repo-path",
        default=".",
        help="Path to the git repo to review (default: current directory)",
    )
    parser.add_argument(
        "--base",
        default="HEAD",
        help="Base branch to compare against, e.g. 'main'. Default is HEAD, i.e. uncommitted changes",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the report to, e.g. review.md. If omitted, the report is printed to the terminal",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Reading diff... (repo: {args.repo_path}, base: {args.base})")
    file_diffs = get_parsed_diff(args.repo_path, args.base)

    if not file_diffs:
        print("No changes detected. Exiting.")
        sys.exit(0)

    print(f"Detected {len(file_diffs)} changed file(s). Building context...")
    file_contexts = build_context_for_files(args.repo_path, file_diffs)

    print("Calling the LLM to review...")
    review_results = review_all_files(file_contexts)

    print("Aggregating results...")
    aggregated = aggregate_results(review_results)

    report_text = render_markdown(aggregated)

    if args.output:
        save_report(report_text, args.output)
        print(f"Report saved to: {args.output}")
    else:
        print("\n" + report_text)


if __name__ == "__main__":
    main()
