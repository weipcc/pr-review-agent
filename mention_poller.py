"""
mention_poller.py
Polls GitHub notifications for @mentions on pull requests, runs the review
pipeline, and posts the result as a PR comment.

Required env vars:
    GITHUB_TOKEN  — Personal Access Token with scopes: notifications, repo
                    (or public_repo for public repos only)

How it works:
    1. Fetch unread notifications where reason=mention and type=PullRequest.
    2. Run the existing review pipeline on the PR.
    3. Post the markdown report as a comment on the PR.
    4. Mark the notification as done — it won't appear in future polls.
"""

import os

from github import Github, GithubException


def _run_review_pipeline(pr_url: str) -> str:
    """Run the full review pipeline and return the rendered markdown report."""
    from aggregator import aggregate_results
    from context_builder import MAX_PR_DESCRIPTION_CHARS, truncate_text
    from github_context_builder import build_context_for_files, get_readme
    from github_diff_reader import get_parsed_pr_diff
    from report import render_markdown
    from reviewer import review_all_files

    file_diffs, metadata = get_parsed_pr_diff(pr_url)
    if not file_diffs:
        raise ValueError("No changed files detected in this PR")

    owner = metadata["_owner"]
    repo = metadata["_repo"]
    head_sha = metadata["_head_sha"]

    file_contexts = build_context_for_files(owner, repo, head_sha, file_diffs)
    project_context = {
        "pr_title": metadata.get("title") or "",
        "pr_description": truncate_text(metadata.get("body") or "", MAX_PR_DESCRIPTION_CHARS),
        "readme": get_readme(owner, repo, head_sha),
    }

    review_results = review_all_files(file_contexts, project_context)
    aggregated = aggregate_results(review_results)
    aggregated["pr_title"] = project_context["pr_title"]
    aggregated["pr_url"] = pr_url
    return render_markdown(aggregated)


def poll_once() -> int:
    """
    Run one poll cycle. Returns the number of PRs successfully reviewed.
    Raises RuntimeError if GITHUB_TOKEN is not set.
    """
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN is required for mention polling. "
            "Create a Personal Access Token with 'notifications' and 'repo' scopes."
        )

    gh = Github(token)
    reviewed_count = 0

    # all=False (default) returns only unread notifications.
    # After mark_as_done(), a notification never reappears — no state file needed.
    notifications = gh.get_user().get_notifications(participating=True, all=False)

    for notification in notifications:
        if notification.reason != "mention":
            continue
        if notification.subject.type != "PullRequest":
            continue

        # Convert API URL → web URL
        # API shape: https://api.github.com/repos/{owner}/{repo}/pulls/{number}
        parts = notification.subject.url.rstrip("/").split("/")
        owner_name, repo_name, pr_number = parts[-4], parts[-3], int(parts[-1])
        pr_url = f"https://github.com/{owner_name}/{repo_name}/pull/{pr_number}"

        print(f"[poller] @mention detected on {pr_url}")

        try:
            markdown_report = _run_review_pipeline(pr_url)
            repo_obj = gh.get_repo(f"{owner_name}/{repo_name}")
            repo_obj.get_pull(pr_number).create_issue_comment(markdown_report)
            print(f"[poller] ✅ Review posted on {pr_url}")
            reviewed_count += 1
        except Exception as exc:
            print(f"[poller] ❌ Failed for {pr_url}: {exc}")
        finally:
            # Always mark as done so this notification never comes back
            try:
                notification.mark_as_done()
            except GithubException:
                pass

    return reviewed_count
