# PR Review Agent

A small tool that uses an LLM (currently the Google Gemini API) to automatically review GitHub PRs.
It offers two ways to use it: a local CLI version, and an API version that is called from a web page.

## Architecture

```
Input (PR URL / local repo)
   → Fetch the diff, file contents, and project context (PR title/description, the opening of the README)
   → Choose the review mode by total token count:
        Small PR → batched mode: all files are sent to Gemini at once, so it sees the whole PR
        Large PR → sequential mode: one call per file, with the other files' diffs attached as reference
   → Gemini returns structured results in a fixed schema (category, confidence, quoted source code, suggestion, ...)
   → Aggregate and sort
   → Produce a Markdown report (in English)
```

## Files

### Core logic (shared by the CLI and API versions)
- `reviewer.py` — Calls the Gemini API: batched / sequential modes, with `review_pr_files()` routing automatically by token count;
  the system prompts, few-shot examples, and Pydantic schemas also live here
- `aggregator.py` — Aggregates the results of all files, sorts by category and verdict, and computes the overall verdict for the PR
- `report.py` — Turns the aggregated result into a Markdown report

### CLI version (reviews a local git repo)
- `diff_reader.py` — Reads the local `git diff` and parses it into structured data
- `context_builder.py` — Reads local file contents and the README to build the LLM's context
- `main.py` — CLI entry point

### API version (reviews a remote GitHub PR)
- `github_diff_reader.py` — Fetches the remote PR's diff and basic info through the GitHub API
- `github_context_builder.py` — Fetches file contents and the README through the GitHub API
- `api.py` — A FastAPI wrapper that provides the `/review` endpoint
- `review_page.html` — A standalone small web page: enter a PR URL and it calls the local API

### Deprecated attempts
- `bookmarklet.js` / `bookmarklet.min.txt` — The original idea was to call the API directly from a GitHub
  page using a browser bookmark, but GitHub's CSP security policy blocks it, so
  `review_page.html` replaced it. Kept for the record.

## Review output

Each issue contains:

| Field | Description |
|---|---|
| `category` | 7 categories: `bug` / `security` / `performance` / `style` / `refactor` / `nitpick` / `question` |
| `confidence` | `high` / `medium` / `low`, how sure the model is about this issue |
| `evidence` | The source code that triggers the issue, quoted verbatim (used to reduce the chance of the model inventing issues) |
| `start_line` / `end_line` | The line range where the issue occurs |
| `comment` / `suggestion` / `suggested_code` | The issue description, the suggested fix, and code that can be applied directly (may be empty) |

Each file also has a `change_intent` (what this file's change is trying to do) and a `recommendation`
(`looks_good` / `minor_comments` / `needs_changes`); the verdict for the whole PR is the most severe one across all files.

The report first lists the PR title (API version only), the overall verdict, and a file overview table; only files with issues
are expanded in detail, while files without issues are collected at the end, one line per file. The report, terminal output,
and web interface are all in English.

## Environment setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=your_Gemini_key
```

## Usage

### CLI version (reviews a local git repo)

```bash
python main.py --repo-path /path/to/repo --base main
python main.py --repo-path . --output review.md
```

Without `--base`, uncommitted changes are reviewed. The CLI version has no PR title/description,
so only the repo's README is passed in as project context.

### API version (reviews a remote GitHub PR)

1. Start the API service:
   ```bash
   uvicorn api:app --reload --port 8000
   ```
2. Open `review_page.html` in Chrome (not Safari, whose file:// permissions are stricter)
3. Paste a PR URL and click "Start review"

Currently only **public repos** are supported, and no GitHub token is sent, so anonymous API rate limits apply
(about 60 requests per hour). Reviewing one PR uses roughly 3 + N requests
(PR info, diff, README, plus the contents of N files), so PRs with many files hit the limit more easily.

## Known limitations / areas for improvement

- GitHub API calls are anonymous, so the rate limit is low and private repos cannot be read
- Reviews are triggered manually; automatic triggering via a webhook is not wired up yet
- Only a report is returned; it is not yet posted back to the GitHub PR as inline comments
- It can only run locally; it has not been deployed to the cloud yet
- When a large PR goes through sequential mode, the model only sees the other files' diffs, not their full contents
- Only the first 6000 characters of the README and the first 3000 characters of the PR description are used; a full project architecture summary has not been built yet
- No benchmark has been built to quantify review quality (precision / recall), and it has not yet been verified how much
  the few-shot examples and project context actually help
- Sequential mode sends the few-shot examples on every call, which adds some token usage for large PRs
- The `refactor` and `question` categories currently have no dedicated few-shot examples
