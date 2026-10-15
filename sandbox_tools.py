"""
sandbox_tools.py
E2B sandbox that holds a checkout of the PR, plus the single `run_bash` tool the review agent
uses to explore the changed files and their dependencies.

Output-size best practices applied here (the same ones coding agents like Claude Code / Codex use):
1. The *prompt* tells the model to bound its own output (grep -m / head / tail / sed -n 'A,Bp' /
   wc -l first), so most commands never produce a lot of output.
2. The *tool* enforces a hard cap (MAX_OUTPUT_CHARS) as a safety net, and says how much was cut
   so the model knows to narrow its command rather than assume it saw everything.
3. Commands have a timeout, and failures are returned as text (exit code + stderr) instead of
   exceptions, so the model can recover.
"""

import base64
import os
import shlex
from contextlib import contextmanager

from langchain_core.tools import tool

REPO_DIR = "/home/user/repo"
SANDBOX_LIFETIME_SECONDS = 15 * 60
COMMAND_TIMEOUT_SECONDS = 30

# run_bash is a file READER, not a terminal: only these commands may run (enforced in check_read_only).
ALLOWED_COMMANDS = ("grep", "egrep", "fgrep", "sed", "head", "tail", "wc", "ls", "find", "cat", "nl", "sort", "uniq", "cut")
_SEPARATORS = {"|", "||", "&&", ";"}
_FIND_FORBIDDEN = {"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprintf", "-fls"}

# Hard cap on the characters a single tool call can return to the model.
MAX_OUTPUT_CHARS = 5000


def _api_key() -> str:
    return os.environ.get("E2B_API_KEY") or os.environ.get("E2B_KEY") or ""


def truncate_output(text: str) -> str:
    """Cap the returned text at MAX_OUTPUT_CHARS and tell the model if anything was cut."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return (
        text[:MAX_OUTPUT_CHARS]
        + f"\n...[output truncated: {len(text) - MAX_OUTPUT_CHARS} more chars. "
        "Use a more targeted command: grep -m N, head, tail, or sed -n 'A,Bp']"
    )


def check_read_only(command: str) -> str | None:
    """
    Enforce that a command only READS files. Returns None if it is allowed, else a message
    explaining why it was blocked. Rules: every command in a pipeline/list must be in
    ALLOWED_COMMANDS (or `git show`); no redirects (except to /dev/null or 2>&1), no command/process
    substitution, no `sed -i`, no `find` actions (-exec, -delete, ...) and no `find` outside the repo.
    """
    if "`" in command or "$(" in command or "<(" in command or ">(" in command or "\n" in command.strip():
        return "command substitution and multi-line scripts are not allowed"
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError as e:
        return f"could not parse command ({e})"
    if not tokens:
        return "empty command"

    segments: list[list[str]] = [[]]
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if tok in _SEPARATORS:
            segments.append([])
        elif tok in (">", ">>"):
            if nxt != "/dev/null":
                return "redirecting output to files is not allowed (only to /dev/null)"
            i += 1
        elif tok == ">&":
            if not nxt.isdigit():
                return "redirecting output to files is not allowed"
            i += 1
        elif tok and set(tok) <= set("<>&();|"):
            return f"shell operator {tok!r} is not allowed"
        else:
            segments[-1].append(tok)
        i += 1

    for seg in segments:
        if not seg:
            return "empty command in pipeline"
        cmd, args = seg[0], seg[1:]
        if cmd == "git":
            if not args or args[0] != "show":
                return "only `git show` is allowed"
        elif cmd not in ALLOWED_COMMANDS:
            return f"`{cmd}` is not allowed"
        if cmd == "sed" and any(a == "--in-place" or (a.startswith("-") and not a.startswith("--") and "i" in a) for a in args):
            return "`sed -i` (in-place edit) is not allowed"
        if cmd == "sort" and any(a == "--output" or a.startswith("--output=") or (a.startswith("-") and not a.startswith("--") and "o" in a) for a in args):
            return "`sort -o` (write to file) is not allowed"
        if cmd == "find":
            if _FIND_FORBIDDEN & set(args):
                return "`find` actions that run commands or write files are not allowed"
            if any(a.startswith("/") or a.startswith("..") or a.startswith("~") for a in args):
                return "`find` must stay inside the repository (use relative paths)"
    return None


TOOL_DESCRIPTION = (
    "Read files of the checked-out pull request repository (working directory = repo root). "
    "Read-only: only " + ", ".join(ALLOWED_COMMANDS) + " and `git show <rev>:<path>` run; anything else is blocked. "
    "Output is truncated, so keep it small (grep | head, sed -n 'A,Bp', head, tail). "
    "Returns the exit code, stdout and stderr."
)


class PRSandbox:
    """An E2B sandbox with the PR's head commit checked out at REPO_DIR."""

    def __init__(self, owner: str, repo: str, pr_number: int, head_sha: str, base_sha: str):
        from e2b import Sandbox

        self.owner, self.repo, self.pr_number = owner, repo, pr_number
        self.head_sha, self.base_sha = head_sha, base_sha
        self._sandbox = Sandbox.create(timeout=SANDBOX_LIFETIME_SECONDS, api_key=_api_key())
        try:
            self._checkout()
            self._block_network()
        except Exception:
            self.close()
            raise

    def _checkout(self) -> None:
        # Fetch only the two commits we need (shallow) instead of cloning the whole history.
        # The token is passed as a per-command header so it is never written into .git/config.
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        auth = ""
        if token:
            basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
            auth = f"-c http.extraheader='Authorization: Basic {basic}' "
        url = f"https://github.com/{self.owner}/{self.repo}.git"
        script = " && ".join(
            [
                f"mkdir -p {REPO_DIR}",
                f"cd {REPO_DIR}",
                "git init -q",
                f"git remote add origin {url}",
                # pull/N/head works for fork PRs too; the base sha is needed for `git diff`.
                f"git {auth}fetch -q --depth=1 origin refs/pull/{self.pr_number}/head",
                f"git {auth}fetch -q --depth=1 origin {self.base_sha}",
                f"git checkout -q --detach {self.head_sha}",
            ]
        )
        result = self._sandbox.commands.run(script, timeout=120)
        if result.exit_code != 0:
            raise RuntimeError(f"Sandbox checkout failed: {result.stderr}")

    def _block_network(self) -> None:
        """
        Cut ALL outbound traffic once the checkout is done, so nothing the agent runs (or any
        prompt-injected command from the PR's code) can exfiltrate data or fetch anything.
        Fails closed: if egress can't be blocked, or a probe shows it still works, we raise and
        the sandbox is destroyed.
        """
        self._sandbox.update_network({"deny_out": ["0.0.0.0/0"]})
        probe = self._sandbox.commands.run(
            "curl -sS -m 5 -o /dev/null https://github.com; echo $?", timeout=20
        )
        if probe.stdout.strip().splitlines()[-1:] == ["0"]:
            raise RuntimeError("Sandbox network is still reachable after deny-all; refusing to continue.")

    def run(self, command: str) -> str:
        """Run a bash command from the repo root (cwd=REPO_DIR) and return exit code + capped output."""
        from e2b.sandbox.commands.command_handle import CommandExitException

        blocked = check_read_only(command)
        if blocked:
            return f"[blocked] {blocked}. run_bash is read-only; allowed: {', '.join(ALLOWED_COMMANDS)}, `git show <rev>:<path>`."

        try:
            result = self._sandbox.commands.run(
                command, cwd=REPO_DIR, timeout=COMMAND_TIMEOUT_SECONDS
            )
            exit_code, stdout, stderr = result.exit_code, result.stdout, result.stderr
        except CommandExitException as e:  # non-zero exit
            exit_code, stdout, stderr = e.exit_code, e.stdout, e.stderr
        except Exception as e:  # timeout, sandbox gone, ...
            return f"[tool error] {type(e).__name__}: {e}"

        parts = [f"[exit code: {exit_code}]"]
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append(f"[stderr]\n{stderr}")
        if not stdout and not stderr:
            parts.append("(no output)")
        return truncate_output("\n".join(parts))

    def close(self) -> None:
        try:
            self._sandbox.kill()
        except Exception as e:
            print(f"Warning: failed to kill sandbox: {e}")

    def make_tool(self):
        """Build the single `run_bash` LangChain tool bound to this sandbox."""

        @tool(description=TOOL_DESCRIPTION)
        def run_bash(command: str) -> str:
            return self.run(command)

        return run_bash

    def guide(self) -> str:
        """System-prompt section telling the agent where the repo is and how to explore it."""
        return SANDBOX_GUIDE.format(
            repo_dir=REPO_DIR,
            base_sha=self.base_sha,
            max_chars=MAX_OUTPUT_CHARS,
            max_steps=MAX_STEPS,
            allowed=", ".join(f"`{c}`" for c in ALLOWED_COMMANDS),
        )


# One step = one round of: LLM turn -> run_bash call(s) -> result(s). After MAX_STEPS rounds the agent
# is stripped of its tools and forced to answer (see StepBudgetMiddleware in reviewer.py).
MAX_STEPS = 10

SANDBOX_GUIDE = """# Reading the repository (run_bash tool)
`run_bash` lets you READ files of the pull request's repository. Use it only to look at code the diffs do not show:
- callers of anything a diff renames, removes or changes (signature, return value, behavior): \
`grep -rn --exclude-dir=.git 'name' . | head -n 20`, then `sed -n` the lines around a suspicious hit;
- the definition of a helper or class the new code uses but the diff does not contain;
- the unchanged code around a hunk, if the hunk alone is not enough to judge it.
Search for identifiers from the diffs, not generic words. Do not re-read the diffs, and do not list the repository or \
the changed files (`ls`, `find`, `git diff --stat`): you already have them.

Setup:
- Working directory: always the repository root, {repo_dir} (every call is a fresh shell, so `cd` does not persist). \
Use relative paths.
- It is a shallow checkout: HEAD is the PR head commit and the files are the post-change versions. \
`git show {base_sha}:<path>` gives the pre-PR version of a file. There is no other git history.
- Read-only, enforced: only {allowed} and `git show <rev>:<path>` run (pipes, `&&`, `;` and redirects to /dev/null are \
fine). Everything else is blocked. Never try to execute or modify anything. There is no network and the project's \
dependencies are not installed, so a package missing here proves nothing. `rg` is not installed; use grep.
- Output is cut off after {max_chars} characters. Keep it small: check size with `wc -l path`, read slices with \
`sed -n '40,90p' path`, `head -n 60 path` or `tail -n 40 path`, and pipe long output through `| head -n 50`. If a \
result says it was truncated, narrow the command.

Budget: {max_steps} steps. One step is one turn in which you call run_bash (several calls in the same turn count as \
one step). You are told your current step; after the last step your tools are removed and you must answer. Stop \
exploring as soon as you have what you need."""



@contextmanager
def open_pr_sandbox(metadata: dict):
    """
    Context manager: yields a PRSandbox for the PR described by `metadata` (as returned by
    get_parsed_pr_diff). Raises if no E2B key is configured or the sandbox can't be created --
    reviews require the sandbox. Always cleans up.
    """
    if not _api_key():
        raise RuntimeError("E2B_API_KEY (or E2B_KEY) is not set; the review agent requires a sandbox.")

    sandbox = PRSandbox(
        owner=metadata["_owner"],
        repo=metadata["_repo"],
        pr_number=metadata["_pr_number"],
        head_sha=metadata["_head_sha"],
        base_sha=metadata["base"]["sha"],
    )
    try:
        yield sandbox
    finally:
        sandbox.close()
