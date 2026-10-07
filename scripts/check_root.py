#!/usr/bin/env python3
"""Enforce PROJECT_RULES.md rule 9 (curated root — the template is the
target, not the status quo).

The allow-lists below are the TEMPLATE: where every root-level entry
belongs, not a description of wherever it currently sits. This means the
gate is expected to FAIL on this repo until each specialist's move lands
(see rule 9's move list) -- a red gate here is the check working, not a bug.

Hard checks (non-zero exit on failure):
  1. Every tracked root-level entry is on the template allow-list. No other
     exception exists for root entries: a current violation fails until its
     move lands; there is no "pending" status that keeps it green.
  2. No tracked file (anywhere in the tree) exceeds 5MB, unless it is a
     named PENDING_LARGE_FILES entry (a tracked-asset rewrite, not a path
     move, and it needs the owner's sign-off regardless).

Report-only (never fails the gate):
  3. Each PENDING_LARGE_FILES exception, with its tag and note.
  4. Other git worktrees and branches already merged into main, for manual
     tidy-up.

Cheap by design: only `git` subprocess calls, no imports beyond stdlib, no
training/rollout. Safe to run on a shared, loaded box.
"""
import subprocess
import sys

# --- Template (target layout) -------------------------------------------
# Root-level FILES the template places at root. Nothing else is added here
# without the matching edit to PROJECT_RULES.md rule 9 in the same commit.
ROOT_FILES_ALLOWED = {
    ".gitignore",
    "README.md",
    "CLAUDE.md",
    "PATHWAY_FORWARD.md",
    "PROJECT_RULES.md",
    "license.md",
    "CITATION.cff",  # GitHub's citation widget only reads this at root
    "Dockerfile",
    "requirements.txt",
    "requirements.dl.txt",
    "enviornment.yml",
    "gns_env.yml",
}

# Root-level DIRECTORIES the template places at root. `evals/` and `data/`
# are template slots that fill as earned (rule 1) -- their absence today is
# not a violation; their presence under another name would be. `utils/` and
# `slurm_scripts/` are deliberately NOT here: the template nests them under
# `scripts/`. `test/` is deliberately NOT here: the template renames it
# `tests/`. `example/` is deliberately NOT here: the template moves it to
# `docs/user/examples/`.
ROOT_DIRS_ALLOWED = {
    ".circleci",
    ".github",
    "docs",
    "evals",
    "data",
    "gns",
    "meshnet",
    "scripts",
    "tests",
}

# Tracked files already over the cap, with a named remediation pending
# owner sign-off (rewrites a tracked asset's history). Not a path move, and
# the only exception mechanism this gate carries -- there is no equivalent
# list for root-entry violations: those fail until moved, full stop.
PENDING_LARGE_FILES = {
    "docs/img/meshnet.gif": ("pending owner", "convert to Git LFS or external hosting"),
}

MAX_BYTES = 5 * 1024 * 1024


def run(cmd):
    return subprocess.run(
        cmd, capture_output=True, text=True, check=False
    ).stdout


def check_root_listing():
    out = run(["git", "ls-tree", "--name-only", "-z", "HEAD"])
    entries = [e for e in out.split("\0") if e]
    failures = []
    for entry in entries:
        ls = subprocess.run(
            ["git", "ls-tree", "HEAD", entry], capture_output=True, text=True
        ).stdout
        is_dir = ls.startswith("040000")
        if is_dir:
            if entry not in ROOT_DIRS_ALLOWED:
                failures.append(
                    f"root dir not on template allow-list: {entry}/ "
                    "(see PROJECT_RULES.md rule 9 for its target)"
                )
        else:
            if entry not in ROOT_FILES_ALLOWED:
                failures.append(
                    f"root file not on template allow-list: {entry} "
                    "(see PROJECT_RULES.md rule 9 for its target)"
                )
    return failures


def check_file_sizes():
    out = run(["git", "ls-tree", "-r", "-l", "HEAD"])
    failures = []
    pending_hits = []
    for line in out.splitlines():
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) < 4 or parts[3] == "-":
            continue
        size = int(parts[3])
        if size <= MAX_BYTES:
            continue
        if path in PENDING_LARGE_FILES:
            pending_hits.append((path, size))
        else:
            failures.append(f"{path}: {size} bytes (> {MAX_BYTES} cap, new)")
    return failures, pending_hits


def report_pending(pending_large):
    print("--- pending exceptions (report-only, does not fail the gate) ---")
    if not pending_large:
        print("  (none)")
    for path, size in pending_large:
        tag, note = PENDING_LARGE_FILES[path]
        print(f"  [{tag}] {path} ({size} bytes) -- {note}")


def report_tidy():
    print("--- tidy (report-only, does not affect exit code) ---")
    worktrees = run(["git", "worktree", "list"]).strip().splitlines()
    for w in worktrees[1:]:
        print(f"  other worktree: {w}")
    merged = run(["git", "branch", "--merged", "main"]).strip().splitlines()
    for b in merged:
        name = b.strip("*+ ").strip()
        if name and name != "main":
            print(f"  merged branch (candidate for deletion): {name}")


def main():
    root_failures = check_root_listing()
    size_failures, size_pending = check_file_sizes()
    failures = root_failures + size_failures

    if failures:
        print("FAIL: rule 9 (curated root -- template target) violations:")
        for f in failures:
            print(f"  {f}")
    else:
        print("PASS: root matches rule 9's template; no new file over cap.")

    report_pending(size_pending)
    report_tidy()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
