#!/usr/bin/env python3
"""Enforce PROJECT_RULES.md rule 9 (curated root).

Hard checks (non-zero exit on failure):
  1. Every tracked root-level entry is on the allow-list, or is one of the
     named, tagged PENDING exceptions below (printed report-only, does not
     fail the gate). A brand-new, unlisted root entry fails the gate.
  2. No tracked file (anywhere in the tree) exceeds 5MB, unless it is a
     named PENDING_LARGE_FILES entry. A brand-new oversized file fails the
     gate.

Report-only (never fails the gate):
  3. Each PENDING exception, with its tag and owner, so the backlog stays
     visible without blocking merges.
  4. Other git worktrees and branches already merged into main, for manual
     tidy-up.

Cheap by design: only `git` subprocess calls, no imports beyond stdlib, no
training/rollout. Safe to run on a shared, loaded box.
"""
import subprocess
import sys

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

ROOT_DIRS_ALLOWED = {
    ".circleci",
    ".github",
    "docs",
    "example",
    "gns",
    "meshnet",
    "scripts",
    "slurm_scripts",
    "utils",
}

# Named, tagged root-level exceptions (rule 9). Report-only: printed every
# run, never fails the gate. Removing an entry here without landing its move
# is a rule-9 violation the next run will catch (it falls through to "new,
# unlisted entry" and fails).
PENDING = {
    "AUTHORS.md": ("safe", "move to .github/", None),
    "CODE_OF_CONDUCT.md": ("safe", "move to .github/", None),
    "CONTRIBUTING.md": ("safe", "move to .github/", None),
    "DCO.md": ("safe", "move to .github/", None),
    "train.sh": ("safe", "move to scripts/", None),
    "run.sh": ("safe", "move to scripts/", None),
    "resume.train.sh": ("safe", "move to scripts/", None),
    "asp.rollout.sh": ("safe", "move to scripts/", None),
    "module.sh": ("safe", "move to scripts/", None),
    "test": (
        "ci-coupled",
        "move to tests/ (shadows stdlib `test`); update "
        ".github/workflows/tests.yml and .circleci/config.yml in the same "
        "commit",
        "iris-vermeulen",
    ),
    "render.sh": (
        "doc-coupled",
        "move to scripts/; update docs/rollout_and_analysis.md in the same "
        "commit",
        None,
    ),
    "render.cpu.sh": (
        "doc-coupled",
        "move to scripts/; update docs/rollout_and_analysis.md in the same "
        "commit",
        None,
    ),
    "build_venv.sh": (
        "breaks outside paths",
        "external job scripts invoke this by root path; waits for the "
        "owner",
        None,
    ),
    "build_venv_frontera.sh": (
        "breaks outside paths",
        "external job scripts invoke this by root path; waits for the "
        "owner",
        None,
    ),
    "start_venv.sh": (
        "breaks outside paths",
        "external job scripts invoke this by root path; waits for the "
        "owner",
        None,
    ),
    "train_cli.py": (
        "breaks outside paths",
        "external job scripts invoke this by root path; waits for the "
        "owner",
        None,
    ),
    "run.process.gns.py": (
        "breaks outside paths",
        "external job scripts invoke this by root path; waits for the "
        "owner",
        None,
    ),
    "scenario.rollout.py": (
        "breaks outside paths",
        "external job scripts invoke this by root path; waits for the "
        "owner",
        None,
    ),
}

# Tracked files already over the cap, with a named remediation pending
# owner sign-off (rewrites a tracked asset's history). Not a path move.
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
    pending_hits = []
    for entry in entries:
        ls = subprocess.run(
            ["git", "ls-tree", "HEAD", entry], capture_output=True, text=True
        ).stdout
        is_dir = ls.startswith("040000")
        if is_dir:
            if entry in ROOT_DIRS_ALLOWED:
                continue
            elif entry in PENDING:
                pending_hits.append(entry)
            else:
                failures.append(f"root dir not on allow-list (new): {entry}/")
        else:
            if entry in ROOT_FILES_ALLOWED:
                continue
            elif entry in PENDING:
                pending_hits.append(entry)
            else:
                failures.append(f"root file not on allow-list (new): {entry}")
    return failures, pending_hits


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


def report_pending(pending_entries, pending_large):
    print("--- pending exceptions (report-only, does not fail the gate) ---")
    for entry in sorted(pending_entries):
        tag, note, owner = PENDING[entry]
        owner_s = f", owner {owner}" if owner else ""
        print(f"  [{tag}] {entry} -- {note}{owner_s}")
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
    root_failures, root_pending = check_root_listing()
    size_failures, size_pending = check_file_sizes()
    failures = root_failures + size_failures

    if failures:
        print("FAIL: rule 9 (curated root) violations:")
        for f in failures:
            print(f"  {f}")
    else:
        print("PASS: root matches rule 9's allow-list + tagged exceptions; "
              "no new file over cap.")

    report_pending(root_pending, size_pending)
    report_tidy()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
