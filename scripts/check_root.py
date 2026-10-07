#!/usr/bin/env python3
"""Enforce PROJECT_RULES.md rule 9 (curated root).

Hard checks (non-zero exit on failure):
  1. Every tracked root-level entry is on the allow-list below, or is a
     named, flagged exception pending a proposed move.
  2. No tracked file (anywhere in the tree) exceeds 5 MB.

Report-only (never fails the gate):
  3. Other git worktrees and branches already merged into main, for manual
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
    "test",
    "utils",
}

# Root files that violate rule 9 but whose move needs a synchronized doc
# edit (README.md and/or rule 5) in the same commit -- routed as doc-coupled.
ROOT_FILES_DOC_COUPLED = {
    "build_venv.sh",
    "build_venv_frontera.sh",
    "start_venv.sh",
    "train_cli.py",
    "run.process.gns.py",
    "scenario.rollout.py",
    "render.sh",
    "render.cpu.sh",
}

# Root files that violate rule 9 with no doc/CI reference found -- a clean
# move to scripts/ once proposed.
ROOT_FILES_CLEAN_MOVE = {
    "AUTHORS.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "DCO.md",
    "train.sh",
    "run.sh",
    "resume.train.sh",
    "asp.rollout.sh",
    "module.sh",
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
        is_dir = subprocess.run(
            ["git", "cat-file", "-e", f"HEAD:{entry}"], capture_output=True
        ).returncode == 0 and subprocess.run(
            ["git", "ls-tree", "HEAD", entry], capture_output=True, text=True
        ).stdout.startswith("040000")
        if is_dir:
            if entry not in ROOT_DIRS_ALLOWED:
                failures.append(f"root dir not on allow-list: {entry}/")
        else:
            if entry in ROOT_FILES_ALLOWED:
                continue
            elif entry in ROOT_FILES_DOC_COUPLED:
                failures.append(
                    f"root file violates rule 9 (doc-coupled move, see "
                    f"README.md / rule 5): {entry}"
                )
            elif entry in ROOT_FILES_CLEAN_MOVE:
                failures.append(
                    f"root file violates rule 9 (clean move to scripts/): "
                    f"{entry}"
                )
            else:
                failures.append(f"root file not on allow-list (new): {entry}")
    return failures


def check_file_sizes():
    out = run(["git", "ls-tree", "-r", "-l", "HEAD"])
    failures = []
    for line in out.splitlines():
        # format: <mode> <type> <sha>\t<path>  OR  <mode> <type> <sha> <size>\t<path>
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) < 4 or parts[3] == "-":
            continue
        size = int(parts[3])
        if size > MAX_BYTES:
            failures.append(f"{path}: {size} bytes (> {MAX_BYTES} cap)")
    return failures


def report_tidy():
    print("--- report-only tidy (does not affect exit code) ---")
    worktrees = run(["git", "worktree", "list"]).strip().splitlines()
    for w in worktrees[1:]:
        print(f"  other worktree: {w}")
    merged = run(["git", "branch", "--merged", "main"]).strip().splitlines()
    for b in merged:
        name = b.strip("*+ ").strip()
        if name and name != "main":
            print(f"  merged branch (candidate for deletion): {name}")


def main():
    failures = []
    failures += [("root-listing", f) for f in check_root_listing()]
    failures += [("file-size", f) for f in check_file_sizes()]

    if failures:
        print("FAIL: rule 9 (curated root) violations:")
        for kind, f in failures:
            print(f"  [{kind}] {f}")
    else:
        print("PASS: root matches rule 9's allow-list; no file over cap.")

    report_tidy()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
