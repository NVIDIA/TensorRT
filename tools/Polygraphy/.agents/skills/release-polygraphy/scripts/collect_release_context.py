#!/usr/bin/env python3
"""Collect the repository evidence needed to prepare a Polygraphy release."""

import argparse
import re
import subprocess
import sys
from pathlib import Path


VERSION_RE = re.compile(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", re.MULTILINE)


def run_git(root, *args, check=True):
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return result


def ref_exists(root, ref):
    result = run_git(
        root, "rev-parse", "--verify", "--quiet", ref, check=False
    )
    return result.returncode == 0


def show_file(root, ref, path):
    return run_git(root, "show", f"{ref}:{path}").stdout


def parse_version(init_text):
    match = VERSION_RE.search(init_text)
    if not match:
        raise RuntimeError("Could not find __version__ in polygraphy/__init__.py")
    return match.group(1)


def increment(version, part):
    pieces = version.split(".")
    if len(pieces) != 3 or not all(piece.isdigit() for piece in pieces):
        return "unavailable for non-numeric version"
    major, minor, patch = map(int, pieces)
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    return f"{major}.{minor + 1}.0"


def resolve_head(root, requested):
    if requested:
        if not ref_exists(root, requested):
            raise RuntimeError(f"Head ref does not exist: {requested}")
        return requested
    for candidate in ("origin/develop", "develop", "HEAD"):
        if ref_exists(root, candidate):
            return candidate
    raise RuntimeError("Could not resolve a release head ref")


def resolve_base(root, version, requested):
    if requested:
        if not ref_exists(root, requested):
            raise RuntimeError(f"Base ref does not exist: {requested}")
        return requested

    version_tag = f"v{version}"
    if ref_exists(root, version_tag):
        return version_tag

    result = run_git(root, "tag", "--list", "v[0-9]*", "--sort=-version:refname")
    tags = [tag for tag in result.stdout.splitlines() if tag]
    if not tags:
        raise RuntimeError("Could not find a previous version tag")
    return tags[0]


def section(title, content):
    print(f"\n== {title} ==")
    print(content.rstrip() or "(none)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", help="Previous release tag or commit (auto-detected by default)"
    )
    parser.add_argument(
        "--head", help="Release source ref (defaults to origin/develop)"
    )
    parser.add_argument(
        "--changelog-lines",
        type=int,
        default=80,
        help="Number of leading CHANGELOG lines to display (default: 80)",
    )
    args = parser.parse_args()

    root_result = run_git(Path.cwd(), "rev-parse", "--show-toplevel")
    root = Path(root_result.stdout.strip())
    head = resolve_head(root, args.head)
    init_text = show_file(root, head, "polygraphy/__init__.py")
    version = parse_version(init_text)
    base = resolve_base(root, version, args.base)

    changelog = show_file(root, head, "CHANGELOG.md")
    has_vnext = bool(re.search(r"^##\s+vNext(?:\s|$)", changelog, re.MULTILINE))
    status = run_git(root, "status", "--short", "--branch").stdout

    print(f"Repository: {root}")
    print(f"Release source: {head}")
    print(f"Previous release: {base}")
    print(f"Current package version: {version}")
    print(f"Suggested patch version: {increment(version, 'patch')}")
    print(f"Suggested minor version: {increment(version, 'minor')}")
    print(f"vNext section present: {'yes' if has_vnext else 'no'}")
    print(f"Audit range: {base}..{head}")

    section("Working tree", status)
    section(
        "First-parent commits",
        run_git(
            root,
            "log",
            "--reverse",
            "--first-parent",
            "--format=%h %s",
            f"{base}..{head}",
        ).stdout,
    )
    section(
        "Non-merge commits",
        run_git(
            root,
            "log",
            "--reverse",
            "--no-merges",
            "--format=%h %s",
            f"{base}..{head}",
        ).stdout,
    )
    section(
        "Changed files",
        run_git(root, "diff", "--name-status", f"{base}..{head}").stdout,
    )
    section("CHANGELOG head", "\n".join(changelog.splitlines()[: args.changelog_lines]))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as err:
        print(f"error: {err}", file=sys.stderr)
        sys.exit(2)
