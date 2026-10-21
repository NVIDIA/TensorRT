---
name: release-polygraphy
description: Prepare a Polygraphy release by auditing all changes since the previous release, finalizing or adding the versioned CHANGELOG section, updating polygraphy/__init__.py, validating the release-only diff, and opening a GitLab merge request targeting develop. Use when asked to create, prepare, cut, or publish a new Polygraphy version or release MR.
---

# Release Polygraphy

Prepare the version-bump merge request only. Do not merge it, create a tag, or publish packages
unless the user separately requests those actions.

## Collect release context

1. Confirm that the working tree is clean. Preserve unrelated user changes and stop if they overlap
   the release files.
2. Fetch `origin/develop` and version tags so the audit uses current remote state.
3. Run:

   ```bash
   python3 .agents/skills/release-polygraphy/scripts/collect_release_context.py
   ```

4. Inspect every non-merge commit and its diff in the reported tag-to-`origin/develop` range. Use
   first-parent merge commits to understand grouping, but do not rely on commit subjects alone.
5. Confirm that the reported previous tag and `polygraphy.__version__` agree. Resolve any mismatch
   before editing release files.

## Choose the version

- If the top of `CHANGELOG.md` contains a `vNext` section, retain its user-visible content and
  replace `vNext` with the selected version and current date.
- Otherwise, select the next version from the unreleased changes, the user's requested release
  scope, and recent Polygraphy release history. Do not assume a compatible feature requires a minor
  increment: Polygraphy has shipped user-visible additions in patch releases. Ask before changing an
  already-selected release version or making a breaking-version decision.
- Use the local date in `YYYY-MM-DD` format and verify that the version/tag does not already exist.

## Write the release entry

Add the new section at the top of `CHANGELOG.md`, after its introduction. Match the established
format exactly:

```markdown
## vX.Y.Z (YYYY-MM-DD)
### Fixed
- User-visible outcome.
```

- Use only applicable headings from `Added`, `Changed`, `Fixed`, `Deprecated`, and `Removed`, in the
  same style and order as nearby releases.
- Keep two blank lines between version sections and wrap bullets consistently with nearby entries.
- Cover every user-visible change in the audited range, including changes whose original commit did
  not update the changelog.
- Describe behavior, affected commands or public APIs, and the user-visible outcome. Omit internal
  bug IDs, commit/MR references, code structure, algorithms, refactors, tests, and CI details.
- Do not list repository tooling, release mechanics, or this skill as Polygraphy product changes.

Update only `__version__` in `polygraphy/__init__.py` to the same version unless repository evidence
shows another release-version source of truth.

## Validate

Run at least:

```bash
python3 -c "import polygraphy; print(polygraphy.__version__)"
git diff --check
git diff -- CHANGELOG.md polygraphy/__init__.py .agents/skills/release-polygraphy
```

Confirm that:

- the imported version exactly matches the changelog heading;
- the release date and previous tag are correct;
- the changelog contains only user-visible changes;
- the diff contains no unrelated edits.

Release-only metadata edits do not require the full runtime test suite. Run additional focused tests
only if the release branch also changes executable product code.

## Open the merge request

1. Create a branch from the refreshed `origin/develop`, using a descriptive name such as
   `dev-<username>-release-X.Y.Z`.
2. Commit the reviewed release diff with a concise release message.
3. Push the branch and create a GitLab merge request targeting `develop`. If no working GitLab CLI
   is available, use Git push options:

   ```bash
   git push -u origin <branch> \
     -o merge_request.create \
     -o merge_request.target=develop \
     -o 'merge_request.title=Release X.Y.Z' \
     -o 'merge_request.description=<summary and validation>'
   ```

4. Verify the pushed branch, target branch, MR title, and MR URL. Report the version, changelog
   summary, validation performed, commit, and MR link to the user.
