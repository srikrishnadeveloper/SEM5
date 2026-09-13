# Cloud Backup — College (SEM5) local history

**Created:** 2026-09-13
**Branch:** `backup/college-pre-merge-2026-09-13`
**Snapshot commit:** `fdf1c11` (fdf1c11c45e4f3d1b88a79b5fa1d28e548278dcc)
**Files in snapshot:** 11714

## Why this backup exists

The local `main` and the GitHub `main` had **diverged into two parallel
histories** — local was 19 commits ahead, remote was 15 commits ahead, with
6,779 files differing between them.

The same coursework appears to have been committed on **two machines and pushed
independently**, producing two lines of history with overlapping-but-different
commit SHAs. For example, both sides contain a commit titled
*"organize: archive zips, rename Placement training, update README/AGENTS/.gitignore"*
but with different hashes.

This branch is a **frozen snapshot of the complete local history**, taken
*before* reconciling the two, so that no local work can be lost if the merge
goes wrong.

## What it contains

- The full local `main` history at the commit above
- 11714 files, including all local-only work that was never pushed

## How to restore from this backup

```bash
git fetch origin
git checkout -B main origin/backup/college-pre-merge-2026-09-13
```

## Notes

- This backup deliberately preserves the local-only commits verbatim.
- `.env` is tracked in the *remote* history but is **not** part of this backup.
- Safe to delete this branch once the reconciliation is verified good.
