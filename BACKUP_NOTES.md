# Cloud Backup — College (SEM5) local history

**Created:** 2026-09-13
**Branch:** `backup/college-pre-merge-2026-09-13`
**Snapshot commit:** `cfb8d0f4` (cfb8d0f4383e9b47819a77a7506b5a02df0bd77d)
**Files in snapshot:** 11715

## Why this backup exists

The local `main` and the GitHub `main` had **diverged into two parallel
histories** — local was 19 commits ahead, remote was 15 ahead, with ~6,800 files
differing between them. The same coursework appears to have been committed on
**two machines and pushed independently**, producing two lines of history with
overlapping-but-different commit SHAs (e.g. both sides contain a commit titled
*"organize: archive zips, rename Placement training, update README/AGENTS/.gitignore"*
but with different hashes).

This branch is a **frozen snapshot of the complete local history**, taken before
reconciling the two, so no local work can be lost if the merge goes wrong.

## History rewrite (important)

The original local history could **not be pushed at all**, because it contained:

```
Machine Learning/Kaggle_Competition_Mini_Project/data/filament-segmentation-2026.zip
```

— **671 MB**, far over GitHub's 100 MB per-file limit. GitHub rejected every push
of that history (`GH001: Large files detected`).

That single blob was therefore purged from history with `git filter-repo`, which
rewrote every commit that contained it. **All commit SHAs from that point forward
changed.** The commits in this branch are the *rewritten* ones.

The **pre-rewrite history is preserved locally** in:

```
C:\Users\srik2\Desktop\College_FULL_HISTORY_backup_2026-09-13.bundle   (2.0 GB)
```

Restore it with: `git clone College_FULL_HISTORY_backup_2026-09-13.bundle restored`

## What it contains

- The full (rewritten) local `main` history at the commit above
- 11715 files, including all local-only work that was never pushed

## How to restore from this backup

```bash
git fetch origin
git checkout -B main origin/backup/college-pre-merge-2026-09-13
```

## Notes

- `.env` is tracked in the *remote* history but is **not** part of this backup.
- Safe to delete this branch once the reconciliation is verified good.
