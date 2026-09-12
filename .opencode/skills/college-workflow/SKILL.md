---
name: college-workflow
description: A runbook skill for the College workspace. Use for continuing sessions, building lab report PDFs, stripping code for PDF, System Design experiments, LMS sync, folder cleanup, code fixes, and browser/MCP setup.
---

# College Workflow Runbook

One place to start any College workspace task without re-reading the full `AGENTS.md` and `WORKLOG.md` each time.

## When to use

- continue, contuine, resume, "where did we leave off", "carry on"
- make PDF, build report, lab report, assignment, submission, report
- strip code, _nocomments, no comments, for-pdf
- sync LMS, lms sync, college portal, course materials
- do the todo, todo, next task
- exp X, experiment X, assignment X
- webcrawler, System Design, SDL
- build fix, fix code, normalize comments
- browser, MCP, connect, login, screenshot

## Identity

- Student: Srikrishna O S
- Class/Section: CSE Section A
- Register number: 3122245001312
- College: SSN College of Engineering
- Profile file (lab reports): `C:\Users\srik2\.lab_report_profile.json`

## Core principles

1. Use subagents in parallel whenever a task splits into independent pieces.
2. Never fabricate code, output, plots, or benchmark numbers.
3. Keep original files untouched; create `_nocomments` variants for PDF code blocks.
4. Compile original source before stripping comments; fix missing imports in the original first.
5. All PDFs must be polished (`plain: false`) for every subject, including System Design Laboratory (UCS3513).
6. Use absolute paths for every deliverable and PDF reference.
7. Build a valid `spec.json`; run `python -c "import json; json.load(open(...))"` before `build_pdf.py`.
8. Never use `paragraphs` (plural) in a spec section. Use `paragraph` (single string).
9. Never put Python `None` into a spec JSON; use a string or omit the field.
10. Append a dated entry to `C:\Users\srik2\Desktop\College\WORKLOG.md` after every session (newest at top).
11. Do not type or read the LMS password; the user logs in manually.

## Workflows

### 1. Continue / resume a session

1. Read the top 3-5 entries of `WORKLOG.md`.
2. Read `AGENTS.md` if the task type is unclear.
3. Propose the most likely continuation; only ask if the log is ambiguous.
4. Pick the matching workflow below and start it.

### 2. Build a lab report PDF

1. Identify subject, experiment/assignment number, and title.
2. Fix code only if broken; preserve the student's casual comment style.
3. For long or comment-heavy files, create a `_nocomments` variant with `strip_comments.py`.
4. Run the code for real:
   - Python: `capture_run.py <file> --cwd <dir> --out-dir <dir>`.
   - Non-Python: run directly and capture terminal output/screenshots.
5. Write a valid `spec.json` using only supported keys: `heading`, `paragraph`, `bullets`, `table`, `plots`, `screenshots`, `code_file`, `output_file`.
6. Build with `lab-report-pdf/scripts/build_pdf.py`.
7. Verify with `verify_pdf.py`.
8. Copy the final PDF to the assignment root and append to `WORKLOG.md`.
9. Combine multi-part assignments into one PDF and one folder by default.

### 3. Strip source files for PDF

1. Build or compile the original file first.
2. Run:
   ```
   python .opencode/skills/college-workflow/scripts/strip_comments.py --input <original> --output <original_nocomments.ext> --lang <java|python>
   ```
3. Keep the original untouched; point the spec's `code_file` to the stripped variant.
4. Re-verify the build if the stripped file is compiled separately.

### 4. System Design experiment

1. Clone the previous experiment folder into a new `expN` folder.
2. Remove `target/`, old report PDFs, and stale docs.
3. Make the minimum changes required (pom.xml, application.properties, controller/service).
4. Build: `mvn clean package -DskipTests`.
5. Run the app and hit endpoints with `curl` or a browser.
6. Benchmark: 100x POST + 100x GET, average ms via `curl -w %{time_total}` or a Stopwatch.
7. Generate the report PDF using Workflow 2 with real run/benchmark output.

### 5. LMS sync

1. Use `browsermcp` (preferred for Edge/LMS) or `claude-in-chrome` if running.
2. Navigate to `https://lms.ssn.edu.in/my/courses.php`.
3. If login page appears, tell the user; when they say credentials are entered, click Login immediately (twice if unclear).
4. Enumerate courses from the dashboard.
5. For each course, read the sidebar with `filter:"all"` and collect `mod/resource/view.php?id=<id>` links.
6. Diff against `LMS/manifest.json`.
7. Download new resources in batches with `forcedownload=1`.
8. Run `sync_manifest.py apply scan.json` to move files to `LMS/<course-folder>/`.
9. Report what was new and append to `WORKLOG.md`.

### 6. College folder organization

1. Keep `LMS/<course-folder>/` one folder per course, tracked by `manifest.json`.
2. Put each assignment in its own folder; per-part code goes in subfolders (`exp3/3a_checksum/`, `exp3/3b_crc/`).
3. Place the final report PDF at the folder root.
4. Use `_archive/` for old zips; do not delete working source.
5. Keep `SD_Lab-main/` in sync with the team repo.

### 7. Code fix + comment-style normalization

1. Fix only syntax errors, missing imports, undefined names, deprecated calls, and real bugs.
2. Do not refactor, rename variables, or rewrite comments into polished AI language.
3. Add comments in the student's casual voice when needed; small typos are fine.
4. Keep multi-line Javadoc/docstrings out of the main file; use `_nocomments` for the PDF.
5. Verify the fixed file still runs.

### 8. Browser / MCP / computer-use setup and repair

1. `browsermcp` is first choice for LMS/Edge. The user must manually connect the extension.
2. If `browsermcp` fails, try `claude-in-chrome` on `127.0.0.1:8001/sse` only if it is running.
3. For Windows apps or file dialogs, use `computer-use`.
4. For global hotkeys, screenshots, or app launching, use `servo`.
5. Never type or read the user's password.

## Tool selection guide

| Tool | Use when |
|---|---|
| `browsermcp` | LMS/Edge is open, the extension is connected, and the target is a web page. |
| `claude-in-chrome` | Chrome with the extension is already running on `127.0.0.1:8001/sse`. |
| `computer-use` | Native Windows UI is needed (file dialogs, apps, screenshots). |
| `servo` | Global hotkeys, mouse/keyboard, app launching/focusing. |
| Subagents | Independent pieces: multiple reports, multi-course scan, parallel research. |

## Subagent & model selection

### Parent model (for heavy / high-stakes work)
- `windsurf/swe-1.7` — default strong model.
- `windsurf/swe-1.7-medium` — faster, lighter reasoning.
- `windsurf/swe-1.6` — small tasks, titles, summaries.
- `windsurf/glm-5.2` — high-context work.
- `opencode/*` — quick fallback only.

### Subagent profiles
- `build` — main coding, executing, driving multi-step work.
- `coder` — writing/editing files.
- `tester` — debugging, testing, verification.
- `explorer` — fast read-only exploration and research.
- `general` — general-purpose helper.

Use subagents in parallel when a task splits into independent research/exploration pieces.

## Verification checklist

- [ ] Build check passes: `mvn clean package -DskipTests`, `python -m py_compile`, or equivalent.
- [ ] Original source compiles/runs before comments are stripped.
- [ ] `spec.json` is valid JSON with no `None` and no `paragraphs` plural key.
- [ ] `build_pdf.py` exits cleanly.
- [ ] `verify_pdf.py --pdf <path>` reports no overflow spans (`x1 > 545 pt` on A4) and a sane image count.
- [ ] `pymupdf` text extraction shows all expected content.
- [ ] `WORKLOG.md` has a new top entry with the absolute deliverable path.

## Common pitfalls

- **PowerShell has no heredocs.** Never use `python << 'EOF'`. Write a `.py` to `C:\Users\srik2\AppData\Local\Temp\opencode\` and run it.
- **`<` redirect** in PowerShell is not the same as bash; pass file paths as arguments.
- **Browser MCP manual connect** cannot be automated; the user must click the extension icon and Connect.
- **`paragraphs` plural** in `build_pdf.py` silently skips the whole section. Use `paragraph`.
- **`None` in JSON** makes `build_pdf.py` crash. Validate the spec first.
- **Missing imports before stripping** propagate into `_nocomments` files. Fix the original first.
- **`plain` mode** is no longer used. `lab-report-pdf/SKILL.md` and `AGENTS.md` now require polished style for all subjects. Never set `plain: true`.
- **Orphan code headings** — do not insert `{"page_break": true}` between consecutive `CODE` sections unless a new page is deliberate.
- **Output overflow** — long unwrapped lines can exceed the right margin; `verify_pdf.py` catches `bbox.x1 > 545 pt`.
- **LMS login friction** — click Login right after the user says credentials are entered; click again if the page does not change.

## WORKLOG convention

Append a new top entry after every working session. Newest entries at the top.

```markdown
---

## YYYY-MM-DD — <one-line goal>

- **Goal:** <what the session was trying to achieve>
- **Deliverable:** `C:\...\full\path\to\file.ext`
- **What was done:**
  - <step 1>
  - <step 2>
- **Lessons learned:** <one or two lines>
```
