# AGENTS.md — College workspace rules

## Identity

- Student: Srikrishna O S
- Class/Section: CSE Section A
- Register number: 3122245001312
- College: SSN College of Engineering
- Profile file (lab reports): `C:\Users\srik2\.lab_report_profile.json`
- **Terminology: "submission PDF" = "report". When the user says "submitting pdf" / "submission pdf", they mean the report PDF produced by the `lab-report-pdf` skill.**

## Coding & comment style

- Casual human tone, with small typos allowed. Reference style: `Machine Learning/exp2/learn.py`.
- Never rewrite existing comments into polished AI-sounding language unless explicitly asked.
- Do not invent code, output, or plots in lab reports. Every code block must come from a real file, every output from a real run, every plot from a real PNG.

## Lab report PDFs

- Use the `.opencode/skills/lab-report-pdf/` skill. Generate with `capture_run.py` + `build_pdf.py`.
- **All subjects = polished style**: Poppins text, Consolas code on dark `#1E1E1E` with VS Code-like token colors, page border 0.9pt inset 9mm, running Name/Class/RegNo header, tables with header `#DDE7F5` and alternating rows `#F7F9FC`, plots capped ~85mm and captioned. No subject uses plain mode — every PDF must look polished.
- **spec.json section keys**: only `heading`, `paragraph` (single string), `bullets` (list), `table`, `plots`/`screenshots`, `code_file`, `output_file` are supported. NEVER use `paragraphs` (plural) — `build_pdf.py` silently skips it and the section vanishes from the PDF.
- **JSON validity**: never put Python `None` in a spec; use a string or omit the field. Validate with `python -c "import json;json.load(open(...))"` before building.
- **Always verify after building**: extract text with pymupdf and check the character count is sane, count embedded images, and scan for spans with bbox x1 > ~545pt. Do not tell the user the PDF is done until content is confirmed present.
- **Work log**: after every working session, append a dated entry to `C:\Users\srik2\Desktop\College\WORKLOG.md` (goal, deliverable full path, what was done, lessons learned). Newest entries at the top.
- Nothing in a report may be fabricated.
- Combine one exercise/assignment into one report by default (e.g., Assignment 3 = 3a checksum + 3b CRC in a single PDF) unless the user specifically asks for separate reports.
- **One folder per assignment**: the report PDF(s) live at the folder root (e.g., `exp3/Assignment_3_...pdf`), and per-part code/outputs go in subfolders (e.g., `exp3/3a_checksum/`, `exp3/3b_crc/`, `exp2/2a/`, `exp2/2b/`). Never split one assignment across multiple top-level folders.

## System Design Lab workflow

- Clone the previous experiment folder and make minimal changes to add the next feature.
- Example: `exp3` was cloned from `exp2`, then only `pom.xml`, `application.properties`, and the controller/service changed.
- No unnecessary config classes (e.g., Spring auto-configures `StringRedisTemplate`).
- Keep benchmark scripts (`benchmark.sh`, `benchmark.ps1`) consistent: 100x POST `/shorten` + 100x GET, average ms via `curl -w %{time_total}` or a Stopwatch.

## MCP servers / browser & computer control

Configured globally in `~/.config/opencode/opencode.jsonc`:

- **`browsermcp`** (Browser MCP Edge extension) — **PREFERRED for LMS/Edge**
  - Use when the extension is connected and the active tab is a web page.
  - Best for: navigating to `lms.ssn.edu.in`, clicking course/resource links, downloading files via `forcedownload=1` URLs, reading page snapshots.
  - Works in the user's normal browser session (cookies/login preserved).
- **`claude-in-chrome`** (disabled by default)
  - Use only when the Claude-in-Chrome extension is running in Chrome on `127.0.0.1:8001/sse`.
  - Faster for DOM tasks in Chrome, but does not work in Edge by default.
- **`computer-use`** (`open-computer-use` package)
  - Use for desktop Windows automation when `browsermcp` is not enough: screenshots, app/window UI state, clicks, typing, scrolling in any app.
  - Good fallback for file dialogs, Windows apps, or if `browsermcp` fails.
- **`servo`**
  - Cross-platform mouse/keyboard/app control. Use as a fallback for global hotkeys, screenshots, or launching/focusing apps.

**Important**: never type or read the user's LMS password. The user logs in manually; when they say credentials are entered, only click the Login button. If the button is unclear, click it, wait, and click again.

## LMS downloads

- Base URL: `https://lms.ssn.edu.in`
- Single file resources: navigate to `mod/resource/view.php?id=<id>`, then run JavaScript `location.href = location.href + '&forcedownload=1'` to download.
- Folder resources: directly use `pluginfile.php/.../filename?forcedownload=1` URLs.
- Downloads land in `~/Downloads`. The `sync_manifest.py` workflow copies them into `LMS/<course-folder>/` and updates `LMS/manifest.json`.
- Always use the logged-in browser; do not touch credentials.

## Organization conventions

- `_archive/` folders hold backup zips and are gitignored (kept on disk, not in git).
- `LMS/` holds course material, one folder per course, tracked by `manifest.json`.
- `SD_Lab-main/` is the shared team repo layout (`as1`..`as4`); keep it synced with `https://github.com/srikrishnadeveloper/System-Desgin-Labortary.git`.

## Git / subagents

- Use subagents in parallel when a task clearly splits into independent research/exploration pieces.
- Do not commit unless explicitly asked.

## PDF layout — avoid orphaned code headings

- Always reference PDFs by their **full absolute path** (e.g. `C:\Users\srik2\Desktop\College\Networks Laboratory\exp6\Assignment_6_Rate_Limiter_Srikrishna_O_S.pdf`). Never use relative paths.
- `build_pdf.py` keeps each full `CODE`/`OUTPUT` block as one `Flowable` and lets `reportlab` call `CodeBlock.split()`/`OutputBlock.split()` to fill the available page space automatically. Do not pre-chunk code into fixed-size pieces.
- Do not insert `{"page_break": True}` between consecutive `CODE` sections; let the split logic fill the leftover space with the next heading + the first lines of the next code block. Only use a page break when a section deliberately needs a new page.
- For `screenshots`/`plots` sections, `build_pdf.py` wraps the heading and the first image in a `KeepTogether` so the heading is not left alone at the bottom of a page. Slightly widen images by capping at `CONTENT_W - 4` instead of `CONTENT_W - 10`.
- Long files with huge comment blocks should also be stripped: create a `_nocomments.java` (or similar) file for the PDF, keep the original file untouched, and point the spec's `code_file` to the stripped version.
- `OutputBlock` shrink-to-fit has a font floor (~5.5pt); lines still wider than the box (e.g. long `Categorical columns: [...]` prints, wide DataFrame misclassification rows) are hard-wrapped into chunks at the box width by `OutputBlock._hard_wrap()`. Never let output text draw past the right margin — after every rebuild, scan spans with pymupdf for any `bbox x1 > ~545pt` on A4.
- Bullet-heavy end sections (OBSERVATIONS / REFERENCES / LEARNING OUTCOMES) easily spill one item onto an orphan page. Keep bullets short enough to be single-line and re-verify page count/`bottom_gap` after edits — a page whose content ends above ~70% of the frame usually needs a bullet trimmed or shortened.

## Avoiding stuck / canceled commands

- If a command is taking too long, hanging, or gets canceled, do not keep retrying the same thing. Pause and use an alternative that does not depend on the failing tool.
- **PowerShell has no heredocs.** Never use `python << 'EOF'` or `python -c` with multi-line/multi-quote code — it fails with parser errors. Instead: Write tool a `.py` file into `C:\Users\srik2\AppData\Local\Temp\opencode\`, then run `python <that file>`.
- Example: converting PDF to DOCX with `win32com` Word COM can hang. Prefer building the DOCX directly from the `spec.json` with `python-docx` instead of trying to convert the PDF.
