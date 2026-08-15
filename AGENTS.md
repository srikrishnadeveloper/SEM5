# AGENTS.md — College workspace rules

## Identity

- Student: Srikrishna O S
- Class/Section: CSE Section A
- Register number: 3122245001312
- College: SSN College of Engineering
- Profile file (lab reports): `C:\Users\srik2\.lab_report_profile.json`

## Coding & comment style

- Casual human tone, with small typos allowed. Reference style: `Machine Learning/exp2/learn.py`.
- Never rewrite existing comments into polished AI-sounding language unless explicitly asked.
- Do not invent code, output, or plots in lab reports. Every code block must come from a real file, every output from a real run, every plot from a real PNG.

## Lab report PDFs

- Use the `.opencode/skills/lab-report-pdf/` skill. Generate with `capture_run.py` + `build_pdf.py`.
- **System Design Lab (UCS3513) = deliberately plain**: Times/Calibri, no page border, minimal shading.
- **All other subjects = polished style**: Poppins text, Consolas code on dark `#1E1E1E` with VS Code-like token colors, page border 0.9pt inset 9mm, running Name/Class/RegNo header, tables with header `#DDE7F5` and alternating rows `#F7F9FC`, plots capped ~85mm and captioned.
- Nothing in a report may be fabricated.

## System Design Lab workflow

- Clone the previous experiment folder and make minimal changes to add the next feature.
- Example: `exp3` was cloned from `exp2`, then only `pom.xml`, `application.properties`, and the controller/service changed.
- No unnecessary config classes (e.g., Spring auto-configures `StringRedisTemplate`).
- Keep benchmark scripts (`benchmark.sh`, `benchmark.ps1`) consistent: 100x POST `/shorten` + 100x GET, average ms via `curl -w %{time_total}` or a Stopwatch.

## MCP servers / browser & computer control

Configured globally in `~/.config/opencode/opencode.jsonc`:

- `computer-use` (enabled, `open-computer-use` package)
  - **When to use**: default for controlling Windows apps and the browser (Edge/Chrome). Does not need any browser extension. Can take screenshots, list apps, get UI state, click, type, scroll, drag, set values.
  - **Good for**: LMS downloads, clicking through already-logged-in browser sessions, file dialogs, desktop automation.
- `claude-in-chrome` (disabled by default)
  - **When to use**: only when the user has the Claude-in-Chrome extension running in Chrome and its MCP server is listening on `127.0.0.1:8001/sse`. Faster for DOM-based web tasks.
  - **Edge**: this extension does not expose an MCP server in Edge by default, so prefer `computer-use` for Edge.

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
