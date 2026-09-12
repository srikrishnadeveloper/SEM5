# College Workflow Cheatsheet

Quick lookup for the `college-workflow` skill. When in doubt, check `SKILL.md` and the latest `AGENTS.md`.

| # | Workflow | Trigger | One-liner goal | Key files | Tools | Subagents? | Common pitfall |
|---|---|---|---|---|---|---|---|
| 1 | Continue / resume | "continue", "contuine", "resume" | Restore context and pick the next task | `WORKLOG.md`, `AGENTS.md` | read | no | Reading too far back in the log instead of the top 3-5 entries |
| 2 | Build lab report PDF | "make PDF", "lab report", "assignment" | Turn real code + output into a submission PDF | `spec.json`, `build_pdf.py`, `verify_pdf.py` | `lab-report-pdf` skill | yes (multiple reports) | Using `paragraphs` plural or `None` in spec; fabricating output |
| 3 | Strip source for PDF | "strip code", "_nocomments" | Remove comments while preserving package/import/code | `strip_comments.py` | python | no | Stripping before the original compiles; `://` in regex strings |
| 4 | System Design experiment | "exp X", "SDL", "webcrawler" | Clone prior exp, build, benchmark, report | `pom.xml`, `application.properties`, controller/service | mvn, curl | yes (build + benchmark + report) | Rewriting from scratch instead of cloning; missing benchmark |
| 5 | LMS sync | "sync LMS", "college portal" | Download new course files and file them locally | `LMS/manifest.json`, `sync_manifest.py` | `browsermcp` | yes (per course folder) | Typing credentials; not using `forcedownload=1` |
| 6 | Folder organization | "cleanup", "organize" | Keep one folder per assignment and course | `LMS/`, `_archive/`, `SD_Lab-main/` | file tools | no | Splitting one assignment across multiple top-level folders |
| 7 | Code fix + comments | "fix code", "normalize comments" | Fix real bugs and keep student-style comments | original source, `_nocomments` variant | python/mvn | no | Over-polishing comments into AI-sounding docstrings |
| 8 | Browser / MCP setup | "browser", "MCP", "connect" | Get the right MCP tool working | `~/.config/opencode/opencode.jsonc` | `browsermcp`, `computer-use`, `servo` | no | Forgetting the manual Browser MCP connect step |

## One-line command reminders

```bash
# Strip Java source for PDF
python .opencode/skills/college-workflow/scripts/strip_comments.py --input C:\...\Foo.java --output C:\...\Foo_nocomments.java --lang java

# Strip Python source for PDF
python .opencode/skills/college-workflow/scripts/strip_comments.py --input C:\...\learn.py --output C:\...\learn_nocomments.py --lang python

# Verify a PDF
python .opencode/skills/college-workflow/scripts/verify_pdf.py --pdf C:\...\Assignment_*.pdf
```
