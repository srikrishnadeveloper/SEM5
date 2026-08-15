---
name: lab-report-pdf
description: Generate a college lab/experiment/assignment report PDF with top-level UI - Poppins text, VS Code dark-theme syntax-colored code blocks, real captured output, real plots, page border, running Name/Class/RegNo header, light tables. Use whenever the user asks to create/generate/make a PDF report for a lab experiment, assignment, or practical for ANY subject (Machine Learning, Computer Networks, System Design Lab, DBMS, etc.), or asks to fix code then turn it into a submission PDF, or attaches a friend's/senior's example PDF to match. ONLY the System Design Laboratory (UCS3513) uses the deliberately plain style; all other subjects get the polished look.
---

# Lab Report PDF Generator

Turns a student's lab code into a submission-ready PDF that looks hand-formatted,
not AI-generated: real printed output, real plots, a page border, and a
specific font pairing (Poppins for text, Consolas for code) instead of the
default fonts most PDF tools fall back to.

## Why this exists

The whole point is that nothing in the PDF is invented. The code block is the
student's actual (bug-fixed) file, the OUTPUT section is what that file
actually printed when run, and every image is a real chart it actually
produced. A report built from placeholder text or hallucinated output numbers
is worse than useless here - it's the one thing that would get a student
caught. Treat "run it for real, capture the real output" as non-negotiable.

## Files in this skill

- `scripts/capture_run.py` - runs a Python lab script, saves figures as PNGs
  (monkeypatches `plt.show()`), captures stdout to `output.txt`.
- `scripts/build_pdf.py` - renders the PDF from a JSON spec (the design
  contract below). Run `python build_pdf.py spec.json`.
- `references/spec_schema.md` - full spec format with an example.
- `assets/fonts/Poppins-*.ttf` - bundled fonts (never redownload).

## Workflow

### 1. Get the code into a runnable, submission-ready state

If the user hands you buggy code, fix it - but fix it the way a student would
fix their own mistakes, not the way a senior engineer would refactor it.
Concretely:
- Keep their comments, wording, typos in print strings, and overall structure
  exactly as they are. Don't add docstrings, don't rename variables to be
  "cleaner", don't restructure loops. A report with suspiciously polished
  code next to a first-year student's name is the opposite of what they want.
- Only touch what's actually broken: syntax errors, undefined names, missing
  imports, wrong function calls, deprecated library arguments. If something
  is a real bug fix, make the smallest edit that fixes it.
- If you're not sure whether something is a bug or intentional, ask rather
  than "improving" it.
- If the user asks for MORE comments: write them in their own casual voice
  (lowercase, small typos, "why X" explanations, personal asides) - see
  `Machine Learning/exp2/learn.py` for the reference style. Never write
  polished AI-sounding comments.

### 2. Get student/report metadata

Check `~/.lab_report_profile.json` for a previously saved `name`,
`class_section`, and `regno`. If it exists, use it without re-asking. If it
doesn't exist, ask once, then write it there so future invocations skip this
step entirely.

Separately, per report, work out:
- **Title / experiment number** (e.g. "ASSIGNMENT - 3", "EXPERIMENT NO. 5") -
  ask if genuinely ambiguous, otherwise infer from context (file names,
  course materials, what the user calls it in chat).
- **AIM** - one paragraph stating the objective, tools/libraries used, and
  what the experiment covers. If the user provided a reference PDF (a
  friend's example, a lab manual), mirror its AIM phrasing closely - this is
  boilerplate set by the instructor, not original work, so matching it is
  expected and not a problem.
- **Institution footer** - "SSN COLLEGE OF ENGINEERING" (visible in reference
  material; otherwise omit).
- **plain flag** - true ONLY for System Design Laboratory (UCS3513). Ask if
  unsure about an unusual subject; never generalize the exception yourself.

### 3. Run the code for real and capture real output

```bash
python scripts/capture_run.py "<path to fixed code file>" \
    --cwd "<directory containing the code, so relative CSV/file paths resolve>" \
    --out-dir "<a scratch directory for this run's images + output.txt>"
```

It prints `OUTPUT_TXT: <path>` and one `PLOT: <path>` line per figure, in
the order `plt.show()` was called - use that order for the `plots` list in
the spec later.

If this errors (missing package, deprecated API, etc.), that's real signal -
install the missing package or apply the smallest compatible fix (same
philosophy as step 1) and rerun. Never fabricate the output text or skip a
plot because the run failed - fix the run instead.

For non-Python or non-plotting experiments (a C sockets program for a
Networks lab, a Spring Boot app for System Design Lab, a shell script,
etc.), skip capture_run.py - run the program directly and capture its
terminal output as `output`/`output_file` in the spec, and pass screenshots
(if the user provides them) as `plots`. For Spring Boot: run with `mvn
spring-boot:run`, then `curl` the endpoints (e.g. `curl -i -X POST
localhost:8080/shorten -H "Content-Type: application/json" -d '{"longUrl":
"..."}'`) and capture the real responses.

### 4. Write the "Tasks Performed" table and "Learning Outcomes"

Base every row of the tasks table on something that's actually visible in
the captured output - real row/column counts, real correlation values, real
class-balance percentages, real shapes after a train/test split. This is
what makes the report read as genuinely completed work rather than a
template. Learning Outcomes is a short (5-8 bullet) reflection on what the
experiment covered, in first person ("I was able to...").

### 5. Build the PDF

Decide the style: **System Design Laboratory (UCS3513) -> plain; every
other subject -> polished.** Write a JSON spec (see
`references/spec_schema.md`) and run:

```bash
python scripts/build_pdf.py spec.json --out "Lab_Exercise_N_<Topic>_<Name>.pdf"
```

Name the output file the way the user's other reports are named (e.g.
`Lab_Exercise_2_URL_Shortener_Srikrishna_O_S.pdf`).

### 6. Proofread before delivering

Read 2-3 pages of the actual generated PDF back (the page with the code
block, the page with the first plot, the last page with the table/outcomes).
Check for: text overlapping the box below it, images overflowing a page,
a table row splitting awkwardly. This step caught real spacing bugs during
development - don't skip it just because the script ran without errors.

### 7. Deliver

Send the finished PDF to the user as a file, not just a "done" message. If
they also want to edit it themselves, produce a `.docx` copy alongside.

## Design contract

**Exception first: System Design Laboratory (UCS3513) reports use a
deliberately plain style, not this one.** For that subject specifically, set
`plain: true` - default-looking fonts (Times/Calibri), no page border, no
running header, minimal-to-no table shading - closer to a normal Word
document a student typed themselves. This was an explicit user instruction,
not a guess: overly polished output for that particular course read as
suspicious. Every other subject (ML, Networks Lab, Computer Networks, DBMS,
whatever comes next) gets the full polished treatment below. Don't generalize
the System Design exception to other courses on your own judgment - if it
seems like another course should also be toned down, ask rather than assume.

These choices are deliberate and shouldn't drift between runs - they're what
makes the output consistently recognizable as "the good format" rather than
looking different every time (build_pdf.py implements all of this):

- **Fonts**: Poppins (regular/medium/semibold/bold) for every text element -
  title, headings, body, table, captions. Consolas for code and output
  blocks. Never fall back to Helvetica/Times/Courier for text elements.
- **Code block**: dark background `#1E1E1E`, light gray text `#D4D4D4`,
  comments `#6A9955` (green), strings `#CE9178` (orange), keywords `#569CD6`
  (blue), numbers `#B5CEA8` - VS Code's default dark palette. Generous
  padding (13pt) and line spacing (leading ~1.5x font size).
- **Output block**: light gray `#F4F4F4` background, same mono font, smaller
  than the code block so the two are visually distinct at a glance. Preserve
  the output's original line structure - don't collapse or re-wrap
  DataFrame prints; let long lines shrink-to-fit.
- **Page border**: a thin (~0.9pt) rectangle inset ~9mm from every edge, on
  every page.
- **Running header**: Name (left) / Class (center) / Reg No (right) at the
  top of every page, with a thin divider line beneath it.
- **Table**: light blue header row (`#DDE7F5`), subtle alternating row tint
  (`#F7F9FC`), thin gray gridlines - not the default reportlab black grid.
- **Plots**: centered, capped at ~85mm tall so 2+ fit per page rather than
  one per page with acres of white space, each with a small caption
  underneath.
- **Headings**: Poppins SemiBold dark slate (`#22314F`) with the subtitle in
  accent blue (`#569CD6`).

If a user asks for a visual tweak (different accent color, different
institution footer, tighter/looser spacing), that's a legitimate one-off
edit to the spec or a copy of `build_pdf.py` - you don't need to preserve
every value above as sacred, just don't silently drift back to default
fonts/colors without being asked.

## When multiple reports are needed

If the user asks for reports for several subjects/experiments at once, use
subagents in parallel: each subagent handles one full report pipeline
(fix code -> capture -> spec -> build -> proofread) using this skill. The
main agent verifies each PDF before delivery.
