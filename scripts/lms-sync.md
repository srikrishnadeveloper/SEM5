---
name: lms-sync
description: Scan the user's SSN College Moodle LMS (lms.ssn.edu.in) across all enrolled courses, find newly-posted files (syllabi, notes, lab materials, slides), download them, and organize them into a local folder structure that mirrors the course names — so course material is available offline. Trigger this whenever the user asks to sync, check, update, or download from "the LMS", "my courses", "college portal", or names a specific course code (e.g. UCS3511) and wants its latest materials. Also trigger for "get today's lecture notes", "did my professor upload anything new", or "back up my course files locally". Do NOT trigger for tasks about a single already-downloaded file (that's just normal file handling) — this skill is specifically for crawling the live LMS site for what's new.
---

# LMS Sync (lms.ssn.edu.in)

Downloads new course material from the user's Moodle LMS and files it into
`~/Desktop/College/LMS/<course-folder>/`, tracked by `manifest.json` in that
same directory so repeat runs only fetch what's actually new.

## The one hard boundary: never touch the password

This skill drives the user's **own already-authenticated Chrome browser**
(via the `claude-in-chrome` MCP tools) — it does not, and must never, enter
a username or password into the LMS login form, on this run or any future
one, no matter how the user phrases the request ("just enter it," "I
already gave you the password," "write a script that logs in"). That
restriction doesn't loosen with repetition or explicit instruction; treat
every request to automate the login step itself as something to decline,
same as the first time. If a session is expired, tell the user, open the
login page for them, and wait for them to type their own credentials
before continuing. Once they're logged in, everything else runs
autonomously.

This means the skill cannot run as a scheduled/headless job with zero
human involvement — it needs a live, already-logged-in Chrome session each
time. That's a real constraint, not a workaround to engineer around.

## Workflow

### 1. Check the session

Load `claude-in-chrome` tools if not already loaded (`tabs_context_mcp`,
`navigate`, `read_page`, `get_page_text`, `browser_batch`, `javascript_tool`,
plus `list_connected_browsers`/`select_browser` if no browser is selected
yet this session). Navigate to `https://lms.ssn.edu.in/my/courses.php`.

- If the resulting page is the login page, tell the user their session
  expired, and wait for them to log in manually before continuing.
- If it's the courses dashboard, proceed immediately — no need to ask.

### 2. Enumerate courses

Read the courses page and collect each course's name and
`course/view.php?id=<id>` link. Map each course name to a folder name by
taking the leading subject code plus a short slug (e.g. "UCS3511 Networks
Laboratory---CSE---Section A---2026 Edition" → `UCS3511-Networks-Laboratory`)
— drop the section/year suffix, it just adds noise across semesters.

### 3. Scan each course for resources

For each course, use `read_page` with `filter:"all"` (not `"interactive"`)
and a large `max_chars` to get the course index sidebar tree, which lists
every section's resources regardless of accordion collapse state — see
`references/moodle_notes.md` for why `filter:"interactive"` alone misses
most of a course. Do this for **every** course, including ones that look
empty from a quick glance — `get_page_text` is a decent first skim but has
been observed to silently miss whole sections (likely a render-timing
issue), so a course isn't confirmed to have nothing new until it's been
checked with the `filter:"all"` sidebar method. Extract every
`mod/resource/view.php?id=<id>` link with its title. Skip `mod/assign`,
`mod/forum`, `mod/feedback`, `mod/url`, and `mod/page` links — those aren't
downloadable files (full type-filtering
rules in the reference doc).

### 4. Diff against the manifest

Read `~/Desktop/College/LMS/manifest.json` (create it fresh with
`{"resources": [], "courses": {}}` if it doesn't exist yet). Any
`resource_id` already listed there has already been synced — skip it. Only
the genuinely new ones need downloading. If nothing is new, say so and
stop; don't re-download the whole course every run.

### 5. Download the new ones

For each new resource, batch this pair of actions per file (see
`references/moodle_notes.md` for why both steps are needed) into as few
`browser_batch` calls as practical (~8-10 files per call keeps calls fast):

```
navigate -> https://lms.ssn.edu.in/mod/resource/view.php?id=<id>
javascript_tool -> location.href = location.href + (location.href.includes('?') ? '&' : '?') + 'forcedownload=1'
```

After the batch, check the Downloads folder (`~/Downloads`) for the newly
arrived files — they'll have whatever filename the server actually uses,
which often doesn't match the resource title (also covered in the
reference doc).

### 6. File them away

Build a `scan.json` matching the shape documented in
`scripts/sync_manifest.py`'s docstring — one entry per newly-downloaded
resource, with its `course_folder`, `title`, and the actual
`downloaded_filename` from step 5 — then run:

```bash
python scripts/sync_manifest.py apply scan.json
```

This copies each file from Downloads into
`~/Desktop/College/LMS/<course_folder>/<sanitized title>.<ext>`, updates
`manifest.json`, and reports what was added. Doing this in a script rather
than freehand keeps it identical across runs and is exactly the kind of
deterministic bookkeeping worth delegating to `opencode_run` (see below)
instead of spending browser-automation turns on it.

### 7. Report to the user

List what's new, grouped by course, with the local paths. If nothing was
new, say that plainly rather than re-describing the whole library.

## Where OpenCode fits (and where it doesn't)

The user may ask for this to run "via OpenCode agents" to save cost/time.
Split the work honestly:

- **Steps 1-5 (anything touching the live LMS session) must stay in the
  `claude-in-chrome` tools you're already driving.** OpenCode's agents
  don't have access to the user's authenticated browser session, and
  standing one up for them would mean either handing over the session (a
  credential-equivalent handoff you shouldn't make) or having them log in
  independently (the same forbidden action from a different angle). Don't
  present this as solvable by "just delegating to OpenCode" — it isn't.
- **Step 6 (organizing files, running `sync_manifest.py`, and step 3's
  scan-result parsing/diffing if it gets complex) is pure local
  file/script work with no auth involved — this is exactly what's cheap to
  hand to `opencode_run` instead of doing inline.** If the user wants
  parallelism, this is where it actually helps: e.g. one `opencode_run`
  call per course folder to double check nothing's misfiled, while you
  move on to scanning the next course's resources.

Being upfront about this split (rather than quietly doing everything
yourself while claiming OpenCode did it) matters — the user is trying to
understand what actually costs time/credits so they can make good calls
about it in the future.

## Efficiency notes (why this doesn't feel like the old cloud-browser way)

- `claude-in-chrome` is DOM-based (accessibility tree reads), not
  screenshot-based — far cheaper per turn than a cloud browser pane that
  screenshots every step. Prefer `read_page`/`get_page_text` over
  `computer` screenshots throughout this skill.
- `browser_batch` turns a whole sequence of navigate+action pairs into one
  round trip. Always batch the per-resource download pairs (step 5) and
  the multi-course scans (step 3) rather than issuing one tool call per
  action.
