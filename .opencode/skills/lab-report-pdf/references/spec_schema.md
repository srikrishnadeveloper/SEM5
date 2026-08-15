# Spec JSON Schema (for build_pdf.py)

The spec is a single JSON file. **Never invent values**: code comes from the
student's actual (bug-fixed) file, output from a real run, plots from real PNGs.

```json
{
  "title": "EXPERIMENT NO. 1 - EDA on Diabetes Dataset",
  "subtitle": "Exploratory Data Analysis and Data Preprocessing",
  "name": "Srikrishna O S",
  "class_section": "CSE Section A",
  "regno": "3122245001312",
  "institution": "SSN COLLEGE OF ENGINEERING",
  "plain": false,

  "sections": [
    { "heading": "AIM",
      "paragraph": "One paragraph: objective, tools/libraries, what the experiment covers." },

    { "heading": "CODE",
      "code_file": "C:/path/to/fixed/learn.py",
      "lang": "python" },

    { "heading": "OUTPUT",
      "output_file": "C:/path/to/capture/output.txt" },

    { "heading": "PLOTS",
      "plots": [
        { "image": "C:/path/to/capture/plots/plot_01.png",
          "caption": "Fig 1 - Correlation matrix heatmap" },
        { "image": "C:/path/to/capture/plots/plot_02.png",
          "caption": "Fig 2 - Feature distributions after scaling" }
      ] },

    { "heading": "TASKS PERFORMED",
      "table": {
        "headers": ["Task", "Details", "Result"],
        "rows": [
          ["Load dataset", "pandas read_csv", "768 rows x 9 columns"],
          ["Handle missing", "fillna with column mean", "0 missing after"]
        ]
      } },

    { "heading": "LEARNING OUTCOMES",
      "bullets": [
        "I was able to load and inspect a real dataset using pandas.",
        "I understood why scaling matters before feeding data to a model."
      ] }
  ]
}
```

## Fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `title` | string | yes | e.g. "ASSIGNMENT - 2", "EXPERIMENT NO. 5" |
| `subtitle` | string | no | small centered line under the title |
| `name`/`class_section`/`regno` | string | no | falls back to `~/.lab_report_profile.json` |
| `institution` | string | no | footer text (e.g. "SSN COLLEGE OF ENGINEERING") |
| `plain` | bool | no | **true ONLY for System Design Laboratory (UCS3513)** |
| `sections` | array | yes | ordered list of content blocks |

## Section block keys

| Key | Value | Notes |
|---|---|---|
| `heading` | string | section title |
| `paragraph` | string \| [string] | body text (AIM, procedure, result analysis) |
| `code_file` | path | the actual code file to embed (syntax-highlighted) |
| `code` | string | inline code instead of a file |
| `lang` | "python" \| anything else | python tokenizer vs generic (Java/C/sh) |
| `output_file` | path | captured real output text |
| `output` | string | inline output |
| `plots` | array | `{"image": path, "caption": string}` |
| `table` | object | `{"headers": [...], "rows": [[...]], "widths": [...]}` |
| `bullets` | array | learning outcomes etc. |
| `page_break` | bool | start a new page |

Rules of thumb:
- One `code_file` per code section; long files are auto-split across pages.
- Real numbers in the tasks table must come from the captured output.
- The AIM mirrors the instructor's boilerplate phrasing when a reference PDF
  was given (that's expected, not plagiarism of the student).
