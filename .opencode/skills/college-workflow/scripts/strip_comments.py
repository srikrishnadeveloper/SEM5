#!/usr/bin/env python3
"""Strip comments from Java or Python source files for PDF code blocks.

Java: removes // line comments and /* ... */ block comments (including /** ... */).
Python: removes whole-line # comments while preserving # inside strings.
Package and import statements are always preserved.
"""

import argparse
import os
import sys


_JAVEX = ".java"
_PYEX = ".py"


def _strip_java(text: str):
    """Return (cleaned_text, chars_removed, line_comments, block_comments)."""
    out = []
    state = "normal"
    i = 0
    n = len(text)
    removed = 0
    line_comments = 0
    block_comments = 0

    while i < n:
        c = text[i]
        if state == "normal":
            if c == "/" and i + 1 < n and text[i + 1] == "/":
                state = "line"
                i += 2
                removed += 2
                line_comments += 1
                continue
            if c == "/" and i + 1 < n and text[i + 1] == "*":
                state = "block"
                i += 2
                removed += 2
                block_comments += 1
                continue
            if c == '"':
                state = "dstr"
            elif c == "'":
                state = "sstr"
            out.append(c)
        elif state == "line":
            if c == "\n":
                out.append("\n")
                state = "normal"
            else:
                removed += 1
        elif state == "block":
            if c == "\n":
                out.append("\n")
            elif c == "*" and i + 1 < n and text[i + 1] == "/":
                state = "normal"
                i += 2
                removed += 2
                continue
            else:
                removed += 1
        elif state == "dstr":
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 1
            elif c == '"':
                state = "normal"
        elif state == "sstr":
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 1
            elif c == "'":
                state = "normal"
        i += 1

    return "".join(out), removed, line_comments, block_comments


def _strip_python(text: str):
    """Return (cleaned_text, chars_removed, comments_removed)."""
    out = []
    state = "normal"
    i = 0
    n = len(text)
    removed = 0
    comments = 0
    line_started = False
    pending_ws = []

    while i < n:
        c = text[i]
        if state == "normal":
            if c == "\n":
                if pending_ws:
                    out.extend(pending_ws)
                    pending_ws = []
                out.append(c)
                line_started = False
            elif not line_started and c in " \t\r":
                pending_ws.append(c)
            elif not line_started and c == "#":
                pending_ws = []
                state = "line"
                removed += 1
                comments += 1
            elif not line_started:
                line_started = True
                out.extend(pending_ws)
                pending_ws = []
                out.append(c)
                if c == "'" and text.startswith("'''", i):
                    state = "triple_s"
                    out.append("'")
                    out.append("'")
                    i += 2
                elif c == "'":
                    state = "single"
                elif c == '"' and text.startswith('"""', i):
                    state = "triple_d"
                    out.append('"')
                    out.append('"')
                    i += 2
                elif c == '"':
                    state = "double"
            else:
                out.append(c)
                if c == "'" and text.startswith("'''", i):
                    state = "triple_s"
                    out.append("'")
                    out.append("'")
                    i += 2
                elif c == "'":
                    state = "single"
                elif c == '"' and text.startswith('"""', i):
                    state = "triple_d"
                    out.append('"')
                    out.append('"')
                    i += 2
                elif c == '"':
                    state = "double"
        elif state == "single":
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 1
            elif c == "'":
                state = "normal"
        elif state == "double":
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 1
            elif c == '"':
                state = "normal"
        elif state == "triple_s":
            out.append(c)
            if c == "'" and text.startswith("'''", i):
                out.append("'")
                out.append("'")
                i += 2
                state = "normal"
        elif state == "triple_d":
            out.append(c)
            if c == '"' and text.startswith('"""', i):
                out.append('"')
                out.append('"')
                i += 2
                state = "normal"
        elif state == "line":
            removed += 1
            if c == "\n":
                out.append(c)
                state = "normal"
                line_started = False
        i += 1

    if pending_ws:
        out.extend(pending_ws)

    return "".join(out), removed, comments


def strip(text: str, lang: str):
    if lang == "java":
        return _strip_java(text)
    if lang == "python":
        return _strip_python(text)
    raise ValueError(f"Unsupported language: {lang}")


def _detect_lang(path: str, lang: str | None):
    if lang:
        return lang
    ext = os.path.splitext(path)[1].lower()
    if ext == _JAVEX:
        return "java"
    if ext == _PYEX:
        return "python"
    raise ValueError(
        f"Cannot detect language from extension {ext!r}; use --lang java|python"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Remove comments from Java or Python source files for PDF code blocks."
    )
    parser.add_argument("--input", required=True, help="source file to strip")
    parser.add_argument("--output", required=True, help="destination file")
    parser.add_argument("--lang", choices=["java", "python"], help="language (default: infer from extension)")
    args = parser.parse_args()

    lang = _detect_lang(args.input, args.lang)
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    cleaned, removed, *rest = strip(text, lang)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(cleaned)

    if lang == "java":
        line_comments, block_comments = rest
        print(
            f"Wrote {args.output}: removed {removed} comment characters "
            f"({line_comments} line comments, {block_comments} block comments)."
        )
    else:
        comments = rest[0]
        print(
            f"Wrote {args.output}: removed {removed} whole-line comment characters "
            f"({comments} whole-line comments)."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
