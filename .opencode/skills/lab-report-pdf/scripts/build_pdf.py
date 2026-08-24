#!/usr/bin/env python3
"""
build_pdf.py - builds a submission-ready lab report PDF from a JSON spec.

This is the renderer behind the `lab-report-pdf` skill. It implements the
"design contract" that makes reports look like the good ML-lab format:

POLISHED MODE (default):
  - Poppins for every text element (bundled in ../assets/fonts)
  - Consolas (Windows) code blocks on VS Code dark #1E1E1E with real
    syntax colors (comments #6A9955, strings #CE9178, keywords #569CD6)
  - Output block on light gray #F4F4F4 in monospace, smaller than code
  - Thin 0.9pt page border inset ~9mm on every page
  - Running header: Name (left) / Class (center) / Reg No (right)
  - Light table theme: header #DDE7F5, alt rows #F7F9FC, thin gray grid
  - Plots centered, capped ~85mm tall, captioned

PLAIN MODE (plain: true) - used ONLY for System Design Laboratory (UCS3513):
  - Times/Helvetica default fonts, no page border, no running header
  - Monospace Courier code on light background, black text
  - Plain black-grid tables, minimal shading

Usage:
    python build_pdf.py spec.json [--out out.pdf]

The spec must NOT invent anything: code/output come from real files captured
by capture_run.py (or the user's actual run), plots are real PNGs.
"""
import argparse
import json
import os
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.normpath(os.path.join(HERE, "..", "assets", "fonts"))
WINDOWS_FONTS = r"C:\Windows\Fonts"

# ---------------------------------------------------------------- palette
BG_CODE = colors.HexColor("#1E1E1E")
FG_CODE = colors.HexColor("#D4D4D4")
C_COMMENT = colors.HexColor("#6A9955")
C_STRING = colors.HexColor("#CE9178")
C_KEYWORD = colors.HexColor("#569CD6")
C_NUMBER = colors.HexColor("#B5CEA8")
BG_OUTPUT = colors.HexColor("#F4F4F4")
HEADER_ROW = colors.HexColor("#DDE7F5")
ALT_ROW = colors.HexColor("#F7F9FC")
GRID = colors.HexColor("#C9CFD8")
HEADING = colors.HexColor("#22314F")
ACCENT = colors.HexColor("#569CD6")
FOOTER_TEXT = colors.HexColor("#5A6472")
BORDER = colors.HexColor("#9AA3B2")

PAGE_W, PAGE_H = A4
MARGIN_L = 20 * mm
MARGIN_R = 20 * mm
MARGIN_T = 24 * mm
MARGIN_B = 18 * mm
BORDER_INSET = 9 * mm
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R
FRAME_H = PAGE_H - MARGIN_T - MARGIN_B
KEEP_GROUP_LIMIT = 180  # pt: keep heading + block together when group is this small

PY_KEYWORDS = {
    "False", "None", "True", "and", "as", "assert", "async", "await", "break",
    "class", "continue", "def", "del", "elif", "else", "except", "finally",
    "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal",
    "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
}
GEN_KEYWORDS = {
    "public", "private", "protected", "class", "interface", "extends",
    "implements", "static", "final", "void", "int", "long", "double",
    "float", "boolean", "char", "String", "new", "return", "if", "else",
    "for", "while", "do", "switch", "case", "break", "continue", "import",
    "package", "try", "catch", "finally", "throw", "throws", "this", "super",
    "null", "true", "false", "var", "const", "let", "function", "export",
    "def", "return", "if", "else", "elif", "for", "while", "import", "from",
}


def _register_fonts():
    fonts = {}
    if os.path.isdir(FONT_DIR):
        for name, fname in [
            ("Poppins", "Poppins-Regular.ttf"),
            ("Poppins-Medium", "Poppins-Medium.ttf"),
            ("Poppins-SemiBold", "Poppins-SemiBold.ttf"),
            ("Poppins-Bold", "Poppins-Bold.ttf"),
            ("Poppins-Italic", "Poppins-Italic.ttf"),
        ]:
            path = os.path.join(FONT_DIR, fname)
            if os.path.isfile(path):
                pdfmetrics.registerFont(TTFont(name, path))
                fonts[name] = path
    if not fonts:
        print("WARNING: Poppins fonts not found in %s - falling back to Helvetica." % FONT_DIR)
    return fonts


FONTS = _register_fonts()


def _has_pf(path):
    return os.path.isfile(path)


def _register_code_fonts():
    reg = bold = None
    for p in (os.path.join(WINDOWS_FONTS, "consola.ttf"),
              os.path.join(WINDOWS_FONTS, "Consola.ttf")):
        if _has_pf(p):
            reg = p
            break
    for p in (os.path.join(WINDOWS_FONTS, "consolab.ttf"),
              os.path.join(WINDOWS_FONTS, "Consolab.ttf")):
        if _has_pf(p):
            bold = p
            break
    if reg:
        pdfmetrics.registerFont(TTFont("Code", reg))
        pdfmetrics.registerFont(TTFont("Code-Bold", bold or reg))
    else:
        print("WARNING: Consolas not found - using Courier.")
        pdfmetrics.registerFont(TTFont("Code", "Courier"))
        pdfmetrics.registerFont(TTFont("Code-Bold", "Courier-Bold"))


_register_code_fonts()


# ------------------------------------------------------------ code tokens
def _tokenize_python(lines):
    """Yield (text, color) per line; handles strings/comments/keywords/numbers."""
    out = []
    in_triple = None  # '""""'' or "''''''"
    for line in lines:
        tokens = []
        i = 0
        n = len(line)
        if in_triple:
            end = line.find(in_triple)
            if end == -1:
                tokens.append((line, C_STRING))
                out.append(tokens)
                continue
            tokens.append((line[: end + 3], C_STRING))
            i = end + 3
            in_triple = None
        while i < n:
            ch = line[i]
            if line.startswith("#", i):
                tokens.append((line[i:], C_COMMENT))
                break
            if line.startswith(('"""', "'''"), i):
                end = line.find(line[i : i + 3], i + 3)
                if end == -1:
                    tokens.append((line[i:], C_STRING))
                    in_triple = line[i : i + 3]
                    break
                tokens.append((line[i : end + 3], C_STRING))
                i = end + 3
                continue
            if ch in ('"', "'"):
                j = i + 1
                while j < n and line[j] != ch:
                    if line[j] == "\\":
                        j += 1
                    j += 1
                tokens.append((line[i : j + 1], C_STRING))
                i = j + 1
                continue
            if ch.isdigit():
                j = i
                while j < n and (line[j].isdigit() or line[j] in ".eE+-"):
                    j += 1
                tokens.append((line[i:j], C_NUMBER))
                i = j
                continue
            if ch.isalpha() or ch == "_":
                j = i
                while j < n and (line[j].isalnum() or line[j] == "_"):
                    j += 1
                word = line[i:j]
                tokens.append((word, C_KEYWORD if word in PY_KEYWORDS else FG_CODE))
                i = j
                continue
            tokens.append((ch, FG_CODE))
            i += 1
        out.append(tokens)
    return out


def _tokenize_generic(lines):
    """Fallback tokenizer for Java/C/sh: line+block comments, strings, keywords."""
    out = []
    in_block = False
    for line in lines:
        tokens = []
        i = 0
        n = len(line)
        while i < n:
            if in_block:
                end = line.find("*/", i)
                if end == -1:
                    tokens.append((line[i:], C_COMMENT))
                    break
                tokens.append((line[i : end + 2], C_COMMENT))
                i = end + 2
                in_block = False
                continue
            if line.startswith("//", i):
                tokens.append((line[i:], C_COMMENT))
                break
            if line.startswith("/*", i):
                end = line.find("*/", i + 2)
                if end == -1:
                    tokens.append((line[i:], C_COMMENT))
                    in_block = True
                    break
                tokens.append((line[i : end + 2], C_COMMENT))
                i = end + 2
                continue
            if line[i] in ('"', "'"):
                j = i + 1
                while j < n and line[j] != line[i]:
                    if line[j] == "\\":
                        j += 1
                    j += 1
                tokens.append((line[i : j + 1], C_STRING))
                i = j + 1
                continue
            if line[i].isalpha() or line[i] == "_":
                j = i
                while j < n and (line[j].isalnum() or line[j] == "_"):
                    j += 1
                word = line[i:j]
                tokens.append((word, C_KEYWORD if word in GEN_KEYWORDS else FG_CODE))
                i = j
                continue
            tokens.append((line[i], FG_CODE))
            i += 1
        out.append(tokens)
    return out


# ---------------------------------------------------------------- styles
def _styles(plain):
    if plain:
        return {
            "title": ParagraphStyle("t", fontName="Times-Bold", fontSize=16, leading=20,
                                    alignment=TA_CENTER, spaceAfter=2),
            "subtitle": ParagraphStyle("s", fontName="Times-Roman", fontSize=12, leading=15,
                                       alignment=TA_CENTER, spaceAfter=10, textColor=colors.black),
            "h2": ParagraphStyle("h2", fontName="Times-Bold", fontSize=13, leading=16,
                                 spaceBefore=12, spaceAfter=5, textColor=colors.black),
            "h3": ParagraphStyle("h3", fontName="Times-Bold", fontSize=12, leading=15,
                                 spaceBefore=10, spaceAfter=4, textColor=colors.black),
            "body": ParagraphStyle("b", fontName="Times-Roman", fontSize=11, leading=15,
                                   alignment=TA_JUSTIFY, spaceAfter=6),
            "caption": ParagraphStyle("c", fontName="Times-Italic", fontSize=9.5, leading=12,
                                      alignment=TA_CENTER, spaceAfter=6, textColor=colors.black),
            "bullet": ParagraphStyle("bl", fontName="Times-Roman", fontSize=11, leading=14,
                                     spaceAfter=2),
            "sec_label": ParagraphStyle("sl", fontName="Times-Bold", fontSize=11, leading=14),
        }
    return {
        "title": ParagraphStyle("t", fontName="Poppins-Bold", fontSize=19, leading=24,
                                alignment=TA_CENTER, spaceAfter=2, textColor=HEADING),
        "subtitle": ParagraphStyle("s", fontName="Poppins-Medium", fontSize=11.5, leading=15,
                                   alignment=TA_CENTER, spaceAfter=12, textColor=ACCENT),
        "h2": ParagraphStyle("h2", fontName="Poppins-SemiBold", fontSize=13, leading=17,
                             spaceBefore=14, spaceAfter=6, textColor=HEADING),
        "h3": ParagraphStyle("h3", fontName="Poppins-SemiBold", fontSize=11.5, leading=15,
                             spaceBefore=12, spaceAfter=5, textColor=HEADING),
        "body": ParagraphStyle("b", fontName="Poppins", fontSize=10.5, leading=15,
                               alignment=TA_JUSTIFY, spaceAfter=6, textColor=colors.HexColor("#1A1A1A")),
        "caption": ParagraphStyle("c", fontName="Poppins-Italic", fontSize=9, leading=12,
                                  alignment=TA_CENTER, spaceAfter=8, textColor=FOOTER_TEXT),
        "bullet": ParagraphStyle("bl", fontName="Poppins", fontSize=10.5, leading=14.5,
                                 spaceAfter=2, textColor=colors.HexColor("#1A1A1A")),
        "sec_label": ParagraphStyle("sl", fontName="Poppins-Medium", fontSize=10.5, leading=13),
    }


# ------------------------------------------------------------ flowables
class CodeBlock(Flowable):
    """Dark (or light in plain mode) monospace block with syntax colors."""

    def __init__(self, tokens_by_line, width, plain=False, font_size=9.0, pad=13):
        super().__init__()
        self.lines = tokens_by_line
        self.width = width
        self.plain = plain
        self.pad = pad
        self.font_size = font_size
        self.leading = font_size * 1.5
        self.line_h = self.leading + 1.5

        # shrink font until the widest line fits
        maxw = max((sum(pdfmetrics.stringWidth(t, "Code", self.font_size) for t, _ in line) for line in self.lines), default=0)
        while maxw > (self.width - 2 * self.pad) and self.font_size > 5.5:
            self.font_size -= 0.25
            self.leading = self.font_size * 1.5
            self.line_h = self.leading + 1.5
            maxw = max((sum(pdfmetrics.stringWidth(t, "Code", self.font_size) for t, _ in line) for line in self.lines), default=0)
        self.bg = BG_CODE if not plain else colors.HexColor("#F2F2F2")
        self.height = len(self.lines) * self.line_h + self.pad * 2 + 2
        if len(self.lines) > 60:
            self.height += 10  # breathing room on long blocks

    def draw(self):
        c = self.canv
        c.saveState()
        c.setFillColor(self.bg)
        c.roundRect(0, 0, self.width, self.height, 4, fill=1, stroke=0)
        x = self.pad
        y = self.height - self.pad - self.font_size
        for line in self.lines:
            for text, color in line:
                c.setFillColor(colors.black if self.plain else color)
                c.setFont("Code", self.font_size)
                c.drawString(x, y, text)
                x += pdfmetrics.stringWidth(text, "Code", self.font_size)
            x = self.pad
            y -= self.line_h
        c.restoreState()

    def split(self, aW, aH):
        if aH >= self.height or len(self.lines) <= 1:
            return []
        pad_eff = max(4.0, min(self.pad, (aH - self.line_h - 2) / 2))
        n = int((aH - 2 * pad_eff - 2) / self.line_h)
        if n < 1:
            return []
        return [
            CodeBlock(self.lines[:n], self.width, self.plain, self.font_size, pad_eff),
            CodeBlock(self.lines[n:], self.width, self.plain, self.font_size, self.pad),
        ]

    def split_fill(self, aH):
        """Split so the top piece fills aH with as many lines as fit, using
        reduced padding when the space is tight. Needs at least 1 line."""
        if aH < self.line_h + 2 * 4 + 2:
            return None
        pad_eff = max(4.0, min(self.pad, (aH - self.line_h - 2) / 2))
        n = int((aH - 2 * pad_eff - 2) / self.line_h)
        if n < 1 or n >= len(self.lines):
            return None
        top = CodeBlock(self.lines[:n], self.width, self.plain, self.font_size, pad_eff)
        bottom = CodeBlock(self.lines[n:], self.width, self.plain, self.font_size, self.pad)
        return (top, bottom)


class OutputBlock(Flowable):
    """Light-gray monospace block for captured real output."""

    def __init__(self, text, width, plain=False, font_size=8.5, pad=10):
        super().__init__()
        self.text = text.rstrip("\n")
        self.width = width
        self.plain = plain
        self.pad = pad
        self.font_size = font_size
        self.leading = font_size * 1.45
        self.line_h = self.leading + 1
        lines = self.text.split("\n")
        maxw = max((pdfmetrics.stringWidth(l, "Code", self.font_size) for l in lines), default=0)
        while maxw > (self.width - 2 * self.pad) and self.font_size > 5.5:
            self.font_size -= 0.25
            self.leading = self.font_size * 1.45
            self.line_h = self.leading + 1
            maxw = max((pdfmetrics.stringWidth(l, "Code", self.font_size) for l in lines), default=0)
        # hard-wrap lines that are still wider than the box (shrink hit its floor)
        max_content_w = self.width - 2 * self.pad
        self.wrapped = []
        for line in lines:
            if pdfmetrics.stringWidth(line, "Code", self.font_size) <= max_content_w:
                self.wrapped.append([line])
            else:
                self.wrapped.append(self._hard_wrap(line, max_content_w, self.font_size))
        n_lines = sum(len(w) for w in self.wrapped)
        self.height = n_lines * self.line_h + self.pad * 2

    @staticmethod
    def _hard_wrap(line, max_w, font_size):
        chunks = []
        start = 0
        n = len(line)
        while start < n:
            end = start + 1
            while end <= n and pdfmetrics.stringWidth(line[start:end], "Code", font_size) <= max_w:
                end += 1
            chunks.append(line[start : end - 1])
            start = end - 1
        return chunks

    def draw(self):
        c = self.canv
        c.saveState()
        c.setFillColor(BG_OUTPUT if not self.plain else colors.HexColor("#F2F2F2"))
        c.rect(0, 0, self.width, self.height, fill=1, stroke=0)
        x = self.pad
        y = self.height - self.pad - self.font_size
        c.setFillColor(colors.black if self.plain else colors.HexColor("#333333"))
        c.setFont("Code", self.font_size)
        for group in self.wrapped:
            for chunk in group:
                c.drawString(x, y, chunk)
                y -= self.line_h
        c.restoreState()

    def split(self, aW, aH):
        if aH >= self.height or len(self.text.split("\n")) <= 1:
            return []
        pad_eff = max(4.0, min(self.pad, (aH - self.line_h - 2) / 2))
        max_wrapped = int((aH - 2 * pad_eff - 2) / self.line_h)
        if max_wrapped < 1:
            return []
        lines = self.text.split("\n")
        n = 0
        used = 0
        while n < len(lines) and used + len(self.wrapped[n]) <= max_wrapped:
            used += len(self.wrapped[n])
            n += 1
        if n == 0 or n >= len(lines):
            return []
        top = "\n".join(lines[:n])
        bottom = "\n".join(lines[n:])
        return [
            OutputBlock(top, self.width, self.plain, self.font_size, pad=pad_eff),
            OutputBlock(bottom, self.width, self.plain, self.font_size),
        ]

    def split_fill(self, aH):
        """Split so the top piece fills aH with as many wrapped lines as fit,
        using reduced padding when the space is tight. Needs at least 1 line."""
        if aH < self.line_h + 2 * 4 + 2:
            return None
        pad_eff = max(4.0, min(self.pad, (aH - self.line_h - 2) / 2))
        max_wrapped = int((aH - 2 * pad_eff - 2) / self.line_h)
        if max_wrapped < 1:
            return None
        lines = self.text.split("\n")
        n = 0
        used = 0
        while n < len(lines) and used + len(self.wrapped[n]) <= max_wrapped:
            used += len(self.wrapped[n])
            n += 1
        if n == 0 or n >= len(lines):
            return None
        top = OutputBlock("\n".join(lines[:n]), self.width, self.plain, self.font_size, pad=pad_eff)
        bottom = OutputBlock("\n".join(lines[n:]), self.width, self.plain, self.font_size)
        return (top, bottom)


def _section_label(text, style, plain):
    if plain:
        return Paragraph(text, style)
    # small pill-ish label with accent underline
    p = Paragraph(text, style)
    return p


class HeadingBlock(Flowable):
    """Heading + block kept together. When the remaining page space can fit the
    heading plus at least two block lines, the block is split (with adaptive
    padding) so the leftover space is filled and the heading is never left
    alone at the bottom of a page. When even two lines do not fit, the whole
    group moves to the next page, leaving only a small tail."""

    def __init__(self, heading_p, block):
        super().__init__()
        self.heading = heading_p
        self.block = block
        sb = getattr(heading_p.style, "spaceBefore", 0)
        sa = getattr(heading_p.style, "spaceAfter", 0)
        _, text_h = heading_p.wrap(block.width, 1e9)
        self.heading_h = text_h + sb + sa
        self.block_h = block.height
        self.total_h = self.heading_h + self.block_h

    def wrap(self, aW, aH):
        self._width = aW
        self._height = self.total_h
        return (aW, self.total_h)

    def draw(self):
        c = self.canv
        sb = getattr(self.heading.style, "spaceBefore", 0)
        text_h = self.heading_h - sb - getattr(self.heading.style, "spaceAfter", 0)
        self.heading.drawOn(c, 0, self.total_h - sb - text_h)
        self.block.drawOn(c, 0, 0)

    def split(self, aW, aH):
        if self.total_h <= aH:
            return [self.heading, self.block]
        space = aH - self.heading_h
        pieces = self.block.split_fill(space)
        if pieces:
            return [self.heading, pieces[0], pieces[1]]
        return []


def _keep_heading_block(heading_p, block, story):
    """Attach a heading to its block when the heading is the last item in the
    story, so the heading can never be left alone at the bottom of a page.
    Uses HeadingBlock (split-aware) for code/output blocks and plain
    KeepTogether for small bullet lists."""
    if heading_p is not None and story and story[-1] is heading_p:
        if isinstance(block, ListFlowable):
            if block.height <= KEEP_GROUP_LIMIT:
                story.pop()
                story.append(KeepTogether([heading_p, block]))
            else:
                story.append(block)
        else:
            story.pop()
            story.append(HeadingBlock(heading_p, block))
    else:
        story.append(block)


# ------------------------------------------------------------ document
def _read(path, desc):
    if not path or not os.path.isfile(path):
        raise SystemExit("ERROR: %s file not found: %s" % (desc, path))
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _normalise_paragraph(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build(spec, out_pdf):
    plain = bool(spec.get("plain", False))
    S = _styles(plain)

    table_header_style = ParagraphStyle(
        "th",
        fontName="Times-Bold" if plain else "Poppins-SemiBold",
        fontSize=10 if plain else 9.5,
        leading=13,
        textColor=colors.black if plain else HEADING,
        alignment=TA_CENTER,
        spaceBefore=0,
        spaceAfter=0,
    )
    table_body_style = ParagraphStyle(
        "tb",
        fontName="Times-Roman" if plain else "Poppins",
        fontSize=10 if plain else 9.5,
        leading=13,
        textColor=colors.black if plain else colors.HexColor("#1A1A1A"),
        spaceBefore=0,
        spaceAfter=0,
    )

    # metadata ---------------------------------------------------------
    name = spec.get("name")
    class_section = spec.get("class_section")
    regno = spec.get("regno")
    if not (name and class_section and regno):
        profile = os.path.join(os.path.expanduser("~"), ".lab_report_profile.json")
        if os.path.isfile(profile):
            with open(profile, encoding="utf-8") as f:
                p = json.load(f)
            name = name or p.get("name")
            class_section = class_section or p.get("class_section")
            regno = regno or p.get("regno")
    institution = spec.get("institution")

    def on_page(canvas, doc):
        canvas.saveState()
        # border --------------------------------------------------------
        if not plain:
            canvas.setStrokeColor(BORDER)
            canvas.setLineWidth(0.9)
            canvas.rect(BORDER_INSET, BORDER_INSET,
                        PAGE_W - 2 * BORDER_INSET, PAGE_H - 2 * BORDER_INSET)
            # running header
            if name:
                canvas.setFillColor(HEADING)
                canvas.setFont("Poppins-Medium", 8.5)
                canvas.drawString(MARGIN_L, PAGE_H - 14 * mm, _normalise_paragraph(name))
            if class_section:
                canvas.drawCentredString(PAGE_W / 2, PAGE_H - 14 * mm, _normalise_paragraph(class_section))
            if regno:
                canvas.drawRightString(PAGE_W - MARGIN_R, PAGE_H - 14 * mm, _normalise_paragraph(regno))
            canvas.setStrokeColor(colors.HexColor("#D8DCE3"))
            canvas.setLineWidth(0.5)
            canvas.line(MARGIN_L, PAGE_H - 15.5 * mm, PAGE_W - MARGIN_R, PAGE_H - 15.5 * mm)
        # footer ---------------------------------------------------------
        if institution:
            canvas.setFillColor(colors.black if plain else FOOTER_TEXT)
            canvas.setFont("Times-Roman" if plain else "Poppins-Medium", 8.5)
            canvas.drawCentredString(PAGE_W / 2, 12 * mm, _normalise_paragraph(institution))
        canvas.restoreState()

    doc = SimpleDocTemplate(
        out_pdf, pagesize=A4,
        leftMargin=MARGIN_L, rightMargin=MARGIN_R,
        topMargin=MARGIN_T, bottomMargin=MARGIN_B,
        title=spec.get("title", "Lab Report"), author=name or "",
    )
    story = []

    # title block --------------------------------------------------------
    story.append(Paragraph(_normalise_paragraph(spec.get("title", "")), S["title"]))
    if spec.get("subtitle"):
        story.append(Paragraph(_normalise_paragraph(spec["subtitle"]), S["subtitle"]))
    story.append(Spacer(1, 4))

    # sections -----------------------------------------------------------
    for sec in spec.get("sections", []):
        if sec.get("page_break"):
            story.append(PageBreak())
            continue
        heading_p = None
        if sec.get("heading"):
            heading_p = Paragraph(_normalise_paragraph(sec["heading"]), S["h2"])
            story.append(heading_p)
        if sec.get("subheading"):
            story.append(Paragraph(_normalise_paragraph(sec["subheading"]), S["h3"]))
        if sec.get("paragraph"):
            for para in (sec["paragraph"] if isinstance(sec["paragraph"], list) else [sec["paragraph"]]):
                story.append(Paragraph(_normalise_paragraph(para), S["body"]))
        if sec.get("code_file"):
            content = _read(sec["code_file"], "code")
            lines = content.rstrip("\n").split("\n")
            tokenized = _tokenize_python(lines) if sec.get("lang", "python") == "python" else _tokenize_generic(lines)
            _keep_heading_block(heading_p, CodeBlock(tokenized, CONTENT_W, plain=plain), story)
            story.append(Spacer(1, 6))
        elif sec.get("code"):
            lines = sec["code"].rstrip("\n").split("\n")
            tokenized = _tokenize_python(lines) if sec.get("lang", "python") == "python" else _tokenize_generic(lines)
            _keep_heading_block(heading_p, CodeBlock(tokenized, CONTENT_W, plain=plain), story)
            story.append(Spacer(1, 6))
        if sec.get("output_file"):
            out_text = _read(sec["output_file"], "output")
            out_lines = out_text.rstrip("\n").split("\n")
            _keep_heading_block(heading_p, OutputBlock("\n".join(out_lines), CONTENT_W, plain=plain), story)
            story.append(Spacer(1, 8))
        elif sec.get("output"):
            out_lines = sec["output"].rstrip("\n").split("\n")
            _keep_heading_block(heading_p, OutputBlock("\n".join(out_lines), CONTENT_W, plain=plain), story)
            story.append(Spacer(1, 8))
        images = sec.get("plots") or sec.get("screenshots")
        if images:
            first_image = True
            for pl in images:
                img_path = pl.get("image", pl if isinstance(pl, str) else "")
                if not os.path.isfile(img_path):
                    print("WARNING: image not found, skipped: %s" % img_path)
                    continue
                img = Image(img_path)
                ratio = img.imageWidth / max(img.imageHeight, 1)
                target_h = 100 * mm if not plain else 85 * mm
                img.drawWidth = target_h * ratio
                img.drawHeight = target_h
                if img.drawWidth > CONTENT_W - 4:
                    img.drawWidth = CONTENT_W - 4
                    img.drawHeight = img.drawWidth / ratio
                cap = Paragraph(_normalise_paragraph(pl["caption"]), S["caption"]) if pl.get("caption") else None
                if first_image and heading_p and story and story[-1] is heading_p:
                    story.pop()
                    group = [heading_p, img]
                    if cap:
                        group.append(cap)
                    story.append(KeepTogether(group))
                    first_image = False
                else:
                    story.append(img)
                    if cap:
                        story.append(cap)
        if sec.get("table"):
            t = sec["table"]
            headers = [Paragraph(_normalise_paragraph(str(h)), table_header_style) for h in t["headers"]]
            rows = [[Paragraph(_normalise_paragraph(str(c)), table_body_style) for c in row] for row in t["rows"]]
            data = [headers] + rows
            widths = t.get("widths")
            table = Table(data, colWidths=widths, repeatRows=1)
            if plain:
                style = [
                    ("GRID", (0, 0), (-1, -1), 0.7, colors.black),
                    ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                    ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAEAEA")),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("LEADING", (0, 0), (-1, -1), 13),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            else:
                style = [
                    ("BACKGROUND", (0, 0), (-1, 0), HEADER_ROW),
                    ("FONTNAME", (0, 0), (-1, 0), "Poppins-SemiBold"),
                    ("TEXTCOLOR", (0, 0), (-1, 0), HEADING),
                    ("FONTNAME", (0, 1), (-1, -1), "Poppins"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                    ("LEADING", (0, 0), (-1, -1), 13),
                    ("GRID", (0, 0), (-1, -1), 0.5, GRID),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ALT_ROW]),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ]
            table.setStyle(TableStyle(style))
            story.append(table)
            story.append(Spacer(1, 8))
        if sec.get("bullets"):
            bullets = [
                ListItem(Paragraph(_normalise_paragraph(b), S["bullet"]), leftIndent=10)
                for b in sec["bullets"]
            ]
            lst = ListFlowable(
                bullets,
                bulletType="bullet",
                leftIndent=16,
                bulletFontName="Poppins-SemiBold" if not plain else "Times-Bold",
            )
            lst.height = len(bullets) * (S["bullet"].leading + S["bullet"].spaceAfter)
            _keep_heading_block(heading_p, lst, story)
            story.append(Spacer(1, 4))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print("PDF written: %s" % out_pdf)


def main():
    ap = argparse.ArgumentParser(description="Build a lab report PDF from a JSON spec.")
    ap.add_argument("spec", help="path to spec.json")
    ap.add_argument("--out", default=None, help="output PDF path (default: next to spec)")
    args = ap.parse_args()

    with open(args.spec, encoding="utf-8") as f:
        spec = json.load(f)

    out_pdf = args.out or os.path.splitext(args.spec)[0] + ".pdf"
    build(spec, out_pdf)


if __name__ == "__main__":
    main()
