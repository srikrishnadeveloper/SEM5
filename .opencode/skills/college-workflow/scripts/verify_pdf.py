#!/usr/bin/env python3
"""Verify a lab report PDF for overflow, image count, and text presence."""

import argparse
import os
import sys


def verify(pdf_path: str, max_x1: float):
    try:
        import pymupdf as fitz
    except ImportError as exc:
        print(
            "Error: pymupdf is required. Install it with: pip install pymupdf",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    if not os.path.isfile(pdf_path):
        print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    doc = fitz.open(pdf_path)
    page_count = len(doc)
    total_chars = 0
    image_count = 0
    overflow_count = 0

    print(f"File: {pdf_path}")
    print(f"Pages: {page_count}")

    for pno in range(page_count):
        page = doc.load_page(pno)
        text = page.get_text()
        total_chars += len(text)

        # Count image references on the page.
        images = page.get_images(full=True)
        image_count += len(images)

        # Scan every text span for right-margin overflow.
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    x1 = span["bbox"][2]
                    if x1 > max_x1:
                        overflow_count += 1
                        snippet = span["text"].strip().replace("\n", " ")[:60]
                        print(
                            f"  Overflow on page {pno + 1}: "
                            f"bbox.x1={x1:.1f} > {max_x1} -- {snippet!r}"
                        )

    print(f"Total characters: {total_chars}")
    print(f"Image references: {image_count}")
    print(f"Overflow spans: {overflow_count}")

    if overflow_count:
        print("Result: FAIL (overflow detected)", file=sys.stderr)
        return 1
    print("Result: PASS")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Verify a PDF with pymupdf: page count, text, images, and overflow."
    )
    parser.add_argument("--pdf", required=True, help="path to PDF file")
    parser.add_argument(
        "--max-x1",
        type=float,
        default=545.0,
        help="maximum allowed right edge for a text span (default: 545 pt for A4)",
    )
    args = parser.parse_args()

    return verify(args.pdf, args.max_x1)


if __name__ == "__main__":
    sys.exit(main())
