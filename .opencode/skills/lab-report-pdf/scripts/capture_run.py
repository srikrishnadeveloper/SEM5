#!/usr/bin/env python3
"""
capture_run.py - run a student's lab script for real and capture its output.

Monkeypatches plt.show() (only if matplotlib is imported) so figures save to
PNG instead of popping up a window, and redirects stdout/stderr to a text
file so the report gets REAL output, never hallucinated numbers.

Usage:
    python capture_run.py <script.py> [--cwd DIR] [--out-dir DIR]

Prints (for the report builder):
    OUTPUT_TXT: <path>
    PLOT: <path>          (one per figure, in plt.show() order)
"""
import argparse
import os
import runpy
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description="Run a lab script and capture its real output and plots.")
    parser.add_argument("script", help="path to the Python script to run")
    parser.add_argument("--cwd", default=None, help="working directory (so relative CSV paths resolve)")
    parser.add_argument("--out-dir", default=None, help="scratch dir for output.txt + PNGs")
    parser.add_argument("--no-plot-capture", action="store_true", help="skip matplotlib monkeypatch")
    args = parser.parse_args()

    script = os.path.abspath(args.script)
    if not os.path.isfile(script):
        print(f"ERROR: script not found: {script}", file=sys.stderr)
        sys.exit(1)

    cwd = os.path.abspath(args.cwd) if args.cwd else os.path.dirname(script)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else tempfile.mkdtemp(prefix="capture_run_")
    os.makedirs(out_dir, exist_ok=True)

    output_txt = os.path.join(out_dir, "output.txt")
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # The plot counter must be visible to the monkeypatched plt.show inside the
    # child process. We pass it through an environment variable.
    env = dict(os.environ)
    env["CAPTURE_PLOT_DIR"] = plot_dir

    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_runner.py"),
           script, output_txt, plot_dir]
    if args.no_plot_capture:
        cmd.append("--no-plot-capture")

    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)

    if proc.returncode != 0:
        print("ERROR: the script failed to run. stderr:", file=sys.stderr)
        print(proc.stderr[-4000:] if proc.stderr else "(no stderr)", file=sys.stderr)

    if os.path.isfile(output_txt) and os.path.getsize(output_txt) > 0:
        print(f"OUTPUT_TXT: {output_txt}")
    else:
        # Script crashed before writing anything useful - still report the error output.
        fallback = os.path.join(out_dir, "output.txt")
        with open(fallback, "w", encoding="utf-8") as f:
            f.write(proc.stdout)
            f.write("\n--- STDERR ---\n")
            f.write(proc.stderr)
        print(f"OUTPUT_TXT: {fallback}")

    plots = sorted(
        os.path.join(plot_dir, p) for p in os.listdir(plot_dir)
        if p.lower().endswith(".png")
    )
    for p in plots:
        print(f"PLOT: {p}")

    if proc.returncode != 0:
        print(f"WARNING: exit code {proc.returncode} - the script had an error.", file=sys.stderr)
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
