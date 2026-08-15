#!/usr/bin/env python3
"""Internal runner for capture_run.py - do not call directly.

Runs the student script with stdout redirected to a file and plt.show()
monkeypatched to save figures to a plot dir instead of opening a window.
"""
import os
import sys


def run(script, output_txt, plot_dir, no_plot_capture):
    real_stdout = sys.stdout

    # ---- monkeypatch matplotlib BEFORE the script imports it ----
    if not no_plot_capture:
        _patch_matplotlib(plot_dir, real_stdout)

    sys.stdout = open(output_txt, "w", encoding="utf-8")
    sys.stderr = open(output_txt + ".err", "w", encoding="utf-8")
    sys.argv = [script]

    try:
        # runpy executes the file in a fresh namespace, imports run normally.
        import runpy
        runpy.run_path(script, run_name="__main__")
    finally:
        try:
            sys.stdout.close()
        except Exception:
            pass
        try:
            sys.stderr.close()
        except Exception:
            pass
        sys.stdout = real_stdout


def _patch_matplotlib(plot_dir, log):
    """Monkeypatch pyplot so show() saves the current figure as a PNG."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        counter = {"n": 0}

        def fake_show(*args, **kwargs):
            fig = plt.gcf()
            counter["n"] += 1
            path = os.path.join(plot_dir, f"plot_{counter['n']:02d}.png")
            try:
                fig.savefig(path, dpi=150, bbox_inches="tight")
                print(f"  [captured plot {counter['n']} -> {os.path.basename(path)}]")
            except Exception as e:  # noqa: BLE001
                print(f"  [failed to save plot {counter['n']}: {e}]")
            plt.close(fig)

        # Patch both the module-level show and pyplot's function
        plt.show = fake_show
        matplotlib.pyplot.show = fake_show
        # Some code does `import matplotlib.pyplot as plt` -> same object.
        log.write("  [plot capture active: plt.show() saves PNGs]\n")
    except ImportError:
        log.write("  [matplotlib not used - no plot capture]\n")


if __name__ == "__main__":
    script = sys.argv[1]
    output_txt = sys.argv[2]
    plot_dir = sys.argv[3]
    no_plot = "--no-plot-capture" in sys.argv
    run(script, output_txt, plot_dir, no_plot)
