"""
scripts/make_figs.py - Convenience alias for scripts/make_paper_figs.py
Generates all paper figures from results/
"""
import os
import runpy

if __name__ == "__main__":
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_paper_figs.py")
    runpy.run_path(target, run_name="__main__")
