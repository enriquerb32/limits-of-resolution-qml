"""
make_figs.py - Root entry point to generate paper figures from results/
Calls scripts/make_paper_figs.py
"""
import sys, os

root = os.path.dirname(os.path.abspath(__file__))
script = os.path.join(root, "scripts", "make_paper_figs.py")

if __name__ == "__main__":
    import runpy
    runpy.run_path(script, run_name="__main__")
