"""Run the exported notebook script from a terminal, without pop-up windows.

    python run_full_analysis.py [path/to/AXA_Use_Case_2_Mental_Health_Prediction.py]

Put depression_data.csv in the folder you run this from. Figures are saved to figures/, tables to outputs/,
and all printed results appear in the terminal. A full run on the real data takes about 5 minutes.
"""
import sys
import matplotlib
matplotlib.use("Agg")                      # draw to files, never to a window

path = sys.argv[1] if len(sys.argv) > 1 else "AXA_Use_Case_2_Mental_Health_Prediction.py"
src = open(path, encoding="utf8").read().replace(
    "from IPython.display import display",
    "display = lambda x: print(x.data.to_string() if hasattr(x, 'data') else x.to_string())")
exec(compile(src, path, "exec"), {"__name__": "__main__"})
