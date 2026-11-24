"""Minimal command-line run: python run_analysis.py path/to/depression_data.csv
For the full narrative analysis (EDA, variable choice, performance, bias, limitations) open notebooks/*.ipynb."""
import sys
from sklearn.model_selection import train_test_split
from src.data import load_csv, prepare, FEATURES, NUM, CAT
from src.models import make_linear
from src.evaluation import evaluate, choose_threshold

df = prepare(load_csv(sys.argv[1]))
Xtr, Xte, ytr, yte = train_test_split(df[FEATURES], df["target"], test_size=0.2, stratify=df["target"], random_state=42)
m = make_linear(NUM, CAT).fit(Xtr, ytr)
thr, label = choose_threshold(ytr, m.predict_proba(Xtr)[:, 1])
print({k: round(float(v), 3) for k, v in evaluate(yte, m.predict_proba(Xte)[:, 1], thr).items()}, label)
