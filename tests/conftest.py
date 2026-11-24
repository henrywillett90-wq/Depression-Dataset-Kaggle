import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd, pytest


@pytest.fixture(scope="session")
def raw():
    r = np.random.RandomState(0); n = 3000
    ch = lambda o: r.choice(o, n)
    d = pd.DataFrame({
        "Name": [f"P{i}" for i in range(n)], "Age": r.randint(18, 80, n),
        "Marital Status": ch(["Single", "Married", "Divorced", "Widowed"]),
        "Education Level": ch(["High School", "Bachelor's Degree", "PhD"]),
        "Number of Children": r.randint(0, 5, n), "Smoking Status": ch(["Non-smoker", "Former", "Current"]),
        "Physical Activity Level": ch(["Sedentary", "Moderate", "Active"]),
        "Employment Status": ch(["Employed", "Unemployed"]), "Income": np.round(r.lognormal(10.3, .9, n), 2),
        "Alcohol Consumption": ch(["Low", "Moderate", "High"]), "Dietary Habits": ch(["Unhealthy", "Moderate", "Healthy"]),
        "Sleep Patterns": ch(["Poor", "Fair", "Good"]), "History of Substance Abuse": ch(["Yes", "No"]),
        "Family History of Depression": ch(["Yes", "No"]), "Chronic Medical Conditions": ch(["Yes", "No"])})
    z = -0.8 + 1.2 * (d["Family History of Depression"] == "Yes")
    d["History of Mental Illness"] = np.where(r.rand(n) < 1 / (1 + np.exp(-z)), "Yes", "No")
    return d
