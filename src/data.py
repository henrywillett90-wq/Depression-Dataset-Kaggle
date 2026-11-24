"""Loading and preparing the Kaggle Depression Dataset."""
import numpy as np
import pandas as pd

TARGET = "History of Mental Illness"
NUM_RAW = ["Age", "Number of Children", "Income"]
NUM = ["Age", "Number of Children", "Log Income"]
CAT = ["Marital Status", "Education Level", "Smoking Status", "Physical Activity Level",
       "Employment Status", "Alcohol Consumption", "Dietary Habits", "Sleep Patterns",
       "History of Substance Abuse", "Family History of Depression", "Chronic Medical Conditions"]
FEATURES = NUM + CAT


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, build the binary target and derived columns. Returns a new frame."""
    expected = ["Name"] + NUM_RAW + CAT + [TARGET]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.copy()
    for c in ["Name"] + CAT + [TARGET]:
        df[c] = df[c].astype(str).str.strip()
    labels = set(df[TARGET].str.lower().unique())
    if not labels <= {"yes", "no"}:
        raise ValueError(f"Unexpected target values: {labels}")
    df["target"] = (df[TARGET].str.lower() == "yes").astype(int)
    df["Log Income"] = np.log1p(df["Income"].clip(lower=0))
    df["Age Band"] = pd.cut(df["Age"], bins=[0, 24, 34, 44, 54, 64, 200],
                            labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"])
    df["Income Quartile"] = pd.qcut(df["Income"], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"],
                                    duplicates="drop")
    return df
