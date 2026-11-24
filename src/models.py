"""Model pipelines: regularised logistic regression and histogram gradient boosting."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

RANDOM_STATE = 42


def make_linear(num, cat, C=1.0):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=C, max_iter=2000))])


def make_hgb(num, cat, **params):
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan))]), cat)])
    mask = [False] * len(num) + [True] * len(cat)
    clf = HistGradientBoostingClassifier(categorical_features=mask, early_stopping=True, validation_fraction=0.1,
                                         n_iter_no_change=20, max_iter=500, random_state=RANDOM_STATE, **params)
    return Pipeline([("pre", pre), ("clf", clf)])
