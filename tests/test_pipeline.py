import numpy as np, pytest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from src.data import prepare, FEATURES, NUM, CAT
from src.models import make_linear, make_hgb
from src.evaluation import evaluate, ece, choose_threshold, boot_auc_diff
from src.fairness import gap_summary


def test_prepare_builds_target_and_features(raw):
    df = prepare(raw)
    assert set(df["target"].unique()) <= {0, 1}
    assert "Log Income" in df and (df["Log Income"] >= 0).all()
    assert df[FEATURES].isna().sum().sum() == 0


def test_prepare_rejects_bad_schema(raw):
    with pytest.raises(ValueError):
        prepare(raw.drop(columns=["Income"]))


def test_prepare_rejects_bad_target(raw):
    bad = raw.copy(); bad.loc[0, "History of Mental Illness"] = "Maybe"
    with pytest.raises(ValueError):
        prepare(bad)


def test_models_beat_chance_and_return_probabilities(raw):
    df = prepare(raw)
    Xtr, Xte, ytr, yte = train_test_split(df[FEATURES], df["target"], test_size=0.3, stratify=df["target"], random_state=1)
    for m in (make_linear(NUM, CAT), make_hgb(NUM, CAT, learning_rate=0.1, max_depth=3)):
        p = m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        assert ((p >= 0) & (p <= 1)).all()
        assert roc_auc_score(yte, p) > 0.6          # synthetic signal from family history


def test_unseen_category_does_not_crash(raw):
    df = prepare(raw)
    m = make_linear(NUM, CAT).fit(df[FEATURES], df["target"])
    row = df[FEATURES].head(1).copy(); row["Marital Status"] = "Unknown"
    assert m.predict_proba(row).shape == (1, 2)


def test_evaluate_and_ece_ranges():
    y = np.array([0, 1] * 50); p = np.linspace(0, 1, 100)
    m = evaluate(y, p, 0.5)
    assert 0 <= m["roc_auc"] <= 1 and m["tp"] + m["fn"] == y.sum()
    assert 0 <= ece(y, p) <= 1


def test_choose_threshold_falls_back_when_signal_is_weak():
    r = np.random.RandomState(0); y = (r.rand(2000) < 0.4).astype(int); p = r.rand(2000) * 0.1 + 0.3
    t, label = choose_threshold(y, p)
    assert (p >= t).mean() <= 0.65


def test_bootstrap_identical_models_has_zero_difference():
    r = np.random.RandomState(1); y = (r.rand(1000) < 0.4).astype(int); p = r.rand(1000)
    m, lo, hi = boot_auc_diff(y, p, p, B=20)
    assert m == 0 and lo == 0 and hi == 0


def test_gap_summary_flags_unequal_selection(raw):
    df = prepare(raw); r = np.random.RandomState(0)
    p = np.where(df["Marital Status"] == "Single", 0.9, 0.1) + r.rand(len(df)) * 0.01
    out = gap_summary(df[["Marital Status"]], df["target"], p, 0.5)
    assert "DI<0.8" in out["flags"].iloc[0]
