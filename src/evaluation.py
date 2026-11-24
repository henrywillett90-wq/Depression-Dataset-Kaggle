"""Metrics, threshold selection and bootstrap helpers."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)


def evaluate(y_true, p, threshold=0.5):
    y_true = np.asarray(y_true)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return dict(roc_auc=roc_auc_score(y_true, p), pr_auc=average_precision_score(y_true, p),
                brier=brier_score_loss(y_true, p), log_loss=log_loss(y_true, np.clip(p, 1e-6, 1 - 1e-6)),
                accuracy=accuracy_score(y_true, pred), balanced_acc=balanced_accuracy_score(y_true, pred),
                precision=precision_score(y_true, pred, zero_division=0),
                recall=recall_score(y_true, pred, zero_division=0),
                specificity=tn / (tn + fp) if (tn + fp) else np.nan,
                f1=f1_score(y_true, pred, zero_division=0), mcc=matthews_corrcoef(y_true, pred),
                tn=tn, fp=fp, fn=fn, tp=tp)


def ece(y_true, p, bins=10):
    """Expected calibration error using equal-count bins."""
    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(p)})
    d["b"] = pd.qcut(d["p"].rank(method="first"), bins, labels=False)
    g = d.groupby("b").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((g.n / g.n.sum() * (g.y - g.p).abs()).sum())


def best_f1_threshold(y_true, p):
    grid = np.linspace(0.02, 0.9, 89)
    f1s = [f1_score(y_true, (p >= t).astype(int), zero_division=0) for t in grid]
    return grid[int(np.argmax(f1s))]


def choose_threshold(y_true, p, max_flag_rate=0.60):
    """F1-optimal threshold, falling back to a prevalence-matched top-share rule if it would flag too many."""
    t = best_f1_threshold(y_true, p)
    if (p >= t).mean() > max_flag_rate:
        prevalence = float(np.mean(y_true))
        return float(np.quantile(p, 1 - prevalence)), f"top-{prevalence:.0%} rule"
    return float(t), "F1-opt"


def boot_auc_diff(y_true, p_a, p_b, B=200, seed=42):
    """Bootstrap mean and 95% interval of AUC(a) - AUC(b)."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    d = []
    for _ in range(B):
        i = rng.randint(0, len(y_true), len(y_true))
        if y_true[i].min() == y_true[i].max():
            continue
        d.append(roc_auc_score(y_true[i], p_a[i]) - roc_auc_score(y_true[i], p_b[i]))
    d = np.array(d)
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)
