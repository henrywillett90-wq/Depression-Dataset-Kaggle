import os, sys, glob, warnings, subprocess, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from IPython.display import display

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score, brier_score_loss,
    log_loss, confusion_matrix, roc_curve, precision_recall_curve, matthews_corrcoef)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

RANDOM_STATE = 42
SAMPLE_N = None          # e.g. 100_000 for a faster dry run; None = use all rows
FIG_DIR, OUT_DIR = "figures", "outputs"
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)

def savefig(name):
    plt.savefig(os.path.join(FIG_DIR, name), dpi=150, bbox_inches="tight")

def say(text, width=110):
    print(textwrap.fill(text, width=width))

# ---- Load the data: local file -> kagglehub -> manual upload (Colab) ----
def find_csv():
    patterns = ["depression_data.csv", "**/depression_data.csv", "/content/**/depression*.csv"]
    for p in patterns:
        hits = glob.glob(p, recursive=True)
        if hits:
            return hits[0]
    return None

path = find_csv()
if path is None:
    try:
        try:
            import kagglehub
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"], check=True)
            import kagglehub
        folder = kagglehub.dataset_download("anthonytherrien/depression-dataset")
        path = glob.glob(os.path.join(folder, "*.csv"))[0]
    except Exception as e:
        print("kagglehub download failed:", repr(e))
if path is None:
    try:
        from google.colab import files
        print("Please upload depression_data.csv (download it from the Kaggle page first).")
        uploaded = files.upload()
        path = list(uploaded.keys())[0]
    except ImportError:
        raise FileNotFoundError("Place depression_data.csv next to this notebook and re-run.")

print("Loading:", path)
df = pd.read_csv(path)
df.columns = df.columns.str.strip()
print("Raw shape:", df.shape)

TARGET = "History of Mental Illness"
NUM_RAW = ["Age", "Number of Children", "Income"]
CAT = ["Marital Status", "Education Level", "Smoking Status", "Physical Activity Level",
       "Employment Status", "Alcohol Consumption", "Dietary Habits", "Sleep Patterns",
       "History of Substance Abuse", "Family History of Depression", "Chronic Medical Conditions"]
expected = ["Name"] + NUM_RAW + CAT + [TARGET]
missing_cols = [c for c in expected if c not in df.columns]
assert not missing_cols, f"Missing columns {missing_cols}. Found: {list(df.columns)}"

for c in ["Name"] + CAT + [TARGET]:
    df[c] = df[c].astype(str).str.strip()

if SAMPLE_N:
    df = df.sample(SAMPLE_N, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Using a random sample of {SAMPLE_N:,} rows")

# --- target ---
assert set(df[TARGET].str.lower().unique()) <= {"yes", "no"}, df[TARGET].unique()
df["target"] = (df[TARGET].str.lower() == "yes").astype(int)

# --- data quality profile ---
profile = pd.DataFrame({
    "dtype": df[expected].dtypes.astype(str),
    "missing": df[expected].isna().sum(),
    "n_unique": df[expected].nunique(),
})
display(profile)
print(f"\nRows: {len(df):,} | exact duplicate rows: {df.duplicated().sum():,} | "
      f"duplicate names: {df['Name'].duplicated().sum():,} of {len(df):,}")
print("\nNumeric summary:")
display(df[NUM_RAW].describe().T)
for c in CAT:
    print(f"\n{c}: " + ", ".join(f"{k} ({v:.1%})" for k, v in df[c].value_counts(normalize=True).items()))

prev = df["target"].mean()
print(f"\nTarget prevalence ('{TARGET}' = Yes): {prev:.1%}  ->  imbalance ratio "
      f"{(1-prev)/prev:.2f} : 1 (negative : positive)")

# --- light cleaning / feature construction ---
if (df["Income"] < 0).any():
    print(f"Warning: {(df['Income'] < 0).sum()} negative incomes clipped to 0")
df["Log Income"] = np.log1p(df["Income"].clip(lower=0))          # Income is heavily right-skewed
df["Age Band"] = pd.cut(df["Age"], bins=[0, 24, 34, 44, 54, 64, 200],
                        labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"])
df["Income Quartile"] = pd.qcut(df["Income"], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"], duplicates="drop")

# natural orderings used for plots and for the correlation check
ORDERS = {
    "Education Level": ["High School", "Associate Degree", "Bachelor's Degree", "Master's Degree", "PhD"],
    "Physical Activity Level": ["Sedentary", "Moderate", "Active"],
    "Alcohol Consumption": ["Low", "Moderate", "High"],
    "Dietary Habits": ["Unhealthy", "Moderate", "Healthy"],
    "Sleep Patterns": ["Poor", "Fair", "Good"],
    "Smoking Status": ["Non-smoker", "Former", "Current"],
}
print("Data ready:", df.shape)

# ---- 3.1 Target balance + numeric predictors by target ----
fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
df["target"].map({0: "No", 1: "Yes"}).value_counts().reindex(["No", "Yes"]).plot.bar(ax=axes[0], color=["#5A6B87", "#D9922B"], rot=0)
axes[0].set_title(f"Target: {TARGET}"); axes[0].set_ylabel("Count")
for i, (col, title) in enumerate([("Age", "Age"), ("Log Income", "Log(1+Income)"), ("Number of Children", "Number of children")]):
    sns.boxplot(data=df, x="target", y=col, ax=axes[i+1], palette=["#5A6B87", "#D9922B"], showfliers=False)
    axes[i+1].set_xticklabels(["No", "Yes"]); axes[i+1].set_xlabel(TARGET); axes[i+1].set_title(title)
plt.tight_layout(); savefig("01_target_and_numeric.png"); plt.show()

# ---- 3.2 Effect sizes: numeric ----
def rank_biserial(x1, x0):
    u, p = stats.mannwhitneyu(x1, x0, alternative="two-sided")
    return 2 * u / (len(x1) * len(x0)) - 1, p      # >0: positives tend to have higher values

def cramers_v(table):
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    n = table.values.sum()
    return np.sqrt(chi2 / (n * (min(table.shape) - 1))), p

def effect_label(x):
    x = abs(x)
    return ("negligible" if x < 0.05 else "very weak" if x < 0.10 else "weak" if x < 0.20
            else "moderate" if x < 0.30 else "strong")

rows = []
for col in ["Age", "Number of Children", "Log Income"]:
    x1, x0 = df.loc[df.target == 1, col], df.loc[df.target == 0, col]
    r, p = rank_biserial(x1, x0)
    rows.append(dict(feature=col, type="numeric", effect_size=abs(r), signed=r, p_value=p,
                     direction=f"higher in 'Yes' (r={r:+.3f})" if r > 0 else f"lower in 'Yes' (r={r:+.3f})"))

# ---- 3.3 Effect sizes: categorical ----
for col in CAT:
    tab = pd.crosstab(df[col], df["target"])
    v, p = cramers_v(tab)
    rate = df.groupby(col)["target"].mean()
    rows.append(dict(feature=col, type="categorical", effect_size=v, signed=np.nan, p_value=p,
                     direction=f"highest rate: {rate.idxmax()} ({rate.max():.1%}); lowest: {rate.idxmin()} ({rate.min():.1%})"))

assoc = pd.DataFrame(rows).sort_values("effect_size", ascending=False).reset_index(drop=True)
assoc["strength"] = assoc["effect_size"].map(effect_label)
assoc["p_bonferroni"] = (assoc["p_value"] * len(assoc)).clip(upper=1)
display(assoc[["feature", "type", "effect_size", "strength", "p_bonferroni", "direction"]]
        .style.format({"effect_size": "{:.3f}", "p_bonferroni": "{:.2e}"}))

# ---- 3.4 Target rate by category (with 95% CI) ----
def rate_ci(g):
    p, n = g.mean(), len(g)
    return pd.Series({"rate": p, "ci": 1.96 * np.sqrt(p * (1 - p) / n), "n": n})

fig, axes = plt.subplots(4, 3, figsize=(17, 15))
for ax, col in zip(axes.ravel(), CAT):
    t = df.groupby(col)["target"].apply(rate_ci).unstack()
    order = [c for c in ORDERS.get(col, []) if c in t.index] or list(t.sort_values("rate").index)
    t = t.loc[order]
    ax.barh(t.index, t["rate"], xerr=t["ci"], color="#2E8B8B", alpha=0.85)
    ax.axvline(df["target"].mean(), color="#B0473F", ls="--", lw=1)
    ax.set_title(col); ax.set_xlabel("P(history of mental illness)")
    lo, hi = (t["rate"] - t["ci"]).min(), (t["rate"] + t["ci"]).max()
    ax.set_xlim(max(0, lo - 0.05), min(1, hi + 0.05))
for ax in axes.ravel()[len(CAT):]:
    ax.axis("off")
plt.suptitle("Rate of 'History of Mental Illness' by category (dashed line = overall rate)", y=1.0, fontsize=14)
plt.tight_layout(); savefig("02_rate_by_category.png"); plt.show()

# ---- 3.5 Continuous predictors binned ----
fig, axes = plt.subplots(1, 3, figsize=(18, 4.2))
for ax, col, title in [(axes[0], "Age Band", "Age band"), (axes[1], "Income Quartile", "Income quartile")]:
    t = df.groupby(col, observed=True)["target"].apply(rate_ci).unstack()
    ax.errorbar(range(len(t)), t["rate"], yerr=t["ci"], marker="o", color="#1B2A4A", capsize=4)
    ax.set_xticks(range(len(t))); ax.set_xticklabels(t.index, rotation=20)
    ax.axhline(df["target"].mean(), color="#B0473F", ls="--", lw=1)
    ax.set_title(f"Rate by {title}"); ax.set_ylabel("P(history of mental illness)")
t = df.groupby("Number of Children")["target"].apply(rate_ci).unstack()
axes[2].errorbar(t.index, t["rate"], yerr=t["ci"], marker="o", color="#1B2A4A", capsize=4)
axes[2].axhline(df["target"].mean(), color="#B0473F", ls="--", lw=1)
axes[2].set_title("Rate by number of children"); axes[2].set_xlabel("Number of children")
plt.tight_layout(); savefig("03_rate_by_continuous.png"); plt.show()

# ---- 3.6 Are predictors redundant with each other? (Spearman on ordinal encodings) ----
enc = pd.DataFrame(index=df.index)
for c in ["Age", "Number of Children", "Log Income"]:
    enc[c] = df[c]
for c in CAT:
    if c in ORDERS:
        enc[c] = df[c].map({v: i for i, v in enumerate(ORDERS[c])})
    else:
        enc[c] = df[c].astype("category").cat.codes
corr = enc.corr(method="spearman")
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="vlag", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, cbar_kws={"shrink": .7})
plt.title("Spearman correlation between predictors (nominal variables coded arbitrarily)")
savefig("04_predictor_correlation.png"); plt.show()
off = corr.where(~np.eye(len(corr), dtype=bool)).abs().stack().sort_values(ascending=False)
print("Largest absolute predictor-predictor correlation:", off.index[0], f"= {off.iloc[0]:.2f}")

# ---- 3.7 Mutual information (captures non-linear / non-monotonic dependence) ----
mi_df = enc.copy()
samp = mi_df.sample(min(50_000, len(mi_df)), random_state=RANDOM_STATE)
disc = [c not in ["Age", "Log Income"] for c in samp.columns]
mi = mutual_info_classif(samp, df.loc[samp.index, "target"], discrete_features=disc, random_state=RANDOM_STATE)
mi_s = pd.Series(mi, index=samp.columns).sort_values(ascending=False)
assoc = assoc.merge(mi_s.rename("mutual_info").reset_index().rename(columns={"index": "feature"}), on="feature", how="left")
plt.figure(figsize=(7, 5)); mi_s.sort_values().plot.barh(color="#D9922B")
plt.title("Mutual information with the target (nats)"); plt.xlabel("MI"); plt.tight_layout()
savefig("05_mutual_information.png"); plt.show()

# ---- 3.8 Two-way view of the two strongest categorical predictors ----
top_cat = assoc[assoc["type"] == "categorical"]["feature"].tolist()[:2]
if len(top_cat) == 2:
    pv = df.pivot_table(index=top_cat[0], columns=top_cat[1], values="target", aggfunc="mean")
    plt.figure(figsize=(6.5, 4.5)); sns.heatmap(pv, annot=True, fmt=".1%", cmap="YlOrBr")
    plt.title(f"P(history of mental illness): {top_cat[0]} x {top_cat[1]}")
    savefig("06_two_way.png"); plt.show()

# ---- 3.9 Auto-generated EDA takeaways ----
print("=" * 110); print("EDA TAKEAWAYS (generated from the data)"); print("=" * 110)
say(f"1. Class balance: {prev:.1%} of people have a history of mental illness (imbalance {(1-prev)/prev:.1f}:1). "
    f"A model that always says 'No' would score {1-prev:.1%} accuracy, so accuracy alone is a misleading metric here; "
    f"I use ROC-AUC, PR-AUC, calibration and threshold-based precision/recall instead.")
strong = assoc[assoc.effect_size >= 0.05]
if len(strong):
    say("2. Predictors with at least a 'very weak' association (effect size >= 0.05): " +
        "; ".join(f"{r.feature} ({r.strength}, {r.effect_size:.2f})" for r in strong.itertuples()) + ".")
else:
    say("2. NO predictor reaches even a 'very weak' association (all effect sizes < 0.05). Expect a model with limited "
        "discrimination; statistical significance at this sample size does not imply practical importance.")
for r in assoc.head(4).itertuples():
    say(f"   - {r.feature}: effect size {r.effect_size:.3f} ({r.strength}); {r.direction}.")
say(f"3. Largest predictor-predictor correlation is {off.iloc[0]:.2f}, so multicollinearity is "
    f"{'not a concern' if off.iloc[0] < 0.5 else 'something to watch'} for the linear model.")
say(f"4. Data quality: {int(df.duplicated().sum())} exact duplicate rows, {int(df[expected].isna().sum().sum())} missing cells. "
    f"'Name' is an identifier ({df['Name'].nunique():,} distinct values) and carries no legitimate predictive information.")

NUM = ["Age", "Number of Children", "Log Income"]
FEATURES = NUM + CAT

reasons = {
    "Name": ("Exclude", "Identifier: no causal meaning, privacy risk, enables memorisation of individuals."),
    "Income": ("Exclude (use Log Income)", "Heavy right skew; log transform used instead, raw column dropped to avoid duplication."),
    "Log Income": ("Include", "Socio-economic position; plausible link to mental health. Possible reverse causation."),
    "Age": ("Include (sensitive)", "Standard demographic risk factor. Protected characteristic: audited in Section 7."),
    "Marital Status": ("Include (sensitive)", "Social-support proxy. Protected characteristic: audited in Section 7."),
    "Number of Children": ("Include", "Household / caring responsibility context."),
    "Education Level": ("Include", "Socio-economic context; ordinal."),
    "Employment Status": ("Include", "Plausibly linked to mental health; possible reverse causation."),
    "Smoking Status": ("Include", "Lifestyle marker associated with mental health in the literature."),
    "Physical Activity Level": ("Include", "Lifestyle marker; ordinal."),
    "Alcohol Consumption": ("Include", "Lifestyle marker; ordinal."),
    "Dietary Habits": ("Include", "Lifestyle marker; ordinal."),
    "Sleep Patterns": ("Include", "Sleep and mental health are closely linked; possible reverse causation."),
    "History of Substance Abuse": ("Include", "Co-occurring condition; may be a consequence as well as a risk factor."),
    "Family History of Depression": ("Include", "Established risk factor."),
    "Chronic Medical Conditions": ("Include", "Comorbidity; associated with mental health outcomes."),
}
decisions = (pd.DataFrame([(k, v[0], v[1]) for k, v in reasons.items()], columns=["feature", "decision", "rationale"])
             .merge(assoc[["feature", "effect_size", "strength", "mutual_info"]], on="feature", how="left"))
pd.set_option("display.max_colwidth", 120)
display(decisions.style.format({"effect_size": "{:.3f}", "mutual_info": "{:.4f}"}, na_rep="-"))

weak = assoc[assoc.effect_size < 0.05]["feature"].tolist()
if weak:
    say("Data-driven note: these kept variables show negligible univariate association (< 0.05): " + ", ".join(weak) +
        ". They are retained for the main model, and a production version should test dropping them.")

X, y = df[FEATURES], df["target"]
X_tr_full, X_te, y_tr_full, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr_full, y_tr_full, test_size=0.25, stratify=y_tr_full, random_state=RANDOM_STATE)
print(f"train {len(X_tr):,} | validation {len(X_va):,} | test {len(X_te):,} | positive rate "
      f"{y_tr.mean():.3f} / {y_va.mean():.3f} / {y_te.mean():.3f}")

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

# ---- Bayesian optimiser (helper code; also in src/tuning.py) ----
import numpy as np
import pandas as pd


def decode_unit(u, space):
    params = {}
    for ui, (name, (lo, hi, kind)) in zip(u, space.items()):
        if kind == "log":
            params[name] = float(np.exp(np.log(lo) + ui * (np.log(hi) - np.log(lo))))
        elif kind == "int":
            params[name] = int(round(lo + ui * (hi - lo)))
        else:
            params[name] = float(lo + ui * (hi - lo))
    return params


def bayes_optimise(objective, space, n_trials, seed=42, n_init=6):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        rows = []

        def obj(trial):
            p = {}
            for name, (lo, hi, kind) in space.items():
                p[name] = (trial.suggest_int(name, lo, hi) if kind == "int"
                           else trial.suggest_float(name, lo, hi, log=(kind == "log")))
            m, s = objective(p)
            rows.append({**p, "cv_auc": m, "cv_sd": s})
            return m

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_init))
        study.optimize(obj, n_trials=n_trials)
        return pd.DataFrame(rows), "Optuna (TPE)"
    except ImportError:
        pass

    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    rng = np.random.RandomState(seed)
    d = len(space)
    U, rows = [], []
    for t in range(n_trials):
        if t < n_init:
            u = rng.rand(d)
        else:
            y = np.array([r["cv_auc"] for r in rows])
            kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(d)) + WhiteKernel(1e-3)
            gp = GaussianProcessRegressor(kernel, normalize_y=True, random_state=seed,
                                          n_restarts_optimizer=2).fit(np.array(U), y)
            cand = rng.rand(3000, d)
            mu, sd = gp.predict(cand, return_std=True)
            sd = np.maximum(sd, 1e-9)
            z = (mu - y.max()) / sd
            ei = (mu - y.max()) * norm.cdf(z) + sd * norm.pdf(z)          # expected improvement
            u = cand[int(np.argmax(ei))]
        p = decode_unit(u, space)
        m, s = objective(p)
        U.append(u)
        rows.append({**p, "cv_auc": m, "cv_sd": s})
    return pd.DataFrame(rows), "Gaussian process + expected improvement"

# ---- run the searches: 5-fold CV on a stratified sample of the training set ----
CV_FOLDS, N_TRIALS_HGB, N_TRIALS_LR, TUNE_ROWS = 5, 25, 12, 100_000
tune_idx = (X_tr.index if len(X_tr) <= TUNE_ROWS else
            train_test_split(X_tr.index, train_size=TUNE_ROWS, stratify=y_tr, random_state=RANDOM_STATE)[0])
X_tune, y_tune = X_tr.loc[tune_idx], y_tr.loc[tune_idx]
cv_tune = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

def cv_auc(model):
    s = cross_val_score(model, X_tune, y_tune, cv=cv_tune, scoring="roc_auc")
    return s.mean(), s.std()

hgb_space = {"learning_rate": (0.01, 0.3, "log"), "max_depth": (2, 8, "int"), "max_leaf_nodes": (8, 64, "int"),
             "min_samples_leaf": (20, 500, "int"), "l2_regularization": (1e-3, 10.0, "log")}
hgb_trials, tuner = bayes_optimise(lambda p: cv_auc(make_hgb(NUM, CAT, **p)), hgb_space, N_TRIALS_HGB)
lin_trials, _ = bayes_optimise(lambda p: cv_auc(make_linear(NUM, CAT, p["C"])), {"C": (1e-3, 100.0, "log")}, N_TRIALS_LR)

def best_row(tr):
    return tr.loc[tr.cv_auc.idxmax()]
b = best_row(hgb_trials)
best_params = dict(learning_rate=float(b.learning_rate), max_depth=int(b.max_depth), max_leaf_nodes=int(b.max_leaf_nodes),
                   min_samples_leaf=int(b.min_samples_leaf), l2_regularization=float(b.l2_regularization))
best_C = float(best_row(lin_trials).C)

print(f"Optimiser: {tuner}. {N_TRIALS_HGB} gradient-boosting trials and {N_TRIALS_LR} logistic trials, {CV_FOLDS}-fold CV on {len(X_tune):,} training rows.")
print("Chosen HGB parameters:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in best_params.items()},
      f"-> CV AUC {b.cv_auc:.4f} (sd {b.cv_sd:.4f})")
print(f"Chosen logistic C: {best_C:.4g} -> CV AUC {best_row(lin_trials).cv_auc:.4f}")
near = int((hgb_trials.cv_auc >= b.cv_auc - b.cv_sd).sum())
print(f"{near} of {len(hgb_trials)} gradient-boosting trials are within one fold-sd of the best, "
      + ("so many settings are equivalent; the exact choice matters little." if near > 3 else "so the search found a clear optimum."))
display(hgb_trials.sort_values("cv_auc", ascending=False).head(8).style.format(
    {"learning_rate": "{:.3f}", "l2_regularization": "{:.3f}", "cv_auc": "{:.4f}", "cv_sd": "{:.4f}"}))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for ax, tr, ttl in [(axes[0], hgb_trials, "Gradient boosting"), (axes[1], lin_trials, "Logistic regression")]:
    ax.errorbar(range(1, len(tr) + 1), tr.cv_auc, yerr=tr.cv_sd, fmt="o", color="#5A6B87", alpha=0.6, label="trial (CV mean, sd)")
    ax.plot(range(1, len(tr) + 1), tr.cv_auc.cummax(), color="#D9922B", lw=2, label="best so far")
    ax.set_title(f"{ttl}: Bayesian optimisation"); ax.set_xlabel("trial"); ax.set_ylabel("5-fold CV ROC-AUC"); ax.legend()
plt.tight_layout(); savefig("13_bayes_optimisation.png"); plt.show()
hgb_trials.to_csv(os.path.join(OUT_DIR, "bayes_trials_hgb.csv"), index=False); lin_trials.to_csv(os.path.join(OUT_DIR, "bayes_trials_logistic.csv"), index=False)

models = {
    "Baseline (prior)": DummyClassifier(strategy="prior"),
    "Logistic regression": make_linear(NUM, CAT, best_C),
    "Gradient boosting (HGB)": make_hgb(NUM, CAT, **best_params),
}
for name, m in models.items():
    m.fit(X_tr, y_tr)
val_p = {n: m.predict_proba(X_va)[:, 1] for n, m in models.items()}
test_p = {n: m.predict_proba(X_te)[:, 1] for n, m in models.items()}

# ---- stability: stratified 3-fold CV on the training set ----
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
cv_rows = []
for name in ["Logistic regression", "Gradient boosting (HGB)"]:
    s = cross_val_score(clone(models[name]), X_tr, y_tr, cv=cv, scoring="roc_auc")
    cv_rows.append({"model": name, "cv_auc_mean": s.mean(), "cv_auc_std": s.std()})
display(pd.DataFrame(cv_rows).style.format({"cv_auc_mean": "{:.4f}", "cv_auc_std": "{:.4f}"}))

def boot_auc_diff(y_true, p_a, p_b, B=200, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed); y_true = np.asarray(y_true); d = []
    for _ in range(B):
        i = rng.randint(0, len(y_true), len(y_true))
        if y_true[i].min() == y_true[i].max():
            continue
        d.append(roc_auc_score(y_true[i], p_a[i]) - roc_auc_score(y_true[i], p_b[i]))
    d = np.array(d); return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)

m_d, lo_d, hi_d = boot_auc_diff(y_va, val_p["Gradient boosting (HGB)"], val_p["Logistic regression"])
print(f"Validation AUC difference (HGB - logistic): {m_d:+.4f}  95% CI [{lo_d:+.4f}, {hi_d:+.4f}]")
if lo_d > 0:
    final_name, other_name = "Gradient boosting (HGB)", "Logistic regression"
    why = "gradient boosting is reliably better on validation data, so the extra complexity is justified"
else:
    final_name, other_name = "Logistic regression", "Gradient boosting (HGB)"
    why = ("the interval does not show gradient boosting to be reliably better" if hi_d >= 0
           else "logistic regression is reliably better on validation data") + ", so the simpler, more transparent model is chosen"
print(f"FINAL MODEL: {final_name}: {why}.")
if final_name == "Logistic regression":
    make_final = lambda num, cat: make_linear(num, cat, best_C)
else:
    make_final = lambda num, cat: make_hgb(num, cat, **best_params)

def evaluate(y_true, p, threshold=0.5):
    y_true = np.asarray(y_true); pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return dict(roc_auc=roc_auc_score(y_true, p), pr_auc=average_precision_score(y_true, p),
                brier=brier_score_loss(y_true, p), log_loss=log_loss(y_true, np.clip(p, 1e-6, 1 - 1e-6)),
                accuracy=accuracy_score(y_true, pred), balanced_acc=balanced_accuracy_score(y_true, pred),
                precision=precision_score(y_true, pred, zero_division=0), recall=recall_score(y_true, pred, zero_division=0),
                specificity=tn / (tn + fp) if (tn + fp) else np.nan, f1=f1_score(y_true, pred, zero_division=0),
                mcc=matthews_corrcoef(y_true, pred), tn=tn, fp=fp, fn=fn, tp=tp)

def ece(y_true, p, bins=10):
    d = pd.DataFrame({"y": np.asarray(y_true), "p": p}); d["b"] = pd.qcut(d["p"].rank(method="first"), bins, labels=False)
    g = d.groupby("b").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((g.n / g.n.sum() * (g.y - g.p).abs()).sum())

def best_f1_threshold(y_true, p):
    grid_t = np.linspace(0.02, 0.9, 89)
    f1s = [f1_score(y_true, (p >= t).astype(int), zero_division=0) for t in grid_t]
    return grid_t[int(np.argmax(f1s))]

def choose_threshold(y_true, p, max_flag_rate=0.60):
    t = best_f1_threshold(y_true, p)
    if (p >= t).mean() > max_flag_rate:             # F1 degenerates to 'flag nearly everyone' when signal is weak
        prevalence = float(np.mean(y_true))
        return float(np.quantile(p, 1 - prevalence)), f"top-{prevalence:.0%} rule"
    return float(t), "F1-opt"

thr, thr_label = {"Baseline (prior)": 0.5}, {"Baseline (prior)": "0.50"}
for n in models:
    if n != "Baseline (prior)":
        thr[n], thr_label[n] = choose_threshold(y_va, val_p[n])
print("Operating thresholds chosen on validation data:",
      {k: f"{v:.2f} ({thr_label[k]})" for k, v in thr.items() if k != "Baseline (prior)"})
if any("rule" in v for v in thr_label.values()):
    say("Note: for at least one model the F1-optimal threshold would flag more than 60% of people, so the prevalence-matched rule was used "
        "instead. This usually happens when the model's signal is weak.")

rows = []
for n in models:
    rows.append({"model": n, "threshold": "0.50", **evaluate(y_te, test_p[n], 0.5)})
    if n != "Baseline (prior)":
        rows.append({"model": n, "threshold": f"{thr[n]:.2f} ({thr_label[n]})", **evaluate(y_te, test_p[n], thr[n])})
perf = pd.DataFrame(rows)
show = ["model", "threshold", "roc_auc", "pr_auc", "brier", "log_loss", "accuracy", "balanced_acc", "precision", "recall", "f1", "mcc"]
print("TEST-SET PERFORMANCE (prevalence = %.3f, majority-class accuracy = %.3f)" % (y_te.mean(), 1 - y_te.mean()))
display(perf[show].style.format({c: "{:.3f}" for c in show[2:]}))
perf.to_csv(os.path.join(OUT_DIR, "test_performance.csv"), index=False)
print("Final model:", final_name)

# ---- ROC, precision-recall, calibration ----
colors = {"Baseline (prior)": "#9AA5B8", "Logistic regression": "#2E8B8B", "Gradient boosting (HGB)": "#D9922B"}
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for n, p in test_p.items():
    fpr, tpr, _ = roc_curve(y_te, p); axes[0].plot(fpr, tpr, color=colors[n], label=f"{n} (AUC {roc_auc_score(y_te, p):.3f})")
    pr, rc, _ = precision_recall_curve(y_te, p); axes[1].plot(rc, pr, color=colors[n], label=f"{n} (AP {average_precision_score(y_te, p):.3f})")
    if n != "Baseline (prior)":
        fr, mp = calibration_curve(y_te, p, n_bins=10, strategy="quantile"); axes[2].plot(mp, fr, marker="o", color=colors[n], label=n)
axes[0].plot([0, 1], [0, 1], "k--", lw=1); axes[0].set_title("ROC curve"); axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
axes[1].axhline(y_te.mean(), color="k", ls="--", lw=1, label="No skill (prevalence)"); axes[1].set_title("Precision-recall curve"); axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[2].plot([0, 1], [0, 1], "k--", lw=1); axes[2].set_title("Calibration (10 quantile bins)"); axes[2].set_xlabel("Mean predicted probability"); axes[2].set_ylabel("Observed rate")
for a in axes: a.legend(fontsize=8)
plt.tight_layout(); savefig("07_roc_pr_calibration.png"); plt.show()

# ---- confusion matrix at the chosen threshold + lift by decile (final model) ----
pf = test_p[final_name]; t_f = thr[final_name]
cm = confusion_matrix(y_te, (pf >= t_f).astype(int))
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=axes[0], xticklabels=["Pred No", "Pred Yes"], yticklabels=["Actual No", "Actual Yes"])
axes[0].set_title(f"{final_name}: confusion matrix at threshold {t_f:.2f}")
dec = pd.DataFrame({"y": y_te.values, "p": pf})
dec["decile"] = pd.qcut(dec["p"].rank(method="first"), 10, labels=range(1, 11))
lift = dec.groupby("decile", observed=True)["y"].agg(rate="mean", n="size").reset_index()
lift["lift"] = lift["rate"] / y_te.mean()
axes[1].bar(lift["decile"].astype(int), lift["lift"], color="#1B2A4A"); axes[1].axhline(1, color="#B0473F", ls="--")
axes[1].set_xlabel("Score decile (10 = highest predicted risk)"); axes[1].set_ylabel("Lift vs overall rate"); axes[1].set_title("Lift by decile (test set)")
plt.tight_layout(); savefig("08_confusion_lift.png"); plt.show()
display(lift.style.format({"rate": "{:.3f}", "lift": "{:.2f}"}))

# shuffled-label control
rs = np.random.RandomState(RANDOM_STATE)
ctrl = make_final(NUM, CAT).fit(X_tr, rs.permutation(y_tr.values))
ctrl_auc = roc_auc_score(y_te, ctrl.predict_proba(X_te)[:, 1])
print(f"Shuffled-label control AUC on test: {ctrl_auc:.3f}  (expected ~0.50)")

# single-variable AUCs (validation), 1-D model per variable
single = []
for f in FEATURES:
    num_f = [f] if f in NUM else []; cat_f = [f] if f in CAT else []
    m1 = make_final(num_f, cat_f).fit(X_tr[[f]], y_tr)
    single.append({"feature": f, "single_var_val_auc": roc_auc_score(y_va, m1.predict_proba(X_va[[f]])[:, 1])})
single = pd.DataFrame(single).sort_values("single_var_val_auc", ascending=False).reset_index(drop=True)
display(single.style.format({"single_var_val_auc": "{:.3f}"}))

# capacity table (test set, final model)
cap = []
order_idx = np.argsort(-pf)
for share in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
    k = int(share * len(pf)); top = y_te.values[order_idx[:k]]
    cap.append({"contact_top_share": f"{share:.0%}", "people_contacted": k, "precision": top.mean(),
                "share_of_all_positives_reached": top.sum() / y_te.sum(), "lift_vs_random": top.mean() / y_te.mean()})
cap = pd.DataFrame(cap)
display(cap.style.format({"precision": "{:.3f}", "share_of_all_positives_reached": "{:.3f}", "lift_vs_random": "{:.2f}"}))
cap.to_csv(os.path.join(OUT_DIR, "capacity_table.csv"), index=False)

from sklearn.tree import DecisionTreeClassifier

def tree_frame(Xin):
    F = Xin[NUM].copy()
    for c in CAT:
        if c in ORDERS:                                   # ordinal -> rank, so one split can mean 'degree or higher'
            F[c + " (rank)"] = Xin[c].map({v: i for i, v in enumerate(ORDERS[c])}).fillna(-1)
        else:
            d = pd.get_dummies(Xin[c], prefix=c, drop_first=(Xin[c].nunique() == 2)).astype(int)
            F = pd.concat([F, d], axis=1)
    return F

F_all = tree_frame(X)
F_tr, F_te = F_all.loc[X_tr.index], F_all.loc[X_te.index]
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=max(200, int(0.02 * len(F_tr))),
                              min_impurity_decrease=2e-4, random_state=RANDOM_STATE).fit(F_tr, y_tr)
auc_tree = roc_auc_score(y_te, tree.predict_proba(F_te)[:, 1])

def leaf_rules(t, names):
    tr_ = t.tree_; out = {}
    def walk(node, conds):
        if tr_.children_left[node] == -1:
            out[node] = " AND ".join(conds) or "everyone"; return
        f, thr_ = names[tr_.feature[node]], tr_.threshold[node]
        walk(tr_.children_left[node], conds + [f"{f} <= {thr_:.1f}"]); walk(tr_.children_right[node], conds + [f"{f} > {thr_:.1f}"])
    walk(0, []); return out

rules = leaf_rules(tree, list(F_tr.columns)); leaf_id = tree.apply(F_tr)
tiers = (pd.DataFrame({"leaf": leaf_id, "y": y_tr.values}).groupby("leaf")["y"].agg(rate="mean", n="size").reset_index())
tiers["share_of_people"] = tiers["n"] / len(F_tr); tiers["rule"] = tiers["leaf"].map(rules)
tiers = tiers.sort_values("rate", ascending=False).drop(columns="leaf").reset_index(drop=True)
display(tiers.style.format({"rate": "{:.3f}", "share_of_people": "{:.1%}", "n": "{:,}"}))
tiers.to_csv(os.path.join(OUT_DIR, "tree_tiers.csv"), index=False)
print(f"Shallow tree ({tree.get_n_leaves()} leaves): test ROC-AUC {auc_tree:.4f}  versus final model {roc_auc_score(y_te, pf):.4f}")

# ---- bootstrap confidence intervals on the TEST set (reporting only; not used for selection) ----
rng = np.random.RandomState(RANDOM_STATE)
yt = y_te.values; po = test_p[other_name]
auc_f, auc_o, ap_f = [], [], []
for _ in range(200):
    i = rng.randint(0, len(yt), len(yt))
    if yt[i].min() == yt[i].max(): continue
    auc_f.append(roc_auc_score(yt[i], pf[i])); auc_o.append(roc_auc_score(yt[i], po[i])); ap_f.append(average_precision_score(yt[i], pf[i]))
auc_f, auc_o, ap_f = map(np.array, (auc_f, auc_o, ap_f)); diff = auc_f - auc_o
ci = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5))
print(f"{final_name}: test ROC-AUC {roc_auc_score(yt, pf):.4f}  95% CI [{ci(auc_f)[0]:.4f}, {ci(auc_f)[1]:.4f}]")
print(f"{final_name}: test PR-AUC  {average_precision_score(yt, pf):.4f}  95% CI [{ci(ap_f)[0]:.4f}, {ci(ap_f)[1]:.4f}]")
print(f"{other_name}: test ROC-AUC {roc_auc_score(yt, po):.4f}  95% CI [{ci(auc_o)[0]:.4f}, {ci(auc_o)[1]:.4f}]")
print(f"Test AUC difference (final - other) {diff.mean():+.4f}  95% CI [{ci(diff)[0]:+.4f}, {ci(diff)[1]:+.4f}]")

# ---- what drives the model? permutation importance (final model) and odds ratios (logistic) ----
final_model = models[final_name]
sub = X_va.sample(min(20_000, len(X_va)), random_state=RANDOM_STATE)
pi = permutation_importance(final_model, sub, y_va.loc[sub.index], scoring="roc_auc", n_repeats=5, random_state=RANDOM_STATE)
imp = pd.DataFrame({"feature": FEATURES, "auc_drop": pi.importances_mean, "sd": pi.importances_std}).sort_values("auc_drop", ascending=False)
lin = models["Logistic regression"]
names = lin.named_steps["pre"].get_feature_names_out()
coefs = pd.DataFrame({"term": names, "odds_ratio": np.exp(lin.named_steps["clf"].coef_[0])})

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
imp.sort_values("auc_drop").plot.barh(x="feature", y="auc_drop", xerr="sd", legend=False, ax=axes[0], color="#D9922B")
axes[0].set_title(f"{final_name}: permutation importance (drop in validation ROC-AUC)")
ext = pd.concat([coefs.sort_values("odds_ratio").head(8), coefs.sort_values("odds_ratio").tail(8)]).drop_duplicates()
axes[1].barh(ext["term"].str.replace("cat__", "").str.replace("num__", ""), ext["odds_ratio"] - 1, left=1, color="#2E8B8B")
axes[1].axvline(1, color="k", lw=1); axes[1].set_title("Logistic regression: largest odds ratios (num = per 1 SD)")
plt.tight_layout(); savefig("09_importance.png"); plt.show()
display(imp.style.format({"auc_drop": "{:.4f}", "sd": "{:.4f}"}))

# partial dependence for the numeric drivers (average effect on predicted probability)
try:
    Xs = sub.sample(min(5000, len(sub)), random_state=0).copy(); Xs[NUM] = Xs[NUM].astype(float)
    PartialDependenceDisplay.from_estimator(final_model, Xs, features=NUM, kind="average", grid_resolution=20)
    plt.suptitle(f"Partial dependence ({final_name})", y=1.02); savefig("10_partial_dependence.png"); plt.show()
except Exception as e:
    print("Partial dependence skipped:", repr(e))

# ---- would more data help?  (learning curve, final model) ----
fracs = [0.05, 0.1, 0.25, 0.5, 1.0]; lc = []
for f in fracs:
    idx = X_tr.sample(frac=f, random_state=RANDOM_STATE).index if f < 1 else X_tr.index
    m = make_final(NUM, CAT).fit(X_tr.loc[idx], y_tr.loc[idx])
    lc.append({"train_rows": len(idx), "train_auc": roc_auc_score(y_tr.loc[idx], m.predict_proba(X_tr.loc[idx])[:, 1]),
               "val_auc": roc_auc_score(y_va, m.predict_proba(X_va)[:, 1])})
lc = pd.DataFrame(lc)
plt.figure(figsize=(7, 4.5)); plt.plot(lc.train_rows, lc.train_auc, marker="o", label="Train", color="#5A6B87")
plt.plot(lc.train_rows, lc.val_auc, marker="o", label="Validation", color="#D9922B"); plt.xscale("log")
plt.xlabel("Training rows"); plt.ylabel("ROC-AUC"); plt.title(f"Learning curve ({final_name})"); plt.legend(); plt.tight_layout()
savefig("11_learning_curve.png"); plt.show(); display(lc.style.format({"train_auc": "{:.4f}", "val_auc": "{:.4f}"}))

# ---- auto-generated performance narrative ----
def auc_band(a):
    return ("no better than chance" if a < 0.55 else "poor" if a < 0.65 else "modest" if a < 0.75
            else "good" if a < 0.85 else "strong")
m_f = evaluate(y_te, pf, thr[final_name]); auc_pt = roc_auc_score(yt, pf); pr_pt = average_precision_score(yt, pf)
ece_f = ece(y_te, pf); base_brier = perf.loc[perf.model == "Baseline (prior)", "brier"].iloc[0]
flag_rate = (pf >= thr[final_name]).mean()
print("=" * 110); print("PERFORMANCE TAKEAWAYS (generated from the data)"); print("=" * 110)
say(f"1. Final model: {final_name}. Discrimination on the held-out test set: ROC-AUC {auc_pt:.3f} ({auc_band(auc_pt)}; 95% CI "
    f"{ci(auc_f)[0]:.3f} to {ci(auc_f)[1]:.3f}); PR-AUC {pr_pt:.3f} versus a no-skill reference of {y_te.mean():.3f}.")
say(f"2. Model choice: on validation data the AUC difference (HGB - logistic) was {m_d:+.4f} (95% CI {lo_d:+.4f} to {hi_d:+.4f}); {why}.")
say(f"3. At the chosen operating point ({thr_label[final_name]}, threshold {thr[final_name]:.2f}) the model flags {flag_rate:.0%} of people, catches "
    f"{m_f['recall']:.0%} of those with a history of mental illness (recall) with {m_f['precision']:.0%} precision (no-skill precision would be "
    f"{y_te.mean():.0%}). Accuracy is {m_f['accuracy']:.1%} versus {1-y_te.mean():.1%} for always predicting 'No', which is why accuracy alone is not used.")
say(f"4. Lift: the top decile of scores has a rate {lift['lift'].iloc[-1]:.2f}x the average; the bottom decile {lift['lift'].iloc[0]:.2f}x.")
say(f"5. Calibration: Brier score {m_f['brier']:.3f} (baseline {base_brier:.3f}); expected calibration error {ece_f:.3f} "
    + ("(well calibrated: predicted probabilities can be read as risks)." if ece_f < 0.02
       else "(noticeable miscalibration: apply isotonic or Platt recalibration before using scores as probabilities)."))
top3 = imp.head(3)
say("6. Most influential variables (permutation importance): " + "; ".join(f"{r.feature} ({r.auc_drop:.4f})" for r in top3.itertuples()) + ".")
say(f"7. Learning curve: validation AUC moves from {lc.val_auc.iloc[0]:.3f} (smallest sample) to {lc.val_auc.iloc[-1]:.3f} (all training data); "
    f"train-validation gap at full size {lc.train_auc.iloc[-1]-lc.val_auc.iloc[-1]:.3f}. "
    + ("The curve has flattened, so more rows of the same data are unlikely to help; better features would."
       if lc.val_auc.iloc[-1] - lc.val_auc.iloc[-2] < 0.003 else "The curve is still rising, so more data would help."))
say(f"8. Sanity checks: a model trained on shuffled labels scores AUC {ctrl_auc:.3f} (should be about 0.50), so there is no sign of leakage. "
    f"Best single variable alone reaches validation AUC {single.single_var_val_auc.iloc[0]:.3f} ({single.feature.iloc[0]}); "
    + ("performance is therefore carried by one or two variables." if single.single_var_val_auc.iloc[0] > 0.9 * (roc_auc_score(y_va, val_p[final_name]) - 0.5) + 0.5 else "performance is spread across several variables, each contributing a little."))
say(f"9. Structure: a {tree.get_n_leaves()}-leaf decision tree reaches test AUC {auc_tree:.3f} against {auc_pt:.3f} for the final model, with group rates from "
    f"{tiers.rate.min():.1%} to {tiers.rate.max():.1%} (table in Section 6.2). "
    + ("The signal is therefore a few simple groups, not a complicated function, and any gain of boosting over logistic regression comes from interactions between those variables."
       if abs(auc_tree - auc_pt) < 0.01 else "The tree is clearly weaker than the final model, so there is more structure than a few simple groups."))
if auc_pt < 0.65:
    say("Overall: the model has limited predictive power. That is itself a finding: the available lifestyle and demographic variables "
        "contain little signal about this outcome, and the model should NOT be used to make individual-level decisions.")

# ---- interaction helpers (also in src/interactions.py) ----
import itertools
import numpy as np
import pandas as pd


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def partial_dependence_at_points(model, S, cols):
    n = len(S)
    base = S.reset_index(drop=True)
    T = base.iloc[np.tile(np.arange(n), n)].reset_index(drop=True)
    T[cols] = base[cols].iloc[np.repeat(np.arange(n), n)].values
    return _logit(model.predict_proba(T)[:, 1]).reshape(n, n).mean(axis=1)


def pairwise_h(model, S, features):
    single = {f: partial_dependence_at_points(model, S, [f]) for f in features}
    single = {f: v - v.mean() for f, v in single.items()}
    rows = []
    for a, b in itertools.combinations(features, 2):
        jk = partial_dependence_at_points(model, S, [a, b])
        jk = jk - jk.mean()
        resid = jk - single[a] - single[b]
        rows.append({"feature_a": a, "feature_b": b,
                     "strength": float(np.sqrt(np.mean(resid ** 2))),
                     "h2": float(np.sum(resid ** 2) / max(np.sum(jk ** 2), 1e-12))})
    return pd.DataFrame(rows).sort_values("strength", ascending=False).reset_index(drop=True)


def make_bins(X_train, num_cols):
    return {c: np.unique(np.quantile(X_train[c], [0.25, 0.5, 0.75])) for c in num_cols}


def add_crosses(X, pairs, bins):
    out = X.copy()
    names = []

    def lab(c):
        if c in bins:
            return pd.Series(np.digitize(X[c], bins[c]), index=X.index).astype(str)
        return X[c].astype(str)

    for a, b in pairs:
        nm = f"{a} x {b}"
        out[nm] = lab(a) + " | " + lab(b)
        names.append(nm)
    return out, names

# ---- 1. do interactions matter?  boosted trees of increasing depth ----
# depth 1 = stumps: the model can only add up single-variable effects (no interactions possible).
rungs = [("Additive only (depth 1)", 1, 2), ("Up to 2-way interactions (depth 2)", 2, 4),
         ("Up to 3-way interactions (depth 3)", 3, 8), ("Tuned model (Section 5.2)", best_params["max_depth"], best_params["max_leaf_nodes"])]
ladder, ladder_p = [], {}
for label, depth, leaves in rungs:
    if label.startswith("Tuned"):
        m = models["Gradient boosting (HGB)"]
    else:
        m = make_hgb(NUM, CAT, **{**best_params, "max_depth": depth, "max_leaf_nodes": leaves}).fit(X_tr, y_tr)
    ladder_p[label] = m.predict_proba(X_va)[:, 1]
    ladder.append({"model": label, "val_auc": roc_auc_score(y_va, ladder_p[label])})
ladder.append({"model": "Logistic regression (additive)", "val_auc": roc_auc_score(y_va, val_p["Logistic regression"])})
ladder = pd.DataFrame(ladder); display(ladder.style.format({"val_auc": "{:.4f}"})); ladder.to_csv(os.path.join(OUT_DIR, "interaction_ladder.csv"), index=False)
d_int, lo_int, hi_int = boot_auc_diff(y_va, ladder_p["Tuned model (Section 5.2)"], ladder_p["Additive only (depth 1)"])
print(f"Validation AUC gain from allowing interactions (tuned - additive): {d_int:+.4f}  95% CI [{lo_int:+.4f}, {hi_int:+.4f}]")
interactions_matter = lo_int > 0
print("=> " + ("interactions add reliable predictive signal, so it is worth finding them." if interactions_matter
               else "no reliable gain from interactions; an additive model is enough."))

# ---- 2a. which pairs interact?  Friedman H-statistic on the tuned boosted model (log-odds scale) ----
S_h = X_tr.sample(300, random_state=RANDOM_STATE)
h_table = pairwise_h(models["Gradient boosting (HGB)"], S_h, FEATURES)
h_table["pair"] = h_table.feature_a + "  x  " + h_table.feature_b
h_table.to_csv(os.path.join(OUT_DIR, "interaction_strength.csv"), index=False)
display(h_table.head(10)[["pair", "strength", "h2"]].style.format({"strength": "{:.4f}", "h2": "{:.3f}"}))

top = h_table.head(10).iloc[::-1]
plt.figure(figsize=(8, 4.5)); plt.barh(top.pair, top.strength, color="#2E8B8B")
plt.xlabel("interaction strength (log-odds; 0 = purely additive)"); plt.title("Strongest pairwise interactions (tuned boosted model)")
plt.tight_layout(); savefig("15_interaction_strength.png"); plt.show()
print("Read this as the size of the 'extra' effect of a pair beyond the sum of the two separate effects. "
      "For scale, the permutation importance table above shows how much each variable matters on its own.")

# ---- 2b. what does an interaction look like?  predicted risk across the top pairs ----
S_g = X_tr.sample(3000, random_state=RANDOM_STATE)
def grid_values(col):
    if col in NUM:
        return list(np.unique(np.round(S_g[col].quantile([.1, .3, .5, .7, .9]).values, 2)))
    present = set(S_g[col]); return [v for v in ORDERS.get(col, sorted(present)) if v in present]

def risk_grid(model, a, b):
    ga, gb = grid_values(a), grid_values(b); out = pd.DataFrame(index=ga, columns=gb, dtype=float)
    for va in ga:
        for vb in gb:
            T = S_g.copy(); T[a] = va; T[b] = vb
            out.loc[va, vb] = model.predict_proba(T)[:, 1].mean()
    return out

pairs_top = [(r.feature_a, r.feature_b) for r in h_table.head(3).itertuples()]
grids = [risk_grid(models["Gradient boosting (HGB)"], a, b) for a, b in pairs_top]
for (a_, b_), g_ in zip(pairs_top, grids):
    g_.to_csv(os.path.join(OUT_DIR, f"risk_grid_{a_}_x_{b_}.csv".replace(" ", "_")))
vmin, vmax = min(g.values.min() for g in grids), max(g.values.max() for g in grids)
fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
for ax, g, (a, b) in zip(axes, grids, pairs_top):
    sns.heatmap(g, annot=True, fmt=".0%", cmap="YlOrBr", vmin=vmin, vmax=vmax, cbar=False, ax=ax)
    ax.set_xlabel(b); ax.set_ylabel(a); ax.set_title(f"Predicted risk: {a} x {b}")
plt.suptitle("If effects were additive, every row would show the same left-to-right pattern", y=1.03)
plt.tight_layout(); savefig("16_interaction_heatmaps.png"); plt.show()

# ---- 3. SHAP with XGBoost (optional packages: pip install xgboost shap) ----
try:
    import xgboost as xgb
    import shap
    HAVE_SHAP = True
except Exception as e:
    HAVE_SHAP = False
    print("SHAP / XGBoost part skipped:", repr(e))
    print("To enable it: pip install xgboost shap   (then restart the kernel and re-run).")

shap_pairs = None; shap_rank = None
if HAVE_SHAP:
    cat_levels = {c: (ORDERS[c] if c in ORDERS else sorted(X_tr[c].dropna().unique())) for c in CAT}
    def to_codes(Xin):
        F = Xin[NUM].astype(float).copy()
        for c in CAT:
            codes = pd.Categorical(Xin[c], categories=cat_levels[c]).codes.astype(float)
            F[c] = np.where(codes < 0, np.nan, codes)
        return F
    Xc_tr, Xc_va = to_codes(X_tr), to_codes(X_va)
    i_fit, i_stop = train_test_split(np.arange(len(Xc_tr)), test_size=0.1, stratify=y_tr, random_state=RANDOM_STATE)
    xp = dict(n_estimators=1500, learning_rate=max(best_params["learning_rate"], 0.03), max_depth=min(best_params["max_depth"], 6),
              min_child_weight=max(1.0, 0.2 * best_params["min_samples_leaf"]), reg_lambda=max(1.0, best_params["l2_regularization"]),
              tree_method="hist", eval_metric="auc", n_jobs=-1, random_state=RANDOM_STATE)
    fit_args = dict(eval_set=[(Xc_tr.iloc[i_stop], y_tr.iloc[i_stop])], verbose=False)
    try:
        xgb_model = xgb.XGBClassifier(early_stopping_rounds=30, **xp).fit(Xc_tr.iloc[i_fit], y_tr.iloc[i_fit], **fit_args)
    except TypeError:                                   # older xgboost: early stopping is a fit() argument
        xgb_model = xgb.XGBClassifier(**xp).fit(Xc_tr.iloc[i_fit], y_tr.iloc[i_fit], early_stopping_rounds=30, **fit_args)
    auc_xgb = roc_auc_score(y_va, xgb_model.predict_proba(Xc_va)[:, 1])
    print(f"XGBoost validation ROC-AUC {auc_xgb:.4f} (sklearn HGB {roc_auc_score(y_va, val_p['Gradient boosting (HGB)']):.4f}): "
          "same model family, used here because SHAP has an exact, fast algorithm for it.")

    Xs = Xc_va.sample(3000, random_state=RANDOM_STATE); Xs_disp = X_va.loc[Xs.index]
    explainer = shap.TreeExplainer(xgb_model)
    sv = explainer.shap_values(Xs)
    sv = sv[-1] if isinstance(sv, list) else sv
    shap_rank = pd.Series(np.abs(sv).mean(0), index=Xs.columns).sort_values(ascending=False)
    print("Mean |SHAP| (average push on log-odds):"); display(shap_rank.round(4).to_frame("mean_abs_shap").T)

    for kind, fname, ttl in [(None, "17_shap_summary.png", "SHAP summary: each dot is a person (colour = value; for category variables, the category code)"),
                             ("bar", "18_shap_importance.png", "SHAP importance: mean absolute effect on log-odds")]:
        try:
            plt.figure(); shap.summary_plot(sv, Xs, plot_type=kind, show=False, max_display=14)
            plt.title(ttl, fontsize=10); plt.tight_layout(); savefig(fname); plt.show()
        except Exception as e:
            print("SHAP plot skipped:", repr(e))

    for f in list(shap_rank.index[:3]):
        try:
            plt.figure(); shap.dependence_plot(f, sv, Xs, display_features=Xs_disp, show=False)
            plt.tight_layout(); savefig(f"19_shap_dependence_{f.replace(' ', '_')}.png"); plt.show()
        except Exception as e:
            try:
                plt.figure(); shap.dependence_plot(f, sv, Xs, show=False); plt.tight_layout(); plt.show()
            except Exception as e2:
                print("SHAP dependence plot skipped for", f, repr(e2))

    try:
        iv = explainer.shap_interaction_values(Xs.iloc[:1500])
        M = pd.DataFrame(np.abs(iv).mean(0), index=Xs.columns, columns=Xs.columns)
        M_off = M.copy(); np.fill_diagonal(M_off.values, 0)
        plt.figure(figsize=(9, 7)); sns.heatmap(M_off, cmap="YlOrBr", annot=False, square=True)
        plt.title("SHAP interaction values (mean absolute; diagonal removed)"); plt.tight_layout(); savefig("20_shap_interactions.png"); plt.show()
        pr = [(a, b, 2 * M_off.loc[a, b]) for i, a in enumerate(M_off.index) for b in M_off.columns[i + 1:]]
        shap_pairs = pd.DataFrame(pr, columns=["feature_a", "feature_b", "shap_interaction"]).sort_values("shap_interaction", ascending=False).reset_index(drop=True)
        display(shap_pairs.head(8).style.format({"shap_interaction": "{:.4f}"}))
        top_h = set(map(frozenset, zip(h_table.feature_a[:5], h_table.feature_b[:5])))
        top_s = set(map(frozenset, zip(shap_pairs.feature_a[:5], shap_pairs.feature_b[:5])))
        print(f"Pairs in both top-5 lists (H-statistic vs SHAP): {len(top_h & top_s)} of 5")
    except Exception as e:
        print("SHAP interaction values skipped:", repr(e))

# ---- 4. give the discovered interactions to the logistic regression ----
# Each chosen pair becomes one cross-feature ('A | B' combined category; numeric variables are cut into quartiles).
# Pairs come from the H-statistic ranking (training data). K is chosen on VALIDATION AUC.
bins = make_bins(X_tr, NUM)
cand_pairs = list(zip(h_table.feature_a, h_table.feature_b))[:6]
def fit_cross_logit(K, C):
    Xa, names = add_crosses(X_tr, cand_pairs[:K], bins)
    return make_linear(NUM, CAT + names, C).fit(Xa, y_tr), names

k_rows = []
for K in range(0, len(cand_pairs) + 1):
    m, names = fit_cross_logit(K, best_C)
    Xv, _ = add_crosses(X_va, cand_pairs[:K], bins)
    k_rows.append({"pairs added (K)": K, "added": ", ".join(names) if names else "-", "val_auc": roc_auc_score(y_va, m.predict_proba(Xv)[:, 1])})
k_tab = pd.DataFrame(k_rows); display(k_tab.style.format({"val_auc": "{:.4f}"}))
K_best = int(min(r["pairs added (K)"] for r in k_rows if r["val_auc"] >= k_tab.val_auc.max() - 0.0005))
print(f"Chosen K = {K_best} (smallest number of interactions within 0.0005 of the best validation AUC).")

# re-tune C for the richer model (Bayesian optimisation, 5-fold CV on the same training sample)
Xt_i, names_best = add_crosses(X_tune, cand_pairs[:K_best], bins)
def cv_auc_cross(C):
    s = cross_val_score(make_linear(NUM, CAT + names_best, C), Xt_i, y_tune, cv=cv_tune, scoring="roc_auc"); return s.mean(), s.std()
C_tr, _ = bayes_optimise(lambda p: cv_auc_cross(p["C"]), {"C": (1e-3, 100.0, "log")}, 10)
C_int = float(best_row(C_tr).C)
lin_int, names_best = fit_cross_logit(K_best, C_int)
Xv_i, _ = add_crosses(X_va, cand_pairs[:K_best], bins); Xte_i, _ = add_crosses(X_te, cand_pairs[:K_best], bins)
pv_int, pt_int = lin_int.predict_proba(Xv_i)[:, 1], lin_int.predict_proba(Xte_i)[:, 1]

# test set: reported once, AFTER every choice above was made on training / validation data
cmp_tab = pd.DataFrame([
    {"model": "Logistic regression (additive)", "val_auc": roc_auc_score(y_va, val_p["Logistic regression"]), "test_auc": roc_auc_score(y_te, test_p["Logistic regression"])},
    {"model": f"Logistic regression + {K_best} interaction(s)", "val_auc": roc_auc_score(y_va, pv_int), "test_auc": roc_auc_score(y_te, pt_int)},
    {"model": "Gradient boosting (HGB)", "val_auc": roc_auc_score(y_va, val_p["Gradient boosting (HGB)"]), "test_auc": roc_auc_score(y_te, test_p["Gradient boosting (HGB)"])}])
display(cmp_tab.style.format({"val_auc": "{:.4f}", "test_auc": "{:.4f}"})); cmp_tab.to_csv(os.path.join(OUT_DIR, "interaction_model_comparison.csv"), index=False)
d1 = boot_auc_diff(y_te, pt_int, test_p["Logistic regression"]); d2 = boot_auc_diff(y_te, test_p["Gradient boosting (HGB)"], pt_int)
print(f"Test AUC, interaction logistic minus additive logistic: {d1[0]:+.4f}  95% CI [{d1[1]:+.4f}, {d1[2]:+.4f}]")
print(f"Test AUC, gradient boosting minus interaction logistic: {d2[0]:+.4f}  95% CI [{d2[1]:+.4f}, {d2[2]:+.4f}]")

# readable coefficients for the interaction terms
if K_best > 0:
    pre = lin_int.named_steps["pre"]; oh = pre.named_transformers_["cat"].named_steps["oh"]
    try: cn = list(oh.get_feature_names_out(CAT + names_best))
    except AttributeError: cn = list(oh.get_feature_names(CAT + names_best))
    co = pd.DataFrame({"term": list(pre.transformers_[0][2]) + cn, "odds_ratio": np.exp(lin_int.named_steps["clf"].coef_[0])})
    co = co[co.term.str.contains(" x ")]
    co = co.reindex((co.odds_ratio - 1).abs().sort_values(ascending=False).index).head(10)
    print("Largest interaction terms (odds ratio relative to the additive effects; >1 = higher risk than the separate effects suggest):")
    display(co.style.format({"odds_ratio": "{:.3f}"})); co.to_csv(os.path.join(OUT_DIR, "interaction_odds_ratios.csv"), index=False)

# ---- full test-set report for the interaction-aware logistic regression (threshold chosen on validation) ----
thr_int, thr_int_label = choose_threshold(y_va, pv_int)
m_int = evaluate(y_te, pt_int, thr_int); ece_int = ece(y_te, pt_int)
dec = pd.DataFrame({"y": y_te.values, "p": pt_int}); dec["d"] = pd.qcut(dec.p.rank(method="first"), 10, labels=False)
lift_int = (dec.groupby("d").y.mean() / y_te.mean()).values
int_report = {"threshold": thr_int, "rule": thr_int_label, "roc_auc": m_int["roc_auc"], "pr_auc": m_int["pr_auc"], "brier": m_int["brier"],
              "ece": ece_int, "accuracy": m_int["accuracy"], "precision": m_int["precision"], "recall": m_int["recall"],
              "flag_rate": float((pt_int >= thr_int).mean()), "lift_top_decile": lift_int[-1], "lift_bottom_decile": lift_int[0]}
pd.Series(int_report).to_frame("interaction_logistic").to_csv(os.path.join(OUT_DIR, "interaction_model_report.csv"))
display(pd.Series(int_report).to_frame("interaction-aware logistic regression (test set)").style.format(precision=4))

# ---- auto-generated interaction narrative ----
print("=" * 110); print("INTERACTION TAKEAWAYS (generated from the data)"); print("=" * 110)
say(f"1. Do interactions matter? Validation AUC rises from {ladder.val_auc.iloc[0]:.4f} (additive boosted trees) to {ladder.val_auc.iloc[3]:.4f} (tuned), "
    f"a gain of {d_int:+.4f} (95% CI {lo_int:+.4f} to {hi_int:+.4f}). "
    + ("The gain is reliable but small: most of the signal is additive, with a modest interaction component on top." if interactions_matter and d_int < 0.02
       else "The gain is substantial." if interactions_matter else "The gain is not reliably different from zero, so the signal is essentially additive."))
tp = h_table.iloc[0]
say(f"2. Strongest interaction (H-statistic): {tp.feature_a} x {tp.feature_b}, size {tp.strength:.3f} log-odds; next: "
    + "; ".join(f"{r.feature_a} x {r.feature_b} ({r.strength:.3f})" for r in h_table.iloc[1:3].itertuples()) + ".")
if HAVE_SHAP and shap_rank is not None:
    say("3. SHAP (XGBoost) ranks the variables by average effect: " + ", ".join(f"{k} ({v:.3f})" for k, v in shap_rank.head(4).items()) + ".")
else:
    say("3. SHAP was skipped because xgboost / shap are not installed (pip install xgboost shap).")
gap = cmp_tab.test_auc.iloc[2] - cmp_tab.test_auc.iloc[0]; got = cmp_tab.test_auc.iloc[1] - cmp_tab.test_auc.iloc[0]
say(f"4. Feeding {K_best} discovered interaction(s) into the logistic regression moves test AUC from {cmp_tab.test_auc.iloc[0]:.4f} to {cmp_tab.test_auc.iloc[1]:.4f} "
    f"(boosting: {cmp_tab.test_auc.iloc[2]:.4f})"
    + (f", recovering {got / gap:.0%} of the gap to boosting while staying fully explainable." if gap > 0.0005 else ". Boosting has no real advantage over the additive model here, so there is little to recover."))
if K_best == 0 or got <= 0.0005:
    rec = ("keep the additive logistic regression. The interactions found did not improve held-out accuracy, so they would add complexity without benefit.")
elif gap > 0.0005 and got / gap > 0.7:
    rec = ("use the interaction-aware logistic regression: it keeps odds ratios that can be explained to non-technical audiences and regulators, and matches the boosted model closely.")
else:
    rec = ("the interaction-aware logistic regression is a useful, explainable improvement; weigh it against boosting using the use case's need for transparency.")
say("5. Recommendation: " + rec)

# ---- how much do three variables carry, and in what model form?  (test set, same split as everything else) ----
THREE = ["Employment Status", "Education Level", "Smoking Status"]
DEGREE = ["Bachelor's Degree", "Master's Degree", "PhD"]
def tier_of(Xin):
    un = Xin["Employment Status"] == "Unemployed"; ns = Xin["Smoking Status"] != "Current"
    deg = Xin["Education Level"].isin(DEGREE)
    return np.where(un & ns, "High", np.where(~un & deg & ns, "Low", "Middle"))
tier_rate = pd.Series(y_tr.values).groupby(tier_of(X_tr)).mean()          # group rates learned on TRAINING data only
auc_tier_rule = roc_auc_score(y_te, pd.Series(tier_of(X_te)).map(tier_rate).values)

def with_three_way(Xin):
    out = Xin.copy(); out["Emp x Edu x Smoke"] = Xin[THREE[0]] + "|" + Xin[THREE[1]] + "|" + Xin[THREE[2]]; return out
three_pairs = [(THREE[0], THREE[1]), (THREE[0], THREE[2]), (THREE[1], THREE[2])]
rows3 = []
def add3(label, model, Xa_tr, Xa_te):
    model.fit(Xa_tr, y_tr); rows3.append({"model": label, "test_auc": roc_auc_score(y_te, model.predict_proba(Xa_te)[:, 1])})
add3("Logistic, 3 variables, additive", make_linear([], THREE, best_C), X_tr[THREE], X_te[THREE])
Xa_tr, nm3 = add_crosses(X_tr[THREE], three_pairs, bins); Xa_te, _ = add_crosses(X_te[THREE], three_pairs, bins)
add3("Logistic, 3 variables + 3 pairwise interactions", make_linear([], THREE + nm3, C_int), Xa_tr, Xa_te)
add3("Logistic, 3 variables + full 3-way combination", make_linear([], THREE + ["Emp x Edu x Smoke"], C_int), with_three_way(X_tr[THREE]), with_three_way(X_te[THREE]))
rows3.append({"model": "Three-tier rule (lookup of training rates, no model)", "test_auc": auc_tier_rule})
rows3.append({"model": "Logistic, all 14 variables, additive", "test_auc": cmp_tab.test_auc.iloc[0]})
rows3.append({"model": "Logistic, all 14 + interactions (recommended)", "test_auc": cmp_tab.test_auc.iloc[1]})
rows3.append({"model": "Gradient boosting, all 14", "test_auc": cmp_tab.test_auc.iloc[2]})
three_tab = pd.DataFrame(rows3); display(three_tab.style.format({"test_auc": "{:.4f}"}))
three_tab.to_csv(os.path.join(OUT_DIR, "three_variable_check.csv"), index=False)

# ---- odds ratios for every term ----
def or_table(model, num_cols, cat_cols):
    pre = model.named_steps["pre"]; oh = pre.named_transformers_["cat"].named_steps["oh"]
    try: cn = list(oh.get_feature_names_out(cat_cols))
    except AttributeError: cn = list(oh.get_feature_names(cat_cols))
    coef = model.named_steps["clf"].coef_[0]
    t = pd.DataFrame({"term": list(num_cols) + cn, "coefficient": coef, "odds_ratio": np.exp(coef)})
    t["term"] = t.term.str.replace("_", " = ", n=1, regex=False)
    return t.reindex(t.coefficient.abs().sort_values(ascending=False).index).reset_index(drop=True)

or_add = or_table(models["Logistic regression"], NUM, CAT)
or_int = or_table(lin_int, NUM, CAT + names_best)
or_add.to_csv(os.path.join(OUT_DIR, "odds_ratios_additive.csv"), index=False); or_int.to_csv(os.path.join(OUT_DIR, "odds_ratios_interaction_model.csv"), index=False)
print(f"Additive logistic regression (C = {best_C:.4g}): odds ratio for every term, largest effects first.")
print("Numeric variables are standardised (odds ratio per 1 standard deviation). Category terms are one per level, so an odds ratio is that level's effect")
print("relative to the average over the other levels, shrunk by the L2 penalty. Not significance-tested: the model is regularised.")
display(or_add.style.format({"coefficient": "{:+.3f}", "odds_ratio": "{:.3f}"}))
print(f"\nInteraction-aware logistic regression (C = {C_int:.4g}): the cross terms (largest first).")
display(or_int[or_int.term.str.contains(" x ")].head(20).style.format({"coefficient": "{:+.3f}", "odds_ratio": "{:.3f}"}))

# ---- hyper-parameters: searched and chosen ----
final_hgb = models["Gradient boosting (HGB)"].named_steps["clf"]
hp = pd.DataFrame([
    ("Gradient boosting", "learning_rate", "0.01 to 0.3 (log scale)", best_params["learning_rate"]),
    ("Gradient boosting", "max_depth", "2 to 8", best_params["max_depth"]),
    ("Gradient boosting", "max_leaf_nodes", "8 to 64", best_params["max_leaf_nodes"]),
    ("Gradient boosting", "min_samples_leaf", "20 to 500", best_params["min_samples_leaf"]),
    ("Gradient boosting", "l2_regularization", "0.001 to 10 (log scale)", best_params["l2_regularization"]),
    ("Gradient boosting", "number of trees", "early stopping (max 500, patience 20)", final_hgb.n_iter_),
    ("Logistic regression (additive)", "C (inverse L2 strength)", "0.001 to 100 (log scale)", best_C),
    ("Logistic regression + interactions", "C (inverse L2 strength)", "0.001 to 100 (log scale)", C_int)],
    columns=["model", "hyper-parameter", "range searched", "chosen"])
hp["chosen"] = hp["chosen"].map(lambda v: f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v))
display(hp); hp.to_csv(os.path.join(OUT_DIR, "hyperparameters.csv"), index=False)
print("Fixed settings: logistic regression uses the L2 penalty, lbfgs solver, max_iter 2000; gradient boosting uses early stopping on a 10% internal validation split.")
print(f"Search: {N_TRIALS_HGB} + {N_TRIALS_LR} Bayesian trials, {CV_FOLDS}-fold CV, {len(X_tune):,} training rows.")

# ---- split sizes, target rate and feature balance across train / validation / test ----
sets = [("Train", X_tr, y_tr), ("Validation", X_va, y_va), ("Test", X_te, y_te)]
split_tab = pd.DataFrame([{"set": n, "rows": len(y), "share_of_data": len(y) / len(df), "with_history": int(y.sum()), "positive_rate": y.mean()} for n, X_, y in sets])
display(split_tab.style.format({"rows": "{:,}", "share_of_data": "{:.1%}", "with_history": "{:,}", "positive_rate": "{:.2%}"}))
split_tab.to_csv(os.path.join(OUT_DIR, "split_summary.csv"), index=False)
print(f"Overall rate {y.mean():.2%} -> spread across sets {100 * (split_tab.positive_rate.max() - split_tab.positive_rate.min()):.2f} percentage points (stratified on the target).")

bal = []
for c in CAT:
    d = pd.DataFrame({n: X_[c].value_counts(normalize=True) for n, X_, _ in sets}).fillna(0)
    bal.append({"feature": c, "type": "category", "largest_gap": 100 * (d.max(axis=1) - d.min(axis=1)).max(), "unit": "points of category share"})
for c in NUM:
    m = [X_[c].mean() for _, X_, _ in sets]
    bal.append({"feature": c, "type": "numeric", "largest_gap": 100 * (max(m) - min(m)) / X_tr[c].std(), "unit": "% of a standard deviation (mean)"})
bal = pd.DataFrame(bal).sort_values("largest_gap", ascending=False); display(bal.style.format({"largest_gap": "{:.2f}"}))
bal.to_csv(os.path.join(OUT_DIR, "feature_balance_across_splits.csv"), index=False)
print(f"Largest feature difference between any two sets: {bal.largest_gap.max():.2f} ({bal.iloc[0].unit}) for {bal.iloc[0].feature}.")

# ---- confusion matrix, lift chart and precision-recall curve: recommended model on the test set ----
pred_i = (pt_int >= thr_int).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, pred_i, labels=[0, 1]).ravel()
pd.DataFrame([{"tn": tn, "fp": fp, "fn": fn, "tp": tp, "threshold": thr_int}]).to_csv(os.path.join(OUT_DIR, "recommended_confusion.csv"), index=False)
dd = pd.DataFrame({"y": y_te.values, "p": pt_int}); dd["decile"] = pd.qcut(dd.p.rank(method="first"), 10, labels=False) + 1
lift_tab = dd.groupby("decile").agg(rate=("y", "mean"), n=("y", "size")); lift_tab["lift"] = lift_tab.rate / y_te.mean()
lift_tab.to_csv(os.path.join(OUT_DIR, "recommended_lift.csv"))
pr_p, pr_r, _ = precision_recall_curve(y_te, pt_int); keep = np.linspace(0, len(pr_p) - 1, 400).astype(int)
pd.DataFrame({"precision": pr_p[keep], "recall": pr_r[keep]}).to_csv(os.path.join(OUT_DIR, "recommended_pr_curve.csv"), index=False)

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
cm_ = np.array([[tn, fp], [fn, tp]])
axes[0].imshow(cm_, cmap="YlOrBr")
for i in range(2):
    for j in range(2): axes[0].text(j, i, f"{cm_[i, j]:,}\n({cm_[i, j] / cm_[i].sum():.0%} of row)", ha="center", va="center", fontsize=12)
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["Predicted No", "Predicted Yes"]); axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["Actual No", "Actual Yes"])
axes[0].set_title(f"Confusion matrix (threshold {thr_int:.2f})")
axes[1].bar(lift_tab.index, lift_tab.lift, color="#2E8B8B"); axes[1].axhline(1, color="k", lw=1, ls="--")
axes[1].set_xlabel("Score decile (1 = lowest 10%)"); axes[1].set_ylabel("Rate / average rate"); axes[1].set_title("Lift by decile")
axes[2].plot(pr_r, pr_p, color="#D9922B", lw=2); axes[2].axhline(y_te.mean(), color="k", ls="--", lw=1, label=f"no skill ({y_te.mean():.2f})")
axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision"); axes[2].set_title(f"Precision-recall (PR-AUC {average_precision_score(y_te, pt_int):.3f})"); axes[2].legend()
plt.tight_layout(); savefig("14_recommended_model_curves.png"); plt.show()

# ---- final settings of both models, in full ----
import json
def pick(d, keys): return {k: d[k] for k in keys if k in d}
final_settings = {
    "logistic_additive": pick(models["Logistic regression"].named_steps["clf"].get_params(), ["penalty", "C", "solver", "max_iter", "tol", "class_weight", "fit_intercept"]),
    "logistic_interactions": {**pick(lin_int.named_steps["clf"].get_params(), ["penalty", "C", "solver", "max_iter", "tol", "class_weight", "fit_intercept"]),
                              "interaction_pairs": [f"{a} x {b}" for a, b in cand_pairs[:K_best]]},
    "gradient_boosting": {**pick(final_hgb.get_params(), ["learning_rate", "max_depth", "max_leaf_nodes", "min_samples_leaf", "l2_regularization", "max_iter", "early_stopping", "validation_fraction",
                                                        "n_iter_no_change", "scoring", "loss", "max_bins", "random_state"]), "trees_actually_built": int(final_hgb.n_iter_)}}
final_settings["preprocessing"] = {"logistic": "numeric: median impute + standardise; categories: most-frequent impute + one-hot (all levels)",
                                   "gradient_boosting": "numeric: median impute; categories: ordinal codes treated natively as categorical"}
for k, v in final_settings.items(): print(k, "->", v)
json.dump(final_settings, open(os.path.join(OUT_DIR, "final_model_settings.json"), "w"), indent=1, default=str)

audit = df.loc[X_te.index, ["Age Band", "Marital Status", "Education Level", "Employment Status", "Income Quartile"]].copy()
for c in audit.columns:
    audit[c] = audit[c].astype(str)

def group_report(attr, y_true, p, threshold, min_n=100):
    d = pd.DataFrame({"g": audit[attr].values, "y": np.asarray(y_true), "p": p})
    rows = []
    for g, s in d.groupby("g"):
        pred = (s.p >= threshold)
        pos, neg = (s.y == 1).sum(), (s.y == 0).sum()
        rows.append(dict(group=g, n=len(s), share=len(s) / len(d), base_rate=s.y.mean(),
                         selection_rate=pred.mean(),
                         TPR=(pred & (s.y == 1)).sum() / pos if pos else np.nan,
                         FPR=(pred & (s.y == 0)).sum() / neg if neg else np.nan,
                         precision=(s.y[pred] == 1).mean() if pred.sum() else np.nan,
                         AUC=roc_auc_score(s.y, s.p) if (pos > 0 and neg > 0 and len(s) >= min_n) else np.nan,
                         mean_pred=s.p.mean(), calib_gap=s.p.mean() - s.y.mean()))
    return pd.DataFrame(rows)

def gap_summary(y_true, p, threshold):
    out = []
    for a in audit.columns:
        r = group_report(a, y_true, p, threshold); r = r[r.n >= 100]
        out.append(dict(attribute=a, groups=len(r),
                        disparate_impact_ratio=r.selection_rate.min() / r.selection_rate.max() if r.selection_rate.max() > 0 else np.nan,
                        TPR_gap=r.TPR.max() - r.TPR.min(), FPR_gap=r.FPR.max() - r.FPR.min(),
                        max_abs_calib_gap=r.calib_gap.abs().max(),
                        AUC_range=r.AUC.max() - r.AUC.min()))
    g = pd.DataFrame(out)
    g["flags"] = g.apply(lambda r: ", ".join([f for f, c in [("DI<0.8", r.disparate_impact_ratio < 0.8), ("TPR gap>5pp", r.TPR_gap > 0.05),
                                              ("FPR gap>5pp", r.FPR_gap > 0.05), ("calib>2pp", r.max_abs_calib_gap > 0.02)] if c]) or "none", axis=1)
    return g

full_gaps = gap_summary(y_te, pf, thr[final_name])
print(f"FAIRNESS SCREEN: {final_name}, test set, threshold {thr[final_name]:.2f} ({thr_label[final_name]}); groups with n >= 100")
display(full_gaps.style.format({c: "{:.3f}" for c in full_gaps.columns if c not in ["attribute", "groups", "flags"]}))

# ---- per-group detail + plots ----
fig, axes = plt.subplots(1, 5, figsize=(26, 4.5))
for ax, a in zip(axes, audit.columns):
    r = group_report(a, y_te, pf, thr[final_name]); r = r[r.n >= 100]
    w = 0.27; x = np.arange(len(r))
    ax.bar(x - w, r.selection_rate, w, label="Selection rate", color="#1B2A4A")
    ax.bar(x, r.TPR, w, label="TPR", color="#2E8B8B")
    ax.bar(x + w, r.FPR, w, label="FPR", color="#D9922B")
    ax.set_xticks(x); ax.set_xticklabels(r.group, rotation=25, ha="right"); ax.set_title(a)
axes[0].legend(fontsize=8)
plt.suptitle(f"Group-level behaviour of the {final_name} model at threshold {thr[final_name]:.2f}", y=1.03)
plt.tight_layout(); savefig("12_fairness_by_group.png"); plt.show()

for a in audit.columns:
    print(f"\n--- {a} ---")
    display(group_report(a, y_te, pf, thr[final_name]).style.format(
        {c: "{:.3f}" for c in ["share", "base_rate", "selection_rate", "TPR", "FPR", "precision", "AUC", "mean_pred", "calib_gap"]}, na_rep="-"))

# ---- does removing the sensitive attributes help?  (fairness through unawareness + proxy test) ----
NUM_R = [c for c in NUM if c != "Age"]; CAT_R = [c for c in CAT if c != "Marital Status"]
restricted = make_final(NUM_R, CAT_R).fit(X_tr[NUM_R + CAT_R], y_tr)
pr_r = restricted.predict_proba(X_te[NUM_R + CAT_R])[:, 1]
thr_r, thr_r_label = choose_threshold(y_va, restricted.predict_proba(X_va[NUM_R + CAT_R])[:, 1])
m_r = evaluate(y_te, pr_r, thr_r)
cmp = pd.DataFrame({"Full model": m_f, "Without Age & Marital Status": m_r}).loc[["roc_auc", "pr_auc", "recall", "precision", "f1"]].T
res_gaps = gap_summary(y_te, pr_r, thr_r)
cmp["worst disparate-impact ratio"] = [full_gaps.disparate_impact_ratio.min(), res_gaps.disparate_impact_ratio.min()]
cmp["largest TPR gap"] = [full_gaps.TPR_gap.max(), res_gaps.TPR_gap.max()]
cmp["largest FPR gap"] = [full_gaps.FPR_gap.max(), res_gaps.FPR_gap.max()]
display(cmp.style.format("{:.3f}"))

# proxy test: can the remaining variables still predict who is 45 or over?
age_old = (df.loc[X_tr.index, "Age"] >= 45).astype(int); age_old_te = (df.loc[X_te.index, "Age"] >= 45).astype(int)
proxy = make_hgb(NUM_R, CAT_R, learning_rate=0.1, max_depth=4).fit(X_tr[NUM_R + CAT_R], age_old)
proxy_auc = roc_auc_score(age_old_te, proxy.predict_proba(X_te[NUM_R + CAT_R])[:, 1])
print(f"Proxy test: predicting 'age >= 45' from the remaining variables gives AUC {proxy_auc:.3f} (0.5 = no proxy information).")

# ---- auto-generated bias narrative ----
print("=" * 110); print("BIAS TAKEAWAYS (generated from the data)"); print("=" * 110)
small = [(a, r.group, r.n) for a in audit.columns for r in group_report(a, y_te, pf, thr[final_name]).itertuples() if r.n < 500]
say("1. Representation: " + (("small test-set groups (n<500) with noisy metrics: " + "; ".join(f"{a}={g} (n={n})" for a, g, n in small) + ".") if small else "no group in the audit has fewer than 500 test rows."))
flagged = full_gaps[full_gaps["flags"] != "none"]
if len(flagged):
    for r in flagged.itertuples():
        say(f"2. {r.attribute}: flags raised -> {r.flags} (DI ratio {r.disparate_impact_ratio:.2f}, TPR gap {r.TPR_gap:.3f}, FPR gap {r.FPR_gap:.3f}, "
            f"max calibration gap {r.max_abs_calib_gap:.3f}).")
else:
    say("2. No attribute breached the screening thresholds. This is reassuring but not proof of fairness: the thresholds are coarse, subgroup sizes limit power, and the audit only covers the attributes listed.")
if auc_pt < 0.60:
    say("   Caution: because the model's scores carry little signal, who lands above the threshold is decided by tiny score differences, so "
        "selection-rate and error-rate gaps between groups are unstable and easy to over-read. Per-group AUC and calibration are more reliable here.")
say(f"3. Removing Age and Marital Status changes ROC-AUC from {m_f['roc_auc']:.3f} to {m_r['roc_auc']:.3f}. "
    f"A proxy test finds the remaining variables predict 'age >= 45' with AUC {proxy_auc:.3f}"
    + (", so age-related information leaks through other variables; removing the column alone does not remove age-related effects." if proxy_auc >= 0.6 else ", so there is little age information left in the other variables."))
say("4. Label bias cannot be measured from this data: if some groups are less likely to be diagnosed or to disclose a history of mental illness, the model learns those reporting patterns, not true prevalence.")
say("5. Synthetic data: group differences here reflect how the dataset was generated and should not be read as real-world differences between groups.")

# ---- the same fairness screen for the interaction-aware logistic regression ----
gaps_int = gap_summary(y_te, pt_int, thr_int)
print(f"FAIRNESS SCREEN: interaction-aware logistic regression, test set, threshold {thr_int:.2f} ({thr_int_label})")
display(gaps_int.style.format(precision=3, na_rep="-")); gaps_int.to_csv(os.path.join(OUT_DIR, "fairness_interaction_model.csv"), index=False)
for a in ["Age Band", "Employment Status"]:
    display(group_report(a, y_te, pt_int, thr_int)[["group", "n", "base_rate", "selection_rate", "TPR", "FPR", "AUC", "mean_pred", "calib_gap"]]
            .style.format(precision=3, na_rep="-"))

# ---- plain-English answer to the three brief questions (generated from this run) ----
print("=" * 110); print("ANSWER SUMMARY (generated from this run; edit the wording before presenting)"); print("=" * 110)
say(f"Q1 Insights: {prev:.1%} of {len(df):,} people have a history of mental illness. The strongest univariate associations are "
    + "; ".join(f"{r.feature} ({r.strength}, {r.effect_size:.2f})" for r in assoc.head(3).itertuples())
    + f". Effect sizes are used instead of p-values because with this many rows almost everything is 'significant'.")
say(f"Q2 Model: {final_name} was chosen on validation data over {other_name} and a majority baseline. On unseen test data it reaches ROC-AUC {auc_pt:.3f} "
    f"(95% CI {ci(auc_f)[0]:.3f} to {ci(auc_f)[1]:.3f}) and PR-AUC {pr_pt:.3f} against a prevalence of {y_te.mean():.3f}; "
    f"the top decile has {lift['lift'].iloc[-1]:.2f}x the average rate. Performance is {auc_band(auc_pt)}"
    + (", so the model is useful for ranking groups at best and not for individual decisions." if auc_pt < 0.75 else "."))
say(f"Q2 Bias: {int((full_gaps['flags'] != 'none').sum())} of {len(full_gaps)} audited attributes raised at least one screening flag; "
    f"the proxy test for age gives AUC {proxy_auc:.3f}. Label bias cannot be measured from this data.")
say("Q3 Limitations: synthetic and cross-sectional data, coarse features, no timing, a recorded-history label, single random split, "
    "and ethical/regulatory constraints on using health data. Improvements: longitudinal and clinical data, external validation, "
    "calibrated probabilities, costs-based thresholds, fairness constraints and governance (Section 8).")

# ---- headline numbers for the slides ----
print("=" * 110); print("HEADLINE NUMBERS FOR THE PRESENTATION"); print("=" * 110)
summary = pd.DataFrame([
    ("Rows analysed", f"{len(df):,}"),
    ("Positive rate (history of mental illness)", f"{prev:.1%}"),
    ("Strongest predictor (effect size)", f"{assoc.feature.iloc[0]} ({assoc.effect_size.iloc[0]:.3f}, {assoc.strength.iloc[0]})"),
    ("Final model", f"{final_name} (chosen on validation data over {other_name} and a majority baseline)"),
    ("Test ROC-AUC (95% CI)", f"{auc_pt:.3f} [{ci(auc_f)[0]:.3f}, {ci(auc_f)[1]:.3f}]"),
    ("Test PR-AUC vs prevalence", f"{pr_pt:.3f} vs {y_te.mean():.3f}"),
    (f"Recall / precision at operating threshold ({thr_label[final_name]})", f"{m_f['recall']:.0%} / {m_f['precision']:.0%}"),
    ("Top-decile lift", f"{lift['lift'].iloc[-1]:.2f}x"),
    ("Worst disparate-impact ratio / largest TPR gap", f"{full_gaps.disparate_impact_ratio.min():.2f} / {full_gaps.TPR_gap.max():.3f}"),
], columns=["Item", "Value"])
display(summary); summary.to_csv(os.path.join(OUT_DIR, "headline_numbers.csv"), index=False)
print("\nFigures saved in ./figures and tables in ./outputs. In Colab, download them from the Files panel.")

