# AXA Health data scientist task: Use Case 2, predicting mental health

Predict `History of Mental Illness` from the Kaggle Depression Dataset
(https://www.kaggle.com/datasets/anthonytherrien/depression-dataset) and assess performance and bias.

## Contents
| Path | What it is |
|---|---|
| `notebooks/AXA_Use_Case_2_Mental_Health_Prediction.ipynb` | The full answer: EDA (Q1), variable narrative, model choice, performance, bias (Q2) and limitations (Q3). Runs in Colab. |
| `notebooks/02_Interaction_Discovery.ipynb` | Standalone deep dive on feature interactions: depth-1/2/3 boosting ladder, Friedman H-statistic, risk heatmaps, and cross-features added to logistic regression. Uses `src/`; set `DATA_PATH` in the first cell. About 2 minutes. |
| `src/` | Reusable code: `data.py` (loading, target, features), `models.py` (logistic regression, gradient boosting), `tuning.py` (Bayesian optimisation with cross-validation), `interactions.py` (H-statistic interaction strength and cross-features), `evaluation.py` (metrics, thresholds, bootstrap), `fairness.py` (group audit). |
| `tests/` | pytest tests on a small synthetic dataset (schema checks, model sanity, metrics, threshold fallback, fairness flags). |
| `scripts/` | Extra analysis used for the presentation: statistical tests, missing-data checks, income/employment overlap, risk tiers and slide charts. Run from the repo root, in this order: `python scripts/eda_for_deck.py depression_data.csv`, then `python scripts/extras_for_deck.py depression_data.csv`, then `python scripts/slide_charts.py depression_data.csv` and `slide_charts2.py` to `slide_charts6.py` (the later ones read `deck_figs/tests.csv` and the tables in `results/tables/`). Output goes to `deck_figs/` and `slide_figs/`, which are created if missing. |
| `results/` | Output of the notebook run on the real dataset (all figures, tables and the printed commentary). |
| `run_analysis.py` | Minimal command-line run with the logistic regression model. |
| `AXA_Use_Case_2_Mental_Health_Prediction.py` | The full notebook exported as a plain Python script. |
| `run_full_analysis.py` | Runs the exported full-analysis script from a terminal without pop-up windows. |
| `requirements.txt` | Libraries needed. |

## Run
```
pip install -r requirements.txt
pytest -q                                   # tests
python run_analysis.py depression_data.csv  # quick run
```
For the full analysis from a terminal, put `depression_data.csv` in the folder and run `python run_full_analysis.py` (it reads `AXA_Use_Case_2_Mental_Health_Prediction.py`, the exported notebook; about 5 minutes; figures go to `figures/`, tables to `outputs/`). Or open the notebook in Colab (Runtime -> Run all). It downloads the data with `kagglehub`, or asks you to upload the CSV.

## Approach in brief
1. Check the data, then EDA ranked by effect size (Cramer's V, rank-biserial) because p-values are meaningless at about 414k rows.
2. Choose variables on principle (drop `Name`, log-transform `Income`, flag `Age` and `Marital Status` as protected characteristics).
3. Compare a majority baseline, regularised logistic regression and histogram gradient boosting on a 60/20/20 split. Hyper-parameters are tuned by Bayesian optimisation (Optuna if installed, otherwise a built-in Gaussian-process optimiser) with 5-fold cross-validation on the training set. The final model is chosen on validation data only, preferring the simpler model unless the complex one is reliably better.
4. Judge performance with ROC-AUC, PR-AUC, calibration, lift, bootstrap intervals, a shuffled-label control and single-variable AUCs.
5. Audit bias by group (selection rate, TPR, FPR, calibration), test fairness through unawareness and test for proxies.
6. Explore what the model is really finding: a shallow decision tree shows the signal is three simple risk tiers (employment, education, smoking); income and age add nothing within a tier.
7. Look for feature interactions (boosted trees by depth, H-statistic, risk heatmaps, SHAP with XGBoost) and feed the strongest ones into the logistic regression as cross-features, keeping an explainable model.
8. State limitations: the data is described as synthetic, cross-sectional, with coarse features and a recorded-history label. Mental-health data is special-category data under UK GDPR, so this exercise must not be used for real decisions.
