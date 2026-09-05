# Running FraudSentry Locally With Real Libraries and Real Data

This sandbox has no internet access, so the version you have was built
with disclosed substitutes (from-scratch SMOTE, HistGradientBoosting,
permutation importance) and synthetic data. This guide gets you to the
real versions on your own machine.

## Which Kaggle dataset to use (and which NOT to)

**Use: IEEE-CIS Fraud Detection** (`ieee-fraud-detection` competition
dataset). It has `DeviceType`/`DeviceInfo`, `ProductCD` (category-like),
card/issuer fields, `addr1`/`addr2` (geography proxy), and
`TransactionDT` (relative time) — these map onto the features this
project already engineers (device novelty, merchant category, geo,
time-of-day, velocity).

**Do NOT use** the simpler "Credit Card Fraud Detection" dataset
(`mlg-ulb/creditcardfraud`) for this project — it only contains
anonymized PCA components (V1-V28) + Amount + Time, with no customer
identity, device, or category fields. It's a fine dataset for a
different, simpler project, but it can't support the feature set
(velocity, geo mismatch, device novelty) this pipeline is built around.

IEEE-CIS has no explicit `customer_id` column either — it's a known
gap in the dataset itself. The Kaggle community's standard workaround
(used in multiple public notebooks for this competition) is to
reconstruct a pseudo-customer identity from a hash of
`card1, card2, card3, card5, addr1, D1` (D1 approximates "days since
account was opened," which stabilizes the grouping). `load_ieee_cis.py`
below implements exactly that, and says so in a comment — this is a
disclosed heuristic, not a guaranteed-correct customer identity.

## Step 1 — Install the real packages

```bash
cd fraudsentry
pip install -r requirements-local.txt
```

`requirements-local.txt` adds `xgboost`, `shap`, `imbalanced-learn`,
and `kaggle` on top of what you already have (pandas, scikit-learn).

## Step 2 — Get Kaggle API credentials

1. Go to https://www.kaggle.com/settings/account -> "Create New API Token"
2. This downloads `kaggle.json`. Move it to `~/.kaggle/kaggle.json`
   (Linux/Mac) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows).
3. You'll also need to accept the competition rules on the
   `ieee-fraud-detection` competition page on Kaggle (required by
   Kaggle before the API will let you download it — it's free, just a
   click-through).

## Step 3 — Download and prepare the real data

```bash
python3 data/load_ieee_cis.py
```

This downloads `train_transaction.csv` and `train_identity.csv` via
the Kaggle API, joins them, reconstructs the pseudo-customer ID
described above, and writes `data/features.csv` in the same schema
`feature_engineering.py` already expects -- so `train_models.py`
downstream does not need to change.

## Step 4 — Run the real-library pipeline

```bash
python3 src/train_models_real.py       # XGBoost + imblearn.SMOTE
python3 src/explainability_real.py     # SHAP TreeExplainer
python3 src/fairness_audit.py          # subgroup false-positive-rate check
```

Each of these is a parallel file to the offline version (not an
in-place edit), so you can compare the two runs directly if you want
to see how much the real libraries change the results.

## What to expect to be different

- **SHAP** will give you real per-alert Shapley-value attributions
  (accounting for feature interactions) instead of the deviation
  heuristic -- this is the most meaningful upgrade of the three.
- **XGBoost** vs. HistGradientBoosting: likely close in performance,
  since they're the same algorithm family, but worth comparing.
- **imblearn's SMOTE** vs. this project's from-scratch version: should
  produce near-identical synthetic samples, since it's the same
  published algorithm -- this is more a "use the standard library"
  upgrade than a results upgrade.
- **Real IEEE-CIS data** will very likely show DIFFERENT top predictive
  features than the synthetic data, since the synthetic fraud signal
  was hand-designed. Don't be surprised if `geo_mismatch` and
  `new_device` aren't the top two anymore -- that's the real dataset
  telling you something the synthetic one couldn't.
