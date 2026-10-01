# Running FraudSentry With IEEE-CIS Data and the Standard Libraries

The default pipeline uses synthetic data and offline substitutes
(from-scratch SMOTE, HistGradientBoosting, permutation importance). This
guide covers the alternative pipeline, which uses the IEEE-CIS dataset
with XGBoost, `imbalanced-learn` and SHAP. It needs internet access and a
Kaggle account.

## Which Kaggle dataset to use

**Use IEEE-CIS Fraud Detection** (the `ieee-fraud-detection` competition
dataset). It has `DeviceType`/`DeviceInfo`, `ProductCD` (category-like),
card/issuer fields, `addr1`/`addr2` (geography proxy), and
`TransactionDT` (relative time). These map onto the features this
project engineers: device novelty, merchant category, geo, time-of-day
and velocity.

**Do not use** the simpler "Credit Card Fraud Detection" dataset
(`mlg-ulb/creditcardfraud`). It contains only anonymized PCA components
(V1-V28) plus Amount and Time, with no customer identity, device, or
category fields, so it can't support the velocity, geo mismatch and
device novelty features this pipeline is built around.

IEEE-CIS has no `customer_id` column either; the gap is in the dataset
itself. A common workaround in public Kaggle notebooks for this
competition is to reconstruct a pseudo-customer identity from a hash of
`card1, card2, card3, card5, addr1, D1` (D1 approximates "days since
account was opened," which stabilizes the grouping). `load_ieee_cis.py`
implements that and says so in a comment. It is a heuristic, not a
guaranteed-correct customer identity.

## Step 1: Install the packages

```bash
cd fraudsentry
pip install -r requirements-local.txt
```

`requirements-local.txt` adds `xgboost`, `shap`, `imbalanced-learn`,
and `kaggle` to the base dependencies (pandas, numpy, scikit-learn).

## Step 2: Get Kaggle API credentials

1. Go to https://www.kaggle.com/settings/account -> "Create New API Token"
2. This downloads `kaggle.json`. Move it to `~/.kaggle/kaggle.json`
   (Linux/Mac) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows).
3. Accept the competition rules on the `ieee-fraud-detection`
   competition page. Kaggle requires this (a free click-through) before
   the API allows the download.

## Step 3: Download and prepare the data

```bash
python3 data/load_ieee_cis.py
```

This downloads `train_transaction.csv` and `train_identity.csv` via the
Kaggle API, joins them, reconstructs the pseudo-customer ID described
above, and writes `data/features.csv` in the schema
`feature_engineering.py` produces, so the downstream training code does
not need to change.

## Step 4: Run the pipeline

```bash
python3 src/train_models_real.py       # XGBoost + imblearn.SMOTE
python3 src/explainability_real.py     # SHAP TreeExplainer
python3 src/fairness_audit.py          # subgroup false-positive-rate check
python3 src/visualize_real.py          # plots in results/plots_real/
```

Each script is a separate file from its offline counterpart, so the two
runs can be compared directly.

## Expected differences from the synthetic run

- **SHAP** gives per-alert Shapley-value attributions that account for
  feature interactions, in place of the deviation heuristic. Of the three
  library swaps, this one changes the output most.
- **XGBoost** and HistGradientBoosting come from the same algorithm
  family, so performance should be close, though it is worth comparing.
- **imblearn's SMOTE** implements the same published algorithm as this
  project's from-scratch version, so the synthetic samples should be
  near-identical. The swap is about using the standard library more than
  about changing results.
- **IEEE-CIS data** is likely to show different top predictive features
  than the synthetic data, since the synthetic fraud signal was
  hand-designed. In the run recorded in the README, the top SHAP features
  were `amount`, `hour_of_day` and `merchant_category_electronics`, and
  `geo_mismatch` carried no signal because of the customer-ID
  reconstruction.
