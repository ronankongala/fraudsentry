# Data Protection Impact Assessment (DPIA)
## FraudSentry: Transaction Fraud Detection Pipeline

**Prepared per GDPR Article 35** (Data Protection Impact Assessment) and
informed by Article 25 (Data Protection by Design and by Default).

**Status:** This DPIA was produced for a personal project. The pipeline
was run on synthetic data and on the public IEEE-CIS Fraud Detection
dataset (see "Data Source Disclosure" in the project README). It follows
the Article 35 structure and reasoning a production deployment would
require, but the processing described is simulated, not live.

---

### 1. Description of Processing

**Nature:** Automated analysis of transaction records to detect potentially
fraudulent activity, using supervised (Logistic Regression, Random Forest,
gradient boosting via HistGradientBoosting or XGBoost) and unsupervised (Isolation Forest) machine learning
models, followed by human analyst review of flagged transactions.

**Scope:** Transaction-level data (amount, merchant category, timestamp,
transaction country) and account-level derived features (transaction
velocity, historical amount deviation, device novelty, home country).
No cardholder names, card numbers, CVVs, or full account numbers are
processed by the model, only the `customer_id` pseudonym and derived
behavioral features.

**Context:** Financial services / fraud prevention. Data subjects are
the organization's customers whose transactions are automatically
screened.

**Purpose:** Reduce financial loss and customer harm from fraudulent
transactions by identifying and interrupting fraud patterns in near
real time, consistent with the legitimate interest and (where
applicable) legal obligation bases for processing under **GDPR Article
6(1)(f)** (legitimate interests) and **Article 6(1)(c)** (compliance
with a legal obligation, e.g., anti-fraud regulatory expectations in
the financial sector).

---

### 2. Necessity and Proportionality Assessment

| Question | Assessment |
|---|---|
| Is automated processing necessary for the purpose? | Yes. Manual review of 100% of transactions at this volume (150,000 in the synthetic dataset) is not operationally feasible, and the base fraud rate (~0.45%) requires automated triage to make investigation feasible at scale. |
| Is the data collected proportionate? | Yes, with a caveat. The feature set (amount, category, timing, device, geography) is the minimum needed to detect the fraud patterns in scope. No unrelated data (e.g., browsing history, social data) is used. |
| Could a less privacy-invasive method achieve the same result? | Partially. Rule-based systems could catch some patterns with less data, but published fraud-detection literature reports that ML models outperform static rules at equivalent false-positive rates. That argues for the data-driven approach despite the added processing, provided the safeguards below are in place. This project did not benchmark a rule-based baseline; `results/metrics.json` compares ML models only. |
| Is there a fully automated *decision* with legal or similarly significant effect (GDPR Art. 22)? | **No.** The system flags transactions for human analyst review (see `case_tracking.py`); it does not autonomously block transactions, close accounts, or make a final fraud determination. This design choice is meant to keep Article 22 out of scope. If a future version moved to fully automated blocking, this DPIA section would need to be revisited and Article 22 safeguards (explanation, human review on request) added explicitly. |

---

### 3. Identified Risks to Data Subjects

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| False positive flags a legitimate transaction, causing customer inconvenience (declined transaction, account friction) | Medium (at the tuned 3% FPR, ~3 in 100 legitimate transactions are flagged) | Low-Medium | Human analyst review before any customer-facing action; local explanation (`explainability.py`) gives the analyst the "why" to resolve quickly; false positives are logged and used to refine the model. |
| Model bias produces disproportionate false-positive rates for a subset of customers (e.g., frequent international travelers, certain merchant categories) | Medium. Subgroup audits were run on synthetic and IEEE-CIS data (see Section 6); the IEEE-CIS run found a 23.7-point FPR spread across merchant categories | Medium | `src/fairness_audit.py` compares subgroup false-positive rates at the 3% FPR operating point. The merchant-category disparity must be investigated before any production use, and the audit should be rerun on each retrained model. Geography could not be tested on IEEE-CIS data (see Section 6). |
| Re-identification of a customer from the pseudonymous `customer_id` combined with transaction metadata | Low in isolation, higher if combined with external data | Medium-High | `customer_id` is a synthetic identifier in the synthetic dataset and a hash of card and address fields in the IEEE-CIS run, with no direct mapping to name/SSN/card number; production systems must ensure this pseudonym cannot be trivially re-linked without going through access-controlled, audited identity-resolution systems. |
| Data retained longer than necessary for the fraud-detection purpose | Not yet defined in this project | Medium | See Section 4 (Retention) below for a proposed policy. |
| Case notes (`case_tracking.py`) contain analyst commentary that could include sensitive inferences about a customer | Low-Medium | Medium | Recommend a periodic manual review of `notes` fields for anything beyond fraud-relevant fact, and restrict `notes` field access to the fraud-ops role only (not general company access). |

---

### 4. Data Minimization, Retention, and Lawful Basis

- **Minimization:** Only behavioral/derived features are modeled; no direct identifiers (name, full card number, SSN) enter the model or the alert database.
- **Retention (proposed policy, not currently implemented in code):** Raw transaction data feeding the model: retained per the organization's existing financial recordkeeping requirements (commonly 5 to 7 years in many jurisdictions for AML/fraud purposes). This is a *legal retention basis*, not indefinite convenience storage. Alert/case records (`alerts` table): recommend retaining resolved false-positive alerts for a much shorter window (e.g., 90 days) once resolved, since they carry lower ongoing fraud-prevention value than confirmed cases, which may need to be retained longer to satisfy the same recordkeeping requirements as the underlying transaction.
- **Lawful basis:** Legitimate interest (Art. 6(1)(f)) for the core fraud-detection processing, balanced against data subject rights via this DPIA; legal obligation (Art. 6(1)(c)) where sector-specific anti-fraud regulation applies.

---

### 5. Data Subject Rights Operationalization

See `data_subject_rights.py` for a working implementation of:
- **Right of access (Art. 15):** export all records tied to a given `customer_id`.
- **Right to erasure (Art. 17):** remove a customer's records from the alert/case database, with the retention-conflict check described in that script (erasure requests can conflict with legal retention obligations above, and the script surfaces that conflict rather than silently ignoring it).

---

### 6. Fairness Audit Results and Known Gaps

**Subgroup fairness audits.** Both audits compare false-positive rates
across subgroups of legitimate transactions at the threshold that gives a
3% overall FPR. They check proxies available in the data (merchant
category, geography mismatch, device novelty), not protected
characteristics, which transaction data does not contain.

- **Synthetic data** (`src/fairness_audit_offline.py`, Random Forest
  model, results in `results/fairness_audit_synthetic.json`). Legitimate
  transactions with a country mismatch had a 54.4% FPR, against 0.30% for
  matched-country transactions, a 54-point spread. Merchant categories
  showed a 6.7-point spread, with online retail (6.9%) and travel (5.4%)
  highest. The synthetic fraud signal was hand-designed to correlate with
  these features, so this result is partly circular and says little about
  customer impact.
- **IEEE-CIS data** (`src/fairness_audit.py`, XGBoost model, results in
  `results/fairness_audit.json`). Merchant category shows a 23.7-point
  FPR spread: legitimate `electronics` transactions are flagged at 23.9%
  and `grocery` at 17.7%, against 0.27% for `travel` and 0.24% for
  `online_retail`. Merchant category is not a protected attribute, and
  part of the spread may reflect differing fraud base rates, but customers
  who shop in those categories would face far more transaction friction.
  This is the open fairness risk to resolve before any production use.
  New-device and existing-device transactions differ by 1.6 points (2.1%
  vs. 3.7%).
- **Geography could not be tested on IEEE-CIS data.** The pseudo-customer
  ID built by `data/load_ieee_cis.py` makes a customer's derived home
  country almost always equal to the transaction country, so every
  legitimate test transaction has `geo_mismatch` = 0 and there is no
  subgroup to compare. This is a limit of the reconstruction, not
  evidence of geographic fairness. A dataset with a stable customer
  identifier would be needed to check whether international or
  frequent-traveler customers are flagged more often.

**Other gaps.**

- **No Data Protection Officer (DPO) or supervisory authority consultation occurred**, since this is a personal project, not a live deployment. A production deployment under GDPR would require the organization's DPO to review this DPIA, and consultation with the supervisory authority if residual risk remains high after mitigation (Art. 36).
- **No Privacy Enhancing Technology (e.g., differential privacy) is implemented in model training.** The safeguards are architectural and procedural only (pseudonymization, minimization, human-in-the-loop review). This limitation is unresolved.
