# Data Protection Impact Assessment (DPIA)
## FraudSentry — Transaction Fraud Detection Pipeline

**Prepared per GDPR Article 35** (Data Protection Impact Assessment) and
informed by Article 25 (Data Protection by Design and by Default).

**Status:** This DPIA was produced for a personal project using synthetic
data (see Data Source Disclosure below). It follows the real Article 35
structure and reasoning a production deployment would require, but the
processing described is simulated, not live.

---

### 1. Description of Processing

**Nature:** Automated analysis of transaction records to detect potentially
fraudulent activity, using supervised (Logistic Regression, Random Forest,
Gradient Boosting) and unsupervised (Isolation Forest) machine learning
models, followed by human analyst review of flagged transactions.

**Scope:** Transaction-level data (amount, merchant category, timestamp,
transaction country) and account-level derived features (transaction
velocity, historical amount deviation, device novelty, home country).
No cardholder names, card numbers, CVVs, or full account numbers are
processed by the model — only the `customer_id` pseudonym and derived
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
| Is automated processing necessary for the purpose? | Yes — manual review of 100% of transactions at this volume (150,000+ in the sample dataset) is not operationally feasible; the base fraud rate (~0.45%) requires automated triage to make investigation feasible at scale. |
| Is the data collected proportionate? | Yes, with a caveat — the feature set (amount, category, timing, device, geography) is the minimum needed to detect the fraud patterns in scope. No unrelated data (e.g., browsing history, social data) is used. |
| Could a less privacy-invasive method achieve the same result? | Partially — rule-based systems could catch some patterns with less data, but published fraud-detection literature and this project's own model comparison show ML models substantially outperform static rules at equivalent false-positive rates (see `results/metrics.json`), which argues for the more data-driven approach *despite* the added processing, provided the safeguards below are in place. |
| Is there a fully automated *decision* with legal or similarly significant effect (GDPR Art. 22)? | **No** — the system flags transactions for human analyst review (see `case_tracking.py`); it does not autonomously block transactions, close accounts, or make a final fraud determination. This is a deliberate design choice specifically to avoid Article 22 applicability. If a future version moved to fully automated blocking, this DPIA section would need to be revisited and Article 22 safeguards (explanation, human review on request) added explicitly. |

---

### 3. Identified Risks to Data Subjects

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| False positive flags a legitimate transaction, causing customer inconvenience (declined transaction, account friction) | Medium (at the tuned 3% FPR, ~3 in 100 legitimate transactions are flagged) | Low-Medium | Human analyst review before any customer-facing action; local explanation (`explainability.py`) gives the analyst the "why" to resolve quickly; false positives are logged and used to refine the model. |
| Model bias produces disproportionate false-positive rates for a subset of customers (e.g., frequent international travelers, certain merchant categories) | Medium — not formally tested in this project (see Section 6) | Medium | **Documented as an open gap** rather than papered over: this project did not run a subgroup fairness audit. A production deployment should evaluate false-positive rate parity across customer segments before launch. |
| Re-identification of a customer from the pseudonymous `customer_id` combined with transaction metadata | Low in isolation, higher if combined with external data | Medium-High | `customer_id` is a synthetic identifier with no direct mapping to name/SSN/card number in this dataset; production systems must ensure this pseudonym cannot be trivially re-linked without going through access-controlled, audited identity-resolution systems. |
| Data retained longer than necessary for the fraud-detection purpose | Not yet defined in this project | Medium | See Section 4 (Retention) below — a concrete policy is proposed rather than left unspecified. |
| Case notes (`case_tracking.py`) contain analyst commentary that could include sensitive inferences about a customer | Low-Medium | Medium | Recommend a periodic manual review of `notes` fields for anything beyond fraud-relevant fact, and restrict `notes` field access to the fraud-ops role only (not general company access). |

---

### 4. Data Minimization, Retention, and Lawful Basis

- **Minimization:** Only behavioral/derived features are modeled; no direct identifiers (name, full card number, SSN) enter the model or the alert database.
- **Retention (proposed policy, not currently implemented in code):** Raw transaction data feeding the model: retained per the organization's existing financial recordkeeping requirements (commonly 5-7 years in many jurisdictions for AML/fraud purposes) — this is a *legal retention basis*, not indefinite convenience storage. Alert/case records (`alerts` table): recommend retaining resolved false-positive alerts for a much shorter window (e.g., 90 days) once resolved, since they carry lower ongoing fraud-prevention value than confirmed cases, which may need to be retained longer to satisfy the same recordkeeping requirements as the underlying transaction.
- **Lawful basis:** Legitimate interest (Art. 6(1)(f)) for the core fraud-detection processing, balanced against data subject rights via this DPIA; legal obligation (Art. 6(1)(c)) where sector-specific anti-fraud regulation applies.

---

### 5. Data Subject Rights Operationalization

See `data_subject_rights.py` for a working implementation of:
- **Right of access (Art. 15):** export all records tied to a given `customer_id`.
- **Right to erasure (Art. 17):** remove a customer's records from the alert/case database, with the retention-conflict check described in that script (erasure requests can conflict with legal retention obligations above, and the script surfaces that conflict rather than silently ignoring it).

---

### 6. Known Gaps in This DPIA (stated plainly, not hidden)

- **A preliminary subgroup fairness check was run on the synthetic data** (`src/fairness_audit_offline.py`, results in `results/fairness_audit_synthetic.json`) and found a substantial disparity worth flagging even on synthetic data: false-positive rate was **54.4% for transactions with a country mismatch vs. 0.30% for matched-country transactions** (a 54-point spread), and a smaller but still notable 6.7-point spread across merchant categories (online retail and travel see the highest false-positive rates). **This is illustrative, not a real audit** — it's checking proxies (geography mismatch, merchant category, device novelty) on synthetic data where the fraud signal was hand-designed to correlate with exactly these features, so the finding is partly circular by construction. The audit methodology itself (subgroup FPR comparison at a fixed overall FPR) is real and would need to be rerun against real IEEE-CIS data (`src/fairness_audit.py`) before it means anything about actual customer impact. What this preliminary run *does* establish: a large FPR disparity across a proxy for "customers who travel internationally" is exactly the kind of pattern a real audit needs to check for, since it would concretely mean international/frequent-traveler customers experience far more transaction friction than others — a genuine customer-harm and potential-discrimination concern, not just a modeling curiosity.
- **No formal Data Protection Officer (DPO) or supervisory authority consultation occurred**, since this is a personal project, not a live deployment. A real deployment under GDPR would require this DPIA to be reviewed by the organization's DPO, and consultation with the supervisory authority if residual risk remains high after mitigation (Art. 36).
- **No Privacy Enhancing Technology (e.g., differential privacy) is implemented in the current model training**, only architectural/procedural safeguards (pseudonymization, minimization, human-in-the-loop review). This is disclosed as a genuine limitation, not resolved.
