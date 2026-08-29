# Model card: temporal IEEE-CIS fraud classifier

## Model details

- **Model family:** LightGBM binary classifier with native categorical features and past-only
  frequency encoding for high-cardinality fields.
- **Probability layer:** sigmoid or isotonic calibration selected on a disjoint chronological
  calibration window.
- **Decision policy:** fixed thresholds representing 0.5%, 1%, 2%, and 5% review capacity; 2% is
  the default deployed operating point.
- **Training data:** the first 50% of elapsed IEEE-CIS `TransactionDT`; the saved bundle also
  records source hashes indirectly through its data version and explicit temporal cutoffs.
- **Versioning:** content-derived data and model versions plus creation time.

## Intended use

This model is an educational and portfolio implementation for ranking transactions for limited
manual fraud review and demonstrating post-deployment monitoring. It is suitable for offline
experimentation, architecture review, and interview discussion.

It is not approved for autonomous payment rejection, law-enforcement referral, credit decisions,
or deployment against a live population without domain validation, governance, and human review.

## Label and imbalance semantics

`isFraud=1` is the competition fraud label. Fraud is rare, so PR-AUC is the primary ranking metric
and review-capacity recall is the principal operating metric. ROC-AUC is reported but is not used
alone to select or promote a model. SMOTE is intentionally not used; class weighting is evaluated
inside each temporal fold.

## Evaluation design

- Five equal-duration development blocks support expanding temporal validation.
- The next 10% of elapsed time is reserved for calibration and threshold selection.
- The following 10% is a locked acceptance/reference period evaluated once.
- Remaining labeled data is production history, not additional acceptance data.
- Bootstrap intervals report uncertainty for acceptance PR-AUC, recall at 2% capacity, and paired
  PR-AUC improvement over logistic regression.
- A reliability table, calibration slope/intercept, Brier score, log loss, ECE, confusion counts,
  error rates, review rate, and captured fraud amount are persisted.

## Verified evaluation

Kaggle notebook Version 5 completed on 2026-08-29 from repository commit `015c1f1` using data
version `805c429ec247`; the resulting model version is `fe47c8a821b3`. The model has 449 features,
including 47 native categoricals and two past-only frequency encodings. Isotonic calibration was
selected over sigmoid on the disjoint calibration window.

On the locked acceptance window, LightGBM achieved 0.5826 PR-AUC (95% bootstrap CI
0.5638–0.6002), compared with 0.1616 for logistic regression and 0.0411 for the dummy prior. The
paired PR-AUC improvement over logistic was 0.4210 (95% bootstrap CI 0.4026–0.4380). ROC-AUC was
0.9080.

At the frozen calibration-derived default threshold of 0.3657, acceptance precision was 0.7397,
recall was 0.4662 (95% bootstrap CI 0.4452–0.4851), and 30.08% of labeled fraud amount was captured.
The fixed threshold yielded a 2.59% acceptance review rate, illustrating that a calibration-window
capacity target does not guarantee the same later-window volume. Acceptance Brier score was 0.0238
and expected calibration error was 0.0015.

Only the allow-listed aggregate monitoring export is committed. Competition rows, transaction
identifiers, the model bundle, full training artifacts, and MLflow runtime data remain private.

## Feature and leakage controls

The model uses numeric `C`, `D`, and `V` families; designated product, card, address, email, match,
device, and identity categoricals; row-level amount/time/missingness features; and causal entity
velocity features. Every aggregate is emitted before the current transaction updates entity state.
Preprocessing, category/frequency maps, zero-variance and missingness removal, calibration,
thresholds, and reference profiles fit only on allowed earlier data.

`TransactionID`, targets, temporal partition labels, and batch IDs are not model features.

## Delayed labels and monitoring

Production is replayed in seven-day windows with a two-batch label delay. Performance for pending
or shadow labels is explicitly unavailable. The system monitors data quality, numeric and
categorical covariate drift, prediction drift, calibration, mature-label performance, and
operational segments against empirical acceptance-period limits.

TreeSHAP summaries are aggregate-only. Investigation records rank likely drivers and identify
segments contributing to false negatives, false positives, and prevalence changes.

The verified replay contains eight production batches and 27 official-test shadow batches. Six
production batches have mature labels and two are pending; all shadow performance is unavailable by
design. The policy recorded 32 investigations and three retraining-evaluation-required states. A
challenger was not run, and replacement was not recommended automatically.

## Retraining policy

Drift alone cannot trigger replacement. Two consecutive mature primary-metric breaches can request
a manually initiated challenger. The challenger uses only earlier matured history, is tested on
later untouched batches, and is recommended only with reliable paired PR-AUC improvement,
non-inferior 2%-capacity recall, and no reliable eligible-segment recall regression. Deployment is
never automatic.

## Risks and limitations

- `TransactionDT` does not identify real calendar dates.
- Competition labels may be delayed, incomplete, or operationally different from live chargeback
  outcomes.
- Anonymized operational segments are not protected classes; fairness cannot be established from
  them.
- Missing identity data may encode changes in collection systems rather than customer behavior.
- Review thresholds assume stable reviewer capacity and costs.
- SHAP explains model behavior, not causal fraud drivers.
- A production deployment would require privacy review, access controls, alert ownership, audit
  retention, feedback-quality checks, and incident runbooks.
