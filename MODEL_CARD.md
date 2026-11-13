# Model Card: Student At-Risk Advisory

## Model details

- **Model:** Sigmoid-calibrated histogram gradient boosting pipeline.
- **Prediction unit:** One student at the end of week 8 of a term.
- **Target:** Whether the week-10 assessment score is below 50/100.
- **Decision threshold:** 0.2286. A prediction at or above this threshold is flagged for human review.
- **Risk bands:** Low is below the decision threshold; Medium is at/above the threshold and below 0.50; High is at/above 0.50. If the threshold is at least 0.50, High starts at `min(threshold + 0.15, 0.95)`. Bands are descriptive categories; the review flag remains based on the decision threshold.
- **Intended users:** Teachers and counsellors considering whether to offer an academic-support check-in.

## Intended use

This prototype estimates the likelihood of a low week-10 assessment score from eight week-8 educational indicators. A flag may prompt a teacher or counsellor to privately check in and consider whether support is appropriate. The estimate is advisory and must be considered alongside the student's circumstances.

## Out-of-scope use

Do not use this model to automate or determine grading, admission, discipline, punishment, eligibility, ranking, or allocation or denial of educational opportunities. Do not treat its risk band as a diagnosis, a measure of a student's ability or character, or evidence of why a student may need support. It is not validated for real students or operational deployment.

## Data

The model was developed and evaluated using reproducibly generated synthetic data: 2,400 records (600 per term), with terms 1-2 used for training, term 3 for model and threshold selection, and term 4 held out for testing. The eight inputs are attendance percentage and trend, assignment completion and late submission rates, three previous assessment scores, and days since last submission. Sensitive group labels are stored separately and are not model inputs. No real student data were used.

The data-generating process encodes artificial relationships and cannot establish that the data, outcome, or model performance represents any real school or student population.

## Evaluation metrics

Held-out term-4 metrics for the selected calibrated histogram gradient boosting model, from [`reports/test_metrics.json`](reports/test_metrics.json):

| Metric | Value |
|---|---:|
| Precision | 0.552 |
| Recall | 0.771 |
| F1 | 0.643 |
| ROC-AUC | 0.900 |
| PR-AUC | 0.733 |
| Brier score | 0.100 |
| Threshold | 0.229 |
| Confusion matrix (TN, FP, FN, TP) | 387, 82, 30, 101 |

These are results on one synthetic held-out term, not guarantees of future or real-world performance.

For completeness, `reports/model_comparison_test.csv` also includes random-forest and histogram-gradient-boosting candidate metrics at thresholds selected on validation with the recall >= 0.80 rule. These rows are marked `reported_after_selection` and were computed only after the selected model and threshold were fixed; they did not influence selection. The histogram-gradient-boosting candidate row is uncalibrated and is distinct from the selected calibrated pipeline.

## Fairness audit summary

The post-hoc term-4 audit in [`reports/fairness_audit.csv`](reports/fairness_audit.csv) compares synthetic gender and school-support groups; these fields were not used for training or prediction. The largest gender-group gap was a 0.117 false-positive-rate gap (Other: 0.281, M: 0.170, F: 0.165); the largest support-level gap was a 0.099 recall gap (high: 0.844, low: 0.750, medium: 0.745). The synthetic Other group has only 42 records. These descriptive results are noisy, are not a fairness certification, and say nothing about impacts on real groups.

## Limitations

- Synthetic data cannot validate real-world accuracy, calibration, fairness, or generalization.
- Educational indicators may proxy for unobserved circumstances such as disability, illness, caregiving, transport access, or resources. Excluding sensitive attributes does not remove proxy bias.
- The test covers only one synthetic term. Changes in schools, curricula, grading, student populations, or data quality may affect performance.
- Missing values are imputed, and unexpected situations after week 8 are not represented. False positives and false negatives are expected.
- Feature signals are model perturbations, not causal explanations. Risk estimates are not statistical confidence intervals.
- The threshold embodies a trade-off between missed at-risk cases and unnecessary check-ins; observed recall is not guaranteed in future data.

## Human oversight

A teacher or counsellor must review context before any supportive action. Use any result only as a prompt for a respectful, private conversation; do not stigmatize students or rely on the estimate as a substitute for professional judgement. This prototype lacks production authentication, authorization, privacy controls, and monitoring. Do not enter real student information.

## Contact

Anuradha Kashaudhan - your-email@example.com
