# Student Performance & At-Risk Prediction

## Overview

This project is a small, reproducible advisory decision-support prototype. At the end of week 8, it estimates whether a student's week-10 assessment score may be below 50, to help a teacher or counsellor consider who could benefit from an academic-support check-in. The data are synthetic; the model is not validated for real students or operational use.

## Problem framing

| Item | Definition |
|---|---|
| Decision | Which students should a teacher/counsellor consider checking in with for extra academic support? |
| Prediction unit | One student at one checkpoint in a term |
| Target | `at_risk = 1` when the next assessment score is **below 50/100**; otherwise `0` |
| Horizon | Two weeks ahead: the week-10 assessment |
| Prediction time | End of week 8; predictors must be available by then |
| Excluded features | Gender, religion, caste, income, disability, location, school-support group, and proxies are never model features; only synthetic gender and support-level labels are kept in a separate post-test audit file |
| Metrics | At-risk recall first, then precision, F1, ROC-AUC, PR-AUC, and confusion matrix |
| Safety rule | Advisory only. Never automate grading, punishment, or admission decisions. A teacher/counsellor reviews before any action. |

## Project structure

```text
.
  .dockerignore
  .env.example
  .gitignore
  .github/workflows/ci.yml
  Dockerfile
  docker-compose.yml
  MODEL_CARD.md
  README.md
  requirements.txt
  api/
    __init__.py
    main.py
  app/
    dashboard.py
    dashboard_utils.py
  data/
    processed/
      .gitkeep
    raw/
      audit_groups.csv           # separate synthetic groups, evaluation only
      students.csv
  docs/
    submission_checklist.md
  models/
    .gitkeep
    metadata.json
    model.joblib
  notebooks/
    01_eda.ipynb
  reports/
    calibration_comparison_val.csv
    capacity_analysis.csv
    fairness_audit.csv
    false_negatives.csv
    false_positives.csv
    global_importance.csv
    model_comparison_test.csv
    model_comparison_val.csv
    robustness.csv
    technical_report.md
    test_metrics.json
    figures/
      at_risk_rate_by_term.png
      calibration_curve.png
      capacity_metrics.png
      confusion_matrix_test.png
      correlation_heatmap.png
      cost_vs_threshold.png
      fairness_audit.png
      feature_distributions_by_class.png
      model_comparison_test.png
      permutation_importance.png
      precision_recall_curve_test.png
      robustness.png
      roc_curve_test.png
      target_class_balance.png
  src/
    __init__.py
    capacity_analysis.py
    audit.py
    explain.py
    evaluate.py
    generate_data.py
    make_global_importance.py
    preprocess.py
    robustness.py
    split.py
    train.py
  tests/
    __init__.py
    .gitkeep
    conftest.py
    test_api.py
    test_artifact.py
    test_audit.py
    test_capacity_analysis.py
    test_dashboard_utils.py
    test_explain_api.py
    test_generate_data.py
    test_preprocess.py
    test_robustness.py
    test_split_no_leakage.py
    test_training_pipeline.py
```

## Environment setup

Use **Python 3.13.12** to match the pinned dependencies, CI workflow, and Docker image. From the repository root, create and activate a virtual environment.

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Reproduce step by step

Run each command from the `student-risk` repository root with the virtual environment activated.

1. **Generate the deterministic synthetic dataset**

   ```bash
   python -m src.generate_data --seed 42
   ```

   Writes `data/raw/students.csv` (default: 600 records in each of four terms) and a separate `data/raw/audit_groups.csv`. The audit file contains `student_id`, `gender`, and `school_support_level`; it is never included in model features.

2. **Train, tune, evaluate, and save the model**

   ```bash
   python -m src.train
   python -m src.make_global_importance
   python -m src.audit
   python -m src.capacity_analysis
   python -m src.robustness
   ```

   Training uses terms 1-2; term 3 is used for model selection, threshold selection, and permutation importance; term 4 is held out for final test evaluation. The fairness audit joins the separate group labels only to term-4 predictions and writes `reports/fairness_audit.csv` and `reports/figures/fairness_audit.png`.

   Capacity analysis ranks term-4 predictions for K=25, 50, 100, and 150, then compares configurable false-negative/false-positive costs across thresholds; defaults are 5 and 1. It writes `reports/capacity_analysis.csv`, `reports/figures/capacity_metrics.png`, and `reports/figures/cost_vs_threshold.png`. To change the relative costs, pass `--cost-false-negative` or `--cost-false-positive` (for example, `python -m src.capacity_analysis --cost-false-negative 5 --cost-false-positive 1`). The selected artifact is a complete preprocessing-plus-classifier pipeline.

   Robustness analysis tests 5%/10% Gaussian noise and 5%/15% additional random missingness on term 4 at the fixed validation threshold. It writes `reports/robustness.csv` and `reports/figures/robustness.png`. The API warns when any input is more than three training standard deviations from that feature's training mean; these training-only statistics are saved in model metadata.

3. **Run the API** (keep this terminal open)

   ```bash
   uvicorn api.main:app --reload
   ```

   Interactive documentation is at `http://localhost:8000/docs`.

4. **Run the dashboard** (in a second terminal)

   ```bash
   streamlit run app/dashboard.py
   ```

   The dashboard calls `http://localhost:8000` by default. To use another API address, set `API_URL` before starting Streamlit (for example, `$env:API_URL="http://localhost:8000"` in PowerShell).

   The **Class overview** tab accepts a downloadable-template CSV with the eight raw features and sends batches of up to 100 rows to `/predict-batch`. The **What-if** tab compares the assessed student's prediction with a scenario changing attendance, assignment completion, and late submissions; this is a model association, not a causal guarantee.

5. **Run tests**

   ```bash
   pytest -q
   ```

   If the model artifact is missing, the test fixture first runs `python -m src.train`.

## Run with Docker Compose

From PowerShell in the repository root, build the image and start the API and dashboard:

```bash
docker compose up --build
```

In a second PowerShell window, check that the API is healthy:

```powershell
Invoke-RestMethod http://localhost:8000/health
docker compose ps
```

Open the dashboard at `http://localhost:8501` and the API documentation at `http://localhost:8000/docs`. The dashboard reaches the healthy API through `http://api:8000` on the Compose network. Stop both services with `Ctrl+C` in the first window, then run this from the repository root:

```powershell
docker compose down
```

The image uses the checked-in model artifacts; the build fails if either model file is missing. To regenerate data and train a model, follow the reproduction steps above before building.

## Example API request/response

With the API running, a high-risk example can be sent with `curl`:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"attendance_pct":55,"attendance_trend_3w":-8,"assignment_completion_rate":0.3,"late_submission_rate":0.6,"prev_score_1":52,"prev_score_2":47,"prev_score_3":41,"days_since_last_submission":12}'
```

The fitted artifact returned this response for that input (probability and signal impacts rounded to three decimals). Signals with impact below 0.01 are omitted; when none qualify, `top_signals` is empty and `signal_note` explains why:

```json
{
  "risk_probability": 0.664,
  "risk_band": "High",
  "flagged": true,
  "uncertainty": "normal",
  "confidence_note": "The estimate is not a certainty; it is outside the borderline range and uses the available feature values.",
  "signal_note": null,
  "top_signals": [
    {"feature": "prev_score_3", "value": 41.0, "effect": "increases risk", "impact": 0.333, "message": "Previous score 3 of 41.0 points is below the typical level and increases risk."},
    {"feature": "attendance_pct", "value": 55.0, "effect": "reduces risk", "impact": 0.087, "message": "Attendance of 55% is below the typical level and reduces risk."},
    {"feature": "assignment_completion_rate", "value": 0.3, "effect": "increases risk", "impact": 0.078, "message": "Assignment completion of 30% is below the typical level and increases risk."}
  ],
  "warnings": ["attendance_pct=55 is more than 3 training standard deviations from the training mean: possible input drift"],
  "human_review_note": "Advisory only. This is a statistical estimate, not a judgement about the student. A teacher or counsellor must review before any action. Never use for grading, punishment or admission decisions.",
  "model_name": "hist_gradient_boosting",
  "threshold": 0.22860668933306647
}
```

## Results

Final held-out test results, read from `reports/model_comparison_test.csv` and `reports/test_metrics.json`:

| Role | Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| selected | Histogram gradient boosting, sigmoid calibrated | 0.229 | 0.552 | 0.771 | 0.643 | 0.900 | 0.733 | 0.100 | 387 | 82 | 30 | 101 |
| baseline | Rule: `prev_score_3 < 50` | 0.500 | 0.637 | 0.443 | 0.523 | 0.686 | 0.404 | 0.177 | 436 | 33 | 73 | 58 |
| baseline | Stratified dummy | 0.500 | 0.187 | 0.198 | 0.193 | 0.479 | 0.212 | 0.363 | 356 | 113 | 105 | 26 |
| reported_after_selection | Logistic regression (tuned) | 0.532 | 0.575 | 0.786 | 0.665 | 0.907 | 0.757 | 0.121 | 393 | 76 | 28 | 103 |
| reported_after_selection | Random forest (untuned) | 0.420 | 0.609 | 0.748 | 0.671 | 0.890 | 0.696 | 0.107 | 406 | 63 | 33 | 98 |
| reported_after_selection | Histogram gradient boosting candidate (tuned, uncalibrated) | 0.467 | 0.552 | 0.771 | 0.643 | 0.900 | 0.733 | 0.118 | 387 | 82 | 30 | 101 |

The selected histogram gradient boosting model's sigmoid calibrator is fitted with five-fold cross-validation on training terms only. Its tuned validation PR-AUC was **0.733**, compared with **0.727** for logistic regression (`reports/model_comparison_val.csv`). On validation, Brier score improved from **0.122448** uncalibrated to **0.098668** calibrated (`reports/calibration_comparison_val.csv` and `reports/figures/calibration_curve.png`). The threshold was re-selected on calibrated validation probabilities to meet the 0.80 recall floor; it is **0.228607**. After model and threshold selection were complete, the tuned logistic-regression candidate, random-forest candidate, and histogram-gradient-boosting candidate were scored on the test term for reporting, each at its own threshold selected on validation to meet recall >= 0.80. These `reported_after_selection` rows do not influence model or threshold selection; the histogram-gradient-boosting candidate row is its tuned uncalibrated pipeline, distinct from the selected calibrated pipeline.

The calibrated and uncalibrated HGB rows share the same confusion matrix because sigmoid calibration is a monotonic transform, so only Brier score differs.

Risk bands are separate from the flag decision. **Low** is below the selected decision threshold; **Medium** is from that threshold (inclusive) to **0.50** (exclusive); **High** starts at **0.50**. If the selected threshold is at least 0.50, the High cutoff is instead `min(threshold + 0.15, 0.95)`. A prediction is flagged for review at or above the decision threshold, so a 25% estimate is Medium but flagged with the current threshold of about 0.229.

The API also returns an uncertainty level and confidence note. Uncertainty is `high` when the probability is within 0.07 of the decision threshold or at least two inputs are missing; otherwise it is `normal`. This is a simple review cue, not a statistical confidence interval or guarantee.

The API also flags individual feature values more than three training standard deviations from the training mean as possible input drift. This simple univariate warning is not a formal distribution-shift detector. Test-set noise and missingness sensitivity is reported in `reports/robustness.csv` and `reports/figures/robustness.png`.

## Responsible AI

- Treat a flag as a possible invitation to offer support, not as a label about a student.
- A teacher or counsellor must review each case and decide whether any action is appropriate.
- Do not use the tool for grading, punishment, admission, discipline, or allocating scarce opportunities automatically.
- Discuss relevant context respectfully with the student; an estimate cannot explain why a pattern occurred.
- Collect only the specified educational indicators and protect any real student information in future work.

## Limitations

- The dataset and its labels are synthetic. Test performance does not establish real-world validity, fairness, calibration, or generalization.
- Attendance and educational measures may act as proxies for unobserved circumstances. Excluded demographic columns do not prove the absence of proxy bias.
- `data/raw/audit_groups.csv` is synthetic evaluation metadata only. It is not joined during training or prediction; the group audit is an exploratory term-4 diagnostic, not proof of fairness or parity.
- The chronological test covers one synthetic term only; school, curriculum, grading, and student populations can differ.
- Median imputation can hide why information is missing. Sudden life changes and other unobserved factors cannot be inferred from these features.
- Local median-replacement signals are model perturbations, not causal explanations. Permutation importance is specific to the validation cohort and correlated features can share or mask importance.
- A threshold trades missed at-risk students against unnecessary check-ins. Its validation recall is not a guarantee of future recall.

## Known incomplete items

- No authentication, authorization, persistent audit log, deployment configuration, or production privacy controls are implemented.
- No real-world, subgroup, or fairness validation is possible with this synthetic dataset.
- Threshold adjustment is an offline metadata/retraining operation, not an interactive dashboard control.
- The dashboard/API are a local demonstration; neither is prepared for production hosting or real student records.
- `src/preprocess.py`, `src/split.py`, and `src/explain.py` support the current workflow; the API is implemented, while unrelated extension points such as automated monitoring and model retraining schedules are not.

## AI tools used

GitHub Copilot was used for code scaffolding; all code was reviewed and understood by me.

## Attribution

This project uses scikit-learn, FastAPI, Streamlit, and pandas (among other packages listed in `requirements.txt`). The data are fully synthetic and generated by `src/generate_data.py`; no real student data are used. No external dataset is included, so there are no dataset licensing issues.
