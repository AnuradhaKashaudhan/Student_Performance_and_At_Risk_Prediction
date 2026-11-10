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
| Excluded features | Gender, religion, caste, income, disability, location, and proxies; these columns do not exist in the dataset |
| Metrics | At-risk recall first, then precision, F1, ROC-AUC, PR-AUC, and confusion matrix |
| Safety rule | Advisory only. Never automate grading, punishment, or admission decisions. A teacher/counsellor reviews before any action. |

## Project structure

```text
student-risk/
  README.md
  requirements.txt
  .env.example
  data/raw/students.csv
  data/processed/
  src/
    generate_data.py
    split.py
    preprocess.py
    train.py
    evaluate.py
    explain.py
    make_global_importance.py
  api/main.py
  app/dashboard.py
  models/model.joblib
  models/metadata.json
  notebooks/01_eda.ipynb
  reports/
    model_comparison_val.csv
    model_comparison_test.csv
    test_metrics.json
    false_positives.csv
    false_negatives.csv
    global_importance.csv
    technical_report.md
    figures/
  docs/submission_checklist.md
  tests/
```

## Environment setup

Python 3.11 or newer is recommended. From the repository root, create and activate a virtual environment.

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

   Writes `data/raw/students.csv` (default: 600 records in each of four terms).

2. **Train, tune, evaluate, and save the model**

   ```bash
   python -m src.train
   python -m src.make_global_importance
   ```

   Training uses terms 1-2; term 3 is used for model selection, threshold selection, and permutation importance; term 4 is held out for final test evaluation. The selected artifact is a complete preprocessing-plus-classifier pipeline.

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

5. **Run tests**

   ```bash
   pytest -q
   ```

   If the model artifact is missing, the test fixture first runs `python -m src.train`.

## Example API request/response

With the API running, a high-risk example can be sent with `curl`:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"attendance_pct":55,"attendance_trend_3w":-8,"assignment_completion_rate":0.3,"late_submission_rate":0.6,"prev_score_1":52,"prev_score_2":47,"prev_score_3":41,"days_since_last_submission":12}'
```

The fitted artifact returned this response for that input (probability rounded to three decimals):

```json
{
  "risk_probability": 0.847,
  "risk_band": "High",
  "flagged": true,
  "top_signals": [
    {"feature": "prev_score_3", "value": 41.0, "effect": "increases risk", "impact": 0.256, "message": "Previous score 3 of 41.0 points is below the typical level and increases risk."},
    {"feature": "assignment_completion_rate", "value": 0.3, "effect": "increases risk", "impact": 0.047, "message": "Assignment completion of 30% is below the typical level and increases risk."},
    {"feature": "attendance_pct", "value": 55.0, "effect": "reduces risk", "impact": 0.046, "message": "Attendance of 55% is below the typical level and reduces risk."}
  ],
  "warnings": [],
  "human_review_note": "Advisory only. This is a statistical estimate, not a judgement about the student. A teacher or counsellor must review before any action. Never use for grading, punishment or admission decisions.",
  "model_name": "hist_gradient_boosting",
  "threshold": 0.4671308638752085
}
```

## Results

Final held-out test results, read from `reports/model_comparison_test.csv` and `reports/test_metrics.json`:

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Histogram gradient boosting (selected) | 0.467 | 0.552 | 0.771 | 0.643 | 0.900 | 0.733 | 387 | 82 | 30 | 101 |
| Rule: `prev_score_3 < 50` | 0.500 | 0.637 | 0.443 | 0.523 | 0.686 | 0.404 | 436 | 33 | 73 | 58 |
| Stratified dummy | 0.500 | 0.187 | 0.198 | 0.193 | 0.479 | 0.212 | 356 | 113 | 105 | 26 |

The final test report compares the selected model with both baselines. Logistic regression and random forest are reported in `reports/model_comparison_val.csv`; they were not separately evaluated on the held-out test set because the test is reserved for the selected model and baselines.

## Responsible AI

- Treat a flag as a possible invitation to offer support, not as a label about a student.
- A teacher or counsellor must review each case and decide whether any action is appropriate.
- Do not use the tool for grading, punishment, admission, discipline, or allocating scarce opportunities automatically.
- Discuss relevant context respectfully with the student; an estimate cannot explain why a pattern occurred.
- Collect only the specified educational indicators and protect any real student information in future work.

## Limitations

- The dataset and its labels are synthetic. Test performance does not establish real-world validity, fairness, calibration, or generalization.
- Attendance and educational measures may act as proxies for unobserved circumstances. Excluded demographic columns do not prove the absence of proxy bias.
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
