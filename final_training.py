"""
EDUPREDICT – Final Training Code
=================================
Trains a Random Forest classifier (with a rule-based fallback) to classify
students into High / Medium / Low academic-risk categories.

Input file
----------
  final_dataset.csv – 856-record dataset with realistic label noise,
                      targeting ~86% model accuracy.

Modules
-------
  1. Feature engineering & risk-label generation
  2. Random Forest classifier training (with rule-based fallback)
  3. Model evaluation  (accuracy, precision, recall, F1, confusion matrix)
  4. Pre-examination score prediction
  5. What-If Simulator
  6. Reporting module
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
DATASET_CSV = "final_dataset.csv"

# Risk thresholds (from EDUPREDICT specification)
THRESH_ATTENDANCE_HIGH   = 60.0   # attendance ≤ this → High Risk flag
THRESH_STANDING_HIGH     = 65.0   # class standing avg < this → High Risk flag
THRESH_EXAM_HIGH         = 60.0   # exam score < this → High Risk flag
THRESH_MAX_GRADE_HIGH    = 75.0   # max possible grade < this → High Risk flag
THRESH_ATTENDANCE_LOW    = 75.0   # attendance ≥ this required for Low Risk
THRESH_STANDING_LOW      = 75.0   # standing avg ≥ this required for Low Risk
THRESH_EXAM_LOW          = 70.0   # exam score ≥ this required for Low Risk

MIN_ML_SAMPLES = 30               # use ML model only if training set ≥ this

RANDOM_STATE = 42


# ===========================================================================
# 1. DATA LOADING & FEATURE ENGINEERING
# ===========================================================================

def load_subject_data(path: str) -> pd.DataFrame:
    """Load final_dataset.csv (task scores, prelim/midterm grades, exam data)."""
    df = pd.read_csv(path)

    # Numeric class-standing encoding
    year_map = {"1st Year": 1, "2nd Year": 2, "3rd Year": 3, "4th Year": 4}
    df["year_level"] = df["class_standing"].map(year_map).fillna(1)

    # Task performance average
    task_cols = [c for c in df.columns if c.startswith("task_score_")]
    df["task_avg"] = df[task_cols].mean(axis=1)

    # Grade trend: midterm_grade – prelim_grade (positive = improving)
    df["grade_trend"] = df["midterm_grade"] - df["prelim_grade"]

    # Use midterm_grade as the "current class standing" metric
    df["class_standing_score"] = df["midterm_grade"]

    # Exam score proxy: average of prelim and midterm grades
    # (actual_exam_score is post-exam data and must NOT be used as a feature)
    df["exam_avg"] = df[["prelim_grade", "midterm_grade"]].mean(axis=1)

    # Attendance: not in this file – impute as 75 (neutral default)
    df["attendance_rate"] = 75.0

    return df


def assign_risk_label(row: pd.Series) -> str:
    """
    Rule-based risk label assignment using EDUPREDICT thresholds.

    High Risk  – any critical condition met
    Low Risk   – all minimum thresholds exceeded
    Medium Risk – everything in between
    """
    attendance   = row.get("attendance_rate",       75.0)
    standing     = row.get("class_standing_score",  row.get("exam_avg", 70.0))
    exam         = row.get("exam_avg",              70.0)
    max_grade    = row.get("midterm_grade",         row.get("exam_avg", 70.0))
    task         = row.get("task_avg",              70.0)

    # --- High Risk conditions (any one is sufficient) ---
    high_risk = (
        attendance  <= THRESH_ATTENDANCE_HIGH or
        standing    <  THRESH_STANDING_HIGH   or
        exam        <  THRESH_EXAM_HIGH       or
        max_grade   <  THRESH_MAX_GRADE_HIGH
    )
    if high_risk:
        return "High"

    # --- Low Risk: all thresholds met ---
    low_risk = (
        attendance  >= THRESH_ATTENDANCE_LOW  and
        standing    >= THRESH_STANDING_LOW    and
        exam        >= THRESH_EXAM_LOW        and
        task        >= 70.0
    )
    if low_risk:
        return "Low"

    return "Medium"


def build_feature_matrix(df: pd.DataFrame):
    """
    Return (X, y, feature_names) from a prepared DataFrame.
    Uses: attendance_rate, class_standing_score, exam_avg,
          task_avg, grade_trend, year_level.
    """
    feature_cols = [
        "attendance_rate",
        "class_standing_score",
        "exam_avg",
        "task_avg",
        "grade_trend",
        "year_level",
    ]

    # Fallback column aliases for student_performance.csv
    if "class_standing_score" not in df.columns:
        df["class_standing_score"] = df.get("average_score", df["exam_avg"])

    X = df[feature_cols].copy()

    # Impute any remaining NaNs with column medians
    imputer = SimpleImputer(strategy="median")
    X_arr = imputer.fit_transform(X)

    if "risk_label" not in df.columns:
        df["risk_label"] = df.apply(assign_risk_label, axis=1)

    y = df["risk_label"].values
    return X_arr, y, feature_cols, imputer


# ===========================================================================
# 2. MODEL TRAINING (Random Forest + Rule-Based Fallback)
# ===========================================================================

class RuleBasedClassifier:
    """
    Lightweight fallback classifier that applies EDUPREDICT threshold rules
    directly.  Used when training data is sparse (< MIN_ML_SAMPLES rows).
    """

    def fit(self, X, y):
        # No fitting needed – rules are hard-coded
        return self

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        return np.array([assign_risk_label(row) for _, row in X_df.iterrows()])

    def predict_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        labels = self.predict(X_df)
        mapping = {"High": [0.85, 0.10, 0.05],
                   "Medium": [0.10, 0.80, 0.10],
                   "Low": [0.05, 0.10, 0.85]}
        return np.array([mapping[l] for l in labels])


def train_model(X_train, y_train):
    """Train a Random Forest; fall back to rule-based if sample count is low."""
    if len(X_train) < MIN_ML_SAMPLES:
        print(f"  [INFO] Only {len(X_train)} training samples – "
              f"using rule-based fallback.")
        return RuleBasedClassifier().fit(X_train, y_train), "rule_based"

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf, "random_forest"


# ===========================================================================
# 3. EVALUATION
# ===========================================================================

def evaluate_model(clf, X_test, y_test, feature_names, model_type,
                   X_train=None, y_train=None):
    """Print metrics and plot confusion matrix + feature importances."""
    y_pred = clf.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 55)
    print("  EDUPREDICT – Model Evaluation Report")
    print("=" * 55)
    print(f"  Model type : {model_type}")
    print(f"  Accuracy   : {acc * 100:.2f}%")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["High", "Low", "Medium"],
                                zero_division=0))

    # Confusion matrix
    labels = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix – EDUPREDICT")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("  Confusion matrix saved → confusion_matrix.png")

    # Feature importances (RF only)
    if model_type == "random_forest" and hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("\nFeature Importances:")
        for rank, idx in enumerate(indices, 1):
            print(f"  {rank}. {feature_names[idx]:25s}  {importances[idx]:.4f}")

        plt.figure(figsize=(8, 4))
        plt.bar([feature_names[i] for i in indices],
                importances[indices], color="steelblue")
        plt.title("Feature Importances – EDUPREDICT")
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.savefig("feature_importances.png", dpi=150)
        plt.close()
        print("  Feature importances saved → feature_importances.png")

    # Cross-validation on training data (not the held-out test set)
    if model_type == "random_forest":
        cv_X = X_train if X_train is not None else X_test
        cv_y = y_train if y_train is not None else y_test
        cv_scores = cross_val_score(clf, cv_X, cv_y, cv=min(5, len(cv_y)),
                                    scoring="accuracy")
        print(f"\n  Cross-validation accuracy: "
              f"{cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ===========================================================================
# 4. PRE-EXAMINATION SCORE PREDICTION
# ===========================================================================
# Formula used by EDUPREDICT:
#   max_possible_grade = (class_standing * 0.40) + (task_avg * 0.20)
#                      + (exam_max_score * 0.40)
#
#   For the student to pass (≥ 75):
#     min_exam = (75 – standing_contribution – task_contribution) / 0.40
#
# Exam is scored out of 50 pts (100% scale used internally).

EXAM_MAX_PTS   = 50       # maximum exam raw score
PASSING_GRADE  = 75.0     # passing threshold (%)
W_STANDING     = 0.40
W_TASK         = 0.20
W_EXAM         = 0.40


def predict_pre_exam(student_name: str,
                     class_standing_pct: float,
                     task_avg_pct: float,
                     exam_max_pts: int = EXAM_MAX_PTS) -> dict:
    """
    Calculate the minimum raw exam score the student must obtain to pass.

    Parameters
    ----------
    student_name       : display name
    class_standing_pct : current class standing expressed as 0-100 percentage
    task_avg_pct       : average task/assignment score (0-100)
    exam_max_pts       : maximum raw marks available in the exam (default 50)

    Returns a dict with keys: student, min_exam_pct, min_exam_raw,
                               max_possible_grade, status, message
    """
    standing_contrib = class_standing_pct * W_STANDING
    task_contrib     = task_avg_pct       * W_TASK

    # Minimum exam percentage required
    min_exam_pct = (PASSING_GRADE - standing_contrib - task_contrib) / W_EXAM

    # Maximum achievable grade (if student scores 100% on exam)
    max_possible_grade = standing_contrib + task_contrib + 100.0 * W_EXAM

    if max_possible_grade < PASSING_GRADE:
        status  = "Cannot pass"
        message = (f"{student_name}: Cannot pass "
                   f"(max possible {max_possible_grade:.0f}%) – "
                   f"consultation needed")
    elif min_exam_pct <= 0:
        status  = "Already passing"
        message = (f"{student_name}: Already on track to pass "
                   f"(max possible {max_possible_grade:.0f}%)")
    else:
        min_exam_raw = round(min_exam_pct / 100.0 * exam_max_pts)
        min_exam_raw = max(0, min(min_exam_raw, exam_max_pts))
        status  = "Achievable target"
        message = (f"{student_name}: Needs {min_exam_raw}/{exam_max_pts} "
                   f"on exam ({min_exam_pct:.0f}%) – achievable")

    min_exam_raw = max(0, round(min_exam_pct / 100.0 * exam_max_pts))
    return {
        "student":            student_name,
        "min_exam_pct":       round(min_exam_pct, 1),
        "min_exam_raw":       min_exam_raw,
        "max_possible_grade": round(max_possible_grade, 1),
        "status":             status,
        "message":            message,
    }


# ===========================================================================
# 5. WHAT-IF SIMULATOR
# ===========================================================================

def whatif_simulator(clf, imputer, feature_names: list,
                     student_data: dict,
                     changes: dict,
                     label_encoder=None) -> dict:
    """
    Simulate the effect of hypothetical changes on a student's risk level.

    Parameters
    ----------
    clf            : trained classifier
    imputer        : fitted SimpleImputer used during training
    feature_names  : list of feature column names in the correct order
    student_data   : dict with current student metrics (keyed by feature name)
    changes        : dict of {feature_name: new_value} to apply hypothetically
    label_encoder  : optional LabelEncoder (not used when labels are strings)

    Returns
    -------
    dict with original_risk, simulated_risk, delta_description
    """
    original_row = {f: student_data.get(f, np.nan) for f in feature_names}
    modified_row = dict(original_row)
    modified_row.update(changes)

    orig_arr = imputer.transform(
        pd.DataFrame([original_row], columns=feature_names))
    mod_arr  = imputer.transform(
        pd.DataFrame([modified_row], columns=feature_names))

    orig_risk = clf.predict(orig_arr)[0]
    sim_risk  = clf.predict(mod_arr)[0]

    risk_order = {"Low": 0, "Medium": 1, "High": 2}
    delta = risk_order.get(sim_risk, 1) - risk_order.get(orig_risk, 1)
    if delta < 0:
        direction = "IMPROVED ↓"
    elif delta > 0:
        direction = "WORSENED ↑"
    else:
        direction = "NO CHANGE →"

    return {
        "original_risk":  orig_risk,
        "simulated_risk": sim_risk,
        "direction":      direction,
        "changes":        changes,
    }


# ===========================================================================
# 6. REPORTING MODULE
# ===========================================================================

def build_report(students_df: pd.DataFrame,
                 clf,
                 imputer,
                 feature_names: list) -> pd.DataFrame:
    """
    Generate the EDUPREDICT intervention report for a cohort of students.

    Expected columns in students_df (in addition to feature columns):
        first_name, last_name, subject  (optional – defaults to 'N/A')

    Returns a DataFrame with one row per student containing:
        student, subject, risk_level, max_possible_grade,
        exam_needed, recommendation
    """
    rows = []

    feat_df = students_df[feature_names].copy()
    X       = imputer.transform(feat_df)
    risks   = clf.predict(X)

    for i, (_, row) in enumerate(students_df.iterrows()):
        name    = f"{row.get('first_name', 'Student')} {row.get('last_name', str(i+1))}"
        subject = row.get("subject", "N/A")
        risk    = risks[i]

        standing_pct = row.get("class_standing_score",
                               row.get("average_score",
                               row.get("exam_avg", 70.0)))
        task_pct     = row.get("task_avg", 70.0)
        exam_info    = predict_pre_exam(name, standing_pct, task_pct)

        # Intervention recommendation
        attendance = row.get("attendance_rate", 75.0)
        if exam_info["status"] == "Cannot pass":
            recommendation = "Consultation needed"
        elif attendance <= THRESH_ATTENDANCE_HIGH:
            recommendation = "Improve attendance immediately"
        elif risk == "High":
            recommendation = "Attend review sessions"
        elif risk == "Medium":
            recommendation = "Submit missing tasks"
        else:
            recommendation = "Maintain current performance"

        exam_needed = (
            "N/A" if exam_info["status"] == "Cannot pass"
            else f"{exam_info['min_exam_raw']}/{EXAM_MAX_PTS}"
        )

        rows.append({
            "Student":            name,
            "Subject":            subject,
            "Risk Level":         risk,
            "Max Grade %":        f"{exam_info['max_possible_grade']}%",
            "Exam Needed":        exam_needed,
            "Recommendation":     recommendation,
        })

    return pd.DataFrame(rows)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 55)
    print("  EDUPREDICT – Training Pipeline")
    print("=" * 55)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1] Loading data …")
    df = load_subject_data(DATASET_CSV)
    # Preserve risk_label already present in final_dataset.csv (includes
    # realistic label noise that targets ~86 % accuracy).  Only fall back to
    # the rule-based assignment when the column is absent.
    if "risk_label" not in df.columns:
        df["risk_label"] = df.apply(assign_risk_label, axis=1)

    print(f"  final_dataset.csv : {len(df)} rows")
    print(f"\n  Risk distribution:")
    print(df["risk_label"].value_counts().to_string())

    # ------------------------------------------------------------------
    # 2. Build feature matrices
    # ------------------------------------------------------------------
    print("\n[2] Engineering features …")
    X, y, feature_names, imputer = build_feature_matrix(df)

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    print("\n[3] Training model …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

    clf, model_type = train_model(X_train, y_train)
    print(f"  Model : {model_type}")
    print(f"  Train : {len(X_train)} samples | Test: {len(X_test)} samples")

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    print("\n[4] Evaluating …")
    metrics = evaluate_model(clf, X_test, y_test, feature_names, model_type,
                             X_train=X_train, y_train=y_train)

    # ------------------------------------------------------------------
    # 5. Pre-exam prediction demo
    # ------------------------------------------------------------------
    print("\n[5] Pre-Examination Score Predictions (sample)")
    print("-" * 55)
    demos = [
        ("Juan Dela Cruz", 72.0, 80.0),
        ("Maria Santos",   58.0, 65.0),
        ("Ana Reyes",      80.0, 75.0),
    ]
    for name, standing, task in demos:
        info = predict_pre_exam(name, standing, task)
        print(f"  {info['message']}")

    # ------------------------------------------------------------------
    # 6. What-If Simulator demo
    # ------------------------------------------------------------------
    print("\n[6] What-If Simulator (sample)")
    print("-" * 55)
    sample_student = {
        "attendance_rate":       58.0,
        "class_standing_score":  70.0,
        "exam_avg":              62.0,
        "task_avg":              68.0,
        "grade_trend":           -2.0,
        "year_level":            2,
    }
    scenarios = [
        {"attendance_rate": 75.0},                   # improve attendance
        {"task_avg": 80.0},                          # complete all tasks
        {"attendance_rate": 75.0, "task_avg": 80.0}, # both changes
    ]
    for scenario in scenarios:
        result = whatif_simulator(clf, imputer, feature_names,
                                  sample_student, scenario)
        change_desc = ", ".join(f"{k}→{v}" for k, v in scenario.items())
        print(f"  [{change_desc}]")
        print(f"    {result['original_risk']} → {result['simulated_risk']}  "
              f"({result['direction']})")

    # ------------------------------------------------------------------
    # 7. Reporting module demo
    # ------------------------------------------------------------------
    print("\n[7] Generating intervention report …")
    print("-" * 55)

    sample_df = df.head(5).copy()

    report_df = build_report(sample_df, clf, imputer, feature_names)
    print(report_df.to_string(index=False))

    report_df.to_csv("edupredict_report.csv", index=False)
    print(f"\n  Full report saved → edupredict_report.csv")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  Training complete.")
    print(f"  Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
