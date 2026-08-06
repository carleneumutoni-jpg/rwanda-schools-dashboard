"""
Rebuilds all data/ files the dashboard needs:
  - schools.csv                 (school-level, same columns as original)
  - cv_results.json             (validated school-level model results — unchanged)
  - feature_importance.json     (validated SHAP/coef results — unchanged)
  - tiering_honesty.json        (validated tiering honesty test — unchanged)
  - ablation.json                (validated ablation results — unchanged)
  - district_baselines.csv      (NEW — district-level aggregates, for the simulator)
  - district_model.joblib       (NEW — Ridge model trained on district aggregates)
  - district_model_meta.json    (NEW — LOOCV performance + feature ranges, for honest display)
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_excel("dataset_final.xlsx")

# ------------------------------------------------------------------
# 1. school-level file (unchanged from original notebook)
# ------------------------------------------------------------------
dashboard_cols = ['school_code', 'school_name', 'province', 'district', 'Sector',
                   'school_location', 'school_status', 'school_settings',
                   'perc_with_50andplus', 'on_grid_electricity', 'has_internet',
                   'has_computer_access', 'avg_classroom_size_2024_25', 'double_shift',
                   'Leadership_marks_avg', 'Num_leader_records']
df_dashboard = df[dashboard_cols].copy()
df_dashboard.to_csv("data/schools.csv", index=False)
print(f"Saved {len(df_dashboard)} schools to data/schools.csv")

# ------------------------------------------------------------------
# 2. validated study reference files (unchanged from original notebook)
# ------------------------------------------------------------------
cv_results = {
    "Linear Regression":   {"MAE": 11.44, "MAE_sd": 0.26, "RMSE": 14.55, "RMSE_sd": 0.45, "R2": 0.215, "R2_sd": 0.095},
    "Random Forest":       {"MAE": 11.18, "MAE_sd": 0.35, "RMSE": 14.34, "RMSE_sd": 0.53, "R2": 0.238, "R2_sd": 0.097},
    "Gradient Boosting":   {"MAE": 11.26, "MAE_sd": 0.36, "RMSE": 14.31, "RMSE_sd": 0.59, "R2": 0.242, "R2_sd": 0.088},
    "Baseline (Province)": {"MAE": 11.38, "MAE_sd": 0,    "RMSE": 14.94, "RMSE_sd": 0,    "R2": 0.047, "R2_sd": 0},
}
feature_importance = [
    {"feature": "Average classroom size", "standardized_coef": -6.860, "shap": 3.679},
    {"feature": "Primary-focused status", "standardized_coef": 3.271, "shap": 0.684},
    {"feature": "Urban location", "standardized_coef": 2.522, "shap": 1.358},
    {"feature": "Province: North", "standardized_coef": -2.591, "shap": 1.079},
    {"feature": "Number of leader records", "standardized_coef": 1.602, "shap": 0.329},
    {"feature": "Province: West", "standardized_coef": -1.432, "shap": 0.359},
    {"feature": "Leadership marks", "standardized_coef": 1.085, "shap": 0.904},
    {"feature": "Double-shift status", "standardized_coef": 0.619, "shap": 0.092},
    {"feature": "Number of education levels", "standardized_coef": 0.590, "shap": 0.538},
    {"feature": "Internet access", "standardized_coef": 0.459, "shap": 0.044},
    {"feature": "Electricity access", "standardized_coef": -0.363, "shap": 0.109},
    {"feature": "Computer access", "standardized_coef": -0.002, "shap": 0.151},
]
# Exact figures pulled from the thesis notebook's Step 16/17 cell outputs
# (04082026_FinalThesis_Analysis_1.ipynb), not re-derived here.
tiering_honesty = {
    "in_sample_accuracy": 0.674, "in_sample_low_precision": 0.92,
    "oof_accuracy": 0.629, "oof_low_precision": 0.38, "oof_low_recall": 0.05,
    "binary_pr_auc": 0.302, "binary_roc_auc": 0.610, "binary_baseline_pr_auc": 0.231,
}
# Cluster-robust OLS (SEs clustered by district) from Step 17 of the notebook.
# R2 is unchanged by clustering (only SEs/p-values shift) -- clustering only
# tests whether significance survives once within-district correlation is
# accounted for.
cluster_robust_check = {
    "r2": 0.281,
    "n_vars_changed_significance": 0,
    "n_vars_tested": 17,
    "still_significant_after_clustering": [
        "Average classroom size", "Primary-focused status", "Leadership marks",
        "Number of leader records", "Province: North", "Province: West",
        "Urban location",
    ],
}
ablation = {"geography_only_r2": 0.118, "infra_leadership_focus_r2": 0.241, "full_model_r2": 0.293}

with open("data/cv_results.json", "w") as f: json.dump(cv_results, f, indent=2)
with open("data/feature_importance.json", "w") as f: json.dump(feature_importance, f, indent=2)
with open("data/tiering_honesty.json", "w") as f: json.dump(tiering_honesty, f, indent=2)
with open("data/ablation.json", "w") as f: json.dump(ablation, f, indent=2)
with open("data/cluster_robust_check.json", "w") as f: json.dump(cluster_robust_check, f, indent=2)
print("Saved validated reference files.")

# ------------------------------------------------------------------
# 3. NEW: district-level aggregates (baselines for the simulator)
# ------------------------------------------------------------------
data = df.dropna(subset=["district"]).copy()

district_agg = data.groupby(["district", "province"]).agg(
    n_schools=("school_code", "count"),
    avg_classroom_size=("avg_classroom_size_2024_25", "mean"),
    pct_double_shift=("double_shift", "mean"),
    pct_electricity=("on_grid_electricity", "mean"),
    pct_internet=("has_internet", "mean"),
    pct_computer=("has_computer_access", "mean"),
    avg_leadership_marks=("Leadership_marks_avg", "mean"),
    avg_leader_records=("Num_leader_records", "mean"),
    pct_urban=("school_location", lambda s: (s == "URBAN").mean()),
    actual_pass_rate=("perc_with_50andplus", "mean"),
).reset_index()

# convert fractions to 0-100 percent scale for display/sliders
for col in ["pct_double_shift", "pct_electricity", "pct_internet", "pct_computer", "pct_urban"]:
    district_agg[col] = (district_agg[col] * 100).round(1)
district_agg["avg_classroom_size"] = district_agg["avg_classroom_size"].round(1)
district_agg["avg_leadership_marks"] = district_agg["avg_leadership_marks"].round(1)
district_agg["avg_leader_records"] = district_agg["avg_leader_records"].round(2)
district_agg["actual_pass_rate"] = district_agg["actual_pass_rate"].round(1)

district_agg.to_csv("data/district_baselines.csv", index=False)
print(f"Saved {len(district_agg)} districts to data/district_baselines.csv")

# ------------------------------------------------------------------
# 4. NEW: district-level Ridge model (levers only — no district identity)
#    Honestly validated with Leave-One-Out CV (appropriate for n=30)
# ------------------------------------------------------------------
# NOTE: pct_electricity is dropped — it's ~100% in every district (no real variance
# to learn from), consistent with the "weak/non-significant" electricity effect
# already noted on the school-level page. It's still shown as context, not a lever.
feat_cols = ["avg_classroom_size", "pct_double_shift", "pct_internet",
             "pct_computer", "avg_leadership_marks", "avg_leader_records", "pct_urban"]
X = district_agg[feat_cols].values
y = district_agg["actual_pass_rate"].values

loo = LeaveOneOut()
oof_preds = np.zeros(len(y))
for train_idx, test_idx in loo.split(X):
    pipe = Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=5.0))])
    pipe.fit(X[train_idx], y[train_idx])
    oof_preds[test_idx] = pipe.predict(X[test_idx])

loocv_r2 = r2_score(y, oof_preds)
loocv_mae = mean_absolute_error(y, oof_preds)
baseline_mae = mean_absolute_error(y, np.full_like(y, y.mean()))

# fit final model on ALL 30 districts for deployment
final_model = Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=5.0))])
final_model.fit(X, y)
joblib.dump(final_model, "data/district_model.joblib")

meta = {
    "feature_order": feat_cols,
    "loocv_r2": round(loocv_r2, 3),
    "loocv_mae": round(loocv_mae, 2),
    "baseline_mae_national_mean": round(baseline_mae, 2),
    "n_districts": int(len(y)),
    "feature_ranges": {
        c: {"min": float(district_agg[c].min()), "max": float(district_agg[c].max()),
            "mean": float(district_agg[c].mean())}
        for c in feat_cols
    },
    "national_avg_pass_rate": round(float(y.mean()), 1),
}
with open("data/district_model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nDistrict model (Leave-One-Out CV, honest test):")
print(f"  R2  = {meta['loocv_r2']}")
print(f"  MAE = {meta['loocv_mae']}  (baseline MAE = {meta['baseline_mae_national_mean']})")
print("\nAll data files ready in data/")
