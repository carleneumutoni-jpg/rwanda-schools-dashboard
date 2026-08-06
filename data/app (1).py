import streamlit as st
import pandas as pd
import json
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Rwanda Primary School Performance Dashboard",
    page_icon="\U0001F3EB",
    layout="wide",
)

@st.cache_data
def load_data():
    schools = pd.read_csv("data/schools.csv")
    with open("data/cv_results.json") as f:
        cv_results = json.load(f)
    with open("data/feature_importance.json") as f:
        feature_importance = json.load(f)
    with open("data/tiering_honesty.json") as f:
        tiering = json.load(f)
    with open("data/ablation.json") as f:
        ablation = json.load(f)
    district_baselines = pd.read_csv("data/district_baselines.csv")
    with open("data/district_model_meta.json") as f:
        district_meta = json.load(f)
    with open("data/cluster_robust_check.json") as f:
        cluster_check = json.load(f)
    return (schools, cv_results, feature_importance, tiering, ablation,
            district_baselines, district_meta, cluster_check)

@st.cache_resource
def load_district_model():
    return joblib.load("data/district_model.joblib")

(schools, cv_results, feature_importance, tiering, ablation,
 district_baselines, district_meta, cluster_check) = load_data()
district_model = load_district_model()

st.sidebar.title("Rwanda Primary Schools")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "What Predicts Performance", "Explore by School",
     "District Funding Simulator",
     "Model Performance & Limitations", "Policy Recommendations"],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Built on 2024/2025 school performance, infrastructure, and leadership data, "
    "supplemented with the MINEDUC Education Statistical Yearbook."
)

# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
if page == "Overview":
    st.title("Primary-Level Student Performance in Rwanda")
    st.markdown("An evidence-based view of what predicts primary pass rates nationally.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Schools Analyzed", f"{len(schools):,}")
    col2.metric("National Mean Pass Rate", f"{schools['perc_with_50andplus'].mean():.1f}%")
    col3.metric("Best Model (Gradient Boosting) R2", "0.242", help="Cross-validated, mean of 5 folds")
    col4.metric("Schools Below 50% Pass Rate", f"{(schools['perc_with_50andplus'] < 50).sum():,}")

    st.markdown("---")
    st.subheader("Average Pass Rate by Province")
    province_avg = schools.groupby("province")["perc_with_50andplus"].mean().sort_values(ascending=False).reset_index()
    fig1 = px.bar(
        province_avg, x="province", y="perc_with_50andplus",
        labels={"perc_with_50andplus": "Average Pass Rate (%)", "province": "Province"},
        color="perc_with_50andplus", color_continuous_scale="Blues",
    )
    fig1.update_layout(coloraxis_showscale=False, height=400)
    st.plotly_chart(fig1, use_container_width=True, key="overview_province_chart")

    st.info(
        "How to read this dashboard: This tool shows validated, "
        "cross-checked patterns from a national predictive model (R2 approx. 0.22-0.24). "
        "It is reliable for identifying provincial and district-level trends and "
        "which factors matter most on average. It is not yet reliable for "
        "flagging individual schools - see the 'Model Performance & Limitations' "
        "page for why."
    )

# ============================================================
# PAGE 2: WHAT PREDICTS PERFORMANCE
# ============================================================
elif page == "What Predicts Performance":
    st.title("What Predicts Primary-Level Performance?")
    st.markdown(
        "Ranked by **SHAP importance** (mean impact on predictions) and cross-checked "
        "against standardized regression coefficients."
    )

    fi_df = pd.DataFrame(feature_importance).sort_values("shap", ascending=True)
    fig2 = px.bar(
        fi_df, x="shap", y="feature", orientation="h",
        labels={"shap": "Mean |SHAP value| (impact on predicted pass rate)", "feature": ""},
        color="shap", color_continuous_scale="Blues",
        height=550,
    )
    fig2.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True, key="feature_importance_chart")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Classroom Overcrowding")
        st.markdown(
            "**The single strongest predictor.** Schools with more pupils per "
            "classroom consistently show lower pass rates - this held true even "
            "under the most conservative statistical tests applied in this study."
        )
        st.subheader("School Focus")
        st.markdown(
            "Schools that only teach Preprimary/Primary levels outperform schools "
            "that also manage secondary or TVET levels - a new finding from this study."
        )
    with col2:
        st.subheader("Leadership")
        st.markdown(
            "Schools with higher leadership evaluation marks, and a **complete** "
            "leadership team (Head Teacher + Deputy), perform significantly better "
            "- though the effect is smaller than classroom size."
        )
        st.subheader("Electricity, Internet, Computers")
        st.markdown(
            "Surprisingly **weak, mostly non-significant effects** once classroom "
            "size and location are accounted for. This does not mean digital "
            "infrastructure is unimportant - only that, in this data, it does not "
            "explain much *additional* variation in pass rates. (Electricity access "
            "is also ~100% across nearly all schools in this dataset, leaving little "
            "variation to learn from either way.)"
        )

    st.markdown("---")
    st.subheader("Infrastructure & Leadership vs. Geography Alone")
    ab_col1, ab_col2, ab_col3 = st.columns(3)
    ab_col1.metric("Geography Only", f"R2 = {ablation['geography_only_r2']}")
    ab_col2.metric("Infrastructure + Leadership + Focus", f"R2 = {ablation['infra_leadership_focus_r2']}")
    ab_col3.metric("Full Model", f"R2 = {ablation['full_model_r2']}")
    st.caption(
        "Infrastructure, leadership, and school-focus variables explain roughly "
        "**2x more variance** than geography (e.g., province) alone - supporting "
        "targeted investment over purely region-based allocation."
    )

# ============================================================
# PAGE 3: EXPLORE BY SCHOOL
# ============================================================
elif page == "Explore by School":
    st.title("Explore Schools")
    st.markdown("Filter and browse actual school records (not model predictions).")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        province_filter = st.multiselect("Province", sorted(schools["province"].dropna().unique()))
    with col2:
        status_filter = st.multiselect("School Status", sorted(schools["school_status"].dropna().unique()))
    with col3:
        location_filter = st.multiselect("Location", sorted(schools["school_location"].dropna().unique()))
    with col4:
        pass_range = st.slider("Pass Rate Range (%)", 0, 100, (0, 100))

    filtered = schools.copy()
    if province_filter:
        filtered = filtered[filtered["province"].isin(province_filter)]
    if status_filter:
        filtered = filtered[filtered["school_status"].isin(status_filter)]
    if location_filter:
        filtered = filtered[filtered["school_location"].isin(location_filter)]
    filtered = filtered[
        (filtered["perc_with_50andplus"] >= pass_range[0]) &
        (filtered["perc_with_50andplus"] <= pass_range[1])
    ]

    st.markdown(f"**{len(filtered):,} schools match your filters** (out of {len(schools):,} total)")

    display_cols = ["school_name", "province", "district", "school_status", "school_location",
                     "perc_with_50andplus", "avg_classroom_size_2024_25", "Leadership_marks_avg"]
    display_df = filtered[display_cols].rename(columns={
        "school_name": "School", "province": "Province", "district": "District",
        "school_status": "Status", "school_location": "Location",
        "perc_with_50andplus": "Pass Rate (%)",
        "avg_classroom_size_2024_25": "Avg. Classroom Size*",
        "Leadership_marks_avg": "Leadership Marks",
    })
    st.dataframe(display_df, use_container_width=True, height=450)

    st.caption(
        "*Average Classroom Size is a national estimate by school ownership type "
        "(from the MINEDUC Yearbook), not a measurement of this specific school's "
        "actual classrooms. See 'Model Performance & Limitations' for details."
    )

    csv = filtered[display_cols].to_csv(index=False)
    st.download_button("Download filtered data as CSV", csv, "filtered_schools.csv", "text/csv")

# ============================================================
# PAGE 4: DISTRICT FUNDING SIMULATOR  (NEW)
# ============================================================
elif page == "District Funding Simulator":
    st.title("District Funding Simulator")
    st.markdown(
        "Estimate how district-average pass rates would change if infrastructure, "
        "leadership, or classroom-size levers were shifted through funding decisions."
    )

    st.warning(
        f"**About this model:** trained on all {district_meta['n_districts']} districts' "
        f"aggregate data and honestly tested with leave-one-out cross-validation "
        f"(each district predicted using a model that never saw it). "
        f"**R2 = {district_meta['loocv_r2']}, average error = \u00b1{district_meta['loocv_mae']} points** "
        f"(vs. \u00b1{district_meta['baseline_mae_national_mean']} points for just guessing the national average). "
        "This is real but modest signal - use it to compare directions and rough "
        "magnitudes across funding options, not as a precise forecast."
    )

    st.markdown("---")
    st.subheader("1. Start from a district's current numbers")
    district_list = sorted(district_baselines["district"].unique())
    selected_district = st.selectbox("Select a district", district_list)
    baseline_row = district_baselines[district_baselines["district"] == selected_district].iloc[0]

    st.caption(
        f"{selected_district} ({baseline_row['province']} Province) has "
        f"{int(baseline_row['n_schools'])} schools and a current actual average "
        f"pass rate of **{baseline_row['actual_pass_rate']}%**."
    )

    st.markdown("---")
    st.subheader("2. Adjust funding levers")
    st.caption("Sliders start at this district's real current values. Move them to simulate a funding scenario.")

    ranges = district_meta["feature_ranges"]

    def slider_bounds(col, pad_ratio=0.15, lo_clip=None, hi_clip=None):
        lo, hi = ranges[col]["min"], ranges[col]["max"]
        pad = max((hi - lo) * pad_ratio, 1.0)
        lo, hi = lo - pad, hi + pad
        if lo_clip is not None:
            lo = max(lo, float(lo_clip))
        if hi_clip is not None:
            hi = min(hi, float(hi_clip))
        lo, hi = round(float(lo), 1), round(float(hi), 1)
        if lo >= hi:  # guard against a degenerate (zero-width) slider
            hi = lo + 1.0
        return lo, hi

    def clamp(value, lo, hi):
        # Guards against a district's own value landing a hair outside its
        # own rounded slider bounds (can happen from independent rounding
        # in build_data.py vs. here), which Streamlit's slider rejects.
        return float(min(max(value, lo), hi))

    c1, c2 = st.columns(2)
    with c1:
        lo, hi = slider_bounds("avg_classroom_size", lo_clip=20)
        avg_classroom_size = st.slider("Avg. classroom size (pupils/class)", lo, hi,
                                        clamp(float(baseline_row["avg_classroom_size"]), lo, hi), step=0.5)
        st.caption("Rwanda's national standard is 46:1.")

        lo, hi = slider_bounds("pct_double_shift", lo_clip=0, hi_clip=100)
        pct_double_shift = st.slider("% schools on double shift", lo, hi,
                                      clamp(float(baseline_row["pct_double_shift"]), lo, hi), step=1.0)

        lo, hi = slider_bounds("pct_internet", lo_clip=0, hi_clip=100)
        pct_internet = st.slider("% schools with internet access", lo, hi,
                                  clamp(float(baseline_row["pct_internet"]), lo, hi), step=1.0)

        lo, hi = slider_bounds("pct_computer", lo_clip=0, hi_clip=100)
        pct_computer = st.slider("% schools with computer access", lo, hi,
                                  clamp(float(baseline_row["pct_computer"]), lo, hi), step=1.0)

    with c2:
        lo, hi = slider_bounds("avg_leadership_marks", lo_clip=0, hi_clip=100)
        avg_leadership_marks = st.slider("Avg. leadership evaluation marks", lo, hi,
                                          clamp(float(baseline_row["avg_leadership_marks"]), lo, hi), step=0.5)

        lo, hi = slider_bounds("avg_leader_records", lo_clip=1.0, hi_clip=2.0)
        avg_leader_records = st.slider("Avg. leadership team completeness (1=Head only, 2=Head+Deputy)", lo, hi,
                                        clamp(float(baseline_row["avg_leader_records"]), lo, hi), step=0.05)

        st.markdown("**Fixed context (not funding-adjustable):**")
        st.text(f"Province: {baseline_row['province']}")
        st.text(f"% urban schools: {baseline_row['pct_urban']}%")
        st.text("% on-grid electricity: ~100% (already universal, not a useful lever)")

    pct_urban = float(baseline_row["pct_urban"])  # fixed, not editable

    # ---- Predict ----
    feat_order = district_meta["feature_order"]
    input_row = pd.DataFrame([{
        "avg_classroom_size": avg_classroom_size,
        "pct_double_shift": pct_double_shift,
        "pct_internet": pct_internet,
        "pct_computer": pct_computer,
        "avg_leadership_marks": avg_leadership_marks,
        "avg_leader_records": avg_leader_records,
        "pct_urban": pct_urban,
    }])[feat_order]

    predicted = float(district_model.predict(input_row)[0])
    predicted = max(0.0, min(100.0, predicted))
    mae = district_meta["loocv_mae"]

    # Extrapolation guardrail: flag if any input falls outside the range of
    # districts the model was actually trained on
    out_of_range = []
    label_map = {
        "avg_classroom_size": "Avg. classroom size", "pct_double_shift": "% double shift",
        "pct_internet": "% internet", "pct_computer": "% computer access",
        "avg_leadership_marks": "Avg. leadership marks", "avg_leader_records": "Leadership team completeness",
    }
    for col, val in input_row.iloc[0].items():
        if col == "pct_urban":
            continue
        r = ranges[col]
        if val < r["min"] or val > r["max"]:
            out_of_range.append(f"{label_map.get(col, col)} ({val:g}, observed range {r['min']:g}\u2013{r['max']:g})")

    st.markdown("---")
    st.subheader("3. Estimated result")

    r1, r2_, r3 = st.columns(3)
    r1.metric("Current actual pass rate", f"{baseline_row['actual_pass_rate']}%")
    r2_.metric(
        "Scenario estimate", f"{predicted:.1f}%",
        delta=f"{predicted - baseline_row['actual_pass_rate']:+.1f} pts vs. current",
    )
    r3.metric("Uncertainty range", f"{max(0,predicted-mae):.1f}% \u2013 {min(100,predicted+mae):.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Current actual", "Scenario estimate"],
        y=[baseline_row["actual_pass_rate"], predicted],
        error_y=dict(type="data", array=[0, mae], visible=True),
        marker_color=["#6c9bd1", "#1f4e8c"],
    ))
    fig.update_layout(yaxis_title="Pass rate (%)", yaxis_range=[0, 100], height=380)
    st.plotly_chart(fig, use_container_width=True, key="district_scenario_chart")

    if out_of_range:
        st.warning(
            "**Extrapolation notice:** this scenario goes beyond what any district in "
            "the training data actually looks like, for: " + "; ".join(out_of_range) + ". "
            "The model is estimating outside its validated range here, so treat this "
            "particular estimate as a rough directional signal only, not the "
            "\u00b1{:.1f}-point accuracy quoted above.".format(mae)
        )

    st.caption(
        "This estimate reflects a district-level pattern, not a guarantee for any "
        "individual school within the district. Use alongside local knowledge before "
        "committing funding."
    )

# ============================================================
# PAGE 5: MODEL PERFORMANCE & LIMITATIONS
# ============================================================
elif page == "Model Performance & Limitations":
    st.title("How Reliable Is This Model?")
    st.markdown(
        "This page explains honestly what this model can and cannot do, "
        "so results are used appropriately."
    )

    st.subheader("Model Accuracy")
    cv_df = pd.DataFrame(cv_results).T.reset_index().rename(columns={"index": "Model"})
    fig3 = px.bar(
        cv_df, x="Model", y="R2", error_y="R2_sd",
        labels={"R2": "R2 (higher = better)"},
        color="R2", color_continuous_scale="Blues",
    )
    fig3.update_layout(coloraxis_showscale=False, height=380)
    st.plotly_chart(fig3, use_container_width=True, key="model_accuracy_chart")

    st.warning(
        "**What R2 means here:** the best model explains about **22-24% of the "
        "variation** in school pass rates. The error bars show real uncertainty: "
        "depending on which schools are tested, this could range from about "
        "**12% to 33%**. The remaining variation is likely driven by factors this "
        "dataset does not capture - teacher quality, student home environment, "
        "and instructional time, among others."
    )

    st.markdown("---")
    st.subheader("Important: This Model Cannot Yet Flag Individual Schools")
    st.markdown(
        "An early version of this tool attempted to automatically flag specific "
        "'high-concern' schools for intervention. **We tested this rigorously and "
        "found it does not work reliably yet** - here is that test, shown honestly:"
    )

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Naive test (misleading)**")
        st.metric("Accuracy", f"{tiering['in_sample_accuracy']:.0%}")
        st.metric("Precision flagging concern schools", f"{tiering['in_sample_low_precision']:.0%}")
        st.caption("This test let the model 'see' the answers in advance - like grading a student on questions they already had the answer key for.")
    with t2:
        st.markdown("**Honest test (trustworthy)**")
        st.metric("Accuracy", f"{tiering['oof_accuracy']:.0%}")
        st.metric("Precision flagging concern schools", f"{tiering['oof_low_precision']:.0%}")
        st.caption("This test only used predictions for schools the model had never seen - the fair, real-world test.")

    st.error(
        "**Bottom line:** province- and district-level patterns in this dashboard "
        "are reliable and safe to act on. Flagging **individual named schools** for "
        "intervention based on this model alone is **not yet reliable** and should "
        "not be done without additional local verification."
    )

    st.markdown("---")
    st.subheader("Robustness Check: Do the Key Predictors Hold Up?")
    st.markdown(
        "Schools within the same district aren't fully independent of each other - "
        "they share the same local administration, funding environment, and "
        "community context. Ordinary statistical tests assume independence, which "
        "can make effects look more certain than they really are. To check this, "
        "the model's coefficients were re-tested with **standard errors clustered "
        "by district**, which is a stricter, more conservative test."
    )
    cr1, cr2 = st.columns(2)
    cr1.metric("Variables tested", cluster_check["n_vars_tested"])
    cr2.metric("Variables that lost significance after clustering",
               cluster_check["n_vars_changed_significance"])
    st.success(
        "**Result: none of the key predictors lost significance.** Classroom size, "
        "leadership marks, leadership-team completeness, primary-focused status, "
        "urban location, and the North/West province effects all remained "
        "statistically significant even under this stricter test - meaning these "
        "aren't artifacts of schools within the same district being correlated "
        "with each other."
    )

    st.markdown("---")
    st.subheader("District Funding Simulator - Model Details")
    st.markdown(
        f"- Trained on **{district_meta['n_districts']} districts'** aggregated data "
        f"(one row per district, averaged across its schools).\n"
        f"- Validated with **leave-one-out cross-validation**: R2 = {district_meta['loocv_r2']}, "
        f"MAE = \u00b1{district_meta['loocv_mae']} points (baseline of guessing the national "
        f"average: \u00b1{district_meta['baseline_mae_national_mean']} points).\n"
        "- District identity itself is **not** used as an input - only funding-relevant levers "
        "(classroom size, connectivity, leadership) - so estimates reflect real relationships, "
        "not memorized district labels.\n"
        "- With only 30 districts, this model is intentionally simple (regularized linear "
        "regression) to avoid overfitting a small sample."
    )

    st.markdown("---")
    st.subheader("Other Known Limitations")
    st.markdown(
        "- Classroom size, shift status, and computer access are **national estimates by school type**, not measurements of individual schools.\n"
        "- Leadership data does not exist for Private schools in the current evaluation system.\n"
        "- Pass rates from very small schools may be noisy (e.g., a school with few students can swing to 0% or 100% easily).\n"
        "- This model covers **primary level (P1-P6) only**."
    )

# ============================================================
# PAGE 6: POLICY RECOMMENDATIONS
# ============================================================
else:
    st.title("Policy Recommendations")
    st.markdown("Evidence-based priorities, ranked by demonstrated impact.")

    st.markdown("### 1. Prioritize Classroom Construction")
    st.markdown(
        "This is the **single highest-impact lever** identified in this study. "
        "Public schools average **64 pupils per classroom** and Government-aided "
        "schools average **58** - both far above Rwanda's national standard of "
        "**46:1**. Reducing classroom overcrowding is likely to have the largest "
        "measurable effect on primary pass rates of any intervention tested. "
        "Use the **District Funding Simulator** page to estimate the effect for a "
        "specific district."
    )

    st.markdown("### 2. Ensure Every School Has a Complete Leadership Team")
    st.markdown(
        "Schools with both a Head Teacher AND a Deputy Head Teacher perform "
        "significantly better than schools with only one leadership position "
        "filled. This is a comparatively **low-cost** intervention relative to "
        "classroom construction."
    )

    st.markdown("### 3. Target Investment by Province, Not by Named School")
    st.markdown(
        "**North Province** shows significantly lower average pass rates even "
        "after accounting for other factors - a real, statistically robust "
        "pattern. However, this dashboard cannot yet reliably identify *which "
        "specific schools* within North Province need the most urgent help "
        "(see Model Performance & Limitations). We recommend province- and "
        "district-level budget prioritization, combined with **local verification** "
        "before committing resources to any single named school."
    )

    st.markdown("### 4. Don't Assume All Infrastructure Investment Is Equal")
    st.markdown(
        "Electricity, internet, and computer access showed **weak, largely "
        "non-significant effects** on pass rates once classroom size and location "
        "were accounted for. These may still be valuable for other policy goals "
        "(digital literacy, administrative efficiency), but **should not be "
        "assumed to be the most effective lever specifically for raising primary "
        "pass rates.**"
    )

    st.markdown("---")
    st.success(
        "This dashboard is based on a peer-reviewable Master's thesis analysis "
        "using national administrative data, cross-validated machine learning "
        "models, and multiple robustness checks. Full methodology available on request."
    )
