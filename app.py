import streamlit as st
import pandas as pd
import json
import plotly.express as px

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
    return schools, cv_results, feature_importance, tiering, ablation

schools, cv_results, feature_importance, tiering, ablation = load_data()

st.sidebar.title("Rwanda Primary Schools")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "What Predicts Performance", "Explore by School",
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
            "explain much *additional* variation in pass rates."
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
# PAGE 4: MODEL PERFORMANCE & LIMITATIONS
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
    st.subheader("Other Known Limitations")
    st.markdown(
        "- Classroom size, shift status, and computer access are **national estimates by school type**, not measurements of individual schools.\n"
        "- Leadership data does not exist for Private schools in the current evaluation system.\n"
        "- Pass rates from very small schools may be noisy (e.g., a school with few students can swing to 0% or 100% easily).\n"
        "- This model covers **primary level (P1-P6) only**."
    )

# ============================================================
# PAGE 5: POLICY RECOMMENDATIONS
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
        "measurable effect on primary pass rates of any intervention tested."
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
