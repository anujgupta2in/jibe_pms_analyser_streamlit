"""Page 11 — Vessel Deep-Dive: Comprehensive Per-Vessel Report Dashboard"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
from data_loader import (
    SATISFACTION_COL, SUPPORT_COL, CORRECTION_COL, TENURE_COL,
    RANK_COL, QUALITY_COLS, SOFTWARE_COLS, LIKED_COL, EXCESSIVE_COL,
    ORDINAL_MAPS, PLOTLY_THEME, PRIMARY,
)

def render(data: dict):
    merged = data["merged"]
    df = data["survey"]

    st.title("🔍 Vessel Wise Analysis")
    st.markdown(
        "Generate a complete, dedicated performance and feedback deep-dive report for any individual vessel. "
        "Select a vessel below to examine its metrics and compare them to fleet averages."
    )
    st.markdown("---")

    # Get list of unique vessels with at least 1 response
    vessels = sorted(merged["Vessel Name"].dropna().unique().tolist())
    if not vessels:
        st.warning("No vessel data available.")
        return

    # Find default index for "Badrinath" if it exists in the list
    default_idx = 0
    if "Badrinath" in vessels:
        default_idx = vessels.index("Badrinath")

    # Vessel Selector
    selected_vessel = st.selectbox(
        "Select Vessel for Deep-Dive Analysis",
        vessels,
        index=default_idx,
        help="Search and select a vessel from the survey data."
    )

    # Filter data for this specific vessel
    v_data = merged[merged["Vessel Name"] == selected_vessel]
    fleet_avg_sat = merged[SATISFACTION_COL].mean()
    fleet_avg_sup = merged[SUPPORT_COL].mean()
    
    # Calculate high correction rates
    high_corr_set = {"10–20%", "More than 20%"}
    fleet_high_corr_pct = (merged[CORRECTION_COL].isin(high_corr_set).sum() / merged[CORRECTION_COL].dropna().count() * 100) if merged[CORRECTION_COL].dropna().count() > 0 else 0
    
    n_resp = len(v_data)
    if n_resp == 0:
        st.warning(f"No survey responses found for vessel: **{selected_vessel}**.")
        return

    # Vessel Metadata
    v_type = v_data["Type"].dropna().mode().iloc[0] if v_data["Type"].notna().any() else "—"
    v_div = v_data["Division Code"].dropna().mode().iloc[0] if v_data["Division Code"].notna().any() else "—"
    v_golive = v_data["PMS Go-Live"].dropna().iloc[0] if "PMS Go-Live" in v_data.columns and v_data["PMS Go-Live"].notna().any() else "—"
    if isinstance(v_golive, pd.Timestamp):
        v_golive = v_golive.strftime('%Y-%m-%d')

    # Metric calculations for this vessel
    v_avg_sat = v_data[SATISFACTION_COL].mean()
    v_avg_sup = v_data[SUPPORT_COL].mean()
    v_corr_series = v_data[CORRECTION_COL].dropna()
    v_high_corr_pct = (v_corr_series.isin(high_corr_set).sum() / len(v_corr_series) * 100) if len(v_corr_series) > 0 else 0

    # Layout: Header cards
    st.markdown(f"### 🚢 Vessel Profile: **{selected_vessel}**")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.info(f"**Vessel Type**\n### {v_type}")
    with m2:
        st.info(f"**Division**\n### {v_div}")
    with m3:
        st.info(f"**PMS Go-Live**\n### {v_golive}")
    with m4:
        st.info(f"**Total Respondents**\n### {n_resp}")

    st.markdown("---")

    # Section 1: Vessel vs Fleet Performance
    st.subheader("📊 Performance Scorecards (Vessel vs. Fleet Average)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        delta_sat = v_avg_sat - fleet_avg_sat if pd.notna(v_avg_sat) else 0.0
        st.metric(
            label="Avg Overall Satisfaction",
            value=f"{v_avg_sat:.2f}/10" if pd.notna(v_avg_sat) else "—",
            delta=f"{delta_sat:+.2f} vs Fleet Avg ({fleet_avg_sat:.2f})",
            delta_color="normal" if delta_sat >= 0 else "inverse"
        )
        
    with col2:
        delta_sup = v_avg_sup - fleet_avg_sup if pd.notna(v_avg_sup) else 0.0
        st.metric(
            label="Avg Support Rating",
            value=f"{v_avg_sup:.2f}/10" if pd.notna(v_avg_sup) else "—",
            delta=f"{delta_sup:+.2f} vs Fleet Avg ({fleet_avg_sup:.2f})",
            delta_color="normal" if delta_sup >= 0 else "inverse"
        )
        
    with col3:
        delta_corr = v_high_corr_pct - fleet_high_corr_pct
        st.metric(
            label="High Correction Rate (≥10%)",
            value=f"{v_high_corr_pct:.1f}%",
            delta=f"{delta_corr:+.1f}% vs Fleet Avg ({fleet_high_corr_pct:.1f}%)",
            delta_color="inverse" if delta_corr >= 0 else "normal"  # lower correction rate is better
        )

    st.markdown("---")

    # Section 2: Visualizations
    st.subheader("📈 Satisfaction & Support Score Distribution")
    dist_col1, dist_col2 = st.columns(2)
    
    with dist_col1:
        # Satisfaction distribution
        sat_counts = v_data[SATISFACTION_COL].value_counts().reindex(range(1, 11), fill_value=0)
        fig_sat = go.Figure(go.Bar(
            x=list(sat_counts.index),
            y=list(sat_counts.values),
            marker_color=PRIMARY,
            text=list(sat_counts.values),
            textposition="outside"
        ))
        fig_sat.update_layout(
            template=PLOTLY_THEME,
            height=280,
            xaxis=dict(tickmode="linear", tick0=1, dtick=1),
            xaxis_title="Overall Satisfaction score (1-10)",
            yaxis_title="Respondents",
            margin=dict(t=30, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_sat, use_container_width=True)

    with dist_col2:
        # Support distribution
        sup_counts = v_data[SUPPORT_COL].value_counts().reindex(range(1, 11), fill_value=0)
        fig_sup = go.Figure(go.Bar(
            x=list(sup_counts.index),
            y=list(sup_counts.values),
            marker_color="#0ea5e9",
            text=list(sup_counts.values),
            textposition="outside"
        ))
        fig_sup.update_layout(
            template=PLOTLY_THEME,
            height=280,
            xaxis=dict(tickmode="linear", tick0=1, dtick=1),
            xaxis_title="Support Rating score (1-10)",
            yaxis_title="Respondents",
            margin=dict(t=30, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_sup, use_container_width=True)

    st.markdown("---")

    # Section 3: Data Quality & Software Capabilities
    dq_col1, dq_col2 = st.columns(2)
    
    with dq_col1:
        st.subheader("📋 Data Quality Score comparison")
        qual_cols_avail = [c for c in QUALITY_COLS if c in v_data.columns]
        if qual_cols_avail:
            ordinal = {"Acceptable": 3, "Needs Improvement": 2, "Poor": 1}
            v_scores = []
            f_scores = []
            labels = []
            
            for orig in qual_cols_avail:
                alias = QUALITY_COLS[orig]
                v_avg = v_data[orig].map(ordinal).mean()
                f_avg = merged[orig].map(ordinal).mean()
                v_scores.append(v_avg if pd.notna(v_avg) else 0)
                f_scores.append(f_avg if pd.notna(f_avg) else 0)
                labels.append(alias)
                
            fig_q = go.Figure()
            fig_q.add_trace(go.Bar(
                y=labels, x=v_scores, name=selected_vessel,
                orientation="h", marker_color=PRIMARY
            ))
            fig_q.add_trace(go.Bar(
                y=labels, x=f_scores, name="Fleet Avg",
                orientation="h", marker_color="#cbd5e1"
            ))
            fig_q.update_layout(
                template=PLOTLY_THEME,
                barmode="group",
                height=300,
                xaxis_title="Avg Rating (1=Poor, 2=Needs Imp, 3=Acceptable)",
                yaxis_title="",
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_q, use_container_width=True)
        else:
            st.info("No data quality columns available.")

    with dq_col2:
        st.subheader("💻 Software Capabilities comparison")
        soft_cols_avail = [c for c in SOFTWARE_COLS if c in v_data.columns]
        if soft_cols_avail:
            ordinal = {"Good": 3, "Needs Improvement": 2, "Poor": 1}
            v_scores = []
            f_scores = []
            labels = []
            
            for orig in soft_cols_avail:
                alias = SOFTWARE_COLS[orig]
                v_avg = v_data[orig].map(ordinal).mean()
                f_avg = merged[orig].map(ordinal).mean()
                v_scores.append(v_avg if pd.notna(v_avg) else 0)
                f_scores.append(f_avg if pd.notna(f_avg) else 0)
                labels.append(alias)
                
            fig_s = go.Figure()
            fig_s.add_trace(go.Bar(
                y=labels, x=v_scores, name=selected_vessel,
                orientation="h", marker_color="#0ea5e9"
            ))
            fig_s.add_trace(go.Bar(
                y=labels, x=f_scores, name="Fleet Avg",
                orientation="h", marker_color="#cbd5e1"
            ))
            fig_s.update_layout(
                template=PLOTLY_THEME,
                barmode="group",
                height=300,
                xaxis_title="Avg Rating (1=Poor, 2=Needs Imp, 3=Good)",
                yaxis_title="",
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_s, use_container_width=True)
        else:
            st.info("No software capability columns available.")

    st.markdown("---")

    # Section 4: Respondent breakdown and open text feedback
    st.subheader("💬 Respondent Ranks & Comments")
    
    comment_col1, comment_col2 = st.columns(2)
    
    with comment_col1:
        st.markdown("**Crew Feedback: What they liked**")
        liked_comments = v_data[LIKED_COL].dropna().tolist()
        if liked_comments:
            for comment in liked_comments:
                if str(comment).strip() and str(comment).strip().lower() != "nan":
                    st.markdown(f"- \"*{comment.strip()}*\"")
        else:
            st.info("No positive feedback logged.")

    with comment_col2:
        st.markdown("**Crew Feedback: Excessive or Repetitive Jobs**")
        ex_comments = v_data[EXCESSIVE_COL].dropna().tolist()
        if ex_comments:
            for comment in ex_comments:
                if str(comment).strip() and str(comment).strip().lower() != "nan":
                    st.markdown(f"- \"*{comment.strip()}*\"")
        else:
            st.info("No excessive jobs reported.")

    st.markdown("---")
    
    # Detail list of all respondents
    st.subheader("📋 Respondent Detail List")
    cols_tbl = [RANK_COL, TENURE_COL, CORRECTION_COL, SATISFACTION_COL, SUPPORT_COL]
    cols_tbl = [c for c in cols_tbl if c in v_data.columns]
    rename_tbl = {
        RANK_COL: "Rank",
        TENURE_COL: "Tenure",
        CORRECTION_COL: "Correction Rate",
        SATISFACTION_COL: "Satisfaction",
        SUPPORT_COL: "Support Rating"
    }
    v_details = v_data[cols_tbl].rename(columns=rename_tbl).reset_index(drop=True)
    v_details.index = v_details.index + 1
    st.dataframe(v_details, use_container_width=True)
