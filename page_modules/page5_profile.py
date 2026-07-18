"""Page 5 — Respondent Profile"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from data_loader import (
    RANK_COL, TENURE_COL, SATISFACTION_COL, CORRECTION_COL,
    PLOTLY_THEME, SEQ_COLOURS, PRIMARY,
)


def render(data: dict):
    df = data["survey"]

    st.title("👥 Respondent Profile")
    st.markdown("Who responded — rank, experience, and how those dimensions relate to satisfaction.")
    st.markdown("---")

    # ── Distribution overview ─────────────────────────────────────────────────
    col_rank, col_tenure = st.columns(2)

    with col_rank:
        st.subheader("Responses by Rank")
        rank_counts = df[RANK_COL].value_counts().reset_index()
        rank_counts.columns = ["Rank", "Count"]
        fig_rank = px.pie(
            rank_counts, names="Rank", values="Count",
            color_discrete_sequence=SEQ_COLOURS,
            hole=0.45, template=PLOTLY_THEME,
        )
        fig_rank.update_traces(textinfo="percent+label", textposition="outside")
        fig_rank.update_layout(height=360, showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig_rank, use_container_width=True)

    with col_tenure:
        st.subheader("PMS Experience Duration")
        tenure_order = ["Less than 6 Months", "6 Months – 1 Year", "1 - 2 Years", "More than 2 Years"]
        tenure_counts = df[TENURE_COL].value_counts().reindex(tenure_order, fill_value=0).reset_index()
        tenure_counts.columns = ["Tenure", "Count"]
        fig_tenure = px.bar(
            tenure_counts, x="Tenure", y="Count",
            color="Count", color_continuous_scale="Blues",
            text="Count", template=PLOTLY_THEME,
        )
        fig_tenure.update_traces(textposition="outside")
        fig_tenure.update_layout(
            height=360, coloraxis_showscale=False,
            xaxis_title="", yaxis_title="Respondents",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_tenure, use_container_width=True)

    st.markdown("---")

    # ── Avg satisfaction by rank (bar) ────────────────────────────────────────
    st.subheader("Average Satisfaction by Rank")
    sat_rank = (
        df.groupby(RANK_COL)[SATISFACTION_COL]
        .agg(["mean", "count", "std"])
        .rename(columns={"mean": "Avg", "count": "N", "std": "Std"})
        .sort_values("Avg", ascending=False)
        .reset_index()
    )
    fig_sat = px.bar(
        sat_rank, x=RANK_COL, y="Avg", error_y="Std",
        color="Avg", color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
        text=sat_rank["Avg"].apply(lambda v: f"{v:.2f}"),
        hover_data={"N": True},
        template=PLOTLY_THEME,
    )
    fig_sat.update_traces(textposition="outside")
    fig_sat.update_layout(
        height=360, coloraxis_showscale=False,
        yaxis_range=[0, 10], xaxis_title="", yaxis_title="Avg Satisfaction",
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_sat, use_container_width=True)

    st.markdown("---")

    # ── Cross-tab: Rank × Satisfaction heat ───────────────────────────────────
    st.subheader("Rank × Satisfaction Score Heatmap")
    st.caption("Number of respondents at each satisfaction score, by rank")
    cross_sat = pd.crosstab(df[RANK_COL], df[SATISFACTION_COL])
    fig_cross_sat = px.imshow(
        cross_sat,
        color_continuous_scale="Blues",
        text_auto=True,
        template=PLOTLY_THEME,
        labels=dict(color="Count"),
        aspect="auto",
    )
    fig_cross_sat.update_layout(
        height=360,
        xaxis_title="Satisfaction Score (1–10)",
        yaxis_title="Rank",
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_cross_sat, use_container_width=True)

    st.markdown("---")

    # ── Cross-tab: Rank × Correction % ───────────────────────────────────────
    st.subheader("Rank × Job Correction Rate")
    corr_order = ["Less than 5%", "5–10%", "10–20%", "More than 20%"]
    cross_corr = (
        pd.crosstab(df[RANK_COL], df[CORRECTION_COL])
        .reindex(columns=[c for c in corr_order if c in df[CORRECTION_COL].unique()], fill_value=0)
    )
    cross_corr_pct = cross_corr.div(cross_corr.sum(axis=1), axis=0) * 100
    fig_cross_corr = px.bar(
        cross_corr_pct.reset_index().melt(id_vars=RANK_COL, var_name="Correction %", value_name="Pct"),
        x=RANK_COL, y="Pct", color="Correction %",
        color_discrete_sequence=["#22c55e", "#86efac", "#f59e0b", "#ef4444"],
        barmode="stack", text_auto=".0f",
        template=PLOTLY_THEME,
    )
    fig_cross_corr.update_layout(
        height=380, xaxis_title="", yaxis_title="% of Rank",
        legend_title="Correction Rate",
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_cross_corr, use_container_width=True)

    st.markdown("---")

    # ── Satisfaction by tenure ────────────────────────────────────────────────
    st.subheader("Satisfaction vs PMS Experience Duration")
    tenure_order = ["Less than 6 Months", "6 Months – 1 Year", "1 - 2 Years", "More than 2 Years"]
    sat_tenure = (
        df.groupby(TENURE_COL)[SATISFACTION_COL]
        .agg(["mean", "count", "std"])
        .rename(columns={"mean": "Avg", "count": "N", "std": "Std"})
        .reindex([t for t in tenure_order if t in df[TENURE_COL].unique()])
        .reset_index()
    )
    fig_st = px.bar(
        sat_tenure, x=TENURE_COL, y="Avg", error_y="Std",
        color="Avg", color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
        text=sat_tenure["Avg"].apply(lambda v: f"{v:.2f}"),
        hover_data={"N": True},
        template=PLOTLY_THEME,
    )
    fig_st.update_traces(textposition="outside")
    fig_st.update_layout(
        height=340, coloraxis_showscale=False,
        yaxis_range=[0, 10], xaxis_title="", yaxis_title="Avg Satisfaction",
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_st, use_container_width=True)
