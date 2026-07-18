"""Page 4 — Fleet Analysis"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from data_loader import (
    SATISFACTION_COL, SUPPORT_COL, VESSEL_COL, QUALITY_COLS, PLOTLY_THEME, PRIMARY, SEQ_COLOURS,
)


def render(data: dict):
    merged = data["merged"]
    fleet  = data["fleet"]

    st.title("🚢 Fleet Analysis")
    st.markdown("Survey responses joined to fleet metadata — satisfaction and quality by vessel type and division.")
    st.markdown("---")

    # ── Fleet overview ────────────────────────────────────────────────────────
    st.subheader("Fleet Composition")
    st.caption("ℹ️ Fleet composition and go-live timeline reflect the **full vessel registry** regardless of active filters. Satisfaction and quality charts below update with filters.")
    c1, c2, c3 = st.columns(3)
    matched = merged["Type"].notna().sum()
    c1.metric("Total Fleet Vessels",    f"{len(fleet):,}")
    c2.metric("Survey Respondents",     f"{len(merged):,}")
    c3.metric("Vessels Matched in Survey", f"{matched:,}", f"{matched/len(merged)*100:.0f}% match rate")

    col_fleet, col_div = st.columns(2)
    with col_fleet:
        type_counts = fleet["Type"].value_counts().reset_index()
        type_counts.columns = ["Vessel Type", "Count"]
        type_counts = type_counts[type_counts["Vessel Type"] != "Type"]
        fig_fleet = px.pie(type_counts, names="Vessel Type", values="Count",
                           color_discrete_sequence=SEQ_COLOURS,
                           hole=0.4, template=PLOTLY_THEME,
                           title="Fleet by Vessel Type")
        fig_fleet.update_traces(textinfo="percent+label")
        fig_fleet.update_layout(height=340, showlegend=False, margin=dict(t=40, b=10))
        st.plotly_chart(fig_fleet, use_container_width=True)

    with col_div:
        div_counts = fleet["Division Code"].value_counts().reset_index()
        div_counts.columns = ["Division", "Vessels"]
        div_counts = div_counts[div_counts["Division"] != "Division Code"]
        fig_div = px.bar(div_counts.head(12), x="Division", y="Vessels",
                         color="Vessels", color_continuous_scale="Blues",
                         text="Vessels", template=PLOTLY_THEME,
                         title="Fleet by Division Code")
        fig_div.update_traces(textposition="outside")
        fig_div.update_layout(height=340, coloraxis_showscale=False,
                              margin=dict(t=40, b=10))
        st.plotly_chart(fig_div, use_container_width=True)

    st.markdown("---")

    # ── Satisfaction by vessel type ───────────────────────────────────────────
    valid = merged[merged["Type"].notna() & (merged["Type"] != "Type")].copy()

    st.subheader("Satisfaction by Vessel Type")
    sat_type = (
        valid.groupby("Type")[SATISFACTION_COL]
        .agg(["mean", "count", "std"])
        .rename(columns={"mean": "Avg Satisfaction", "count": "Responses", "std": "StdDev"})
        .sort_values("Avg Satisfaction", ascending=False)
        .reset_index()
    )
    fig_sat_type = px.bar(
        sat_type, x="Avg Satisfaction", y="Type", orientation="h",
        error_x="StdDev",
        color="Avg Satisfaction", color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
        text=sat_type["Avg Satisfaction"].apply(lambda v: f"{v:.2f}"),
        hover_data={"Responses": True},
        template=PLOTLY_THEME,
    )
    fig_sat_type.update_traces(textposition="outside")
    fig_sat_type.update_layout(height=360, coloraxis_showscale=False,
                                xaxis_range=[0, 10],
                                xaxis_title="Avg Satisfaction (1–10)", yaxis_title="",
                                margin=dict(t=10, b=10, r=60))
    st.plotly_chart(fig_sat_type, use_container_width=True)

    st.markdown("---")

    # ── Satisfaction by division ──────────────────────────────────────────────
    st.subheader("Satisfaction by Division")
    sat_div = (
        valid.groupby("Division Code")[SATISFACTION_COL]
        .agg(["mean", "count"])
        .rename(columns={"mean": "Avg Satisfaction", "count": "Responses"})
        .sort_values("Avg Satisfaction", ascending=False)
        .reset_index()
    )
    fig_sat_div = px.bar(
        sat_div, x="Division Code", y="Avg Satisfaction",
        color="Avg Satisfaction", color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
        text=sat_div["Avg Satisfaction"].apply(lambda v: f"{v:.2f}"),
        hover_data={"Responses": True},
        template=PLOTLY_THEME,
    )
    fig_sat_div.update_traces(textposition="outside")
    fig_sat_div.update_layout(height=340, coloraxis_showscale=False,
                               yaxis_range=[0, 10],
                               xaxis_title="Division Code", yaxis_title="Avg Satisfaction",
                               margin=dict(t=10, b=10))
    st.plotly_chart(fig_sat_div, use_container_width=True)

    st.markdown("---")

    # ── PMS Go-Live timeline ──────────────────────────────────────────────────
    st.subheader("PMS Go-Live Timeline")
    st.caption("When each vessel activated JiBe PMS — shows rollout pace")
    timeline_df = fleet.dropna(subset=["PMS Go-Live"]).copy()
    timeline_df["Month"] = timeline_df["PMS Go-Live"].dt.to_period("M").astype(str)
    monthly = timeline_df.groupby(["Month", "Type"]).size().reset_index(name="Vessels")
    monthly = monthly.sort_values("Month")
    fig_tl = px.bar(
        monthly, x="Month", y="Vessels", color="Type",
        color_discrete_sequence=SEQ_COLOURS,
        template=PLOTLY_THEME,
        title="Vessels Going Live per Month by Type",
    )
    fig_tl.update_layout(height=380, xaxis_tickangle=-45,
                          legend_title="Vessel Type",
                          margin=dict(t=40, b=10))
    st.plotly_chart(fig_tl, use_container_width=True)

    st.markdown("---")

    # ── Quality by vessel type ────────────────────────────────────────────────
    st.subheader("Data Quality Scores by Vessel Type")
    ordinal_map = {"Acceptable": 3, "Needs Improvement": 2, "Poor": 1}
    qual_type = valid.copy()
    for orig, alias in QUALITY_COLS.items():
        qual_type[alias] = qual_type[orig].map(ordinal_map)
    qual_avg = (
        qual_type.groupby("Type")[list(QUALITY_COLS.values())]
        .mean()
        .reset_index()
        .melt(id_vars="Type", var_name="Dimension", value_name="Score")
    )
    fig_qual = px.bar(
        qual_avg, x="Dimension", y="Score", color="Type",
        barmode="group",
        color_discrete_sequence=SEQ_COLOURS,
        template=PLOTLY_THEME,
    )
    fig_qual.update_layout(height=380, yaxis_range=[1, 3],
                            xaxis_title="", yaxis_title="Avg Score (1=Poor, 3=Good)",
                            legend_title="Vessel Type",
                            margin=dict(t=10, b=10))
    st.plotly_chart(fig_qual, use_container_width=True)
