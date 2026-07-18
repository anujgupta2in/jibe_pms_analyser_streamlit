"""Page 2 — Data Quality Assessment"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from data_loader import (
    QUALITY_COLS, RANK_COL, ISSUES_COL, CORRECTION_COL, PLOTLY_THEME, COLOUR_MAP, SEQ_COLOURS, PRIMARY,
)


QUALITY_ORDER = ["Acceptable", "Needs Improvement", "Poor"]
STACKED_COLOURS = {"Acceptable": "#22c55e", "Needs Improvement": "#f59e0b", "Poor": "#ef4444"}


def _stacked_bar(df: pd.DataFrame, col: str, title: str):
    counts = df[col].value_counts().reindex(QUALITY_ORDER, fill_value=0).reset_index()
    counts.columns = ["Rating", "Count"]
    total = counts["Count"].sum()
    counts["Pct"] = counts["Count"] / total * 100
    fig = px.bar(
        counts, x="Pct", y="Rating", orientation="h",
        color="Rating", color_discrete_map=STACKED_COLOURS,
        text=counts["Pct"].apply(lambda v: f"{v:.0f}%"),
        template=PLOTLY_THEME,
    )
    fig.update_traces(textposition="inside", insidetextanchor="middle", showlegend=False)
    fig.update_layout(
        height=160, title=title, title_font_size=13,
        yaxis_title="", xaxis_title="% of respondents",
        margin=dict(t=30, b=10, l=10, r=10),
    )
    return fig


def render(data: dict):
    df = data["survey"]
    issues_counts = data["issues"]

    st.title("🔍 Data Quality Assessment")
    st.markdown("Ratings for the five core data quality dimensions and most common data issues.")
    st.markdown("---")

    # ── Overview score card ───────────────────────────────────────────────────
    st.subheader("Dimension Summary")
    cols_summary = st.columns(len(QUALITY_COLS))
    for i, (orig, alias) in enumerate(QUALITY_COLS.items()):
        pct_good = (df[orig] == "Acceptable").mean() * 100
        pct_poor = (df[orig] == "Poor").mean() * 100
        cols_summary[i].metric(alias, f"{pct_good:.0f}% Acceptable",
                               f"{pct_poor:.0f}% Poor", delta_color="inverse")

    st.markdown("---")

    # ── Stacked bar per dimension ─────────────────────────────────────────────
    st.subheader("Rating Breakdown per Dimension")
    for orig, alias in QUALITY_COLS.items():
        st.plotly_chart(_stacked_bar(df, orig, alias), use_container_width=True)

    st.markdown("---")

    # ── Heatmap by rank ───────────────────────────────────────────────────────
    st.subheader("Rating Heatmap by Rank")
    st.caption("Average score (Acceptable=3, Needs Improvement=2, Poor=1) per rank per dimension")
    ordinal_map = {"Acceptable": 3, "Needs Improvement": 2, "Poor": 1}
    heat_df = df[[RANK_COL] + list(QUALITY_COLS.keys())].copy()
    heat_df = heat_df[heat_df[RANK_COL].notna()]
    for orig in QUALITY_COLS:
        heat_df[orig] = heat_df[orig].map(ordinal_map)
    heat_df = heat_df.rename(columns=QUALITY_COLS)
    heat_pivot = heat_df.groupby(RANK_COL)[list(QUALITY_COLS.values())].mean()
    fig_heat = px.imshow(
        heat_pivot,
        color_continuous_scale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#22c55e"]],
        zmin=1, zmax=3,
        text_auto=".2f",
        template=PLOTLY_THEME,
        labels=dict(color="Avg Score"),
    )
    fig_heat.update_layout(
        height=350,
        xaxis_title="Quality Dimension",
        yaxis_title="Rank",
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # ── Data issues frequency ─────────────────────────────────────────────────
    st.subheader("Most Frequently Reported Data Issues")
    st.caption("Multi-select — respondents could choose all that apply")

    # Filter out "No significant issues" for clarity, show separately
    no_issues_count = issues_counts.get("No significant issues/ Data issues are rare", 0)
    filtered = issues_counts[issues_counts.index != "No significant issues/ Data issues are rare"]
    filtered = filtered.head(12).reset_index()
    filtered.columns = ["Issue", "Count"]
    filtered["Pct"] = filtered["Count"] / len(df) * 100

    fig_issues = px.bar(
        filtered, x="Pct", y="Issue", orientation="h",
        text=filtered["Pct"].apply(lambda v: f"{v:.0f}%"),
        color="Pct", color_continuous_scale=["#bfdbfe", PRIMARY],
        template=PLOTLY_THEME,
    )
    fig_issues.update_traces(textposition="outside")
    fig_issues.update_layout(
        height=480, coloraxis_showscale=False,
        xaxis_title="% of respondents", yaxis_title="",
        yaxis=dict(categoryorder="total ascending"),
        margin=dict(t=10, b=10, r=80),
    )
    st.plotly_chart(fig_issues, use_container_width=True)
    st.caption(f"ℹ️ {no_issues_count} respondents ({no_issues_count/len(df)*100:.0f}%) reported no significant data issues.")

    st.markdown("---")

    # ── Correction % by rank ──────────────────────────────────────────────────
    st.subheader("Job Correction Rate by Rank")
    corr_order = ["Less than 5%", "5–10%", "10–20%", "More than 20%"]
    cross = pd.crosstab(df[RANK_COL], df[CORRECTION_COL]).reindex(columns=corr_order, fill_value=0)
    cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
    fig_cross = px.bar(
        cross_pct.reset_index().melt(id_vars=RANK_COL, var_name="Correction %", value_name="Pct"),
        x=RANK_COL, y="Pct", color="Correction %",
        color_discrete_sequence=["#22c55e", "#86efac", "#f59e0b", "#ef4444"],
        barmode="stack", text_auto=".0f",
        template=PLOTLY_THEME,
    )
    fig_cross.update_layout(
        height=380, xaxis_title="", yaxis_title="% of Rank",
        legend_title="Correction Rate",
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_cross, use_container_width=True)
