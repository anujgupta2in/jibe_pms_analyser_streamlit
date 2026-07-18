"""Page 6 — Issues & Improvements"""

import plotly.express as px
import streamlit as st
import pandas as pd
from data_loader import (
    IMPROVEMENTS_COL, ISSUES_COL, PERF_COL, PARTS_DELAY_COL, CHANGE_FREQ_COL,
    PLOTLY_THEME, PRIMARY, SEQ_COLOURS,
)


FREQ_ORDER_PERF  = ["Always", "Daily", "Weekly", "Monthly", "Occasionally", "Rarely"]
FREQ_ORDER_PARTS = ["Daily", "Weekly", "Monthly", "Rarely", "Never"]
CHANGE_ORDER     = ["Often [Every 15 Days]", "Sometimes [Monthly]", "Rarely [Quarterly]", "Never"]


def render(data: dict):
    df             = data["survey"]
    improvements   = data["improvements"]
    issues_counts  = data["issues"]

    st.title("⚠️ Issues & Improvement Priorities")
    st.markdown("What respondents want fixed most — and how often problems surface.")
    st.markdown("---")

    # ── Top 3 improvements ────────────────────────────────────────────────────
    st.subheader("Top Requested Improvements")
    st.caption("Respondents selected their top 3; chart shows overall popularity rank")

    imp_df = improvements.head(15).reset_index()
    imp_df.columns = ["Improvement", "Votes"]
    imp_df["Pct"] = imp_df["Votes"] / len(df) * 100

    fig_imp = px.bar(
        imp_df, x="Pct", y="Improvement", orientation="h",
        text=imp_df["Pct"].apply(lambda v: f"{v:.0f}%"),
        color="Pct", color_continuous_scale=["#bfdbfe", PRIMARY],
        template=PLOTLY_THEME,
    )
    fig_imp.update_traces(textposition="outside")
    fig_imp.update_layout(
        height=520,
        coloraxis_showscale=False,
        xaxis_title="% of respondents who selected this",
        yaxis_title="",
        yaxis=dict(categoryorder="total ascending"),
        margin=dict(t=10, b=10, r=70),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")

    # ── System performance & parts delay ─────────────────────────────────────
    col_perf, col_parts = st.columns(2)

    with col_perf:
        st.subheader("System Performance Issues")
        st.caption("How often respondents experience slow system performance")
        perf_counts = (
            df[PERF_COL].value_counts()
            .reindex([f for f in FREQ_ORDER_PERF if f in df[PERF_COL].unique()], fill_value=0)
            .reset_index()
        )
        perf_counts.columns = ["Frequency", "Count"]
        perf_counts["Pct"] = perf_counts["Count"] / len(df) * 100
        colours_perf = ["#ef4444", "#ef4444", "#f97316", "#f59e0b", "#86efac", "#22c55e"]
        fig_perf = px.bar(
            perf_counts, x="Frequency", y="Pct",
            text=perf_counts["Pct"].apply(lambda v: f"{v:.0f}%"),
            color="Frequency",
            color_discrete_sequence=colours_perf[:len(perf_counts)],
            template=PLOTLY_THEME,
        )
        fig_perf.update_traces(textposition="outside", showlegend=False)
        fig_perf.update_layout(
            height=360, xaxis_title="", yaxis_title="% of respondents",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_perf, use_container_width=True)

    with col_parts:
        st.subheader("Missing Parts Delays")
        st.caption("How often missing PMS parts delay maintenance or requisitions")
        parts_counts = (
            df[PARTS_DELAY_COL].value_counts()
            .reindex([f for f in FREQ_ORDER_PARTS if f in df[PARTS_DELAY_COL].unique()], fill_value=0)
            .reset_index()
        )
        parts_counts.columns = ["Frequency", "Count"]
        parts_counts["Pct"] = parts_counts["Count"] / len(df) * 100
        colours_parts = ["#ef4444", "#f97316", "#f59e0b", "#86efac", "#22c55e"]
        fig_parts = px.bar(
            parts_counts, x="Frequency", y="Pct",
            text=parts_counts["Pct"].apply(lambda v: f"{v:.0f}%"),
            color="Frequency",
            color_discrete_sequence=colours_parts[:len(parts_counts)],
            template=PLOTLY_THEME,
        )
        fig_parts.update_traces(textposition="outside", showlegend=False)
        fig_parts.update_layout(
            height=360, xaxis_title="", yaxis_title="% of respondents",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_parts, use_container_width=True)

    st.markdown("---")

    # ── Change request frequency ──────────────────────────────────────────────
    st.subheader("PMS Change Request Frequency")
    st.caption("How often respondents request changes from the office")
    change_counts = (
        df[CHANGE_FREQ_COL].value_counts()
        .reindex([c for c in CHANGE_ORDER if c in df[CHANGE_FREQ_COL].unique()], fill_value=0)
        .reset_index()
    )
    change_counts.columns = ["Frequency", "Count"]
    change_counts["Pct"] = change_counts["Count"] / len(df) * 100
    fig_change = px.bar(
        change_counts, x="Pct", y="Frequency", orientation="h",
        text=change_counts["Pct"].apply(lambda v: f"{v:.0f}%"),
        color="Pct", color_continuous_scale=["#22c55e", "#ef4444"],
        template=PLOTLY_THEME,
    )
    fig_change.update_traces(textposition="outside")
    fig_change.update_layout(
        height=280, coloraxis_showscale=False,
        xaxis_title="% of respondents", yaxis_title="",
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(CHANGE_ORDER))),
        margin=dict(t=10, b=10, r=60),
    )
    st.plotly_chart(fig_change, use_container_width=True)

    st.markdown("---")

    # ── Data issues cross-ref (top 10 vs improvement priority) ───────────────
    st.subheader("Data Issues Frequency Ranking")
    filtered_issues = issues_counts[
        ~issues_counts.index.str.startswith("No significant")
    ].head(12).reset_index()
    filtered_issues.columns = ["Issue", "Count"]
    filtered_issues["Pct"] = filtered_issues["Count"] / len(df) * 100

    fig_iss = px.bar(
        filtered_issues, x="Pct", y="Issue", orientation="h",
        text=filtered_issues["Pct"].apply(lambda v: f"{v:.0f}%"),
        color="Pct", color_continuous_scale=["#bfdbfe", "#1e40af"],
        template=PLOTLY_THEME,
    )
    fig_iss.update_traces(textposition="outside")
    fig_iss.update_layout(
        height=460, coloraxis_showscale=False,
        xaxis_title="% of respondents", yaxis_title="",
        yaxis=dict(categoryorder="total ascending"),
        margin=dict(t=10, b=10, r=70),
    )
    st.plotly_chart(fig_iss, use_container_width=True)
