"""Page 3 — Software Capabilities"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
from data_loader import SOFTWARE_COLS, RANK_COL, TENURE_COL, PLOTLY_THEME, PRIMARY, SEQ_COLOURS


SOFTWARE_COLOURS = {"Good": "#22c55e", "Needs Improvement": "#f59e0b", "Poor": "#ef4444"}
SOFTWARE_ORDER   = ["Good", "Needs Improvement", "Poor"]


def render(data: dict):
    df = data["survey"]

    st.title("💻 Software Capabilities")
    st.markdown("Ratings across 9 JiBe PMS capability areas — from ease-of-use to e-form reporting.")
    st.markdown("---")

    # ── Score summary cards ───────────────────────────────────────────────────
    ordinal_map = {"Good": 3, "Needs Improvement": 2, "Poor": 1}
    scores = {}
    for orig, alias in SOFTWARE_COLS.items():
        scores[alias] = df[orig].map(ordinal_map).mean()

    best  = max(scores, key=scores.get)
    worst = min(scores, key=scores.get)

    st.subheader("Capability Overview")
    col_b, col_w = st.columns(2)
    col_b.success(f"✅ **Best rated:** {best}  —  avg score {scores[best]:.2f}/3")
    col_w.error(  f"⚠️ **Needs most attention:** {worst}  —  avg score {scores[worst]:.2f}/3")

    st.markdown("---")

    # ── Radar chart ───────────────────────────────────────────────────────────
    st.subheader("Overall Capability Radar")
    labels = list(scores.keys())
    vals   = [scores[l] for l in labels]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor=f"rgba(30,64,175,0.15)",
        line=dict(color=PRIMARY, width=2),
        name="All Respondents",
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[1, 3],
                                   tickvals=[1, 1.5, 2, 2.5, 3],
                                   ticktext=["Poor", "", "Avg", "", "Good"])),
        template=PLOTLY_THEME,
        height=500,
        showlegend=False,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")

    # ── Stacked bar overview ──────────────────────────────────────────────────
    st.subheader("Detailed Rating Breakdown")
    rows = []
    for orig, alias in SOFTWARE_COLS.items():
        vc = df[orig].value_counts(normalize=True) * 100
        for rating in SOFTWARE_ORDER:
            rows.append({"Capability": alias, "Rating": rating, "Pct": vc.get(rating, 0)})
    detail_df = pd.DataFrame(rows)

    fig_detail = px.bar(
        detail_df, x="Pct", y="Capability", color="Rating",
        color_discrete_map=SOFTWARE_COLOURS,
        orientation="h",
        barmode="stack",
        text=detail_df["Pct"].apply(lambda v: f"{v:.0f}%" if v > 5 else ""),
        template=PLOTLY_THEME,
    )
    fig_detail.update_traces(textposition="inside", insidetextanchor="middle")
    fig_detail.update_layout(
        height=420,
        xaxis_title="% of respondents",
        yaxis_title="",
        yaxis=dict(categoryorder="total ascending"),
        legend_title="Rating",
        margin=dict(t=10, b=10, r=30),
    )
    st.plotly_chart(fig_detail, use_container_width=True)

    st.markdown("---")

    # ── By rank ───────────────────────────────────────────────────────────────
    st.subheader("Average Scores by Rank")
    rank_df = df[[RANK_COL] + list(SOFTWARE_COLS.keys())].copy()
    for orig in SOFTWARE_COLS:
        rank_df[orig] = rank_df[orig].map({"Good": 3, "Needs Improvement": 2, "Poor": 1})
    rank_df = rank_df.rename(columns=SOFTWARE_COLS)
    rank_avg = rank_df.groupby(RANK_COL)[list(SOFTWARE_COLS.values())].mean()

    fig_rank = px.imshow(
        rank_avg,
        color_continuous_scale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#22c55e"]],
        zmin=1, zmax=3,
        text_auto=".2f",
        template=PLOTLY_THEME,
        labels=dict(color="Avg Score"),
        aspect="auto",
    )
    fig_rank.update_layout(
        height=360,
        xaxis_title="Capability",
        yaxis_title="Rank",
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_rank, use_container_width=True)

    st.markdown("---")

    # ── By PMS tenure ─────────────────────────────────────────────────────────
    st.subheader("Average Scores by PMS Experience Duration")
    tenure_order = ["Less than 6 Months", "6 Months – 1 Year", "1 - 2 Years", "More than 2 Years"]
    tenure_df = df[[TENURE_COL] + list(SOFTWARE_COLS.keys())].copy()
    for orig in SOFTWARE_COLS:
        tenure_df[orig] = tenure_df[orig].map({"Good": 3, "Needs Improvement": 2, "Poor": 1})
    tenure_df = tenure_df.rename(columns=SOFTWARE_COLS)
    tenure_avg = tenure_df.groupby(TENURE_COL)[list(SOFTWARE_COLS.values())].mean()
    tenure_avg = tenure_avg.reindex([t for t in tenure_order if t in tenure_avg.index])

    fig_tenure = px.imshow(
        tenure_avg,
        color_continuous_scale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#22c55e"]],
        zmin=1, zmax=3,
        text_auto=".2f",
        template=PLOTLY_THEME,
        labels=dict(color="Avg Score"),
        aspect="auto",
    )
    fig_tenure.update_layout(
        height=280,
        xaxis_title="Capability",
        yaxis_title="PMS Tenure",
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_tenure, use_container_width=True)
