"""Page 7 — Satisfaction & Support"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from data_loader import (
    SATISFACTION_COL, SUPPORT_COL, RANK_COL, TRAINING_COL,
    SUPPORT_RESOURCES_COL, PLOTLY_THEME, PRIMARY, SEQ_COLOURS,
)


def render(data: dict):
    df           = data["survey"]
    support_res  = data["support_res"]

    st.title("😊 Satisfaction & Support")
    st.markdown("Deep dive into satisfaction scores, support ratings, and training impact.")
    st.markdown("---")

    # ── Key metrics ───────────────────────────────────────────────────────────
    total = len(df)
    promoters  = (df[SATISFACTION_COL] >= 8).sum()
    passives   = df[SATISFACTION_COL].between(6, 7).sum()
    detractors = (df[SATISFACTION_COL] <= 5).sum()
    nps        = round((promoters - detractors) / total * 100)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Satisfaction",  f"{df[SATISFACTION_COL].mean():.2f} / 10")
    c2.metric("Avg Support Rating",f"{df[SUPPORT_COL].mean():.2f} / 10")
    c3.metric("Net Promoter Score",f"{nps}")
    c4.metric("Training Attended", f"{(df[TRAINING_COL]=='Yes').sum()} / {total}")

    st.markdown("---")

    # ── Side-by-side histograms ───────────────────────────────────────────────
    col_s, col_sup = st.columns(2)

    with col_s:
        st.subheader("Satisfaction Distribution")
        sat_counts = df[SATISFACTION_COL].value_counts().sort_index().reset_index()
        sat_counts.columns = ["Score", "Count"]
        sat_counts["Category"] = sat_counts["Score"].apply(
            lambda x: "Promoter" if x >= 8 else ("Passive" if x >= 6 else "Detractor")
        )
        fig_sat = px.bar(
            sat_counts, x="Score", y="Count", color="Category",
            color_discrete_map={"Promoter": "#22c55e", "Passive": "#f59e0b", "Detractor": "#ef4444"},
            text="Count", template=PLOTLY_THEME,
        )
        fig_sat.update_traces(textposition="outside")
        fig_sat.update_layout(height=320, showlegend=True,
                               xaxis_title="Score", yaxis_title="Respondents",
                               margin=dict(t=10, b=10))
        st.plotly_chart(fig_sat, use_container_width=True)

    with col_sup:
        st.subheader("Support Rating Distribution")
        sup_counts = df[SUPPORT_COL].value_counts().sort_index().reset_index()
        sup_counts.columns = ["Score", "Count"]
        fig_sup = px.bar(
            sup_counts, x="Score", y="Count",
            color="Score", color_continuous_scale="Blues",
            text="Count", template=PLOTLY_THEME,
        )
        fig_sup.update_traces(textposition="outside")
        fig_sup.update_layout(height=320, coloraxis_showscale=False,
                               xaxis_title="Score", yaxis_title="Respondents",
                               margin=dict(t=10, b=10))
        st.plotly_chart(fig_sup, use_container_width=True)

    st.markdown("---")

    # ── Box plots by rank ─────────────────────────────────────────────────────
    st.subheader("Satisfaction & Support Ratings by Rank")
    col_box1, col_box2 = st.columns(2)

    with col_box1:
        fig_box_sat = px.box(
            df, x=RANK_COL, y=SATISFACTION_COL,
            color=RANK_COL, color_discrete_sequence=SEQ_COLOURS,
            points="outliers", template=PLOTLY_THEME,
        )
        fig_box_sat.update_layout(height=380, showlegend=False,
                                   xaxis_title="", yaxis_title="Satisfaction Score",
                                   xaxis_tickangle=-20,
                                   margin=dict(t=10, b=10))
        st.plotly_chart(fig_box_sat, use_container_width=True)

    with col_box2:
        fig_box_sup = px.box(
            df, x=RANK_COL, y=SUPPORT_COL,
            color=RANK_COL, color_discrete_sequence=SEQ_COLOURS,
            points="outliers", template=PLOTLY_THEME,
        )
        fig_box_sup.update_layout(height=380, showlegend=False,
                                   xaxis_title="", yaxis_title="Support Rating",
                                   xaxis_tickangle=-20,
                                   margin=dict(t=10, b=10))
        st.plotly_chart(fig_box_sup, use_container_width=True)

    st.markdown("---")

    # ── Training impact ───────────────────────────────────────────────────────
    st.subheader("Training Attendance vs Satisfaction")
    trained     = df[df[TRAINING_COL] == "Yes"][SATISFACTION_COL]
    not_trained = df[df[TRAINING_COL] == "No"][SATISFACTION_COL]
    avg_trained     = trained.mean()
    avg_not_trained = not_trained.mean()

    t1, t2 = st.columns(2)
    t1.metric("Trained respondents — avg satisfaction",     f"{avg_trained:.2f}",
              f"{avg_trained - avg_not_trained:+.2f} vs untrained")
    t2.metric("Not trained — avg satisfaction",             f"{avg_not_trained:.2f}")

    train_df = pd.DataFrame({
        "Group": ["Attended Training", "Did Not Attend"],
        "Score": [avg_trained, avg_not_trained],
        "N":     [len(trained), len(not_trained)],
    })
    fig_train = px.bar(
        train_df, x="Group", y="Score",
        text=train_df["Score"].apply(lambda v: f"{v:.2f}"),
        color="Score", color_continuous_scale=["#f59e0b", "#22c55e"],
        hover_data={"N": True},
        template=PLOTLY_THEME,
    )
    fig_train.update_traces(textposition="outside")
    fig_train.update_layout(
        height=300, coloraxis_showscale=False,
        yaxis_range=[0, 10], xaxis_title="", yaxis_title="Avg Satisfaction",
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_train, use_container_width=True)

    st.markdown("---")

    # ── Support resource awareness ────────────────────────────────────────────
    st.subheader("Support Resource Awareness")
    st.caption("Which JiBe support channels respondents knew about (multi-select)")
    res_df = support_res.reset_index()
    res_df.columns = ["Resource", "Count"]
    res_df["Pct"] = res_df["Count"] / len(df) * 100

    fig_res = px.bar(
        res_df, x="Pct", y="Resource", orientation="h",
        text=res_df["Pct"].apply(lambda v: f"{v:.0f}%"),
        color="Pct", color_continuous_scale=["#bfdbfe", PRIMARY],
        template=PLOTLY_THEME,
    )
    fig_res.update_traces(textposition="outside")
    fig_res.update_layout(
        height=400, coloraxis_showscale=False,
        xaxis_title="% of respondents", yaxis_title="",
        yaxis=dict(categoryorder="total ascending"),
        margin=dict(t=10, b=10, r=70),
    )
    st.plotly_chart(fig_res, use_container_width=True)
