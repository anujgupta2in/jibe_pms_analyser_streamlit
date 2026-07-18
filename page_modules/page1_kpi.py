"""Page 1 — KPI Dashboard"""

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd
from data_loader import (
    SATISFACTION_COL, SUPPORT_COL, CORRECTION_COL, TRAINING_COL,
    PLOTLY_THEME, PRIMARY, SEQ_COLOURS,
)


def render(data: dict):
    df     = data["survey"]
    merged = data["merged"]
    st.title("📊 KPI Dashboard")
    st.markdown("High-level health metrics from the JiBe PMS survey.")
    st.markdown("---")

    total        = len(df)
    vessel_count = merged["Vessel Name"].nunique()
    avg_sat   = df[SATISFACTION_COL].mean()
    avg_sup   = df[SUPPORT_COL].mean()
    high_corr = df[CORRECTION_COL].isin(["10–20%", "More than 20%"]).sum() / total * 100
    trained   = (df[TRAINING_COL] == "Yes").sum() / total * 100

    # NPS-style segmentation
    promoters  = (df[SATISFACTION_COL] >= 8).sum()
    passives   = df[SATISFACTION_COL].between(6, 7).sum()
    detractors = (df[SATISFACTION_COL] <= 5).sum()
    nps = round((promoters - detractors) / total * 100)

    # ── Headline metrics ──────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Responses",        f"{total:,}")
    c2.metric("Vessels Covered",        f"{vessel_count:,}")
    c3.metric("Avg. Satisfaction",      f"{avg_sat:.1f} / 10",
              delta=f"{avg_sat - 7:.1f} vs target 7")
    c4.metric("Avg. Support Rating",    f"{avg_sup:.1f} / 10")
    c5.metric("Jobs Need >10% Correction", f"{high_corr:.0f}%",
              delta=f"{high_corr - 20:.0f}pp vs 20% bench", delta_color="inverse")
    c6.metric("Training Attended",      f"{trained:.0f}%")

    st.markdown("---")

    # ── NPS + satisfaction dist ───────────────────────────────────────────────
    col_nps, col_dist = st.columns([1, 2])

    with col_nps:
        st.subheader("NPS-style Sentiment")
        st.caption("Promoters ≥8 · Passives 6–7 · Detractors ≤5")

        # Score card
        nps_color  = "#22c55e" if nps >= 0 else "#ef4444"
        nps_bg     = "#f0fdf4" if nps >= 0 else "#fef2f2"
        nps_border = "#86efac" if nps >= 0 else "#fca5a5"
        nps_label  = "Good" if nps >= 30 else ("Positive" if nps >= 0 else "Needs attention")
        st.markdown(
            f"""
<div style="background:{nps_bg};border:2px solid {nps_border};border-radius:12px;
     padding:18px 20px;text-align:center;margin-bottom:12px;">
  <div style="font-size:0.8rem;color:#64748b;font-weight:600;letter-spacing:.05em;
       text-transform:uppercase;margin-bottom:4px;">Net Promoter Score</div>
  <div style="font-size:3rem;font-weight:800;color:{nps_color};line-height:1;">
    {nps:+d}
  </div>
  <div style="font-size:0.8rem;color:{nps_color};font-weight:600;margin-top:4px;">
    {nps_label}
  </div>
</div>""",
            unsafe_allow_html=True,
        )

        # Stacked horizontal bar
        p_pct  = promoters  / total * 100
        pa_pct = passives   / total * 100
        d_pct  = detractors / total * 100
        fig_stack = go.Figure()
        for label, pct, colour in [
            (f"Promoters {p_pct:.0f}%",  p_pct,  "#22c55e"),
            (f"Passives {pa_pct:.0f}%",  pa_pct, "#f59e0b"),
            (f"Detractors {d_pct:.0f}%", d_pct,  "#ef4444"),
        ]:
            fig_stack.add_bar(
                x=[pct], y=[""], orientation="h",
                name=label, marker_color=colour,
                text=f"{pct:.0f}%", textposition="inside",
                insidetextanchor="middle",
            )
        fig_stack.update_layout(
            barmode="stack", template=PLOTLY_THEME,
            height=45, margin=dict(t=5, b=5, l=0, r=0),
            xaxis=dict(visible=False, range=[0, 100]),
            yaxis=dict(visible=False),
            showlegend=False,
        )
        st.plotly_chart(fig_stack, use_container_width=True, key="nps_stack")

        nps_c1, nps_c2, nps_c3 = st.columns(3)
        nps_c1.metric("😊 Promoters",  f"{promoters} ({p_pct:.0f}%)")
        nps_c2.metric("😐 Passives",   f"{passives} ({pa_pct:.0f}%)")
        nps_c3.metric("😟 Detractors", f"{detractors} ({d_pct:.0f}%)")

    with col_dist:
        st.subheader("Overall Satisfaction Distribution")
        st.caption("Scale 1–10 across all 367 respondents")
        score_counts = df[SATISFACTION_COL].value_counts().sort_index().reset_index()
        score_counts.columns = ["Score", "Count"]
        score_counts["Category"] = score_counts["Score"].apply(
            lambda x: "Promoter" if x >= 8 else ("Passive" if x >= 6 else "Detractor")
        )
        colour_map = {"Promoter": "#22c55e", "Passive": "#f59e0b", "Detractor": "#ef4444"}
        fig_dist = px.bar(
            score_counts, x="Score", y="Count", color="Category",
            color_discrete_map=colour_map,
            text="Count",
            template=PLOTLY_THEME,
        )
        fig_dist.update_traces(textposition="outside")
        fig_dist.update_layout(
            height=340, showlegend=True,
            xaxis_title="Satisfaction Score", yaxis_title="Respondents",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")

    # ── Rank & correction split ───────────────────────────────────────────────
    col_rank, col_corr = st.columns(2)

    with col_rank:
        st.subheader("Responses by Rank")
        rank_counts = df["Select your Rank"].value_counts().reset_index()
        rank_counts.columns = ["Rank", "Count"]
        fig_rank = px.pie(rank_counts, names="Rank", values="Count",
                          color_discrete_sequence=SEQ_COLOURS,
                          hole=0.4, template=PLOTLY_THEME)
        fig_rank.update_traces(textinfo="percent+label")
        fig_rank.update_layout(height=320, showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig_rank, use_container_width=True)

    with col_corr:
        st.subheader("Job Correction Rate Required")
        corr_order = ["Less than 5%", "5–10%", "10–20%", "More than 20%"]
        corr_counts = df[CORRECTION_COL].value_counts().reindex(corr_order).reset_index()
        corr_counts.columns = ["Range", "Count"]
        corr_counts["Colour"] = ["#22c55e", "#86efac", "#f59e0b", "#ef4444"]
        fig_corr = px.bar(
            corr_counts, x="Count", y="Range", orientation="h",
            color="Colour", color_discrete_map="identity",
            text="Count", template=PLOTLY_THEME,
        )
        fig_corr.update_traces(textposition="outside", showlegend=False)
        fig_corr.update_layout(height=320, yaxis_title="", xaxis_title="Respondents",
                               margin=dict(t=10, b=10))
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── Support rating dist ───────────────────────────────────────────────────
    st.subheader("Support Team Rating Distribution")
    sup_counts = df[SUPPORT_COL].value_counts().sort_index().reset_index()
    sup_counts.columns = ["Score", "Count"]
    fig_sup = px.bar(sup_counts, x="Score", y="Count", text="Count",
                     color="Score", color_continuous_scale="Blues",
                     template=PLOTLY_THEME)
    fig_sup.update_traces(textposition="outside")
    fig_sup.update_layout(height=300, coloraxis_showscale=False,
                          xaxis_title="Support Rating (1–10)", yaxis_title="Respondents",
                          margin=dict(t=10, b=10))
    st.plotly_chart(fig_sup, use_container_width=True)
