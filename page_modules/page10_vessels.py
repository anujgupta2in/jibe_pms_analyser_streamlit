"""Page 10 — Vessel Drill-Down: Correction Rate & Per-Vessel Feedback"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from data_loader import (
    CORRECTION_COL, SATISFACTION_COL, SUPPORT_COL, RANK_COL,
    QUALITY_COLS, PLOTLY_THEME, PRIMARY,
)

CORR_ORDER    = ["Less than 5%", "5–10%", "10–20%", "More than 20%"]
HIGH_CORR     = {"10–20%", "More than 20%"}
CORR_COLOURS  = {
    "Less than 5%":  "#22c55e",
    "5–10%":         "#86efac",
    "10–20%":        "#f59e0b",
    "More than 20%": "#ef4444",
}


def _vessel_summary(merged: pd.DataFrame) -> pd.DataFrame:
    """One row per vessel with correction and satisfaction aggregates."""
    rows = []
    for vessel, grp in merged.groupby("Vessel Name"):
        corr = grp[CORRECTION_COL].dropna()
        n    = len(grp)
        n_corr = len(corr)
        high = corr.isin(HIGH_CORR).sum()
        high_pct = high / n_corr * 100 if n_corr > 0 else 0

        avg_sat = grp[SATISFACTION_COL].mean() if SATISFACTION_COL in grp else None
        avg_sup = grp[SUPPORT_COL].mean()       if SUPPORT_COL in grp else None

        vtype = grp["Type"].dropna().mode().iloc[0] if "Type" in grp.columns and grp["Type"].notna().any() else "—"
        div   = grp["Division Code"].dropna().mode().iloc[0] if "Division Code" in grp.columns and grp["Division Code"].notna().any() else "—"

        rows.append({
            "Vessel":          vessel,
            "Type":            vtype,
            "Division":        div,
            "Respondents":     n,
            "With Correction Data": n_corr,
            "High Correction (>10%)": high,
            "High Correction %": round(high_pct, 1),
            "Avg Satisfaction": round(avg_sat, 2) if avg_sat is not None and pd.notna(avg_sat) else None,
            "Avg Support":      round(avg_sup, 2) if avg_sup is not None and pd.notna(avg_sup) else None,
        })

    df = pd.DataFrame(rows)
    df = df[df["With Correction Data"] > 0].sort_values("High Correction %", ascending=False)
    return df.reset_index(drop=True)


def render(data: dict):
    merged = data["merged"]

    st.title("🚢 Vessel Drill-Down")
    st.markdown(
        "Identify vessels where job correction rates are high and explore "
        "individual respondent feedback vessel by vessel."
    )
    st.markdown("---")

    summary = _vessel_summary(merged)
    total_vessels = len(summary)

    # ── Threshold slider ──────────────────────────────────────────────────────
    col_thresh, col_type, col_div = st.columns([1, 1, 1])
    with col_thresh:
        threshold = st.slider(
            "Flag vessels where high-correction respondents exceed:",
            min_value=0, max_value=100, value=10, step=5, format="%d%%"
        )
    with col_type:
        types = ["All"] + sorted(summary["Type"].dropna().unique().tolist())
        sel_type = st.selectbox("Vessel Type", types)
    with col_div:
        divs = ["All"] + sorted(summary["Division"].dropna().unique().tolist())
        sel_div = st.selectbox("Division", divs)

    # Apply filters
    view = summary.copy()
    if sel_type != "All":
        view = view[view["Type"] == sel_type]
    if sel_div != "All":
        view = view[view["Division"] == sel_div]

    flagged = view[view["High Correction %"] >= threshold]
    ok      = view[view["High Correction %"] <  threshold]

    # ── KPIs ─────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vessels shown",        len(view))
    c2.metric(f"Flagged (≥ {threshold}%)", len(flagged),
              delta=f"{len(flagged)/max(len(view),1)*100:.0f}% of vessels",
              delta_color="inverse")
    c3.metric("Within threshold",     len(ok))
    c4.metric("Avg satisfaction (flagged)",
              f"{flagged['Avg Satisfaction'].mean():.1f}" if len(flagged) > 0 else "—")

    st.markdown("---")

    # ── Bar chart: stacked correction breakdown per vessel ────────────────────
    st.subheader("Correction Rate Breakdown by Vessel (respondent counts)")

    # Build per-vessel counts for each correction band
    stacked_rows = []
    for _, row in view.sort_values("High Correction %", ascending=True).iterrows():
        v_data = merged[merged["Vessel Name"] == row["Vessel"]]
        for band in CORR_ORDER:
            stacked_rows.append({
                "Vessel": row["Vessel"],
                "Band":   band,
                "Count":  int((v_data[CORRECTION_COL] == band).sum()),
            })
    stacked_df = pd.DataFrame(stacked_rows)

    fig_bar = go.Figure()
    for band in CORR_ORDER:
        band_df = stacked_df[stacked_df["Band"] == band]
        fig_bar.add_bar(
            y=band_df["Vessel"],
            x=band_df["Count"],
            name=band,
            orientation="h",
            marker_color=CORR_COLOURS[band],
            text=band_df["Count"].apply(lambda v: str(v) if v > 0 else ""),
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate=f"<b>%{{y}}</b><br>{band}: %{{x}} respondents<extra></extra>",
        )
    fig_bar.update_layout(
        barmode="stack",
        template=PLOTLY_THEME,
        height=max(350, len(view) * 22 + 80),
        xaxis_title="Number of respondents",
        yaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font=dict(size=11)),
        margin=dict(t=50, b=20, r=30),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ── Summary table ─────────────────────────────────────────────────────────
    st.subheader("Vessel Summary Table")

    def _flag(row):
        if row["High Correction %"] >= threshold:
            return "🔴 Flagged"
        elif row["High Correction %"] >= threshold * 0.5:
            return "🟡 Watch"
        return "🟢 OK"

    display = view.copy()
    display.insert(0, "Status", display.apply(_flag, axis=1))
    display["Avg Satisfaction"] = display["Avg Satisfaction"].apply(
        lambda v: f"{v:.1f}" if pd.notna(v) else "—"
    )
    display["Avg Support"] = display["Avg Support"].apply(
        lambda v: f"{v:.1f}" if pd.notna(v) else "—"
    )
    display["High Correction %"] = display["High Correction %"].apply(lambda v: f"{v:.1f}%")

    st.dataframe(
        display[["Status", "Vessel", "Type", "Division",
                 "Respondents", "With Correction Data",
                 "High Correction %", "Avg Satisfaction", "Avg Support"]].rename(
            columns={"High Correction %": "High Corr %"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # ── Per-vessel drill-down ─────────────────────────────────────────────────
    st.subheader("Per-Vessel Respondent Detail")
    st.caption("Expand a vessel to see individual respondent feedback.")

    # Show flagged first, then the rest
    ordered_vessels = (
        flagged["Vessel"].tolist() + ok["Vessel"].tolist()
    )

    quality_short = {orig: alias[:18] for orig, alias in QUALITY_COLS.items()}

    for vessel in ordered_vessels:
        v_data = merged[merged["Vessel Name"] == vessel]
        high_pct_val = summary.loc[summary["Vessel"] == vessel, "High Correction %"].values
        high_pct_val = high_pct_val[0] if len(high_pct_val) > 0 else 0
        n_resp = len(v_data)

        status_icon = "🔴" if high_pct_val >= threshold else ("🟡" if high_pct_val >= threshold * 0.5 else "🟢")
        vtype  = v_data["Type"].dropna().mode().iloc[0] if v_data["Type"].notna().any() else "—"
        div    = v_data["Division Code"].dropna().mode().iloc[0] if v_data["Division Code"].notna().any() else "—"
        avg_s  = v_data[SATISFACTION_COL].mean() if v_data[SATISFACTION_COL].notna().any() else None

        label = (
            f"{status_icon} **{vessel}** — {vtype} · {div} · "
            f"{n_resp} respondent{'s' if n_resp != 1 else ''} · "
            f"High corr: {high_pct_val:.0f}% · "
            f"Avg sat: {avg_s:.1f}/10" if avg_s else f"{status_icon} **{vessel}**"
        )

        with st.expander(
            f"{status_icon} {vessel}  |  {vtype}  |  {div}  |  "
            f"{n_resp} resp  |  High corr: {high_pct_val:.0f}%"
            + (f"  |  Avg sat: {avg_s:.1f}/10" if avg_s and pd.notna(avg_s) else ""),
            expanded=False,
        ):
            if n_resp == 0:
                st.info("No responses for this vessel.")
                continue

            # Mini KPIs
            mk1, mk2, mk3, mk4 = st.columns(4)
            mk1.metric("Respondents", n_resp)
            if avg_s and pd.notna(avg_s):
                mk2.metric("Avg Satisfaction", f"{avg_s:.1f}/10")
            avg_sup = v_data[SUPPORT_COL].mean() if v_data[SUPPORT_COL].notna().any() else None
            if avg_sup and pd.notna(avg_sup):
                mk3.metric("Avg Support", f"{avg_sup:.1f}/10")
            mk4.metric("High Correction %", f"{high_pct_val:.0f}%")

            # Correction breakdown bar
            corr_counts = v_data[CORRECTION_COL].value_counts().reindex(CORR_ORDER, fill_value=0)
            fig_corr = go.Figure(go.Bar(
                x=corr_counts.index,
                y=corr_counts.values,
                marker_color=[CORR_COLOURS[c] for c in corr_counts.index],
                text=corr_counts.values,
                textposition="outside",
            ))
            fig_corr.update_layout(
                template=PLOTLY_THEME, height=220,
                title="Correction Rate Distribution",
                xaxis_title="", yaxis_title="Respondents",
                margin=dict(t=30, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_corr, use_container_width=True, key=f"corr_{vessel}")

            # Respondent table
            cols_show = [RANK_COL, CORRECTION_COL, SATISFACTION_COL, SUPPORT_COL]
            cols_show = [c for c in cols_show if c in v_data.columns]
            rename_map = {
                RANK_COL:        "Rank",
                CORRECTION_COL:  "Correction %",
                SATISFACTION_COL: "Satisfaction",
                SUPPORT_COL:     "Support",
            }
            tbl = v_data[cols_show].rename(columns=rename_map).reset_index(drop=True)
            tbl.index = tbl.index + 1
            st.dataframe(tbl, use_container_width=True)

            # Quality heatmap row
            qual_cols = [c for c in QUALITY_COLS if c in v_data.columns]
            if qual_cols:
                ordinal = {"Acceptable": 3, "Needs Improvement": 2, "Poor": 1}
                scores  = {quality_short[c]: v_data[c].map(ordinal).mean() for c in qual_cols}
                heat_df = pd.DataFrame([scores])
                fig_h = px.imshow(
                    heat_df,
                    color_continuous_scale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#22c55e"]],
                    zmin=1, zmax=3, text_auto=".2f",
                    template=PLOTLY_THEME,
                    labels=dict(color="Avg Score"),
                )
                fig_h.update_layout(
                    height=120, margin=dict(t=5, b=5, l=5, r=5),
                    title="Data Quality Avg Scores",
                    yaxis=dict(showticklabels=False),
                )
                st.plotly_chart(fig_h, use_container_width=True, key=f"heat_{vessel}")
