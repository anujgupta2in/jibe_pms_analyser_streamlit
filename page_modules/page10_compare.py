"""Page 10 — Compare Mode: side-by-side vessel / division / rank comparison"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
from data_loader import (
    VESSEL_COL, RANK_COL, SATISFACTION_COL, SUPPORT_COL,
    QUALITY_COLS, SOFTWARE_COLS, PLOTLY_THEME, PRIMARY, SEQ_COLOURS,
)

# Colour pair for the two selections
COLOUR_A = "#1e40af"   # deep blue
COLOUR_B = "#f97316"   # orange

ORDINAL_MAP_QUALITY   = {"Poor": 1, "Needs Improvement": 2, "Acceptable": 3}
ORDINAL_MAP_SOFTWARE  = {"Poor": 1, "Needs Improvement": 2, "Good": 3}


def _filter_group(merged: pd.DataFrame, dim: str, value: str) -> pd.DataFrame:
    """Return rows from merged that match dim == value."""
    if dim == "Vessel":
        col = VESSEL_COL
    elif dim == "Division":
        col = "Division Code"
    else:  # Rank
        col = RANK_COL
    return merged[merged[col] == value].copy()


def _quality_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for orig, alias in QUALITY_COLS.items():
        if orig in df.columns:
            s = df[orig].map(ORDINAL_MAP_QUALITY).dropna()
            rows.append({"Dimension": alias, "Score": s.mean(), "N": len(s)})
    return pd.DataFrame(rows)


def _software_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for orig, alias in SOFTWARE_COLS.items():
        if orig in df.columns:
            s = df[orig].map(ORDINAL_MAP_SOFTWARE).dropna()
            rows.append({"Dimension": alias, "Score": s.mean(), "N": len(s)})
    return pd.DataFrame(rows)


def _bar_comparison(title: str, dims: list[str],
                    scores_a: list[float], scores_b: list[float],
                    label_a: str, label_b: str,
                    y_range: tuple) -> go.Figure:
    """Grouped bar chart comparing two sets of scores."""
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Bar(
        name=label_a, x=dims, y=scores_a,
        marker_color=COLOUR_A,
        text=[f"{v:.2f}" if pd.notna(v) else "—" for v in scores_a],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name=label_b, x=dims, y=scores_b,
        marker_color=COLOUR_B,
        text=[f"{v:.2f}" if pd.notna(v) else "—" for v in scores_b],
        textposition="outside",
    ))
    fig.update_layout(
        title_text=title,
        barmode="group",
        template=PLOTLY_THEME,
        yaxis_range=[y_range[0], y_range[1]],
        legend=dict(orientation="h", y=1.12),
        margin=dict(t=60, b=10, l=10, r=10),
        height=360,
    )
    return fig


def _radar_comparison(dims: list[str],
                      scores_a: list[float], scores_b: list[float],
                      label_a: str, label_b: str,
                      r_max: float) -> go.Figure:
    """Radar chart comparing two groups."""
    # Close the polygon
    theta = dims + [dims[0]]
    vals_a = scores_a + [scores_a[0]]
    vals_b = scores_b + [scores_b[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_a, theta=theta, fill="toself",
        name=label_a,
        line_color=COLOUR_A, fillcolor=COLOUR_A,
        opacity=0.35,
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals_b, theta=theta, fill="toself",
        name=label_b,
        line_color=COLOUR_B, fillcolor=COLOUR_B,
        opacity=0.35,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, r_max])),
        template=PLOTLY_THEME,
        legend=dict(orientation="h", y=-0.1),
        height=360,
        margin=dict(t=20, b=40, l=20, r=20),
    )
    return fig


def _delta_table(dims: list[str],
                 scores_a: list[float], scores_b: list[float],
                 label_a: str, label_b: str,
                 scale_label: str) -> pd.DataFrame:
    rows = []
    for d, a, b in zip(dims, scores_a, scores_b):
        if pd.isna(a) and pd.isna(b):
            continue
        delta = (a - b) if pd.notna(a) and pd.notna(b) else float("nan")
        rows.append({
            "Dimension": d,
            label_a: f"{a:.2f}" if pd.notna(a) else "—",
            label_b: f"{b:.2f}" if pd.notna(b) else "—",
            f"Δ ({label_a} − {label_b})": (
                f"{delta:+.2f}" if pd.notna(delta) else "—"
            ),
            "_delta_raw": delta,
        })
    return pd.DataFrame(rows)


def render(data: dict):
    merged = data["merged"]

    st.title("⚖️ Compare Mode")
    st.markdown(
        "Select a grouping dimension and two values to see a side-by-side breakdown "
        "of data quality, software capabilities, and satisfaction scores."
    )
    st.markdown("---")

    # ── Dimension & value selection ───────────────────────────────────────────
    dim_col, val_a_col, val_b_col = st.columns([1, 2, 2])

    with dim_col:
        dimension = st.selectbox(
            "Compare by",
            ["Division", "Rank", "Vessel"],
            help="Group respondents by Division Code, Rank, or Vessel name.",
            key="cmp_dimension",
        )

    if dimension == "Vessel":
        options = sorted(merged[VESSEL_COL].dropna().unique().tolist())
        label_dim = "Vessel"
    elif dimension == "Division":
        options = sorted(merged["Division Code"].dropna().unique().tolist())
        options = [o for o in options if o != "Division Code"]
        label_dim = "Division"
    else:  # Rank
        options = sorted(merged[RANK_COL].dropna().unique().tolist())
        label_dim = "Rank"

    if len(options) < 2:
        st.warning("Not enough distinct values in the current dataset to compare. Try removing filters.")
        return

    with val_a_col:
        sel_a = st.selectbox(f"Primary {label_dim}", options, index=0, key="cmp_a")
    with val_b_col:
        remaining = [o for o in options if o != sel_a]
        sel_b = st.selectbox(f"Secondary {label_dim}", remaining, index=0, key="cmp_b")

    if sel_a == sel_b:
        st.info("Select two different values to compare.")
        return

    df_a = _filter_group(merged, dimension, sel_a)
    df_b = _filter_group(merged, dimension, sel_b)

    n_a, n_b = len(df_a), len(df_b)
    if n_a == 0 or n_b == 0:
        st.warning("One of the selected groups has no respondents in the current filtered dataset.")
        return

    label_a = f"{sel_a} (n={n_a})"
    label_b = f"{sel_b} (n={n_b})"

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.markdown("### Summary")
    m1, m2, m3, m4, m5, m6 = st.columns(6)

    sat_a  = df_a[SATISFACTION_COL].mean()
    sat_b  = df_b[SATISFACTION_COL].mean()
    sup_a  = df_a[SUPPORT_COL].mean()
    sup_b  = df_b[SUPPORT_COL].mean()

    m1.metric(f"Satisfaction — {sel_a}", f"{sat_a:.2f}")
    m2.metric(f"Satisfaction — {sel_b}", f"{sat_b:.2f}", f"{sat_a - sat_b:+.2f} Δ")
    m3.metric(f"Support — {sel_a}", f"{sup_a:.2f}")
    m4.metric(f"Support — {sel_b}", f"{sup_b:.2f}", f"{sup_a - sup_b:+.2f} Δ")
    m5.metric(f"Respondents — {sel_a}", f"{n_a:,}")
    m6.metric(f"Respondents — {sel_b}", f"{n_b:,}")

    st.markdown("---")

    # ── Data Quality comparison ───────────────────────────────────────────────
    st.markdown("### 🔍 Data Quality Dimensions")
    qa = _quality_scores(df_a)
    qb = _quality_scores(df_b)

    if not qa.empty and not qb.empty:
        q_merged = qa.merge(qb, on="Dimension", suffixes=("_a", "_b"))
        dims_q   = q_merged["Dimension"].tolist()
        scores_qa = q_merged["Score_a"].tolist()
        scores_qb = q_merged["Score_b"].tolist()

        bar_col, radar_col = st.columns([3, 2])
        with bar_col:
            fig_q = _bar_comparison(
                "Quality Score Comparison (1=Poor, 3=Acceptable)",
                dims_q, scores_qa, scores_qb, label_a, label_b, (0, 3.5),
            )
            st.plotly_chart(fig_q, use_container_width=True)
        with radar_col:
            fig_r = _radar_comparison(dims_q, scores_qa, scores_qb, label_a, label_b, 3)
            st.plotly_chart(fig_r, use_container_width=True)

        # Delta table
        dt_q = _delta_table(dims_q, scores_qa, scores_qb, sel_a, sel_b, "1–3")
        _render_delta_table(dt_q, sel_a, sel_b)
    else:
        st.info("Insufficient quality data for one or both selections.")

    st.markdown("---")

    # ── Software Capabilities comparison ──────────────────────────────────────
    st.markdown("### 💻 Software Capability Dimensions")
    sa = _software_scores(df_a)
    sb = _software_scores(df_b)

    if not sa.empty and not sb.empty:
        s_merged = sa.merge(sb, on="Dimension", suffixes=("_a", "_b"))
        dims_s   = s_merged["Dimension"].tolist()
        scores_sa = s_merged["Score_a"].tolist()
        scores_sb = s_merged["Score_b"].tolist()

        bar_col2, radar_col2 = st.columns([3, 2])
        with bar_col2:
            fig_s = _bar_comparison(
                "Software Score Comparison (1=Poor, 3=Good)",
                dims_s, scores_sa, scores_sb, label_a, label_b, (0, 3.5),
            )
            st.plotly_chart(fig_s, use_container_width=True)
        with radar_col2:
            fig_rs = _radar_comparison(dims_s, scores_sa, scores_sb, label_a, label_b, 3)
            st.plotly_chart(fig_rs, use_container_width=True)

        dt_s = _delta_table(dims_s, scores_sa, scores_sb, sel_a, sel_b, "1–3")
        _render_delta_table(dt_s, sel_a, sel_b)
    else:
        st.info("Insufficient software data for one or both selections.")

    st.markdown("---")

    # ── Satisfaction comparison ───────────────────────────────────────────────
    st.markdown("### 😊 Satisfaction & Support")

    sat_dims   = ["Overall Satisfaction", "Support Rating"]
    scores_sat_a = [sat_a, sup_a]
    scores_sat_b = [sat_b, sup_b]

    sat_bar_col, sat_radar_col = st.columns([3, 2])
    with sat_bar_col:
        fig_sat = _bar_comparison(
            "Satisfaction & Support Comparison (1–10)",
            sat_dims, scores_sat_a, scores_sat_b, label_a, label_b, (0, 10.5),
        )
        st.plotly_chart(fig_sat, use_container_width=True)
    with sat_radar_col:
        fig_rsat = _radar_comparison(sat_dims, scores_sat_a, scores_sat_b, label_a, label_b, 10)
        st.plotly_chart(fig_rsat, use_container_width=True)

    dt_sat = _delta_table(sat_dims, scores_sat_a, scores_sat_b, sel_a, sel_b, "1–10")
    _render_delta_table(dt_sat, sel_a, sel_b)


def _render_delta_table(dt: pd.DataFrame, sel_a: str, sel_b: str):
    """Render delta table with colour highlighting on the delta column."""
    if dt.empty:
        return

    display = dt.drop(columns=["_delta_raw"])

    delta_col = f"Δ ({sel_a} − {sel_b})"

    def _style_delta(val: str) -> str:
        try:
            v = float(val.replace("+", ""))
            if v > 0.05:
                return "color: #16a34a; font-weight: 600"
            if v < -0.05:
                return "color: #dc2626; font-weight: 600"
        except (ValueError, AttributeError):
            pass
        return "color: #64748b"

    styled = display.style.map(_style_delta, subset=[delta_col])
    st.dataframe(styled, use_container_width=True, hide_index=True)
