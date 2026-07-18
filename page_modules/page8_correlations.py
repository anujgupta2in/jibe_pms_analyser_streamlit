"""Page 8 — Advanced Correlations"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from data_loader import (
    SATISFACTION_COL, SUPPORT_COL, CORRECTION_COL, TENURE_COL,
    RANK_COL, QUALITY_COLS, SOFTWARE_COLS,
    ORDINAL_MAPS, PLOTLY_THEME, PRIMARY,
)


def render(data: dict):
    df  = data["survey"]
    enc = data["encoded"]

    st.title("📈 Advanced Correlations")
    st.markdown("Statistical relationships between survey dimensions — correlation matrix, scatter plots, and chi-square tests.")
    st.markdown("---")

    # ── Correlation method selector ───────────────────────────────────────────
    corr_method = st.selectbox(
        "Correlation Method",
        ["Spearman Rank Correlation (Recommended for rating scales)", "Pearson Linear Correlation"],
        index=0,
        help="Spearman is rank-based and handles non-linear ordinal ratings (like Poor, Acceptable, Good) correctly. Pearson measures strict linear correlation.",
        key="corr_method",
    )
    method_key = "spearman" if "Spearman" in corr_method else "pearson"
    method_label = "Spearman ρ" if method_key == "spearman" else "Pearson r"

    st.subheader(f"Correlation Heatmap ({method_label})")
    st.caption("All rating columns encoded to integers. Values closer to ±1 show stronger relationships.")
    corr_matrix = enc.corr(method=method_key, numeric_only=True)

    fig_corr = px.imshow(
        corr_matrix,
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        text_auto=".2f",
        template=PLOTLY_THEME,
        labels=dict(color=method_label),
        aspect="auto",
    )
    fig_corr.update_layout(
        height=680,
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")

    # ── Key scatter plots ──────────────────────────────────────────────────────
    st.subheader("Key Relationship Scatter Plots")

    def scatter_with_trendline(x_col, y_col, label_x, label_y, colour_col=None):
        plot_df = enc[[x_col, y_col]].dropna()
        if len(plot_df) < 3:
            return None
            
        if method_key == "spearman":
            r, p = stats.spearmanr(plot_df[x_col], plot_df[y_col])
            coeff_label = "ρ"
        else:
            r, p = stats.pearsonr(plot_df[x_col], plot_df[y_col])
            coeff_label = "r"

        # Build jittered data for visualization only
        plot_df_jittered = plot_df.copy()
        plot_df_jittered[x_col] = plot_df_jittered[x_col] + np.random.normal(0, 0.08, size=len(plot_df))
        plot_df_jittered[y_col] = plot_df_jittered[y_col] + np.random.normal(0, 0.08, size=len(plot_df))

        # Build scatter manually + add OLS trendline using numpy (no statsmodels needed)
        fig = px.scatter(
            plot_df_jittered, x=x_col, y=y_col,
            opacity=0.35,
            color_discrete_sequence=[PRIMARY],
            template=PLOTLY_THEME,
            labels={x_col: label_x, y_col: label_y},
        )
        # Add trendline via numpy polyfit (using original non-jittered values for trend accuracy)
        x_vals = plot_df[x_col].values
        y_vals = plot_df[y_col].values
        m, b = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_line = m * x_line + b
        fig.add_scatter(x=x_line, y=y_line, mode="lines",
                        line=dict(color="#ef4444", width=2), name="Trend")
        fig.update_layout(
            height=320,
            title=f"{coeff_label} = {r:.3f}  |  p {'< 0.001' if p < 0.001 else f'= {p:.3f}'}",
            title_font_size=13,
            showlegend=False,
            margin=dict(t=40, b=10),
        )
        return fig

    sc1, sc2, sc3 = st.columns(3)

    with sc1:
        st.markdown("**Satisfaction vs Support Rating**")
        fig1 = scatter_with_trendline("Satisfaction", "Support Rating",
                                       "Satisfaction", "Support Rating")
        if fig1:
            st.plotly_chart(fig1, use_container_width=True)

    with sc2:
        st.markdown("**Satisfaction vs Correction %**")
        fig2 = scatter_with_trendline("Satisfaction", "Correction %",
                                       "Satisfaction", "Correction % (encoded)")
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)

    with sc3:
        st.markdown("**Satisfaction vs PMS Tenure**")
        fig3 = scatter_with_trendline("Satisfaction", "PMS Tenure",
                                       "Satisfaction", "PMS Tenure (encoded)")
        if fig3:
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ── Additional scatters ───────────────────────────────────────────────────
    sc4, sc5 = st.columns(2)

    with sc4:
        st.markdown("**System Performance vs Satisfaction**")
        fig4 = scatter_with_trendline("Sys Performance", "Satisfaction",
                                       "Sys Performance (encoded)", "Satisfaction")
        if fig4:
            st.plotly_chart(fig4, use_container_width=True)

    with sc5:
        st.markdown("**Support Rating vs Correction %**")
        fig5 = scatter_with_trendline("Support Rating", "Correction %",
                                       "Support Rating", "Correction % (encoded)")
        if fig5:
            st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")

    # ── Chi-square table for categorical pairs ────────────────────────────────
    st.subheader("Chi-Square Association Tests (Categorical Variables)")
    st.caption("p < 0.05 = statistically significant association. p < 0.001 = highly significant.")

    cat_cols = {
        "Rank":           RANK_COL,
        "Tenure":         TENURE_COL,
        "Correction %":   CORRECTION_COL,
        "Training":       "Have you attended any of the JiBe PMS training sessions? [either on Training Centre/ Online session with PMS Team/ Marineflix Videos]",
    }

    pairs = [
        ("Rank",         "Tenure"),
        ("Rank",         "Correction %"),
        ("Rank",         "Training"),
        ("Tenure",       "Correction %"),
        ("Tenure",       "Training"),
        ("Training",     "Correction %"),
    ]

    chi_rows = []
    for a, b in pairs:
        col_a = cat_cols[a]
        col_b = cat_cols[b]
        sub = df[[col_a, col_b]].dropna()
        if len(sub) < 5:
            continue
        ct = pd.crosstab(sub[col_a], sub[col_b])
        try:
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "—"))
            chi_rows.append({
                "Variable A": a, "Variable B": b,
                "χ²": round(chi2, 2), "df": dof,
                "p-value": f"{p:.4f}", "Significant": sig,
            })
        except Exception:
            pass

    if chi_rows:
        chi_df = pd.DataFrame(chi_rows)
        st.dataframe(chi_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Correlation ranking ───────────────────────────────────────────────────
    st.subheader("What Drives Satisfaction Most?")
    st.caption("Top correlations with Overall Satisfaction score")
    if "Satisfaction" in corr_matrix.columns:
        sat_corr = corr_matrix["Satisfaction"].drop("Satisfaction").sort_values(key=abs, ascending=False)
        corr_df = pd.DataFrame({"Feature": sat_corr.index, "Correlation": sat_corr.values})
        corr_df["Direction"] = corr_df["Correlation"].apply(
            lambda v: "Positive" if v > 0 else "Negative"
        )
        fig_rank = px.bar(
            corr_df.head(14), x="Correlation", y="Feature", orientation="h",
            color="Direction",
            color_discrete_map={"Positive": "#22c55e", "Negative": "#ef4444"},
            text=corr_df["Correlation"].head(14).apply(lambda v: f"{v:.3f}"),
            template=PLOTLY_THEME,
        )
        fig_rank.update_traces(textposition="outside")
        fig_rank.update_layout(
            height=460,
            xaxis_title="Pearson Correlation with Satisfaction",
            yaxis_title="",
            yaxis=dict(categoryorder="total ascending"),
            margin=dict(t=10, b=10, r=70),
        )
        st.plotly_chart(fig_rank, use_container_width=True)
