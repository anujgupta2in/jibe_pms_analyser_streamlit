"""
JiBe PMS Data Quality & Product Health Assessment — Survey Analyser
"""

import os
import pathlib
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="JiBe PMS Survey Analyser",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide Streamlit's auto-generated multipage navigation (we use our own radio nav)
st.markdown(
    """
    <style>
    [data-testid="stSidebarNavItems"],
    [data-testid="stSidebarNavSeparator"],
    [data-testid="stSidebarNav"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

from data_loader import load_data, apply_filters, RANK_COL, TENURE_COL
import importlib
import report_export
importlib.reload(report_export)
from report_export import generate_pdf, generate_excel, DEFAULT_PDF_OPTIONS

# ── Pages ─────────────────────────────────────────────────────────────────────
from page_modules import (
    page1_kpi,
    page2_data_quality,
    page3_software,
    page4_fleet,
    page5_profile,
    page6_issues,
    page7_satisfaction,
    page8_correlations,
    page9_text,
    page10_vessels,
    page10_compare,
)

PAGES = {
    "📊 KPI Dashboard":             page1_kpi,
    "🔍 Data Quality Assessment":   page2_data_quality,
    "💻 Software Capabilities":     page3_software,
    "🚢 Fleet Analysis":            page4_fleet,
    "👥 Respondent Profile":        page5_profile,
    "⚠️ Issues & Improvements":     page6_issues,
    "😊 Satisfaction & Support":    page7_satisfaction,
    "📈 Advanced Correlations":     page8_correlations,
    "💬 Open Text Feedback":        page9_text,
    "🔎 Vessel Drill-Down":         page10_vessels,
    "⚖️ Compare":                   page10_compare,
}

# Ordered tenure labels for display
TENURE_ORDER = [
    "Less than 6 Months",
    "6 Months – 1 Year",
    "1 - 2 Years",
    "More than 2 Years",
]


def main():
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/anchor.png", width=60)
        st.title("JiBe PMS Analyser")
        st.markdown("---")

        st.subheader("📁 Data Source")
        uploaded = st.file_uploader(
            "Upload survey Excel file",
            type=["xlsx", "xls"],
            help="Must contain 'Sheet1' (survey responses) and 'Fleet' (vessel list)",
        )

        if uploaded is not None:
            file_bytes = uploaded.read()
            st.success(f"Loaded: **{uploaded.name}**")
        else:
            st.warning("Please upload an Excel file to begin.")
            st.stop()

        st.markdown("---")
        st.subheader("🗂️ Navigate")
        page_name = st.radio("Select page", list(PAGES.keys()), label_visibility="collapsed")
        st.markdown("---")

        # ── Load data (cached) ─────────────────────────────────────────────
        # Load here so filter options can be derived from real data
        with st.spinner("Loading data…"):
            data = load_data(file_bytes)

        # ── Filters ────────────────────────────────────────────────────────
        st.subheader("🔽 Filters")

        merged_full = data["merged"]
        survey_full = data["survey"]

        # Derive sorted unique values from the full (unfiltered) dataset
        all_ranks = sorted(survey_full[RANK_COL].dropna().unique().tolist())
        all_types = sorted(merged_full["Type"].dropna().unique().tolist())
        all_divs  = sorted(merged_full["Division Code"].dropna().unique().tolist())
        all_tenure = [t for t in TENURE_ORDER if t in survey_full[TENURE_COL].dropna().unique().tolist()]

        # Initialise session-state keys so the clear button can reset them
        for key, default in [
            ("f_rank",   []),
            ("f_type",   []),
            ("f_div",    []),
            ("f_tenure", []),
        ]:
            if key not in st.session_state:
                st.session_state[key] = default

        def _clear_filters():
            for k in ("f_rank", "f_type", "f_div", "f_tenure"):
                st.session_state[k] = []

        sel_rank   = st.multiselect("Rank",         all_ranks,  key="f_rank")
        sel_type   = st.multiselect("Vessel Type",  all_types,  key="f_type")
        sel_div    = st.multiselect("Division Code", all_divs,  key="f_div")
        sel_tenure = st.multiselect("PMS Tenure",   all_tenure, key="f_tenure")

        active_filters = {
            k: v for k, v in {
                "rank":        sel_rank,
                "vessel_type": sel_type,
                "division":    sel_div,
                "tenure":      sel_tenure,
            }.items() if v
        }

        n_active = len(active_filters)
        if n_active:
            st.markdown(
                f'<span style="background:#1e40af;color:white;padding:2px 8px;'
                f'border-radius:12px;font-size:0.8rem;">🔵 {n_active} active filter'
                f'{"s" if n_active > 1 else ""}</span>',
                unsafe_allow_html=True,
            )
            st.button("✖ Clear all filters", on_click=_clear_filters, use_container_width=True)

        # Compute filtered data here so export buttons can use it
        if active_filters:
            filtered_data = apply_filters(data, active_filters)
        else:
            filtered_data = data

        n_filtered = len(filtered_data["survey"])

        st.markdown("---")

        # ── Export ─────────────────────────────────────────────────────────
        st.subheader("📥 Export")

        if n_filtered == 0:
            st.warning("No data to export — the current filters return zero respondents.")
        else:
            # Excel — always available
            excel_bytes = generate_excel(filtered_data)
            st.download_button(
                label="⬇️ Download Excel (raw data)",
                data=excel_bytes,
                file_name="jibe_pms_survey_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

            # ── PDF options ────────────────────────────────────────────────
            with st.expander("⚙️ PDF Options", expanded=False):
                st.markdown("**Report metadata**")
                pdf_title    = st.text_input("Report title",
                    DEFAULT_PDF_OPTIONS["report_title"], key="_pdf_title")
                pdf_subtitle = st.text_input("Subtitle",
                    DEFAULT_PDF_OPTIONS["report_subtitle"], key="_pdf_subtitle")
                pdf_author   = st.text_input("Prepared by",
                    DEFAULT_PDF_OPTIONS["author_org"], key="_pdf_author")

                st.markdown("**Layout & Branding**")
                include_cover = st.checkbox("Include cover page",
                    value=DEFAULT_PDF_OPTIONS["include_cover"], key="_pdf_include_cover")
                exec_summary = st.text_area("Executive Summary / Custom Notes",
                    value=DEFAULT_PDF_OPTIONS["executive_summary"], key="_pdf_executive_summary",
                    help="Custom text to display on the cover page (or top of first page if cover is excluded).")
                color_theme = st.selectbox("Color Theme",
                    ["Classic Blue", "Teal / Emerald", "Charcoal", "Crimson"],
                    index=0, key="_pdf_color_theme")

                st.markdown("**Data Limits**")
                vessel_limit = st.slider("Max vessels in table",
                    5, 100, DEFAULT_PDF_OPTIONS["vessel_limit"], key="_pdf_vessel_limit")
                keyword_limit = st.slider("Max open text keywords",
                    5, 25, DEFAULT_PDF_OPTIONS["keyword_limit"], key="_pdf_keyword_limit")

                st.markdown("**Visuals to include**")
                
                col_a, col_b = st.columns(2)
                vis_vals = {}
                
                VISUAL_LABELS = [
                    # KPIs & Satisfaction / Support
                    ("vis_kpis", "Headline KPIs Metrics"),
                    ("vis_sat_dist", "Satisfaction Distribution (Chart)"),
                    ("vis_nps_bar", "NPS-style Sentiment (Chart)"),
                    ("vis_sup_dist", "Support Rating Distribution (Chart)"),
                    ("vis_sat_box_rank", "Satisfaction Box Plot by Rank"),
                    ("vis_sup_box_rank", "Support Rating Box Plot by Rank"),
                    ("vis_train_impact", "Training Attendance vs Sat (Chart)"),
                    ("vis_sup_awareness", "Support Resource Awareness (Chart)"),
                    
                    # Data Quality & Software
                    ("vis_qual_chart", "Data Quality Ratings (Chart)"),
                    ("vis_qual_table", "Data Quality Ratings (Table)"),
                    ("vis_soft_radar", "Software Capability Radar (Chart)"),
                    ("vis_soft_table", "Software Capability Averages (Table)"),
                    ("vis_improvements", "Top Improvements (Chart)"),
                    ("vis_issues", "Most Reported Issues (Chart)"),
                    
                    # Correlations & Drivers
                    ("vis_corr_heatmap", "Correlation Heatmap (Chart)"),
                    ("vis_scatters", "Key Relationship Scatters (Chart)"),
                    ("vis_chisquare_table", "Chi-Square Tests (Table)"),
                    ("vis_drivers_chart", "Satisfaction Drivers (Chart)"),
                    
                    # Respondent Profile & Fleet
                    ("vis_profile_rank", "Respondents by Rank (Chart)"),
                    ("vis_profile_tenure", "PMS Tenure Distribution (Chart)"),
                    ("vis_profile_sat", "Avg Satisfaction by Rank (Chart)"),
                    ("vis_fleet_vtype", "Avg Sat by Vessel Type (Chart)"),
                    ("vis_fleet_div", "Avg Sat by Division (Chart)"),
                    
                    # Correction & Vessels
                    ("vis_corr_rank", "Correction Rate by Rank (Chart)"),
                    ("vis_corr_table", "Correction Rate Bands (Table)"),
                    ("vis_vessels_table", "Vessels High Correction (Table)"),
                    ("vis_vessels_stacked_bar", "Vessels Stacked Correction (Chart)"),
                    
                    # Open Text Key Themes
                    ("vis_opentext_lists", "Open Text Keyword Themes (Lists)"),
                    ("vis_opentext_pos_bar", "Positive Themes Frequency (Chart)"),
                    ("vis_opentext_neg_bar", "Concern Themes Frequency (Chart)"),
                    ("vis_opentext_compare", "Positive vs Concern Themes (Chart)"),
                    
                    # Compare Mode
                    ("vis_compare_summary", "Compare Mode Summary Metrics"),
                    ("vis_compare_quality", "Compare Mode Data Quality Comparison"),
                    ("vis_compare_software", "Compare Mode Software Comparison"),
                    ("vis_compare_sat", "Compare Mode Satisfaction Comparison"),
                ]
                
                for i, (key, label) in enumerate(VISUAL_LABELS):
                    col = col_a if i % 2 == 0 else col_b
                    vis_vals[key] = col.checkbox(
                        label, value=DEFAULT_PDF_OPTIONS[key], key=f"_pdf_{key}"
                    )

            pdf_options = {
                "report_title":      pdf_title,
                "report_subtitle":   pdf_subtitle,
                "author_org":        pdf_author,
                "include_cover":     include_cover,
                "executive_summary": exec_summary,
                "color_theme":       color_theme,
                "vessel_limit":      vessel_limit,
                "keyword_limit":     keyword_limit,
                "cmp_dimension":     st.session_state.get("cmp_dimension"),
                "cmp_val_a":         st.session_state.get("cmp_a"),
                "cmp_val_b":         st.session_state.get("cmp_b"),
                **vis_vals,
            }

            # Invalidate cached PDF when filters OR options change
            filter_key = str(sorted(
                (k, tuple(v)) for k, v in active_filters.items()
            )) + str(sorted(pdf_options.items()))
            if st.session_state.get("_pdf_filter_key") != filter_key:
                st.session_state["_pdf_bytes"]      = None
                st.session_state["_pdf_filter_key"] = filter_key

            if st.button("📄 Generate PDF Report", use_container_width=True):
                with st.spinner("Generating PDF — this may take a few seconds…"):
                    try:
                        st.session_state["_pdf_bytes"] = generate_pdf(
                            filtered_data,
                            active_filters,
                            pdf_options=pdf_options,
                        )
                    except ValueError as exc:
                        st.error(str(exc))
                        st.session_state["_pdf_bytes"] = None

            if st.session_state.get("_pdf_bytes"):
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=st.session_state["_pdf_bytes"],
                    file_name="jibe_pms_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

        st.markdown("---")
        st.caption("JiBe PMS Survey Analysis Tool · Anglo Eastern")

    # ── Validate filter result and render page ────────────────────────────────
    if active_filters:
        n_shown = len(filtered_data["survey"])
        n_total = len(data["survey"])

        if n_shown == 0:
            st.warning(
                "⚠️ **No responses match the current filters.** "
                "The selected combination returns zero respondents — try removing or "
                "broadening one or more filters using the sidebar."
            )
            st.stop()

        st.info(
            f"🔵 **Filters active** — showing **{n_shown:,}** of **{n_total:,}** responses. "
            f"Use the sidebar to adjust or clear filters."
        )

    PAGES[page_name].render(filtered_data)


if __name__ == "__main__":
    main()
