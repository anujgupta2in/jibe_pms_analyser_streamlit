"""Page 9 — Open Text Feedback: Thematic Analysis"""

import re
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from collections import Counter, defaultdict
from data_loader import LIKED_COL, EXCESSIVE_COL, VESSEL_COL, RANK_COL, PLOTLY_THEME, PRIMARY, SEQ_COLOURS

# ── Stop-words ────────────────────────────────────────────────────────────────
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "is", "it", "are", "was", "be", "as", "this", "that", "have", "has",
    "from", "by", "we", "i", "our", "its", "very", "also", "all", "can", "not",
    "no", "do", "so", "more", "some", "they", "their", "there", "which", "when",
    "will", "been", "had", "would", "could", "should", "may", "than", "if", "up",
    "out", "about", "what", "would", "into", "who", "how", "any", "just", "use",
    "used", "using", "one", "my", "your", "his", "her", "he", "she", "us",
    "each", "other", "such", "does", "did", "were", "am", "over", "new", "time",
    "get", "only", "still", "need", "well", "good", "like", "make", "most", "per",
    "etc", "na", "nan", "pms", "jibe", "system",
}

# ── Positive themes (what respondents liked) ──────────────────────────────────
POSITIVE_THEMES = {
    "✅ Easy to Use / User Friendly": [
        "easy", "user friendly", "user-friendly", "simple", "intuitive",
        "straightforward", "convenient", "accessible", "navigate", "friendly",
        "self-explanatory", "clear interface",
    ],
    "📋 Job Tracking & Monitoring": [
        "track", "tracking", "monitor", "monitoring", "deficiency", "finding",
        "overdue", "due date", "status", "visibility", "overview",
    ],
    "🔧 Spare Parts Management": [
        "spare", "parts", "spares", "inventory", "stock", "linked", "material",
        "consumable",
    ],
    "📝 Defect & Maintenance Reporting": [
        "defect", "report", "reporting", "maintenance", "observation",
        "non-conformance", "nc", "finding",
    ],
    "🌐 Real-Time Data & Accessibility": [
        "real-time", "real time", "online", "data", "access", "synchron",
        "cloud", "anywhere", "remote", "live",
    ],
    "⏰ Work & Rest Hours (WRH)": [
        "work", "rest", "hours", "wrh", "fatigue", "watchkeeping", "working hours",
    ],
    "✅ Compliance & Standards": [
        "comply", "compliance", "standard", "ism", "regulation", "audit",
        "certificate", "statutory", "flag", "class",
    ],
    "📚 Job History & Records": [
        "history", "record", "previous", "archive", "log", "past", "documented",
        "documentation",
    ],
    "🏢 Office–Vessel Integration": [
        "office", "integration", "vessel", "communication", "sync", "superintendent",
        "shore", "ship",
    ],
    "⚡ Performance & Speed": [
        "speed", "fast", "performance", "quick", "efficient", "smooth",
        "responsive",
    ],
    "📦 Comprehensive Job Coverage": [
        "comprehensive", "complete", "all jobs", "covered", "available",
        "included", "missing jobs", "orpms",
    ],
    "📊 Reporting & Analytics": [
        "analytics", "graph", "chart", "dashboard", "kpi", "statistics",
        "analysis", "summary",
    ],
}

# ── Negative themes (excessive / repetitive concerns) ────────────────────────
NEGATIVE_THEMES = {
    "📅 Too Many Weekly Jobs": [
        "weekly", "week", "weekly jobs", "weekly routine", "weekly task",
    ],
    "🔁 Repetitive / Duplicate Tasks": [
        "repetitive", "repeat", "duplicate", "similar", "same job", "same task",
        "redundant", "overlap",
    ],
    "⏱ Job Frequency Too High": [
        "frequency", "interval", "too often", "frequent", "daily", "increased",
        "reduce frequency", "optimise interval",
    ],
    "📦 Equipment-Level Task Splitting": [
        "eebd", "equipment", "individual", "club", "combine", "separate task",
        "one task", "grouped",
    ],
    "👨‍✈️ Master / Officer Verification Overload": [
        "master", "verification", "verify", "captain", "officer", "approve",
        "approval", "sign off",
    ],
    "⚙️ Deck / Engine Machinery Jobs": [
        "deck", "machinery", "winch", "crane", "anchor", "mooring", "windlass",
        "engine", "generator",
    ],
    "📋 Unclear Task Descriptions": [
        "description", "unclear", "vague", "not clear", "confusing",
        "ambiguous", "task name",
    ],
    "🔢 Too Many Sub-Tasks per Job": [
        "multiple", "sub-task", "subtask", "sub task", "many tasks",
        "too many tasks", "breakdown",
    ],
    "🛢️ Lube Oil / Chemical Routines": [
        "lube", "lubricating", "oil", "chemical", "grease", "lubrication",
    ],
    "🚨 Safety / Emergency Equipment Checks": [
        "safety", "emergency", "lifeboat", "fire", "drill", "extinguisher",
        "eba", "breathing", "survival",
    ],
}


def _clean_series(series: pd.Series, min_len: int = 5) -> pd.Series:
    """Drop nulls, obvious non-responses, and very short strings."""
    s = series.dropna()
    s = s[s.str.strip().str.len() >= min_len]
    non_responses = {"none", "nil", "n/a", "na", "no", "-", "nothing",
                     "no comment", "not applicable", "nil so far"}
    s = s[~s.str.strip().str.lower().isin(non_responses)]
    return s  # preserve original df index so vessel/rank lookup works


def _theme_counts(series: pd.Series, themes: dict) -> pd.DataFrame:
    """Count how many responses mention each theme (case-insensitive substring match).
    Returns matched indices alongside the text so callers can look up vessel/rank."""
    rows = []
    for theme, keywords in themes.items():
        pattern = "|".join(re.escape(k) for k in keywords)
        matches = series[series.str.lower().str.contains(pattern, na=False)]
        rows.append({
            "Theme":    theme,
            "Count":    len(matches),
            "Responses": matches.tolist(),
            "Indices":  matches.index.tolist(),
        })
    df = pd.DataFrame(rows).sort_values("Count", ascending=False)
    return df


def _tokenize(text_series: pd.Series) -> Counter:
    words = []
    for text in text_series.dropna():
        tokens = re.findall(r"[a-zA-Z]{3,}", str(text).lower())
        words.extend([t for t in tokens if t not in STOPWORDS])
    return Counter(words)


def _try_wordcloud(word_freq: dict, title: str, colormap: str = "Blues"):
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        wc = WordCloud(
            width=800, height=350,
            background_color="white",
            colormap=colormap,
            max_words=80,
        ).generate_from_frequencies(word_freq)
        fig_wc, ax = plt.subplots(figsize=(10, 3.5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=13, pad=8)
        st.pyplot(fig_wc, use_container_width=True)
        plt.close(fig_wc)
    except ImportError:
        pass


def _theme_bar(theme_df: pd.DataFrame, colour: str, title: str):
    fig = px.bar(
        theme_df[theme_df["Count"] > 0],
        x="Count", y="Theme", orientation="h",
        text="Count",
        color="Count",
        color_continuous_scale=["#e0f2fe", colour],
        template=PLOTLY_THEME,
        title=title,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=max(300, len(theme_df[theme_df["Count"] > 0]) * 42),
        coloraxis_showscale=False,
        xaxis_title="Responses mentioning theme",
        yaxis=dict(categoryorder="total ascending"),
        margin=dict(t=40, b=10, r=80),
    )
    return fig


def _render_quote(r: str, idx, df: pd.DataFrame):
    """Render a single quote card with vessel + rank attribution."""
    vessel = df.at[idx, VESSEL_COL] if idx in df.index and VESSEL_COL in df.columns else None
    rank   = df.at[idx, RANK_COL]   if idx in df.index and RANK_COL   in df.columns else None

    tag_parts = []
    if vessel and pd.notna(vessel) and str(vessel).strip():
        tag_parts.append(f"🚢 {str(vessel).strip()}")
    if rank and pd.notna(rank) and str(rank).strip():
        tag_parts.append(f"👤 {str(rank).strip()}")
    tag_line = "  ·  ".join(tag_parts) if tag_parts else ""

    st.markdown(
        f'<div style="border-left:3px solid #cbd5e1;padding:8px 14px 4px 14px;'
        f'background:#f8fafc;border-radius:4px;margin-bottom:2px;">'
        f'{r.strip()}</div>',
        unsafe_allow_html=True,
    )
    if tag_line:
        st.markdown(
            f'<div style="font-size:0.78rem;color:#64748b;'
            f'padding-left:16px;margin-bottom:10px;">{tag_line}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)


def _quotes_expander(
    label: str,
    responses: list,
    indices: list,
    df: pd.DataFrame,
    n: int = 5,
    icon: str = "💬",
    state_key: str = "",
):
    """Show top N representative quotes with vessel name + rank context.
    A 'Show N more' button reveals the remaining quotes in place."""
    if not responses:
        return

    show_all_key = f"_show_all_{state_key}"
    if show_all_key not in st.session_state:
        st.session_state[show_all_key] = False

    with st.expander(f"{icon} {label} — {len(responses)} responses", expanded=False):
        # Always show the first n
        for r, idx in zip(responses[:n], indices[:n]):
            _render_quote(r, idx, df)

        remaining = len(responses) - n
        if remaining > 0:
            if st.session_state[show_all_key]:
                # Show all remaining
                for r, idx in zip(responses[n:], indices[n:]):
                    _render_quote(r, idx, df)
                if st.button(f"▲ Show fewer", key=f"_btn_less_{state_key}"):
                    st.session_state[show_all_key] = False
                    st.rerun()
            else:
                if st.button(f"▼ Show {remaining} more", key=f"_btn_more_{state_key}"):
                    st.session_state[show_all_key] = True
                    st.rerun()


def render(data: dict):
    df = data["survey"]

    st.title("💬 Open Text Feedback Analysis")
    st.markdown(
        "Qualitative themes extracted from two open-ended survey questions. "
        "Responses are grouped by topic so patterns are easy to compare."
    )
    st.markdown("---")

    liked_series  = _clean_series(df[LIKED_COL],    min_len=4)
    excess_series = _clean_series(df[EXCESSIVE_COL], min_len=5)

    # ── Overview KPIs ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total respondents", len(df))
    c2.metric("Positive feedback", f"{len(liked_series)} ({len(liked_series)/len(df)*100:.0f}%)")
    c3.metric("Concerns raised", f"{len(excess_series)} ({len(excess_series)/len(df)*100:.0f}%)")
    no_concern = len(df) - len(excess_series)
    c4.metric("No concerns noted", f"{no_concern} ({no_concern/len(df)*100:.0f}%)")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 1 — POSITIVE FEEDBACK
    # ═══════════════════════════════════════════════════════════════
    st.header("✅ What Respondents Valued About JiBe PMS")
    st.caption(f"Based on {len(liked_series)} responses to: \"{LIKED_COL}\"")

    liked_themes = _theme_counts(liked_series, POSITIVE_THEMES)
    covered = liked_themes[liked_themes["Count"] > 0]["Count"].sum()
    st.markdown(
        f"**{covered}** mentions across **{(liked_themes['Count']>0).sum()}** distinct themes. "
        f"Responses can match multiple themes."
    )

    # Theme frequency bar
    st.plotly_chart(
        _theme_bar(liked_themes, "#0ea5e9", "Positive Themes — Mention Frequency"),
        use_container_width=True,
    )

    # Representative quotes per theme
    st.subheader("Representative Quotes by Positive Theme")
    for _, row in liked_themes[liked_themes["Count"] > 0].iterrows():
        safe_key = re.sub(r"[^a-z0-9]", "_", row["Theme"].lower())
        _quotes_expander(row["Theme"], row["Responses"], row["Indices"], df, n=5, icon="✅", state_key=f"pos_{safe_key}")

    # Word cloud
    st.subheader("Positive Feedback — Word Cloud")
    liked_counter = _tokenize(liked_series)
    col_wc, col_bar = st.columns([1.5, 1])
    with col_wc:
        _try_wordcloud(dict(liked_counter), "What respondents liked", colormap="Blues")
    with col_bar:
        kw_df = pd.DataFrame(liked_counter.most_common(15), columns=["Keyword", "Mentions"])
        fig_kw = px.bar(
            kw_df, x="Mentions", y="Keyword", orientation="h",
            text="Mentions", color="Mentions",
            color_continuous_scale=["#bfdbfe", "#0ea5e9"],
            template=PLOTLY_THEME,
        )
        fig_kw.update_traces(textposition="outside")
        fig_kw.update_layout(
            height=440, coloraxis_showscale=False,
            xaxis_title="Mentions", yaxis_title="",
            yaxis=dict(categoryorder="total ascending"),
            margin=dict(t=10, b=10, r=60),
        )
        st.plotly_chart(fig_kw, use_container_width=True)

    with st.expander("📄 View all positive responses", expanded=False):
        for i, r in enumerate(liked_series.tolist(), 1):
            st.markdown(f"**{i}.** {r.strip()}")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 2 — CONCERNS / NEGATIVE FEEDBACK
    # ═══════════════════════════════════════════════════════════════
    st.header("⚠️ Concerns & Areas for Improvement")
    st.caption(f"Based on {len(excess_series)} responses to: \"{EXCESSIVE_COL}\"")

    excess_themes = _theme_counts(excess_series, NEGATIVE_THEMES)
    covered_neg = excess_themes[excess_themes["Count"] > 0]["Count"].sum()
    st.markdown(
        f"**{covered_neg}** mentions across **{(excess_themes['Count']>0).sum()}** concern categories. "
        f"Responses can match multiple categories."
    )

    # Theme frequency bar
    st.plotly_chart(
        _theme_bar(excess_themes, "#ef4444", "Concern Themes — Mention Frequency"),
        use_container_width=True,
    )

    # Representative quotes per theme
    st.subheader("Representative Quotes by Concern Theme")
    for _, row in excess_themes[excess_themes["Count"] > 0].iterrows():
        safe_key = re.sub(r"[^a-z0-9]", "_", row["Theme"].lower())
        _quotes_expander(row["Theme"], row["Responses"], row["Indices"], df, n=5, icon="⚠️", state_key=f"neg_{safe_key}")

    # Word cloud
    st.subheader("Concerns — Word Cloud")
    excess_counter = _tokenize(excess_series)
    col_wc2, col_bar2 = st.columns([1.5, 1])
    with col_wc2:
        _try_wordcloud(dict(excess_counter), "Excessive / repetitive routines", colormap="Reds")
    with col_bar2:
        kw_df2 = pd.DataFrame(excess_counter.most_common(15), columns=["Keyword", "Mentions"])
        fig_kw2 = px.bar(
            kw_df2, x="Mentions", y="Keyword", orientation="h",
            text="Mentions", color="Mentions",
            color_continuous_scale=["#fee2e2", "#ef4444"],
            template=PLOTLY_THEME,
        )
        fig_kw2.update_traces(textposition="outside")
        fig_kw2.update_layout(
            height=440, coloraxis_showscale=False,
            xaxis_title="Mentions", yaxis_title="",
            yaxis=dict(categoryorder="total ascending"),
            margin=dict(t=10, b=10, r=60),
        )
        st.plotly_chart(fig_kw2, use_container_width=True)

    with st.expander("📄 View all concern responses", expanded=False):
        for i, r in enumerate(excess_series.tolist(), 1):
            st.markdown(f"**{i}.** {r.strip()}")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 3 — POSITIVE vs NEGATIVE THEME COMPARISON
    # ═══════════════════════════════════════════════════════════════
    st.header("📊 Positive vs Concern Theme Overview")

    top_pos = liked_themes[liked_themes["Count"] > 0].head(8)[["Theme", "Count"]].copy()
    top_pos["Type"] = "Positive"
    top_neg = excess_themes[excess_themes["Count"] > 0].head(8)[["Theme", "Count"]].copy()
    top_neg["Type"] = "Concern"
    compare_df = pd.concat([top_pos, top_neg], ignore_index=True)

    fig_compare = px.bar(
        compare_df, x="Count", y="Theme", color="Type",
        orientation="h", barmode="group",
        color_discrete_map={"Positive": "#22c55e", "Concern": "#ef4444"},
        template=PLOTLY_THEME,
        text="Count",
    )
    fig_compare.update_traces(textposition="outside")
    fig_compare.update_layout(
        height=500,
        xaxis_title="Responses mentioning theme",
        yaxis_title="",
        legend_title="",
        margin=dict(t=10, b=10, r=80),
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    # ── Key Insights box ──────────────────────────────────────────────────────
    st.subheader("🔍 Key Insights")
    top_positive = liked_themes.iloc[0]["Theme"] if len(liked_themes) > 0 else "N/A"
    top_concern  = excess_themes.iloc[0]["Theme"] if len(excess_themes) > 0 else "N/A"
    top_pos_n    = liked_themes.iloc[0]["Count"]  if len(liked_themes) > 0 else 0
    top_neg_n    = excess_themes.iloc[0]["Count"] if len(excess_themes) > 0 else 0

    st.markdown(
        f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
  <div style="background:#f0fdf4;border:1px solid #86efac;border-radius:8px;padding:16px;">
    <div style="font-size:0.8rem;color:#16a34a;font-weight:600;margin-bottom:4px;">TOP POSITIVE THEME</div>
    <div style="font-size:1.1rem;font-weight:700;">{top_positive}</div>
    <div style="font-size:0.85rem;color:#15803d;margin-top:4px;">{top_pos_n} respondents mentioned this</div>
  </div>
  <div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:8px;padding:16px;">
    <div style="font-size:0.8rem;color:#dc2626;font-weight:600;margin-bottom:4px;">TOP CONCERN THEME</div>
    <div style="font-size:1.1rem;font-weight:700;">{top_concern}</div>
    <div style="font-size:0.85rem;color:#b91c1c;margin-top:4px;">{top_neg_n} respondents raised this</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )
