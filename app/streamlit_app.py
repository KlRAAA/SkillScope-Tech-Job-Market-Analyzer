"""SkillScope: Tech Job Market Analyzer (Streamlit dashboard).

Run locally from the repo root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import viz
from src.features.skills import SKILL_CATEGORY, SKILL_NAMES
from src.models.predict import MIN_WORDS, analyze, load_models

st.set_page_config(
    page_title="SkillScope: Tech Job Market Analyzer",
    page_icon=":material/insights:",
    layout="wide",
)

LEVELS = viz.SENIORITY_ORDER


# --------------------------------------------------------------------------- data


@st.cache_data
def load_jobs() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "processed" / "app_jobs.parquet")


@st.cache_resource
def get_models() -> dict:
    return load_models(ROOT / "models")


jobs = load_jobs()


def style(fig: go.Figure, height: int = 420) -> go.Figure:
    """Shared Plotly look: recessive grid, left-aligned title, readable hover."""
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 20, "t": 50, "b": 10},
        font={"family": "sans-serif", "size": 13, "color": viz.TEXT_PRIMARY},
        title={"x": 0, "xanchor": "left", "font": {"size": 16}},
        plot_bgcolor=viz.SURFACE,
        paper_bgcolor=viz.SURFACE,
        hoverlabel={"bgcolor": "white", "font_size": 12},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "traceorder": "normal",
        },
    )
    fig.update_xaxes(gridcolor=viz.GRID, zeroline=False, linecolor=viz.GRID)
    fig.update_yaxes(
        gridcolor=viz.GRID, zeroline=False, linecolor=viz.GRID, automargin=True
    )
    # Horizontal bars with value labels at the end: leave room so labels aren't cut off.
    labeled = [
        t
        for t in fig.data
        if t.type == "bar" and t.orientation == "h" and t.textposition == "outside"
    ]
    if labeled and fig.layout.xaxis.range is None:
        top = max(max(t.x) for t in labeled)
        fig.update_xaxes(range=[0, top * 1.18])
        fig.update_traces(cliponaxis=False, selector={"type": "bar"})
    return fig


def filter_row(df: pd.DataFrame, key: str, show_state: bool = True) -> pd.DataFrame:
    """One row of filters above the charts. Empty selection means 'all'."""
    cols = st.columns(3 if show_state else 2)
    levels = cols[0].multiselect(
        "Seniority", LEVELS, key=f"{key}_level", placeholder="All levels"
    )
    roles = cols[1].multiselect(
        "Role cluster",
        sorted(df["cluster"].unique()),
        key=f"{key}_role",
        placeholder="All roles",
    )
    states = []
    if show_state:
        top_states = df["state"].value_counts().index.tolist()
        states = cols[2].multiselect(
            "State", top_states, key=f"{key}_state", placeholder="All states"
        )
    mask = pd.Series(True, index=df.index)
    if levels:
        mask &= df["seniority"].isin(levels)
    if roles:
        mask &= df["cluster"].isin(roles)
    if states:
        mask &= df["state"].isin(states)
    return df.loc[mask]


def too_few(df: pd.DataFrame, minimum: int = 30) -> bool:
    if len(df) < minimum:
        st.warning(
            f"Only {len(df)} postings match these filters. Widen the selection to see reliable numbers."
        )
        return True
    return False


# --------------------------------------------------------------------------- pages


def overview_page():
    st.title("SkillScope: Tech Job Market Analyzer")
    st.caption(
        "10,040 US tech job postings from LinkedIn (April 2024, Kaggle `arshkon/linkedin-job-postings`), "
        "filtered to software, data, cloud, security and IT roles."
    )

    salaried = jobs["salary_yearly"].dropna()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tech postings", f"{len(jobs):,}", border=True)
    k2.metric("Companies", f"{jobs['company_name'].nunique():,}", border=True)
    k3.metric(
        "Median salary",
        f"${salaried.median() / 1000:.0f}k",
        help=f"Yearly, USD. Based on the {len(salaried):,} postings ({len(salaried) / len(jobs):.0%}) that list a salary.",
        border=True,
    )
    k4.metric(
        "Remote",
        f"{(jobs['work_type'] == 'remote').mean():.0%}",
        help="Share of postings marked remote. Another 18% mention hybrid work.",
        border=True,
    )

    left, right = st.columns(2)
    with left:
        daily = (
            jobs.set_index("listed_date")
            .resample("D")
            .size()
            .rename("postings")
            .reset_index()
        )
        daily = daily[daily["listed_date"] >= "2024-04-01"]
        fig = px.line(
            daily, x="listed_date", y="postings", markers=True, title="Postings per day"
        )
        fig.update_traces(
            line_color=viz.BLUE,
            hovertemplate="%{x|%a %b %d}: %{y:,} postings<extra></extra>",
        )
        fig.update_layout(xaxis_title=None, yaxis_title="Postings")
        st.plotly_chart(style(fig, 380), use_container_width=True)
        st.caption(
            "The daily counts show when the data was collected, not hiring demand: 55% of postings were "
            "listed on April 18–19. Use this dataset for comparisons, not trends."
        )
    with right:
        roles = (
            jobs["cluster"]
            .value_counts()
            .sort_values()
            .rename("postings")
            .reset_index()
        )
        fig = px.bar(
            roles,
            x="postings",
            y="cluster",
            orientation="h",
            title="Postings by role cluster",
        )
        fig.update_traces(
            marker_color=viz.BLUE, hovertemplate="%{y}: %{x:,} postings<extra></extra>"
        )
        fig.update_layout(xaxis_title="Postings", yaxis_title=None)
        st.plotly_chart(style(fig, 380), use_container_width=True)

    with st.expander("About the data and its limitations"):
        st.markdown("""
- **Snapshot:** almost all postings were listed between April 5 and 20, 2024.
- **US-heavy:** California and Texas alone hold 23% of postings. 17% only say "United States".
- **Salary** is known for 29% of postings, converted to yearly USD, with outliers removed (IQR rule).
- **Seniority** comes from LinkedIn's experience level, which is missing for 29% of postings. Entry = Internship + Entry level + Associate, Mid = Mid-Senior level, Senior = Director + Executive.
- **Skills** are detected with a 166-skill regex dictionary. **Role clusters** come from unsupervised text clustering (10 clusters).
""")


def skills_page():
    st.title("Skills explorer")
    st.caption(
        "Share of postings that mention each skill. Filter by seniority, role cluster and state."
    )
    df = filter_row(jobs, "skills")
    top_n = st.slider("Number of skills", 10, 40, 20, step=5)
    st.markdown(f"**{len(df):,} postings** match the filters.")
    if too_few(df):
        return

    share = (df[SKILL_NAMES].mean() * 100).sort_values(ascending=False).head(top_n)
    table = pd.DataFrame(
        {
            "skill": share.index,
            "share": share.values,
            "postings": df[share.index].sum().values,
            "category": [SKILL_CATEGORY[s] for s in share.index],
            "overall": (jobs[share.index].mean() * 100).values,
        }
    ).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=table["share"],
            y=table["skill"],
            orientation="h",
            marker_color=viz.BLUE,
            text=[f"{v:.0f}%" for v in table["share"]],
            textposition="outside",
            customdata=table[["postings", "category", "overall"]],
            hovertemplate=(
                "<b>%{y}</b> (%{customdata[1]})<br>%{x:.1f}% of filtered postings (%{customdata[0]:,})"
                "<br>%{customdata[2]:.1f}% of all postings<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"Top {top_n} skills",
        xaxis_title="% of postings mentioning the skill",
        yaxis_title=None,
    )
    st.plotly_chart(style(fig, 28 * top_n + 90), use_container_width=True)

    with st.expander("Table view"):
        st.dataframe(
            table.iloc[::-1]
            .rename(columns={"share": "% of filtered", "overall": "% of all"})
            .round(1),
            hide_index=True,
            use_container_width=True,
        )


def salary_page():
    st.title("Salary & location")
    st.caption("Yearly salary in USD, from the 29% of postings that list one.")
    df = filter_row(jobs, "salary", show_state=False)
    sal = df.dropna(subset=["salary_yearly"])
    st.markdown(
        f"**{len(df):,} postings** match the filters, **{len(sal):,}** with a salary."
    )
    if too_few(sal):
        return

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for level in LEVELS:
            values = sal.loc[sal["seniority"] == level, "salary_yearly"]
            if len(values):
                fig.add_trace(
                    go.Box(
                        x=values,
                        name=f"{level} (n={len(values):,})",
                        marker_color=viz.SENIORITY_COLORS[level],
                        boxpoints=False,
                        hovertemplate="%{x:$,.0f}<extra></extra>",
                    )
                )
        fig.update_layout(
            title="Salary by seniority",
            xaxis_title="Yearly salary (USD)",
            showlegend=False,
        )
        fig.update_xaxes(tickformat="$,.0s")
        st.plotly_chart(style(fig, 360), use_container_width=True)

    with right:
        by_role = sal.groupby("cluster", observed=True)["salary_yearly"].agg(
            ["median", "size"]
        )
        by_role = by_role[by_role["size"] >= 15].sort_values("median")
        fig = go.Figure(
            go.Bar(
                x=by_role["median"],
                y=by_role.index,
                orientation="h",
                marker_color=viz.BLUE,
                customdata=by_role["size"],
                text=[f"${v / 1000:.0f}k" for v in by_role["median"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>median %{x:$,.0f} (n=%{customdata})<extra></extra>",
            )
        )
        fig.update_layout(
            title="Median salary by role cluster",
            xaxis_title="Median yearly salary (USD)",
            yaxis_title=None,
        )
        fig.update_xaxes(tickformat="$,.0s")
        st.plotly_chart(style(fig, 360), use_container_width=True)

    st.subheader("Location")
    min_n = st.slider(
        "Minimum salaried postings per state (for the salary chart)",
        10,
        100,
        30,
        step=10,
    )
    left, right = st.columns(2)
    with left:
        states = df["state"].value_counts().head(15).sort_values()
        fig = go.Figure(
            go.Bar(
                x=states.values,
                y=states.index,
                orientation="h",
                marker_color=viz.BLUE,
                hovertemplate="%{y}: %{x:,} postings<extra></extra>",
            )
        )
        fig.update_layout(
            title="Postings by state (top 15)", xaxis_title="Postings", yaxis_title=None
        )
        st.plotly_chart(style(fig, 460), use_container_width=True)
    with right:
        by_state = sal.groupby("state", observed=True)["salary_yearly"].agg(
            ["median", "size"]
        )
        by_state = by_state[by_state["size"] >= min_n].sort_values("median").tail(15)
        if by_state.empty:
            st.info("No state has enough salaried postings for these filters.")
        else:
            fig = go.Figure(
                go.Bar(
                    x=by_state["median"],
                    y=by_state.index,
                    orientation="h",
                    marker_color=viz.BLUE,
                    customdata=by_state["size"],
                    hovertemplate="<b>%{y}</b><br>median %{x:$,.0f} (n=%{customdata})<extra></extra>",
                )
            )
            fig.update_layout(
                title="Median salary by state",
                xaxis_title="Median yearly salary (USD)",
                yaxis_title=None,
            )
            fig.update_xaxes(tickformat="$,.0s")
            st.plotly_chart(style(fig, 460), use_container_width=True)

    work = (
        df["work_type"]
        .value_counts(normalize=True)
        .reindex(viz.WORK_TYPE_ORDER)
        .fillna(0)
        * 100
    )
    fig = go.Figure()
    for wt in viz.WORK_TYPE_ORDER:
        fig.add_trace(
            go.Bar(
                x=[work[wt]],
                y=["Filtered postings"],
                orientation="h",
                name=wt,
                marker={
                    "color": viz.WORK_TYPE_COLORS[wt],
                    "line": {"color": viz.SURFACE, "width": 2},
                },
                text=f"{wt} {work[wt]:.0f}%",
                textposition="inside",
                insidetextfont={"color": viz.WORK_TYPE_TEXT[wt]},
                hovertemplate=f"{wt}: %{{x:.1f}}%<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        title="Work type",
        xaxis_title="% of postings",
        yaxis_title=None,
    )
    fig.update_xaxes(range=[0, 100])
    st.plotly_chart(style(fig, 200), use_container_width=True)
    st.caption(
        '"On-site" means the posting is not marked remote and does not mention hybrid work.'
    )


EXAMPLE = {
    "title": "Junior Data Analyst",
    "description": (
        "We are looking for a Junior Data Analyst to join our analytics team. You will clean and analyze "
        "sales data using SQL and Excel, build weekly dashboards in Tableau, and present findings to "
        "business stakeholders. Requirements: Bachelor's degree in Statistics, Economics, Computer Science "
        "or a related field; 0-2 years of experience; strong attention to detail. Nice to have: Python "
        "(pandas) and experience with A/B testing. This is an entry-level role with mentorship from "
        "experienced analysts."
    ),
}


def analyzer_page():
    st.title("Job description analyzer")
    st.caption(
        "Paste a job posting to get its predicted seniority, role cluster, the skills it mentions, and "
        "skills that similar postings commonly ask for."
    )

    if st.button("Load an example", icon=":material/description:"):
        st.session_state["an_title"] = EXAMPLE["title"]
        st.session_state["an_desc"] = EXAMPLE["description"]

    title = st.text_input(
        "Job title (optional)", key="an_title", placeholder="e.g. Backend Engineer"
    )
    description = st.text_area(
        "Job description",
        key="an_desc",
        height=240,
        placeholder="Paste the full description: responsibilities, requirements, nice-to-haves ...",
    )
    if not st.button("Analyze", type="primary", icon=":material/search:"):
        return
    if not description.strip():
        st.info(f"Paste a job description first (at least {MIN_WORDS} words).")
        return

    with st.spinner("Analyzing ..."):
        result = analyze(
            title, description, get_models(), jobs[SKILL_NAMES], jobs["cluster"]
        )
    if result.seniority is None:
        st.warning(result.warning)
        return
    if result.warning:
        st.info(result.warning)

    confidence = result.seniority_proba[result.seniority]
    # Bordered cards instead of st.metric: metric values truncate long cluster names.
    cards = [
        ("Predicted seniority", result.seniority, f"{confidence:.0%} probability"),
        ("Role cluster", result.cluster, "from unsupervised clustering"),
        ("Skills detected", str(len(result.skills)), "from a 166-skill dictionary"),
    ]
    for col, (label, value, note) in zip(st.columns(3), cards):
        with col.container(border=True):
            st.caption(label)
            st.markdown(f"#### {value}")
            st.caption(note)

    left, right = st.columns([2, 3])
    with left:
        proba = pd.Series(result.seniority_proba).reindex(LEVELS)
        fig = go.Figure(
            go.Bar(
                x=proba.values * 100,
                y=proba.index,
                orientation="h",
                marker_color=[viz.SENIORITY_COLORS[level] for level in proba.index],
                text=[f"{p:.0%}" for p in proba.values],
                textposition="outside",
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            )
        )
        fig.update_layout(
            title="Seniority probabilities", xaxis_title="%", yaxis_title=None
        )
        fig.update_xaxes(range=[0, 110])
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig, 240), use_container_width=True)
    with right:
        st.markdown("**Skills mentioned**")
        if result.skills:
            st.markdown(" ".join(f":blue-badge[{s}]" for s in result.skills))
        else:
            st.caption("No skills from the 166-skill dictionary were found.")
        st.markdown(f"**Often listed alongside these in *{result.cluster}* postings**")
        if result.related:
            for skill, share in result.related:
                st.markdown(f"- **{skill}**: in {share:.0%} of similar postings")
        else:
            st.caption("Not enough similar postings to suggest related skills.")

    with st.expander("How this works (and its limits)"):
        st.markdown("""
- **Seniority:** an XGBoost model on TF-IDF text features plus hand-crafted features (years of experience,
  skills, keywords such as *lead* or *intern*). On companies it never saw during training it reaches a
  **macro F1 of about 0.6** (0.26 for always guessing "Mid"). It is best at Mid (F1 0.79) and weakest at
  Senior (0.45), which it often mistakes for Mid. The labels come from LinkedIn and are noisy, so treat the
  output as a hint, not a verdict.
- **Known limitation:** the model sees words, not their meaning in context. In the example above, changing
  "*mentorship from experienced analysts*" to "*... senior analysts*" flips the prediction from Entry to Mid,
  because "senior" is one of the strongest Mid signals.
- **Role cluster:** unsupervised grouping of postings (TF-IDF → SVD → KMeans, 10 clusters). The three
  software-engineering clusters overlap a lot.
- **Related skills:** among postings in the same cluster that share at least one of your skills (weighted by
  how many they share), the skills they list most often that your text doesn't mention.
""")


pages = st.navigation(
    [
        st.Page(
            overview_page, title="Overview", icon=":material/dashboard:", default=True
        ),
        st.Page(skills_page, title="Skills explorer", icon=":material/psychology:"),
        st.Page(salary_page, title="Salary & location", icon=":material/payments:"),
        st.Page(
            analyzer_page, title="Job description analyzer", icon=":material/search:"
        ),
    ]
)
pages.run()
