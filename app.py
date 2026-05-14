"""
Batman Actor Classifier — Streamlit Dashboard
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scraper import get_all_films
from data_processor import (
    ACTOR_COLORS,
    ERA_ORDER,
    actor_summary,
    build_dataframe,
    cast_actor_vote,
    cast_film_vote,
    get_sentiment_details,
    load_votes,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Batman Actor Classifier",
    page_icon="🦇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Batman dark theme CSS ──────────────────────────────────────────────────────
st.markdown(
    """
<style>
  /* ---- Global ---- */
  html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
  }
  .stApp { background-color: #0a0a0f; color: #e8e8e8; }
  [data-testid="stSidebar"] { background-color: #0f0f1a; border-right: 1px solid #2a2a3a; }

  /* ---- Hero banner ---- */
  .hero-banner {
    background: linear-gradient(135deg, #0d0d0d 0%, #1a1a2e 50%, #0d0d0d 100%);
    border: 1px solid #FFD700;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    text-align: center;
  }
  .hero-title {
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: 6px;
    color: #FFD700;
    text-shadow: 0 0 30px rgba(255,215,0,0.4);
    margin: 0;
  }
  .hero-subtitle {
    font-size: 1rem;
    color: #aaaaaa;
    letter-spacing: 3px;
    margin-top: 8px;
  }

  /* ---- Metric cards ---- */
  .metric-card {
    background: #12121f;
    border: 1px solid #2a2a3a;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
  }
  .metric-value { font-size: 2rem; font-weight: 700; color: #FFD700; }
  .metric-label { font-size: 0.75rem; color: #888; letter-spacing: 1px; text-transform: uppercase; }

  /* ---- Score badge ---- */
  .score-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
  }

  /* ---- Actor card ---- */
  .actor-card {
    background: linear-gradient(180deg, #12121f, #0d0d18);
    border-radius: 12px;
    padding: 20px;
    border-left: 4px solid;
    margin-bottom: 12px;
  }
  .actor-name { font-size: 1.2rem; font-weight: 700; margin: 0 0 4px 0; }
  .actor-rank { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }

  /* ---- Vote button override ---- */
  div[data-testid="stButton"] > button {
    background: #1a1a2e;
    color: #FFD700;
    border: 1px solid #FFD700;
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 1px;
    transition: all 0.2s;
    width: 100%;
  }
  div[data-testid="stButton"] > button:hover {
    background: #FFD700;
    color: #0a0a0f;
  }

  /* ---- Tab styling ---- */
  [data-testid="stTabs"] button {
    color: #aaa !important;
    font-weight: 600;
    letter-spacing: 0.5px;
  }
  [data-testid="stTabs"] button[aria-selected="true"] {
    color: #FFD700 !important;
    border-bottom-color: #FFD700 !important;
  }

  /* ---- Divider ---- */
  hr { border-color: #2a2a3a; }

  /* ---- Info / success / warning ---- */
  .stAlert { background-color: #12121f; }
</style>
""",
    unsafe_allow_html=True,
)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0a0a0f",
    plot_bgcolor="#0a0a0f",
    font=dict(color="#e8e8e8", family="Segoe UI"),
    xaxis=dict(gridcolor="#1e1e2e", zerolinecolor="#1e1e2e"),
    yaxis=dict(gridcolor="#1e1e2e", zerolinecolor="#1e1e2e"),
)


# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading Batman data …")
def load_data():
    films = get_all_films(enrich=False)
    df = build_dataframe(films)
    actors_df = actor_summary(df)
    sentiment_df = get_sentiment_details(df)
    return df, actors_df, sentiment_df


df, actors_df, sentiment_df = load_data()

# Session-state: track whether current session has voted
if "voted_actor" not in st.session_state:
    st.session_state.voted_actor = None
if "voted_film" not in st.session_state:
    st.session_state.voted_film = None

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero-banner">
  <p class="hero-title">🦇 BATMAN CLASSIFIER</p>
  <p class="hero-subtitle">RATINGS · SENTIMENT · PERFORMANCE SCORES · AUDIENCE VOTE</p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦇 Filters")

    all_actors = sorted(df["actor"].unique())
    selected_actors = st.multiselect(
        "Actors", all_actors, default=all_actors, key="actor_filter"
    )

    all_eras = [e for e in ERA_ORDER if e in df["era"].unique()]
    selected_eras = st.multiselect(
        "Era", all_eras, default=all_eras, key="era_filter"
    )

    year_min, year_max = int(df["year"].min()), int(df["year"].max())
    year_range = st.slider("Release Year", year_min, year_max, (year_min, year_max))

    st.divider()
    st.markdown("### Score Weights")
    st.caption("Adjust how composite score is computed (display only)")

    w_imdb = st.slider("IMDb", 0, 40, 22)
    w_rt_c = st.slider("RT Critics", 0, 40, 22)
    w_rt_a = st.slider("RT Audience", 0, 40, 22)
    w_meta = st.slider("Metacritic", 0, 40, 16)
    w_sent = st.slider("Sentiment", 0, 30, 12)
    w_bo   = st.slider("Box Office", 0, 20, 6)

    total_w = w_imdb + w_rt_c + w_rt_a + w_meta + w_sent + w_bo
    if total_w > 0:
        custom_weights = {
            "imdb_norm": w_imdb / total_w,
            "rt_critics_norm": w_rt_c / total_w,
            "rt_audience_norm": w_rt_a / total_w,
            "metacritic_norm": w_meta / total_w,
            "sentiment_norm": w_sent / total_w,
            "box_office_norm": w_bo / total_w,
        }
    else:
        custom_weights = None

    st.divider()
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# ── Apply filters ─────────────────────────────────────────────────────────────
mask = (
    df["actor"].isin(selected_actors)
    & df["era"].isin(selected_eras)
    & df["year"].between(*year_range)
)
fdf = df[mask].copy()

# Recompute composite with custom weights if adjusted
if custom_weights:
    fdf["composite_score"] = sum(
        fdf[col] * w for col, w in custom_weights.items()
    ).round(2)

fact_df = actor_summary(fdf)

# ── Top metrics row ───────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    (c1, str(len(fdf)), "Films"),
    (c2, str(len(fdf["actor"].unique())), "Actors"),
    (c3, f"{fdf['imdb_rating'].mean():.1f}", "Avg IMDb"),
    (c4, f"{fdf['rt_critics'].mean():.0f}%", "Avg RT Critics"),
    (c5, f"{fdf['composite_score'].mean():.1f}", "Avg Score"),
]
for col, val, lbl in metrics:
    col.markdown(
        f'<div class="metric-card"><div class="metric-value">{val}</div>'
        f'<div class="metric-label">{lbl}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Film Overview", "🏆 Actor Rankings", "🗳️ Vote!", "🧠 Sentiment Analysis"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Film Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("All Batman Films")

    # Ratings heatmap-style table
    display_cols = [
        "film_label", "actor", "era",
        "imdb_rating", "rt_critics", "rt_audience",
        "metacritic", "sentiment_score", "composite_score",
    ]
    display_df = fdf[display_cols].copy().sort_values("composite_score", ascending=False)
    display_df.columns = [
        "Film", "Actor", "Era",
        "IMDb", "RT Critics %", "RT Audience %",
        "Metacritic", "Sentiment", "Composite ↑",
    ]

    def _color_score(val):
        if val >= 80:
            return "color: #4caf50; font-weight: 700"
        if val >= 60:
            return "color: #FFD700; font-weight: 600"
        if val >= 40:
            return "color: #ff9800"
        return "color: #f44336"

    styled = (
        display_df.style
        .applymap(_color_score, subset=["IMDb", "RT Critics %", "RT Audience %",
                                         "Metacritic", "Sentiment", "Composite ↑"])
        .format(
            {
                "IMDb": "{:.1f}",
                "RT Critics %": "{:.0f}%",
                "RT Audience %": "{:.0f}%",
                "Metacritic": "{:.0f}",
                "Sentiment": "{:.1f}",
                "Composite ↑": "{:.1f}",
            }
        )
        .set_properties(**{"background-color": "#12121f", "border-color": "#2a2a3a"})
        .set_table_styles(
            [{"selector": "th", "props": [("background-color", "#0f0f1a"),
                                           ("color", "#FFD700"),
                                           ("font-weight", "700")]}]
        )
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()

    # Multi-metric bar chart
    col_a, col_b = st.columns([3, 2])
    with col_a:
        metric = st.selectbox(
            "Metric to visualise",
            ["composite_score", "imdb_rating", "rt_critics", "rt_audience",
             "metacritic", "sentiment_score", "box_office_million"],
            format_func=lambda x: x.replace("_", " ").title(),
        )
        sorted_fdf = fdf.sort_values(metric, ascending=False)
        fig = px.bar(
            sorted_fdf,
            x="film_label",
            y=metric,
            color="actor",
            color_discrete_map=ACTOR_COLORS,
            text_auto=".1f",
            title=f"{metric.replace('_', ' ').title()} by Film",
            labels={"film_label": "", metric: ""},
        )
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=True,
                          legend=dict(bgcolor="#0a0a0f"),
                          xaxis_tickangle=-35)
        fig.update_traces(textfont_size=10)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Score breakdown — top film**")
        top_film = fdf.loc[fdf["composite_score"].idxmax()]
        dims = ["IMDb (norm)", "RT Critics", "RT Audience", "Metacritic", "Sentiment", "Box Office (norm)"]
        vals = [
            top_film["imdb_norm"],
            top_film["rt_critics_norm"],
            top_film["rt_audience_norm"],
            top_film["metacritic_norm"],
            top_film["sentiment_norm"],
            top_film["box_office_norm"],
        ]
        fig2 = go.Figure(go.Bar(
            x=vals, y=dims,
            orientation="h",
            marker_color="#FFD700",
            text=[f"{v:.1f}" for v in vals],
            textposition="outside",
        ))
        fig2.update_layout(
            **PLOTLY_LAYOUT,
            title=f"{top_film['film_label']}",
            xaxis_range=[0, 110],
            height=300,
            margin=dict(l=0, r=20, t=40, b=0),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Timeline scatter
        st.markdown("**Composite score timeline**")
        fig3 = px.scatter(
            fdf,
            x="year", y="composite_score",
            color="actor",
            color_discrete_map=ACTOR_COLORS,
            size="imdb_votes",
            hover_name="film_label",
            size_max=30,
        )
        fig3.update_layout(**PLOTLY_LAYOUT, showlegend=False,
                           xaxis_title="", yaxis_title="Score",
                           height=250, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Actor Rankings
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Actor Performance Rankings")

    # Actor cards
    for _, row in fact_df.iterrows():
        color = ACTOR_COLORS.get(row["actor"], "#FFD700")
        rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣"][min(row["rank"] - 1, 5)]
        st.markdown(
            f"""
<div class="actor-card" style="border-left-color:{color}">
  <span class="actor-name" style="color:{color}">{rank_emoji} {row['actor']}</span>
  <span class="actor-rank"> — rank #{row['rank']} of {len(fact_df)}</span><br>
  <small>
    🎬 {row['films']} film(s) &nbsp;|&nbsp;
    ⭐ IMDb {row['avg_imdb']:.1f} &nbsp;|&nbsp;
    🍅 RT {row['avg_rt_critics']:.0f}% / {row['avg_rt_audience']:.0f}% &nbsp;|&nbsp;
    🎯 Meta {row['avg_metacritic']:.0f} &nbsp;|&nbsp;
    💬 Sentiment {row['avg_sentiment']:.1f} &nbsp;|&nbsp;
    🏆 Composite <strong style="color:{color}">{row['avg_composite']:.1f}</strong>
  </small><br>
  <small style="color:#aaa">Best film: {row['best_film']}</small>
</div>
""",
            unsafe_allow_html=True,
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        # Composite score bar chart
        fig = px.bar(
            fact_df,
            x="avg_composite",
            y="actor",
            orientation="h",
            color="actor",
            color_discrete_map=ACTOR_COLORS,
            text="avg_composite",
            title="Composite Performance Score (avg across films)",
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(
            **PLOTLY_LAYOUT,
            showlegend=False,
            xaxis_range=[0, 105],
            xaxis_title="Score /100",
            yaxis_title="",
            yaxis=dict(categoryorder="total ascending", gridcolor="#1e1e2e"),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Radar chart — multi-dimension comparison
        dimensions = ["avg_imdb", "avg_rt_critics", "avg_rt_audience",
                      "avg_metacritic", "avg_sentiment"]
        dim_labels = ["IMDb (×10)", "RT Critics", "RT Audience", "Metacritic", "Sentiment"]

        fig_radar = go.Figure()
        for _, row in fact_df.iterrows():
            vals = [
                row["avg_imdb"] * 10,
                row["avg_rt_critics"],
                row["avg_rt_audience"],
                row["avg_metacritic"],
                row["avg_sentiment"],
            ]
            vals += [vals[0]]  # close the polygon
            labels = dim_labels + [dim_labels[0]]
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=vals,
                    theta=labels,
                    fill="toself",
                    name=row["actor"],
                    line_color=ACTOR_COLORS.get(row["actor"], "#fff"),
                    opacity=0.7,
                )
            )
        fig_radar.update_layout(
            **PLOTLY_LAYOUT,
            polar=dict(
                bgcolor="#0d0d1a",
                radialaxis=dict(visible=True, range=[0, 100], color="#555"),
                angularaxis=dict(color="#aaa"),
            ),
            showlegend=True,
            legend=dict(bgcolor="#0a0a0f"),
            title="Multi-Dimension Radar (avg per actor)",
            height=400,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Film-by-film grouped comparison
    st.markdown("### Film-by-Film Composite Score")
    fig4 = px.bar(
        fdf.sort_values("year"),
        x="film_label",
        y="composite_score",
        color="actor",
        color_discrete_map=ACTOR_COLORS,
        barmode="group",
        text_auto=".1f",
    )
    fig4.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_tickangle=-35,
        xaxis_title="",
        yaxis_title="Composite Score",
        showlegend=True,
        legend=dict(bgcolor="#0a0a0f"),
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Box office vs score scatter
    st.markdown("### Box Office vs Composite Score")
    bo_df = fdf[fdf["box_office_million"] > 0].copy()
    if not bo_df.empty:
        fig5 = px.scatter(
            bo_df,
            x="box_office_million",
            y="composite_score",
            color="actor",
            color_discrete_map=ACTOR_COLORS,
            size="imdb_votes",
            hover_name="film_label",
            trendline="ols",
            trendline_scope="overall",
            labels={"box_office_million": "Box Office ($M)", "composite_score": "Score"},
            size_max=40,
        )
        fig5.update_layout(**PLOTLY_LAYOUT, legend=dict(bgcolor="#0a0a0f"))
        st.plotly_chart(fig5, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Voting Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Cast Your Vote")

    votes = load_votes()

    col_vote, col_results = st.columns([1, 1])

    with col_vote:
        st.markdown("#### Who is your favourite Batman?")
        if st.session_state.voted_actor:
            st.success(f"You voted for **{st.session_state.voted_actor}** this session!")
        else:
            actor_cols = st.columns(2)
            for i, actor in enumerate(sorted(ACTOR_COLORS.keys())):
                col = actor_cols[i % 2]
                if col.button(f"🦇 {actor}", key=f"vote_actor_{actor}"):
                    updated = cast_actor_vote(actor)
                    st.session_state.voted_actor = actor
                    votes = updated
                    st.rerun()

        st.divider()
        st.markdown("#### Best Batman film?")
        if st.session_state.voted_film:
            st.success(f"You voted for **{st.session_state.voted_film}**!")
        else:
            film_options = fdf["film_label"].tolist()
            chosen_film = st.selectbox("Pick a film", film_options, index=None,
                                       placeholder="Select a film …")
            if st.button("Cast Film Vote", disabled=chosen_film is None):
                updated = cast_film_vote(chosen_film)
                st.session_state.voted_film = chosen_film
                votes = updated
                st.rerun()

    with col_results:
        st.markdown("#### Live Actor Votes")
        actor_vote_data = votes.get("actor_votes", {})
        if any(actor_vote_data.values()):
            av_df = pd.DataFrame(
                list(actor_vote_data.items()), columns=["Actor", "Votes"]
            ).sort_values("Votes", ascending=True)
            av_df["Color"] = av_df["Actor"].map(ACTOR_COLORS).fillna("#aaa")

            fig_votes = px.bar(
                av_df,
                x="Votes",
                y="Actor",
                orientation="h",
                color="Actor",
                color_discrete_map=ACTOR_COLORS,
                text="Votes",
            )
            fig_votes.update_layout(
                **PLOTLY_LAYOUT,
                showlegend=False,
                height=320,
                xaxis_title="Votes",
                yaxis_title="",
                margin=dict(l=0, r=20, t=20, b=0),
            )
            fig_votes.update_traces(textposition="outside")
            st.plotly_chart(fig_votes, use_container_width=True)
        else:
            st.info("No actor votes yet — be the first!")

        st.markdown("#### Live Film Votes")
        film_vote_data = votes.get("film_votes", {})
        if any(film_vote_data.values()):
            fv_df = (
                pd.DataFrame(list(film_vote_data.items()), columns=["Film", "Votes"])
                .query("Votes > 0")
                .sort_values("Votes", ascending=False)
            )
            fig_fv = px.bar(
                fv_df,
                x="Film",
                y="Votes",
                color="Votes",
                color_continuous_scale=[[0, "#1a1a2e"], [1, "#FFD700"]],
                text="Votes",
            )
            fig_fv.update_layout(
                **PLOTLY_LAYOUT,
                showlegend=False,
                xaxis_tickangle=-35,
                coloraxis_showscale=False,
                height=280,
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(fig_fv, use_container_width=True)
        else:
            st.info("No film votes yet!")

    st.divider()
    total = votes.get("total_votes", 0)
    st.markdown(f"**Total actor votes cast:** `{total}`")

    # Fan Score integration: recompute actor rank incorporating votes
    if any(actor_vote_data.values()):
        st.markdown("### Rankings with Fan Vote Incorporated")
        fan_df = pd.DataFrame(
            list(actor_vote_data.items()), columns=["actor", "fan_votes"]
        )
        max_fv = fan_df["fan_votes"].max() or 1
        fan_df["fan_score_norm"] = (fan_df["fan_votes"] / max_fv) * 100

        combined = fact_df.merge(fan_df, on="actor", how="left").fillna(0)
        combined["fan_composite"] = (
            combined["avg_composite"] * 0.85 + combined["fan_score_norm"] * 0.15
        ).round(2)
        combined = combined.sort_values("fan_composite", ascending=False)

        fig_fan = px.bar(
            combined,
            x="fan_composite",
            y="actor",
            orientation="h",
            color="actor",
            color_discrete_map=ACTOR_COLORS,
            text="fan_composite",
            title="Score (85% critic/audience + 15% fan vote)",
        )
        fig_fan.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_fan.update_layout(
            **PLOTLY_LAYOUT,
            showlegend=False,
            xaxis_range=[0, 105],
            yaxis=dict(categoryorder="total ascending", gridcolor="#1e1e2e"),
            height=320,
        )
        st.plotly_chart(fig_fan, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Sentiment Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Review Sentiment Analysis")
    st.caption("TextBlob polarity applied to curated review excerpts per film.")

    fsent = sentiment_df[
        sentiment_df["actor"].isin(fdf["actor"].unique())
    ].copy()

    col_sa, col_sb = st.columns(2)

    with col_sa:
        # Sentiment distribution per actor
        fig_sent = px.box(
            fsent,
            x="actor",
            y="polarity",
            color="actor",
            color_discrete_map=ACTOR_COLORS,
            points="all",
            hover_data=["review"],
            title="Polarity Distribution by Actor",
            labels={"polarity": "Polarity (–1 negative → +1 positive)", "actor": ""},
        )
        fig_sent.add_hline(y=0, line_dash="dash", line_color="#888",
                           annotation_text="Neutral", annotation_font_color="#888")
        fig_sent.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=380)
        st.plotly_chart(fig_sent, use_container_width=True)

    with col_sb:
        # Label breakdown stacked bar
        label_counts = (
            fsent.groupby(["actor", "label"])
            .size()
            .reset_index(name="count")
        )
        fig_labels = px.bar(
            label_counts,
            x="actor",
            y="count",
            color="label",
            color_discrete_map={
                "Positive": "#4caf50",
                "Neutral": "#FFD700",
                "Negative": "#f44336",
            },
            barmode="stack",
            title="Review Sentiment Labels per Actor",
            labels={"actor": "", "count": "Reviews"},
        )
        fig_labels.update_layout(**PLOTLY_LAYOUT, legend=dict(bgcolor="#0a0a0f"),
                                  height=380)
        st.plotly_chart(fig_labels, use_container_width=True)

    st.divider()

    # Per-film sentiment heatmap
    film_sent_avg = (
        fsent.groupby(["film", "actor"])["polarity"]
        .mean()
        .reset_index()
        .sort_values("polarity", ascending=False)
    )
    fig_heat = px.bar(
        film_sent_avg,
        x="film",
        y="polarity",
        color="actor",
        color_discrete_map=ACTOR_COLORS,
        text_auto=".2f",
        title="Average Sentiment Polarity per Film",
        labels={"film": "", "polarity": "Avg Polarity"},
    )
    fig_heat.add_hline(y=0, line_dash="dot", line_color="#888")
    fig_heat.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_tickangle=-35,
        showlegend=True,
        legend=dict(bgcolor="#0a0a0f"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # Browse individual reviews
    st.markdown("### Browse Reviews")
    selected_actor_sent = st.selectbox(
        "Filter by actor", ["All"] + sorted(fsent["actor"].unique()), key="sent_actor"
    )
    sent_filter = fsent if selected_actor_sent == "All" else fsent[fsent["actor"] == selected_actor_sent]

    sentiment_label_filter = st.radio(
        "Label", ["All", "Positive", "Neutral", "Negative"], horizontal=True
    )
    if sentiment_label_filter != "All":
        sent_filter = sent_filter[sent_filter["label"] == sentiment_label_filter]

    for _, row in sent_filter.iterrows():
        color = "#4caf50" if row["label"] == "Positive" else "#f44336" if row["label"] == "Negative" else "#FFD700"
        st.markdown(
            f"""
<div style="background:#12121f;border-left:3px solid {color};
            border-radius:6px;padding:10px 14px;margin-bottom:8px;">
  <small style="color:#888">{row['film']} — {row['actor']}</small><br>
  <span style="color:#e8e8e8">"{row['review']}"</span><br>
  <small style="color:{color}">
    {row['label']} &nbsp;|&nbsp; polarity: {row['polarity']:.3f} &nbsp;|&nbsp; subjectivity: {row['subjectivity']:.3f}
  </small>
</div>
""",
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:#555;font-size:0.8rem;'>"
    "🦇 Batman Actor Classifier &nbsp;·&nbsp; "
    "Data: IMDb / RT / Metacritic (curated) &nbsp;·&nbsp; "
    "Sentiment: TextBlob &nbsp;·&nbsp; "
    "Scraping: BeautifulSoup + Wikipedia API"
    "</p>",
    unsafe_allow_html=True,
)
