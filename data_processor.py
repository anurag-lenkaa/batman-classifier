"""
Data processing and scoring model for the Batman classifier.

Scoring formula (all dimensions normalised 0-100):
  composite_score =
      imdb_norm   * 0.22
    + rt_critics  * 0.22
    + rt_audience * 0.22
    + metacritic  * 0.16
    + sentiment   * 0.12
    + box_office_norm * 0.06
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from textblob import TextBlob

VOTES_PATH = Path(__file__).parent / "data" / "votes.json"

WEIGHTS = {
    "imdb_norm": 0.22,
    "rt_critics_norm": 0.22,
    "rt_audience_norm": 0.22,
    "metacritic_norm": 0.16,
    "sentiment_norm": 0.12,
    "box_office_norm": 0.06,
}

ACTOR_COLORS = {
    "Michael Keaton": "#FFD700",
    "Val Kilmer": "#C0C0C0",
    "George Clooney": "#CD7F32",
    "Christian Bale": "#4FC3F7",
    "Ben Affleck": "#EF5350",
    "Robert Pattinson": "#AB47BC",
}

ERA_ORDER = ["Burton Era", "Schumacher Era", "Nolan Trilogy", "DCEU Era", "Reeves Era"]


def _sentiment_score(reviews: list[str]) -> float:
    """Mean TextBlob polarity of a list of review strings, scaled to 0-100."""
    if not reviews:
        return 50.0
    polarities = [TextBlob(r).sentiment.polarity for r in reviews]
    mean_polarity = np.mean(polarities)  # -1 … +1
    return round((mean_polarity + 1) / 2 * 100, 2)


def _sentiment_detail(reviews: list[str]) -> dict:
    """Return per-review polarity + subjectivity details."""
    results = []
    for r in reviews:
        blob = TextBlob(r)
        results.append(
            {
                "review": r,
                "polarity": round(blob.sentiment.polarity, 3),
                "subjectivity": round(blob.sentiment.subjectivity, 3),
                "label": (
                    "Positive"
                    if blob.sentiment.polarity > 0.05
                    else "Negative"
                    if blob.sentiment.polarity < -0.05
                    else "Neutral"
                ),
            }
        )
    return results


def build_dataframe(films: list[dict]) -> pd.DataFrame:
    """Convert raw film dicts into a processed DataFrame with all scores."""
    rows = []
    for f in films:
        sentiment = _sentiment_score(f.get("reviews", []))
        rows.append(
            {
                "title": f["title"],
                "year": f["year"],
                "actor": f["actor"],
                "director": f.get("director", ""),
                "era": f.get("era", ""),
                "imdb_rating": f.get("imdb_rating", 0),
                "imdb_votes": f.get("imdb_votes", 0),
                "rt_critics": f.get("rt_critics", 0),
                "rt_audience": f.get("rt_audience", 0),
                "metacritic": f.get("metacritic", 0),
                "box_office_million": f.get("box_office_million", 0),
                "budget_million": f.get("budget_million", 0),
                "runtime_min": f.get("runtime_min", 0),
                "sentiment_score": sentiment,
                "reviews": f.get("reviews", []),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("year").reset_index(drop=True)

    # Normalise to 0-100
    df["imdb_norm"] = (df["imdb_rating"] / 10) * 100
    df["rt_critics_norm"] = df["rt_critics"].astype(float)
    df["rt_audience_norm"] = df["rt_audience"].astype(float)
    df["metacritic_norm"] = df["metacritic"].astype(float)
    df["sentiment_norm"] = df["sentiment_score"].astype(float)

    # Box office: log-normalise so outliers don't dominate
    bo = df["box_office_million"].replace(0, np.nan)
    log_bo = np.log1p(bo)
    df["box_office_norm"] = (
        (log_bo - log_bo.min()) / (log_bo.max() - log_bo.min()) * 100
    ).fillna(0)

    # Composite performance score
    df["composite_score"] = sum(
        df[col] * w for col, w in WEIGHTS.items()
    ).round(2)

    # Tidy display columns
    df["film_label"] = df["title"] + " (" + df["year"].astype(str) + ")"
    df["actor_color"] = df["actor"].map(ACTOR_COLORS).fillna("#FFFFFF")

    return df


def actor_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate film-level scores up to actor-level statistics."""
    agg = (
        df.groupby("actor")
        .agg(
            films=("title", "count"),
            avg_imdb=("imdb_rating", "mean"),
            avg_rt_critics=("rt_critics", "mean"),
            avg_rt_audience=("rt_audience", "mean"),
            avg_metacritic=("metacritic", "mean"),
            avg_sentiment=("sentiment_score", "mean"),
            avg_composite=("composite_score", "mean"),
            best_film=("composite_score", "idxmax"),
            total_box_office=("box_office_million", "sum"),
        )
        .reset_index()
    )

    # Resolve best_film index to title
    agg["best_film"] = agg["best_film"].apply(lambda i: df.loc[i, "film_label"])

    # Round numerics
    for col in agg.select_dtypes(include="float").columns:
        agg[col] = agg[col].round(2)

    agg["actor_color"] = agg["actor"].map(ACTOR_COLORS).fillna("#FFFFFF")

    # Rank actors by composite score (1 = best)
    agg["rank"] = agg["avg_composite"].rank(ascending=False).astype(int)
    agg = agg.sort_values("rank")

    return agg.reset_index(drop=True)


def get_sentiment_details(df: pd.DataFrame) -> pd.DataFrame:
    """Explode per-review sentiment details into a flat DataFrame."""
    rows = []
    for _, film in df.iterrows():
        for detail in _sentiment_detail(film["reviews"]):
            rows.append(
                {
                    "film": film["film_label"],
                    "actor": film["actor"],
                    "era": film["era"],
                    **detail,
                }
            )
    return pd.DataFrame(rows)


def load_votes() -> dict:
    if VOTES_PATH.exists():
        with open(VOTES_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"actor_votes": {}, "film_votes": {}, "total_votes": 0}


def save_votes(votes: dict) -> None:
    with open(VOTES_PATH, "w", encoding="utf-8") as f:
        json.dump(votes, f, indent=2)


def cast_actor_vote(actor: str) -> dict:
    votes = load_votes()
    votes["actor_votes"][actor] = votes["actor_votes"].get(actor, 0) + 1
    votes["total_votes"] = votes.get("total_votes", 0) + 1
    save_votes(votes)
    return votes


def cast_film_vote(film_label: str) -> dict:
    votes = load_votes()
    votes["film_votes"][film_label] = votes["film_votes"].get(film_label, 0) + 1
    save_votes(votes)
    return votes
