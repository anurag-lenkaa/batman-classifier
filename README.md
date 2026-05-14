# 🦇 Batman Actor Classifier

A data-driven dashboard that scores every live-action Batman performance using official ratings, audience sentiment, and a weighted composite model — with a live voting system.

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-2.1+-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## Live Demo
https://batman-classifier-562aaautjbkjffw3eiyr4d.streamlit.app/

> Deploy your own free instance → [share.streamlit.io](https://share.streamlit.io)

---

## What It Does

Covers all 11 Batman films across 6 actors:

| Actor | Films |
|---|---|
| Michael Keaton | Batman (1989), Batman Returns (1992) |
| Val Kilmer | Batman Forever (1995) |
| George Clooney | Batman & Robin (1997) |
| Christian Bale | Batman Begins (2005), The Dark Knight (2008), The Dark Knight Rises (2012) |
| Ben Affleck | Batman v Superman (2016), Justice League (2017), Zack Snyder's Justice League (2021) |
| Robert Pattinson | The Batman (2022) |

---

## Scoring Model

All sources normalised to 0–100, then combined with weighted average:

```
Composite Score =
    IMDb Rating        × 22%
  + RT Critics         × 22%
  + RT Audience        × 22%
  + Metacritic         × 16%
  + Sentiment (NLP)    × 12%
  + Box Office (log)   ×  6%
```

Weights are adjustable live in the sidebar.

---

## Features

- **📊 Film Overview** — sortable ratings table with colour-coded scores, metric bar charts, score timeline
- **🏆 Actor Rankings** — ranked actor cards, radar chart across 5 dimensions, box office vs score scatter
- **🗳️ Voting Dashboard** — vote for favourite actor and film, live bar charts, fan votes blended into final ranking
- **🧠 Sentiment Analysis** — TextBlob NLP on 8 review excerpts per film, polarity box plots, positive/neutral/negative breakdown, browsable review explorer

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `Streamlit` | Web app framework |
| `Pandas` | Data wrangling and aggregation |
| `BeautifulSoup4` | HTML scraping from Wikipedia |
| `Requests` | HTTP requests |
| `TextBlob` | NLP sentiment analysis |
| `Plotly` | Interactive charts |
| `NumPy` | Score normalisation |

---

## Project Structure

```
batman_classifier/
├── app.py                  # Streamlit dashboard
├── scraper.py              # BeautifulSoup + Wikipedia API scraper
├── data_processor.py       # Pandas scoring model + TextBlob sentiment
├── requirements.txt        # Dependencies
└── data/
    ├── batman_films.json   # Film ratings and review excerpts dataset
    └── votes.json          # Persisted vote counts
```

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/batman-classifier.git
cd batman-classifier

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app**
4. Select your repo, branch `main`, file `app.py`
5. Click **Deploy**

Done — live public URL in ~2 minutes.

---

## Results

Current composite scores (default weights):

| Rank | Actor | Score |
|---|---|---|
| 🥇 1 | Christian Bale | 82.97 |
| 🥈 2 | Robert Pattinson | 78.09 |
| 🥉 3 | Michael Keaton | 68.82 |
| 4 | Ben Affleck | 60.40 |
| 5 | Val Kilmer | 45.02 |
| 6 | George Clooney | 23.92 |

---

## License

MIT
