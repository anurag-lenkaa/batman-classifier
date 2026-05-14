"""
Batman film scraper — pulls supplementary data from Wikipedia.
Falls back to the local JSON dataset if any request fails.
"""
import json
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
DATA_PATH = Path(__file__).parent / "data" / "batman_films.json"
WIKI_SEARCH_BASE = "https://en.wikipedia.org/w/api.php"


def _load_local() -> dict:
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def _fetch_wiki_summary(title: str) -> str:
    """Return the opening paragraph of a Wikipedia article for a film title."""
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "redirects": 1,
    }
    try:
        resp = requests.get(WIKI_SEARCH_BASE, params=params, headers=HEADERS, timeout=8)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "")
            if extract:
                # Return only the first paragraph
                return extract.split("\n")[0]
    except Exception:
        pass
    return ""


def _scrape_batman_filmography() -> list[dict]:
    """
    Scrape the Batman-in-film Wikipedia article and parse the
    'Live-action theatrical films' table to get cast / year data.
    Returns a list of dicts: {title, year, actor, director}.
    """
    url = "https://en.wikipedia.org/wiki/Batman_in_film"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[scraper] Wikipedia fetch failed: {exc}")
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    films = []

    # The page has multiple wikitables; find the one with theatrical films
    for table in soup.find_all("table", class_="wikitable"):
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        # We want a table that mentions both 'film' and 'actor'
        if not any("film" in h for h in headers):
            continue
        if not any("batman" in h or "actor" in h or "cast" in h for h in headers):
            continue

        rows = table.find_all("tr")
        col_headers = [th.get_text(strip=True) for th in rows[0].find_all("th")]

        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            row_data = {}
            for idx, cell in enumerate(cells):
                if idx < len(col_headers):
                    row_data[col_headers[idx]] = cell.get_text(strip=True)

            # Try to extract year from title cell or dedicated column
            title_text = (
                row_data.get("Film", "")
                or row_data.get("Title", "")
                or cells[0].get_text(strip=True)
            )
            year_text = row_data.get("Year", row_data.get("Release", ""))
            actor_text = (
                row_data.get("Batman", "")
                or row_data.get("Actor", "")
                or row_data.get("Cast", "")
            )
            director_text = row_data.get("Director", "")

            if title_text:
                films.append(
                    {
                        "title": title_text,
                        "year": year_text,
                        "actor": actor_text,
                        "director": director_text,
                    }
                )

        if films:
            break  # found our table

    return films


def enrich_with_wiki_summaries(films: list[dict]) -> list[dict]:
    """Add a wiki_summary field to each film dict."""
    enriched = []
    for film in films:
        query = f"{film['title']} film {film['year']}"
        summary = _fetch_wiki_summary(query)
        enriched.append({**film, "wiki_summary": summary})
        time.sleep(0.3)  # be polite to Wikipedia
    return enriched


def get_all_films(enrich: bool = False) -> list[dict]:
    """
    Primary entry point.
    1. Load local JSON (always reliable).
    2. Optionally attempt to scrape Wikipedia for supplementary data.
    3. Merge scraped rows into local data where titles match.
    Returns the final list of film dicts.
    """
    local_data = _load_local()
    films = local_data["films"]

    if enrich:
        print("[scraper] Attempting Wikipedia scrape …")
        scraped = _scrape_batman_filmography()
        if scraped:
            print(f"[scraper] Scraped {len(scraped)} rows from Wikipedia.")
            scraped_by_year = {
                str(s.get("year", "")): s for s in scraped if s.get("year")
            }
            for film in films:
                match = scraped_by_year.get(str(film["year"]))
                if match:
                    # Only pull fields not already in local data
                    for k, v in match.items():
                        if k not in film and v:
                            film[k] = v
        else:
            print("[scraper] Scrape returned no data; using local dataset.")

    return films


if __name__ == "__main__":
    films = get_all_films(enrich=True)
    for f in films:
        print(f"{f['year']}  {f['title']:<45}  {f['actor']}")
