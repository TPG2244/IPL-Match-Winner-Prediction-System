"""
data_collection.py
──────────────────
Handles all data ingestion:
  • Downloads the Kaggle IPL dataset (CSV fallback bundled)
  • Scrapes live / recent match results from ESPN Cricinfo
  • Merges new results into the master dataset automatically
"""

import os
import time
import json
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data"
RAW_DIR    = DATA_DIR / "raw"
CACHE_DIR  = DATA_DIR / "cache"

for d in [DATA_DIR, RAW_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MATCHES_CSV   = RAW_DIR / "matches.csv"
DELIVERIES_CSV = RAW_DIR / "deliveries.csv"
LIVE_CACHE    = CACHE_DIR / "live_matches.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── ESPN Cricinfo scraper ──────────────────────────────────────────────────────
ESPNCRICINFO_IPL_URL = (
    "https://www.espncricinfo.com/series/ipl-2024-1410320/match-results"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


def _cache_is_fresh(path: Path, ttl_minutes: int = 60) -> bool:
    """Return True if cache file exists and is younger than ttl_minutes."""
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(minutes=ttl_minutes)


def fetch_live_matches(force: bool = False) -> list[dict]:
    """
    Fetch recent IPL match results from ESPN Cricinfo.
    Returns a list of match dicts.  Falls back to cache on failure.
    """
    if not force and _cache_is_fresh(LIVE_CACHE, ttl_minutes=60):
        log.info("Using cached live matches (< 60 min old).")
        with open(LIVE_CACHE) as f:
            return json.load(f)

    log.info("Fetching live match data from ESPN Cricinfo …")
    matches = []
    try:
        resp = requests.get(ESPNCRICINFO_IPL_URL, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # ESPN Cricinfo match cards – selector may vary by year
        cards = soup.select("div.ds-p-4")
        for card in cards:
            try:
                teams   = card.select("p.ds-text-tight-m")
                scores  = card.select("div.ds-text-compact-s")
                result  = card.select_one("p.ds-text-tight-s")
                venue   = card.select_one("span.ds-text-compact-xxs")

                team1 = teams[0].get_text(strip=True) if len(teams) > 0 else None
                team2 = teams[1].get_text(strip=True) if len(teams) > 1 else None
                winner_text = result.get_text(strip=True) if result else ""

                if team1 and team2:
                    matches.append(
                        {
                            "team1":   team1,
                            "team2":   team2,
                            "result":  winner_text,
                            "venue":   venue.get_text(strip=True) if venue else "",
                            "fetched_at": datetime.now().isoformat(),
                        }
                    )
            except Exception:
                continue

        if matches:
            with open(LIVE_CACHE, "w") as f:
                json.dump(matches, f, indent=2)
            log.info(f"Fetched {len(matches)} live match records.")
        else:
            log.warning("No matches parsed from ESPN Cricinfo (page structure may have changed).")

    except requests.RequestException as e:
        log.warning(f"Live fetch failed: {e}. Falling back to cache.")
        if LIVE_CACHE.exists():
            with open(LIVE_CACHE) as f:
                matches = json.load(f)

    return matches


# ── Kaggle / bundled dataset loader ───────────────────────────────────────────

def load_matches() -> pd.DataFrame:
    """
    Load historical IPL matches.
    Priority:
      1. data/raw/matches.csv  (user-placed Kaggle file)
      2. Synthetic seed dataset bundled here (for demo purposes)
    """
    if MATCHES_CSV.exists():
        log.info(f"Loading matches from {MATCHES_CSV}")
        return pd.read_csv(MATCHES_CSV)

    log.warning("matches.csv not found – generating synthetic seed dataset.")
    return _generate_seed_dataset()


def load_deliveries() -> pd.DataFrame | None:
    """Load ball-by-ball data if present."""
    if DELIVERIES_CSV.exists():
        log.info(f"Loading deliveries from {DELIVERIES_CSV}")
        return pd.read_csv(DELIVERIES_CSV)
    log.info("deliveries.csv not found – skipping ball-by-ball analysis.")
    return None


# ── Synthetic seed dataset ─────────────────────────────────────────────────────
IPL_TEAMS = [
    "Mumbai Indians",
    "Chennai Super Kings",
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Delhi Capitals",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Punjab Kings",
    "Lucknow Super Giants",
    "Gujarat Titans",
]

VENUES = [
    "Wankhede Stadium, Mumbai",
    "MA Chidambaram Stadium, Chennai",
    "M Chinnaswamy Stadium, Bangalore",
    "Eden Gardens, Kolkata",
    "Arun Jaitley Stadium, Delhi",
    "Sawai Mansingh Stadium, Jaipur",
    "Rajiv Gandhi Intl Stadium, Hyderabad",
    "Punjab Cricket Association Stadium, Mohali",
    "Ekana Cricket Stadium, Lucknow",
    "Narendra Modi Stadium, Ahmedabad",
]

def _generate_seed_dataset() -> pd.DataFrame:
    """Create a realistic-looking synthetic IPL dataset (2008-2023)."""
    import numpy as np
    rng = np.random.default_rng(42)

    rows = []
    match_id = 1

    # Team strength weights (higher = slightly better win rate)
    strengths = {
        "Mumbai Indians": 0.62,
        "Chennai Super Kings": 0.60,
        "Kolkata Knight Riders": 0.54,
        "Royal Challengers Bangalore": 0.48,
        "Rajasthan Royals": 0.50,
        "Delhi Capitals": 0.47,
        "Sunrisers Hyderabad": 0.50,
        "Punjab Kings": 0.43,
        "Lucknow Super Giants": 0.52,
        "Gujarat Titans": 0.55,
    }

    for season in range(2008, 2024):
        available = IPL_TEAMS if season >= 2022 else IPL_TEAMS[:8]
        matchups = [
            (t1, t2)
            for i, t1 in enumerate(available)
            for t2 in available[i + 1:]
        ]
        for team1, team2 in matchups:
            venue = rng.choice(VENUES)
            toss_winner = rng.choice([team1, team2])
            toss_decision = rng.choice(["bat", "field"])

            # Win probability: blend team strength + small random noise
            p1 = (strengths.get(team1, 0.5) + rng.uniform(-0.1, 0.1))
            p1 = max(0.15, min(0.85, p1))
            winner = team1 if rng.random() < p1 else team2
            loser  = team2 if winner == team1 else team1

            win_by_runs   = int(rng.integers(1, 80))  if rng.random() > 0.45 else 0
            win_by_wickets = int(rng.integers(1, 10)) if win_by_runs == 0 else 0
            player_of_match = f"Player_{rng.integers(100, 999)}"

            rows.append(
                {
                    "id":                match_id,
                    "season":            season,
                    "city":              venue.split(",")[-1].strip(),
                    "date":              f"{season}-04-{rng.integers(1,30):02d}",
                    "team1":             team1,
                    "team2":             team2,
                    "toss_winner":       toss_winner,
                    "toss_decision":     toss_decision,
                    "result":            "normal",
                    "dl_applied":        0,
                    "winner":            winner,
                    "win_by_runs":       win_by_runs,
                    "win_by_wickets":    win_by_wickets,
                    "player_of_match":   player_of_match,
                    "venue":             venue,
                    "umpire1":           "Umpire A",
                    "umpire2":           "Umpire B",
                }
            )
            match_id += 1

    df = pd.DataFrame(rows)
    df.to_csv(MATCHES_CSV, index=False)
    log.info(f"Seed dataset written to {MATCHES_CSV} ({len(df)} rows).")
    return df


# ── Auto-updater ───────────────────────────────────────────────────────────────

def append_live_to_dataset(live_matches: list[dict]) -> bool:
    """
    Checks live_matches against the historical CSV and appends genuinely
    new rows.  Returns True if the dataset was updated.
    """
    if not live_matches:
        return False

    df = load_matches()
    existing_keys = set(
        zip(df["team1"].str.lower(), df["team2"].str.lower(), df["season"].astype(str))
    )

    new_rows = []
    current_year = str(datetime.now().year)

    for m in live_matches:
        t1 = m.get("team1", "")
        t2 = m.get("team2", "")
        key = (t1.lower(), t2.lower(), current_year)
        if key not in existing_keys and t1 and t2:
            new_rows.append(
                {
                    "id":             df["id"].max() + len(new_rows) + 1,
                    "season":         int(current_year),
                    "city":           m.get("venue", ""),
                    "date":           datetime.now().strftime("%Y-%m-%d"),
                    "team1":          t1,
                    "team2":          t2,
                    "toss_winner":    "",
                    "toss_decision":  "",
                    "result":         "normal",
                    "dl_applied":     0,
                    "winner":         _parse_winner(m.get("result", ""), t1, t2),
                    "win_by_runs":    0,
                    "win_by_wickets": 0,
                    "player_of_match": "",
                    "venue":          m.get("venue", ""),
                    "umpire1":        "",
                    "umpire2":        "",
                }
            )

    if new_rows:
        updated = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        updated.to_csv(MATCHES_CSV, index=False)
        log.info(f"✅  Appended {len(new_rows)} new matches to dataset.")
        return True

    log.info("No new matches to append.")
    return False


def _parse_winner(result_text: str, team1: str, team2: str) -> str:
    """Best-effort winner extraction from result description."""
    for team in [team1, team2]:
        if team.lower() in result_text.lower():
            return team
    return ""


# ── Public entry point ─────────────────────────────────────────────────────────

def run_updater() -> dict:
    """
    Full update pipeline:
      1. Fetch live data
      2. Check for new matches
      3. Update CSV if needed
    Returns a status dict.
    """
    live = fetch_live_matches()
    updated = append_live_to_dataset(live)
    df = load_matches()
    return {
        "live_matches_fetched": len(live),
        "dataset_updated": updated,
        "total_historical_matches": len(df),
        "seasons_covered": sorted(df["season"].unique().tolist()),
    }


if __name__ == "__main__":
    status = run_updater()
    print(json.dumps(status, indent=2, default=str))
