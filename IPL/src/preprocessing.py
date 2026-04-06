"""
preprocessing.py
────────────────
Cleans the raw IPL dataset and engineers features used by the model:
  • Null handling & type coercion
  • Team name normalisation (franchise renames over the years)
  • Feature engineering: win_rate, toss_advantage, venue_strength, h2h_ratio
  • Train / test split (stratified by season)
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger(__name__)

# ── Team name aliases (IPL franchises have rebranded multiple times) ───────────
TEAM_ALIASES: dict[str, str] = {
    "Deccan Chargers":          "Sunrisers Hyderabad",
    "Rising Pune Supergiants":  "Chennai Super Kings",   # replacement team
    "Rising Pune Supergiant":   "Chennai Super Kings",
    "Pune Warriors":            "Delhi Capitals",         # absorbed
    "Kochi Tuskers Kerala":     "Royal Challengers Bangalore",
    "Delhi Daredevils":         "Delhi Capitals",
}

# ── Column schema we expect from load_matches() ───────────────────────────────
REQUIRED_COLS = [
    "id", "season", "team1", "team2",
    "toss_winner", "toss_decision",
    "winner", "venue",
]


# ── Main pipeline ──────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1 – Raw cleaning.
    """
    df = df.copy()

    # Drop rows with no winner (abandoned / no result)
    df = df[df["winner"].notna() & (df["winner"] != "")]

    # Normalise team names
    for col in ["team1", "team2", "toss_winner", "winner"]:
        if col in df.columns:
            df[col] = df[col].map(lambda x: TEAM_ALIASES.get(str(x).strip(), str(x).strip()))

    # Season as int
    df["season"] = pd.to_numeric(df["season"], errors="coerce").fillna(0).astype(int)

    # Toss decision binary
    df["bat_first"] = (df["toss_decision"].str.lower() == "bat").astype(int)

    # Toss winner == match winner?
    df["toss_winner_won"] = (df["toss_winner"] == df["winner"]).astype(int)

    # Win margin (unified)
    df["win_by_runs"]     = pd.to_numeric(df.get("win_by_runs", 0),     errors="coerce").fillna(0)
    df["win_by_wickets"]  = pd.to_numeric(df.get("win_by_wickets", 0),  errors="coerce").fillna(0)

    log.info(f"After cleaning: {len(df)} matches, {df['season'].nunique()} seasons.")
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2 – Feature engineering.
    Adds columns used by the model.
    """
    df = df.copy()

    # ── Win rate per team (rolling over all historical data) ──────────────────
    all_teams = pd.unique(df[["team1", "team2"]].values.ravel())
    win_counts = df["winner"].value_counts()
    # matches played = appearances as team1 + team2
    played = df["team1"].value_counts().add(df["team2"].value_counts(), fill_value=0)
    win_rate = (win_counts / played).fillna(0.5).to_dict()

    df["team1_win_rate"] = df["team1"].map(win_rate).fillna(0.5)
    df["team2_win_rate"] = df["team2"].map(win_rate).fillna(0.5)

    # ── Head-to-head ratio ────────────────────────────────────────────────────
    h2h: dict[tuple, int] = {}   # (t1, t2) -> wins by t1
    for _, row in df.iterrows():
        key = tuple(sorted([row["team1"], row["team2"]]))
        h2h[key] = h2h.get(key, 0)
        if row["winner"] == key[0]:
            h2h[key] += 1

    played_h2h: dict[tuple, int] = {}
    for _, row in df.iterrows():
        key = tuple(sorted([row["team1"], row["team2"]]))
        played_h2h[key] = played_h2h.get(key, 0) + 1

    def h2h_ratio(row):
        key = tuple(sorted([row["team1"], row["team2"]]))
        total = played_h2h.get(key, 1)
        wins  = h2h.get(key, 0)
        # from team1's perspective
        if row["team1"] == key[0]:
            return round(wins / total, 4)
        else:
            return round((total - wins) / total, 4)

    df["h2h_ratio"] = df.apply(h2h_ratio, axis=1)

    # ── Venue win rate for each team ──────────────────────────────────────────
    venue_wins   = df.groupby(["venue", "winner"]).size().reset_index(name="vw")
    venue_played = (
        df.groupby("venue")["team1"].count().reset_index(name="vp")
    )
    venue_rate_df = venue_wins.merge(venue_played, on="venue")
    venue_rate_df["venue_win_rate"] = (venue_rate_df["vw"] / venue_rate_df["vp"]).round(4)
    venue_win_rate: dict[tuple, float] = {
        (r["venue"], r["winner"]): r["venue_win_rate"]
        for _, r in venue_rate_df.iterrows()
    }

    df["team1_venue_rate"] = df.apply(
        lambda r: venue_win_rate.get((r["venue"], r["team1"]), 0.5), axis=1
    )
    df["team2_venue_rate"] = df.apply(
        lambda r: venue_win_rate.get((r["venue"], r["team2"]), 0.5), axis=1
    )

    # ── Recent form (last 5 matches win rate per team) ────────────────────────
    df_sorted = df.sort_values(["season", "id"]).reset_index(drop=True)
    recent_form: dict[str, list[int]] = {}

    form1_list, form2_list = [], []
    for _, row in df_sorted.iterrows():
        t1, t2 = row["team1"], row["team2"]
        f1 = np.mean(recent_form.get(t1, [0.5])[-5:])
        f2 = np.mean(recent_form.get(t2, [0.5])[-5:])
        form1_list.append(round(f1, 4))
        form2_list.append(round(f2, 4))
        # update
        for team in [t1, t2]:
            recent_form.setdefault(team, [])
        recent_form[t1].append(1 if row["winner"] == t1 else 0)
        recent_form[t2].append(1 if row["winner"] == t2 else 0)

    df_sorted["team1_recent_form"] = form1_list
    df_sorted["team2_recent_form"] = form2_list

    # Merge form columns back on original index order
    df = df.merge(
        df_sorted[["id", "team1_recent_form", "team2_recent_form"]],
        on="id", how="left"
    )

    # ── Target label: did team1 win? ──────────────────────────────────────────
    df["team1_won"] = (df["winner"] == df["team1"]).astype(int)

    log.info("Feature engineering complete.")
    return df


def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Step 3 – Encode categorical columns for ML.
    Returns (df_encoded, encoders_dict).
    """
    encoders: dict[str, LabelEncoder] = {}
    cat_cols = ["team1", "team2", "venue", "toss_winner"]

    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    return df, encoders


FEATURE_COLS = [
    "team1_enc",
    "team2_enc",
    "venue_enc",
    "toss_winner_enc",
    "bat_first",
    "team1_win_rate",
    "team2_win_rate",
    "h2h_ratio",
    "team1_venue_rate",
    "team2_venue_rate",
    "team1_recent_form",
    "team2_recent_form",
]

TARGET_COL = "team1_won"


def get_train_test(
    df: pd.DataFrame,
    encoders: dict,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Step 4 – Return X_train, X_test, y_train, y_test (stratified by season).
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    log.info(
        f"Train: {len(X_train)} | Test: {len(X_test)} | "
        f"Features: {len(available)}"
    )
    return X_train, X_test, y_train, y_test


# ── Convenience wrapper ────────────────────────────────────────────────────────

def full_pipeline(df_raw: pd.DataFrame):
    """
    Run the complete preprocessing pipeline.
    Returns (df_processed, encoders, X_train, X_test, y_train, y_test).
    """
    df = clean(df_raw)
    df = engineer_features(df)
    df, encoders = encode_labels(df)
    X_train, X_test, y_train, y_test = get_train_test(df, encoders)
    return df, encoders, X_train, X_test, y_train, y_test


# ── Analytics helpers (used by graphs module) ─────────────────────────────────

def season_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Return team win rates broken down by season."""
    results = []
    for season, grp in df.groupby("season"):
        for team in pd.unique(grp[["team1", "team2"]].values.ravel()):
            played = ((grp["team1"] == team) | (grp["team2"] == team)).sum()
            won    = (grp["winner"] == team).sum()
            if played > 0:
                results.append({"season": season, "team": team,
                                 "played": played, "won": won,
                                 "win_pct": round(won / played * 100, 1)})
    return pd.DataFrame(results)


def toss_impact(df: pd.DataFrame) -> pd.DataFrame:
    """How often does the toss winner go on to win the match?"""
    grp = df.groupby("toss_decision")["toss_winner_won"].agg(["sum", "count"])
    grp["win_pct"] = (grp["sum"] / grp["count"] * 100).round(1)
    return grp.reset_index()


def venue_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Matches played and most successful team per venue."""
    rows = []
    for venue, grp in df.groupby("venue"):
        top_team = grp["winner"].mode().iloc[0] if not grp["winner"].mode().empty else ""
        rows.append({
            "venue": venue,
            "matches_played": len(grp),
            "top_team": top_team,
            "top_team_wins": (grp["winner"] == top_team).sum(),
        })
    return pd.DataFrame(rows).sort_values("matches_played", ascending=False)


def head_to_head(df: pd.DataFrame, team1: str, team2: str) -> dict:
    """Return H2H stats between two teams."""
    mask = (
        ((df["team1"] == team1) & (df["team2"] == team2)) |
        ((df["team1"] == team2) & (df["team2"] == team1))
    )
    matches = df[mask]
    total   = len(matches)
    t1_wins = (matches["winner"] == team1).sum()
    t2_wins = (matches["winner"] == team2).sum()
    return {
        "total_matches":  int(total),
        f"{team1}_wins":  int(t1_wins),
        f"{team2}_wins":  int(t2_wins),
        "no_result":      int(total - t1_wins - t2_wins),
    }
