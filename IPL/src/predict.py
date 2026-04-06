"""
predict.py
──────────
Inference layer:
  • build_feature_vector()  – converts two team names into a model input
  • predict_match_prob()    – returns win probabilities + confidence band
  • win_probability_over_overs() – simulates a prob line graph during a chase
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

log = logging.getLogger(__name__)


# ── Feature vector builder ─────────────────────────────────────────────────────

def build_feature_vector(
    team1: str,
    team2: str,
    venue: str,
    toss_winner: str,
    df: pd.DataFrame,
    encoders: dict,
    feature_names: list[str],
    toss_decision: str = "field",
) -> np.ndarray:
    """
    Construct the feature vector for a single match prediction.
    Unknown categories fall back to the most-common encoded value.
    """

    def safe_encode(encoder, val):
        classes = list(encoder.classes_)
        if val in classes:
            return encoder.transform([val])[0]
        # fallback: most frequent class
        return encoder.transform([classes[0]])[0]

    # Win rates from df
    win_counts = df["winner"].value_counts()
    played     = df["team1"].value_counts().add(df["team2"].value_counts(), fill_value=0)
    win_rate   = (win_counts / played).fillna(0.5)

    t1_win_rate = win_rate.get(team1, 0.5)
    t2_win_rate = win_rate.get(team2, 0.5)

    # H2H ratio
    mask = (
        ((df["team1"] == team1) & (df["team2"] == team2)) |
        ((df["team1"] == team2) & (df["team2"] == team1))
    )
    h2h_matches = df[mask]
    total_h2h = len(h2h_matches)
    t1_h2h_wins = (h2h_matches["winner"] == team1).sum()
    h2h_ratio = (t1_h2h_wins / total_h2h) if total_h2h > 0 else 0.5

    # Venue rates
    v_mask   = df["venue"] == venue
    v_total  = v_mask.sum()
    t1_venue = (df[v_mask]["winner"] == team1).sum() / v_total if v_total > 0 else 0.5
    t2_venue = (df[v_mask]["winner"] == team2).sum() / v_total if v_total > 0 else 0.5

    # Recent form (last 10 matches)
    def recent_form(team):
        team_matches = df[(df["team1"] == team) | (df["team2"] == team)].tail(10)
        if len(team_matches) == 0:
            return 0.5
        return (team_matches["winner"] == team).sum() / len(team_matches)

    feat = {
        "team1_enc":          safe_encode(encoders["team1"], team1),
        "team2_enc":          safe_encode(encoders["team2"], team2),
        "venue_enc":          safe_encode(encoders["venue"], venue) if venue else 0,
        "toss_winner_enc":    safe_encode(encoders["toss_winner"], toss_winner) if toss_winner else 0,
        "bat_first":          1 if toss_decision == "bat" else 0,
        "team1_win_rate":     float(t1_win_rate),
        "team2_win_rate":     float(t2_win_rate),
        "h2h_ratio":          float(h2h_ratio),
        "team1_venue_rate":   float(t1_venue),
        "team2_venue_rate":   float(t2_venue),
        "team1_recent_form":  float(recent_form(team1)),
        "team2_recent_form":  float(recent_form(team2)),
    }

    return np.array([feat.get(f, 0.5) for f in feature_names]).reshape(1, -1)


# ── Single-match prediction ────────────────────────────────────────────────────

def predict_match_prob(
    team1: str,
    team2: str,
    venue: str,
    toss_winner: str,
    df: pd.DataFrame,
    encoders: dict,
    feature_names: list[str],
    toss_decision: str = "field",
    model_choice: str = "random_forest",   # or "logistic_regression"
) -> dict:
    """
    Predict win probabilities for a single match.

    Returns:
        {
          "team1": ...,
          "team2": ...,
          "team1_win_prob": 0.63,
          "team2_win_prob": 0.37,
          "confidence": "high" | "medium" | "low",
          "model": ...,
          "disclaimer": ...,
        }
    """
    from src.model import load_models

    lr, rf, _, _ = load_models()
    model = rf if model_choice == "random_forest" else lr

    X = build_feature_vector(
        team1, team2, venue, toss_winner,
        df, encoders, feature_names, toss_decision
    )

    prob = model.predict_proba(X)[0]
    p1   = float(prob[1])  # prob that team1 wins
    p2   = 1.0 - p1

    # Confidence band
    margin = abs(p1 - 0.5)
    if margin > 0.20:
        confidence = "high"
    elif margin > 0.10:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "team1":           team1,
        "team2":           team2,
        "team1_win_prob":  round(p1, 4),
        "team2_win_prob":  round(p2, 4),
        "confidence":      confidence,
        "model":           model_choice,
        "disclaimer": (
            "Prediction based on historical patterns and model estimation. "
            "Cricket outcomes are inherently uncertain."
        ),
    }


# ── Win probability over overs (line graph data) ──────────────────────────────

def win_probability_over_overs(
    team1: str,
    team2: str,
    target: int,
    current_score: int,
    wickets_lost: int,
    overs_done: float,
    total_overs: int = 20,
) -> list[dict]:
    """
    Simulate win probability progression for a chase.
    Uses a simplified Duckworth-Lewis-style resource model.

    Returns list of {over: int, win_prob: float} for current → total_overs.
    """
    results = []
    remaining_runs = max(0, target - current_score)
    remaining_overs = max(0.1, total_overs - overs_done)
    remaining_wickets = max(0, 10 - wickets_lost)

    for over in np.arange(overs_done, total_overs + 0.5, 0.5):
        overs_left = max(0.1, total_overs - over)
        runs_needed = remaining_runs
        balls_left  = overs_left * 6
        wickets_left = remaining_wickets

        # Required run rate
        rrr = runs_needed / overs_left if overs_left > 0 else 99

        # Resource-based probability
        # Batting resource: wickets left × overs left normalised
        resource = (wickets_left / 10) * (overs_left / total_overs)

        # Logistic transformation
        difficulty = (rrr - 7.5) / 3.0   # 7.5 = typical T20 average
        raw_prob   = 1 / (1 + np.exp(difficulty - resource * 2))

        # Clamp
        prob = float(np.clip(raw_prob, 0.02, 0.98))
        results.append({"over": round(float(over), 1), "win_prob": round(prob, 4)})

    return results


# ── Confidence interval bootstrap ─────────────────────────────────────────────

def prediction_with_ci(
    team1: str,
    team2: str,
    venue: str,
    toss_winner: str,
    df: pd.DataFrame,
    encoders: dict,
    feature_names: list[str],
    n_bootstrap: int = 200,
) -> dict:
    """
    Bootstrap confidence intervals for win probability.
    Samples rows from df with replacement each iteration.
    """
    from src.model import _build_rf

    probs = []
    for _ in range(n_bootstrap):
        sample = df.sample(frac=1, replace=True, random_state=None)
        from src.preprocessing import engineer_features, encode_labels, FEATURE_COLS, TARGET_COL
        try:
            sample = engineer_features(sample)
            sample, enc_s = encode_labels(sample)
            avail = [c for c in FEATURE_COLS if c in sample.columns]
            X_s = sample[avail].values
            y_s = sample[TARGET_COL].values
            if len(np.unique(y_s)) < 2:
                continue
            m = _build_rf()
            m.fit(X_s, y_s)
            X_pred = build_feature_vector(
                team1, team2, venue, toss_winner, sample, enc_s, avail
            )
            p1 = m.predict_proba(X_pred)[0][1]
            probs.append(float(p1))
        except Exception:
            continue

    if not probs:
        return {"error": "Bootstrap failed – insufficient data"}

    arr = np.array(probs)
    return {
        "mean_prob":  round(float(arr.mean()), 4),
        "ci_lower":   round(float(np.percentile(arr, 2.5)), 4),
        "ci_upper":   round(float(np.percentile(arr, 97.5)), 4),
        "std":        round(float(arr.std()), 4),
        "n_samples":  len(probs),
    }
