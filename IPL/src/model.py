"""
model.py
────────
Trains, evaluates, and persists two models:
  • Logistic Regression  – fast baseline
  • Random Forest        – higher accuracy

Also exposes:
  • tournament_winner_probabilities() – simulates the full knockout bracket
  • get_feature_importance()          – for explainability graph
"""

import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    classification_report, confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LR_PATH  = MODELS_DIR / "logistic_regression.pkl"
RF_PATH  = MODELS_DIR / "random_forest.pkl"
META_PATH = MODELS_DIR / "model_meta.pkl"   # encoders + feature list


# ── Model definitions ──────────────────────────────────────────────────────────

def _build_lr() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    CalibratedClassifierCV(
            LogisticRegression(
                max_iter=500, C=1.0,
                solver="lbfgs", random_state=42,
            ), cv=5
        )),
    ])


def _build_rf() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def _build_gb() -> Pipeline:
    """Optional Gradient Boosting – best accuracy."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
        )),
    ])


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    encoders: dict,
    feature_names: list[str],
) -> dict:
    """
    Train both models.  Returns an evaluation report dict.
    Persists models to disk.
    """
    models = {
        "logistic_regression": _build_lr(),
        "random_forest":       _build_rf(),
    }

    report = {}

    for name, pipeline in models.items():
        log.info(f"Training {name} …")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cr  = classification_report(y_test, y_pred, output_dict=True)
        cm  = confusion_matrix(y_test, y_pred).tolist()

        report[name] = {
            "accuracy":         round(acc, 4),
            "roc_auc":          round(auc, 4),
            "precision_macro":  round(cr["macro avg"]["precision"], 4),
            "recall_macro":     round(cr["macro avg"]["recall"], 4),
            "f1_macro":         round(cr["macro avg"]["f1-score"], 4),
            "confusion_matrix": cm,
        }
        log.info(f"  {name}: ACC={acc:.3f}  AUC={auc:.3f}")

    # Persist
    with open(LR_PATH, "wb") as f:
        pickle.dump(models["logistic_regression"], f)
    with open(RF_PATH, "wb") as f:
        pickle.dump(models["random_forest"], f)
    with open(META_PATH, "wb") as f:
        pickle.dump({"encoders": encoders, "feature_names": feature_names}, f)

    log.info("Models saved.")
    return report


# ── Loading ────────────────────────────────────────────────────────────────────

def load_models() -> tuple[Pipeline, Pipeline, dict, list]:
    """Load trained models + metadata.  Raises FileNotFoundError if missing."""
    with open(LR_PATH, "rb")  as f: lr = pickle.load(f)
    with open(RF_PATH, "rb")  as f: rf = pickle.load(f)
    with open(META_PATH, "rb") as f: meta = pickle.load(f)
    return lr, rf, meta["encoders"], meta["feature_names"]


def models_exist() -> bool:
    return LR_PATH.exists() and RF_PATH.exists() and META_PATH.exists()


# ── Feature importance ────────────────────────────────────────────────────────

def get_feature_importance(feature_names: list[str]) -> pd.DataFrame:
    """Return feature importances from the RF model."""
    with open(RF_PATH, "rb") as f:
        rf_pipe: Pipeline = pickle.load(f)

    rf_clf = rf_pipe.named_steps["clf"]
    importances = rf_clf.feature_importances_

    fi = pd.DataFrame(
        {"feature": feature_names[:len(importances)], "importance": importances}
    ).sort_values("importance", ascending=False)
    return fi


# ── Tournament simulation ─────────────────────────────────────────────────────

def tournament_winner_probabilities(
    teams: list[str],
    df_processed: pd.DataFrame,
    encoders: dict,
    feature_names: list[str],
    n_simulations: int = 1000,
) -> dict[str, float]:
    """
    Monte Carlo simulation of tournament bracket.
    Each match winner is sampled proportionally to predicted win probability.
    Returns {team: probability_of_winning_tournament}.
    """
    from src.predict import build_feature_vector, predict_match_prob

    win_counts: dict[str, int] = {t: 0 for t in teams}

    for _ in range(n_simulations):
        remaining = list(teams)

        # Simulate until 1 team remains (round-robin elimination approach)
        while len(remaining) > 1:
            next_round = []
            # Shuffle and pair up
            np.random.shuffle(remaining)
            for i in range(0, len(remaining) - 1, 2):
                t1, t2 = remaining[i], remaining[i + 1]
                p1 = predict_match_prob(
                    t1, t2, "", "",
                    df_processed, encoders, feature_names
                )["team1_win_prob"]
                winner = t1 if np.random.random() < p1 else t2
                next_round.append(winner)
            # If odd number, last team gets a bye
            if len(remaining) % 2 == 1:
                next_round.append(remaining[-1])
            remaining = next_round

        win_counts[remaining[0]] += 1

    total = sum(win_counts.values())
    probs = {t: round(c / total, 4) for t, c in win_counts.items()}
    # Sort descending
    return dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
