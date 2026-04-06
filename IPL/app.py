"""
app.py — IPL Match & Winner Prediction System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run:  streamlit run app.py
"""

import sys
import logging
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_collection import load_matches, run_updater
from src.preprocessing   import (
    full_pipeline, season_win_rates,
    toss_impact, venue_summary, head_to_head,
    FEATURE_COLS,
)
from src.model   import train, models_exist, load_models, tournament_winner_probabilities
from src.predict import predict_match_prob, win_probability_over_overs

logging.basicConfig(level=logging.WARNING)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Prediction System",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Inter:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Rajdhani', sans-serif; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0f0c29, #302b63, #24243e);
    }
    [data-testid="stSidebar"] * { color: #e8e8f0 !important; }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: #1a1a2e;
        border: 1px solid #e94560;
        border-radius: 12px;
        padding: 12px 18px;
    }

    /* ── Prediction box ── */
    .pred-box {
        background: linear-gradient(135deg, #0f3460, #16213e);
        border-left: 5px solid #e94560;
        border-radius: 10px;
        padding: 20px 24px;
        margin: 10px 0;
    }
    .pred-box h2 { color: #e94560; font-family: 'Rajdhani', sans-serif; margin:0; }
    .pred-box p  { color: #c8d0e0; margin: 4px 0; }

    /* ── Divider ── */
    hr { border-color: #e94560; opacity: 0.3; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(90deg, #e94560, #c62a47);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover { opacity: 0.88; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading & processing dataset …")
def get_pipeline_data():
    df_raw = load_matches()
    df, encoders, X_train, X_test, y_train, y_test = full_pipeline(df_raw)
    return df, encoders, X_train, X_test, y_train, y_test


@st.cache_resource(show_spinner="Training models (first run) …")
def get_trained_models(_X_train, _y_train, _X_test, _y_test, _encoders):
    feature_names = [c for c in FEATURE_COLS if True]  # all cols
    if models_exist():
        lr, rf, enc, feat = load_models()
        return lr, rf, enc, feat, {}
    report = train(_X_train, _y_train, _X_test, _y_test, _encoders, feature_names)
    lr, rf, enc, feat = load_models()
    return lr, rf, enc, feat, report


df, encoders, X_train, X_test, y_train, y_test = get_pipeline_data()
lr, rf, enc, feat, model_report = get_trained_models(
    X_train, y_train, X_test, y_test, encoders
)
feature_names = [c for c in FEATURE_COLS if c in df.columns]

ALL_TEAMS = sorted(df["team1"].unique().tolist())
ALL_VENUES = sorted(df["venue"].unique().tolist())


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/8/8e/Indian_Premier_League_Official_Logo.svg/200px-Indian_Premier_League_Official_Logo.svg.png", width=120)
    st.title("🏏 IPL Predictor")
    st.markdown("---")

    nav = st.radio(
        "Navigate",
        [
            "🏠  Overview",
            "🔮  Match Predictor",
            "📊  Team Analytics",
            "🆚  Head-to-Head",
            "🏆  Tournament Simulator",
            "📈  Live Probability",
            "⚙️  Data Updater",
        ],
    )
    st.markdown("---")
    st.caption(f"Dataset: **{len(df):,}** matches · Seasons **{df['season'].min()}–{df['season'].max()}**")
    st.caption("Predictions based on historical patterns & model estimation.")


# ══════════════════════════════════════════════════════════════════════════════
# Matplotlib theme helper
# ══════════════════════════════════════════════════════════════════════════════
def dark_fig(figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.tick_params(colors="#8b949e")
    ax.xaxis.label.set_color("#8b949e")
    ax.yaxis.label.set_color("#8b949e")
    ax.title.set_color("#e6edf3")
    return fig, ax


PALETTE = [
    "#e94560", "#4fc3f7", "#a5d6a7", "#ffcc80",
    "#ce93d8", "#80deea", "#f48fb1", "#bcaaa4",
    "#ffe082", "#b0bec5",
]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════════════════════════
if nav == "🏠  Overview":
    st.title("IPL Match & Winner Prediction System")
    st.markdown(
        "> Powered by **Logistic Regression** + **Random Forest** · "
        "All predictions are based on historical patterns and model estimation."
    )
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Matches",   f"{len(df):,}")
    c2.metric("Seasons Covered", f"{df['season'].nunique()}")
    c3.metric("Teams",           f"{df['team1'].nunique()}")
    c4.metric("Venues",          f"{df['venue'].nunique()}")

    st.markdown("---")
    st.subheader("All-time Win Leaderboard")

    wins_df = (
        df["winner"].value_counts()
        .reset_index()
        .head(10)
    )
    wins_df.columns = ["team", "wins"]

    fig, ax = dark_fig((10, 4))
    bars = ax.barh(wins_df["team"], wins_df["wins"], color=PALETTE[:len(wins_df)])
    ax.set_xlabel("Total Wins")
    ax.set_title("All-time IPL Win Leaderboard", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    for bar in bars:
        ax.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            str(int(bar.get_width())), va="center", color="#e6edf3", fontsize=9
        )
    st.pyplot(fig)
    plt.close(fig)

    # Season trends
    st.subheader("Season-by-season Winner Dominance")
    sw = season_win_rates(df)
    top5 = wins_df["team"].head(5).tolist()
    sw_top = sw[sw["team"].isin(top5)]

    fig2, ax2 = dark_fig((11, 5))
    for i, team in enumerate(top5):
        td = sw_top[sw_top["team"] == team]
        ax2.plot(td["season"], td["win_pct"], marker="o", linewidth=2,
                 label=team, color=PALETTE[i])
    ax2.set_xlabel("Season")
    ax2.set_ylabel("Win %")
    ax2.set_title("Win % Over Seasons (Top 5 Teams)", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=8, facecolor="#0d1117", labelcolor="#e6edf3")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
    st.pyplot(fig2)
    plt.close(fig2)

    # Model accuracy
    if model_report:
        st.subheader("Model Performance")
        cols = st.columns(2)
        for i, (mname, stats) in enumerate(model_report.items()):
            with cols[i % 2]:
                st.markdown(f"**{mname.replace('_', ' ').title()}**")
                st.metric("Accuracy", f"{stats['accuracy']*100:.1f}%")
                st.metric("ROC-AUC",  f"{stats['roc_auc']:.3f}")
                st.metric("F1 (Macro)", f"{stats['f1_macro']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Match Predictor
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "🔮  Match Predictor":
    st.title("🔮 Match Predictor")
    st.caption("Select two teams and match conditions to get win probability estimates.")

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("🏏 Team 1", ALL_TEAMS, index=0)
    with col2:
        team2 = st.selectbox("🏏 Team 2", [t for t in ALL_TEAMS if t != team1], index=1)

    col3, col4, col5 = st.columns(3)
    with col3:
        venue = st.selectbox("🏟️ Venue", ALL_VENUES)
    with col4:
        toss_winner = st.selectbox("🪙 Toss Winner", [team1, team2])
    with col5:
        toss_decision = st.radio("Toss Decision", ["bat", "field"])

    model_choice = st.radio("Model", ["random_forest", "logistic_regression"], horizontal=True)

    if st.button("Predict Winner →"):
        result = predict_match_prob(
            team1, team2, venue, toss_winner, df,
            encoders, feature_names, toss_decision, model_choice
        )

        st.markdown("---")
        p1 = result["team1_win_prob"]
        p2 = result["team2_win_prob"]
        winner_label = team1 if p1 >= p2 else team2
        winner_prob  = max(p1, p2)

        st.markdown(
            f"""<div class="pred-box">
            <h2>Predicted Winner: {winner_label}</h2>
            <p>Win Probability: <strong style='color:#4fc3f7'>{winner_prob*100:.1f}%</strong></p>
            <p>Confidence: <strong>{result['confidence'].upper()}</strong></p>
            <p><em>{result['disclaimer']}</em></p>
            </div>""",
            unsafe_allow_html=True,
        )

        # Probability bar
        fig, ax = dark_fig((8, 1.8))
        bar_data = [p1 * 100, p2 * 100]
        colors   = ["#e94560", "#4fc3f7"]
        ax.barh([team1, team2], bar_data, color=colors, height=0.5)
        for i, v in enumerate(bar_data):
            ax.text(v + 0.5, i, f"{v:.1f}%", va="center", color="#e6edf3", fontsize=11)
        ax.set_xlim(0, 115)
        ax.set_xlabel("Win Probability (%)")
        ax.set_title("Win Probability Breakdown", fontsize=12, fontweight="bold")
        st.pyplot(fig)
        plt.close(fig)

        # H2H inset
        st.subheader("Head-to-Head Record")
        h2h = head_to_head(df, team1, team2)
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{team1} wins",  h2h.get(f"{team1}_wins", 0))
        c2.metric(f"{team2} wins",  h2h.get(f"{team2}_wins", 0))
        c3.metric("Total Matches",  h2h["total_matches"])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Team Analytics
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "📊  Team Analytics":
    st.title("📊 Team Analytics")

    team = st.selectbox("Select Team", ALL_TEAMS)
    team_df = df[(df["team1"] == team) | (df["team2"] == team)]

    col1, col2, col3 = st.columns(3)
    total   = len(team_df)
    wins    = (team_df["winner"] == team).sum()
    win_pct = wins / total * 100 if total > 0 else 0

    col1.metric("Matches Played", total)
    col2.metric("Wins",           wins)
    col3.metric("Win Rate",       f"{win_pct:.1f}%")

    st.markdown("---")

    # Season-by-season win %
    sw = season_win_rates(df)
    td = sw[sw["team"] == team]

    fig, ax = dark_fig((10, 4))
    ax.fill_between(td["season"], td["win_pct"], alpha=0.2, color="#e94560")
    ax.plot(td["season"], td["win_pct"], marker="o", color="#e94560", linewidth=2)
    ax.set_xlabel("Season")
    ax.set_ylabel("Win %")
    ax.set_title(f"{team} — Win % Per Season", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    st.pyplot(fig)
    plt.close(fig)

    # Toss impact for this team
    st.subheader("Toss Impact")
    toss_df = team_df.copy()
    toss_df["toss_won_by_team"] = toss_df["toss_winner"] == team
    toss_df["match_won"]        = toss_df["winner"] == team

    grp = toss_df.groupby("toss_won_by_team")["match_won"].agg(["sum", "count"])
    grp.columns = ["Wins", "Played"]
    grp["Win %"] = (grp["Wins"] / grp["Played"] * 100).round(1)
    grp.index = grp.index.map({True: "Won Toss", False: "Lost Toss"})
    st.dataframe(grp.style.format({"Win %": "{:.1f}%"}), use_container_width=True)

    # Venue performance
    st.subheader("Venue Win Rates")
    v_rows = []
    for venue in team_df["venue"].unique():
        v_df  = team_df[team_df["venue"] == venue]
        v_win = (v_df["winner"] == team).sum()
        v_rows.append({"venue": venue, "played": len(v_df), "wins": int(v_win),
                       "win_pct": round(v_win / len(v_df) * 100, 1)})
    venue_perf = pd.DataFrame(v_rows).sort_values("played", ascending=False).head(8)

    fig2, ax2 = dark_fig((10, 4))
    colors = ["#e94560" if x >= 50 else "#4fc3f7" for x in venue_perf["win_pct"]]
    ax2.bar(range(len(venue_perf)), venue_perf["win_pct"], color=colors)
    ax2.set_xticks(range(len(venue_perf)))
    ax2.set_xticklabels(
        [v[:20] + "…" if len(v) > 20 else v for v in venue_perf["venue"]],
        rotation=30, ha="right", fontsize=8
    )
    ax2.axhline(50, color="#ffffff", linestyle="--", alpha=0.4, linewidth=1)
    ax2.set_ylabel("Win %")
    ax2.set_title(f"{team} — Win % by Venue (top 8)", fontsize=12, fontweight="bold")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
    st.pyplot(fig2)
    plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Head-to-Head
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "🆚  Head-to-Head":
    st.title("🆚 Head-to-Head Comparison")

    col1, col2 = st.columns(2)
    with col1: team_a = st.selectbox("Team A", ALL_TEAMS, index=0)
    with col2: team_b = st.selectbox("Team B", [t for t in ALL_TEAMS if t != team_a], index=1)

    h2h = head_to_head(df, team_a, team_b)
    ta_wins = h2h.get(f"{team_a}_wins", 0)
    tb_wins = h2h.get(f"{team_b}_wins", 0)
    total   = h2h["total_matches"]

    c1, c2, c3 = st.columns(3)
    c1.metric(f"🏆 {team_a}", ta_wins)
    c2.metric("Total Matches", total)
    c3.metric(f"🏆 {team_b}", tb_wins)

    if total > 0:
        # Pie
        fig, ax = dark_fig((5, 5))
        ax.pie(
            [ta_wins, tb_wins],
            labels=[team_a, team_b],
            colors=["#e94560", "#4fc3f7"],
            autopct="%1.1f%%",
            startangle=90,
            textprops={"color": "#e6edf3"},
        )
        ax.set_title(f"H2H Win Share", fontsize=13, fontweight="bold")
        st.pyplot(fig)
        plt.close(fig)

        # Season-by-season H2H
        st.subheader("Season-by-season Results")
        mask = (
            ((df["team1"] == team_a) & (df["team2"] == team_b)) |
            ((df["team1"] == team_b) & (df["team2"] == team_a))
        )
        h2h_df = df[mask].copy()
        h2h_df["winner_short"] = h2h_df["winner"].apply(
            lambda x: "A" if x == team_a else "B"
        )
        st.dataframe(
            h2h_df[["season", "venue", "toss_winner", "winner"]].sort_values("season"),
            use_container_width=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Tournament Simulator
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "🏆  Tournament Simulator":
    st.title("🏆 Tournament Winner Simulator")
    st.caption("Monte Carlo bracket simulation using historical win patterns (1000 iterations).")

    default_teams = ALL_TEAMS[:8]
    selected_teams = st.multiselect(
        "Select participating teams (8 or 10 recommended)",
        ALL_TEAMS, default=default_teams
    )

    if st.button("Run Simulation →"):
        if len(selected_teams) < 2:
            st.warning("Select at least 2 teams.")
        else:
            with st.spinner("Running 1000 simulations …"):
                probs = tournament_winner_probabilities(
                    selected_teams, df, encoders, feature_names,
                    n_simulations=1000
                )

            st.markdown("---")
            st.subheader("Tournament Win Probability")

            labels  = list(probs.keys())
            values  = [v * 100 for v in probs.values()]
            cmap    = PALETTE[:len(labels)]

            fig, ax = dark_fig((10, max(4, len(labels) * 0.6)))
            bars = ax.barh(labels, values, color=cmap)
            ax.set_xlabel("Win Probability (%)")
            ax.set_title("Tournament Winner Probability (Monte Carlo)", fontsize=13, fontweight="bold")
            ax.invert_yaxis()
            for bar in bars:
                ax.text(
                    bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.1f}%", va="center",
                    color="#e6edf3", fontsize=9
                )
            ax.xaxis.set_major_formatter(mticker.PercentFormatter())
            st.pyplot(fig)
            plt.close(fig)

            st.caption("Based on historical patterns and model estimation. Not a guarantee of outcome.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Live Probability
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "📈  Live Probability":
    st.title("📈 Live Win Probability Tracker")
    st.caption("Simulate win probability during a T20 chase based on current match state.")

    col1, col2 = st.columns(2)
    with col1: batting_team = st.selectbox("Batting Team (Chasing)", ALL_TEAMS, index=0)
    with col2: bowling_team  = st.selectbox("Bowling Team (Defending)", [t for t in ALL_TEAMS if t != batting_team], index=1)

    col3, col4, col5, col6 = st.columns(4)
    with col3: target       = st.number_input("Target", min_value=50, max_value=300, value=175)
    with col4: current_score= st.number_input("Current Score", min_value=0, max_value=target - 1, value=80)
    with col5: wickets_lost = st.number_input("Wickets Lost", min_value=0, max_value=9, value=2)
    with col6: overs_done   = st.number_input("Overs Completed", min_value=0.0, max_value=19.5, value=9.0, step=0.1)

    if st.button("Show Probability Curve →"):
        data = win_probability_over_overs(
            batting_team, bowling_team, target,
            current_score, wickets_lost, overs_done
        )
        overs_list = [d["over"] for d in data]
        probs_list = [d["win_prob"] * 100 for d in data]

        fig, ax = dark_fig((11, 5))
        ax.fill_between(overs_list, probs_list, alpha=0.15, color="#4fc3f7")
        ax.plot(overs_list, probs_list, color="#4fc3f7", linewidth=2.5)
        ax.axhline(50, color="#ffffff", linestyle="--", alpha=0.3, linewidth=1, label="50% line")
        ax.axvline(overs_done, color="#e94560", linestyle=":", alpha=0.7, linewidth=1.5,
                   label=f"Current: Over {overs_done}")
        ax.set_xlabel("Over")
        ax.set_ylabel("Win Probability (%)")
        ax.set_title(
            f"{batting_team} vs {bowling_team} — Chasing {target} | {current_score}/{wickets_lost} after {overs_done} overs",
            fontsize=12, fontweight="bold"
        )
        ax.legend(facecolor="#0d1117", labelcolor="#e6edf3", fontsize=9)
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        st.pyplot(fig)
        plt.close(fig)

        # Snapshot metrics
        curr_prob = data[0]["win_prob"] if data else 0.5
        runs_req  = target - current_score
        balls_left = (20 - overs_done) * 6
        rrr = (runs_req / ((20 - overs_done))) if overs_done < 20 else 99

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Win Prob", f"{curr_prob*100:.1f}%")
        c2.metric("Runs Required",    runs_req)
        c3.metric("Balls Remaining",  int(balls_left))
        c4.metric("Required RR",      f"{rrr:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Data Updater
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "⚙️  Data Updater":
    st.title("⚙️ Live Data Updater")
    st.markdown(
        "Fetches recent IPL match results from ESPN Cricinfo and merges "
        "new matches into the historical dataset automatically."
    )

    if st.button("🔄  Fetch & Update Now"):
        with st.spinner("Connecting to ESPN Cricinfo …"):
            status = run_updater()

        st.success("Update complete!")
        st.json(status)

        if status.get("dataset_updated"):
            st.info("New matches added — clear cache and reload for updated predictions.")
            st.cache_data.clear()

    st.markdown("---")
    st.subheader("Current Dataset Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Matches",    len(df))
    c2.metric("Seasons",          df["season"].nunique())
    c3.metric("Latest Season",    int(df["season"].max()))

    st.subheader("Recent Matches")
    recent_cols = ["season", "team1", "team2", "venue", "winner"]
    available_cols = [c for c in recent_cols if c in df.columns]
    st.dataframe(
        df[available_cols].tail(20).sort_values("season", ascending=False),
        use_container_width=True
    )
