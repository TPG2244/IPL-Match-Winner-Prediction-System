# 🏏 IPL Match & Winner Prediction System

> Predict IPL match outcomes using historical patterns + Machine Learning.  
> Built with Python · Scikit-learn · Streamlit · Matplotlib.

---

## 📐 Architecture

```
IPL-Prediction/
├── data/
│   ├── raw/               ← matches.csv / deliveries.csv (Kaggle)
│   ├── models/            ← trained .pkl files (auto-generated)
│   └── cache/             ← live scrape cache
│
├── src/
│   ├── data_collection.py ← Kaggle loader + ESPN Cricinfo scraper + auto-updater
│   ├── preprocessing.py   ← Cleaning, feature engineering, train/test split
│   ├── model.py           ← Logistic Regression + Random Forest training & eval
│   └── predict.py         ← Inference: match prob, live win curve, bootstrap CI
│
├── app.py                 ← Streamlit UI (6 pages)
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & install
```bash
git clone https://github.com/yourname/IPL-Prediction.git
cd IPL-Prediction
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optional) Add real Kaggle data
Download from [Kaggle IPL Dataset](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020):
```
data/raw/matches.csv
data/raw/deliveries.csv   # optional – ball-by-ball
```
Without the CSV the app generates a synthetic seed dataset automatically.

### 3. Launch the app
```bash
streamlit run app.py
```

---

## 🧩 Module Overview

### `data_collection.py`
| Function | Purpose |
|---|---|
| `load_matches()` | Loads `matches.csv` or generates synthetic seed |
| `fetch_live_matches()` | Scrapes recent IPL results from ESPN Cricinfo |
| `append_live_to_dataset()` | Merges genuinely new matches into the CSV |
| `run_updater()` | One-call pipeline: fetch → diff → append |

### `preprocessing.py`
| Function | Purpose |
|---|---|
| `clean()` | Nulls, team-name normalisation (renames), type coercion |
| `engineer_features()` | Win rate, H2H ratio, venue strength, recent form |
| `encode_labels()` | LabelEncoder for categorical columns |
| `full_pipeline()` | End-to-end: raw → X_train / X_test / y_train / y_test |
| `season_win_rates()` | Analytics helper for trend graphs |
| `head_to_head()` | H2H stats between any two teams |

### `model.py`
| Function | Purpose |
|---|---|
| `train()` | Fits LR + RF, saves `.pkl`, returns evaluation report |
| `load_models()` | Loads persisted pipelines + metadata |
| `get_feature_importance()` | RF feature importance DataFrame |
| `tournament_winner_probabilities()` | Monte Carlo bracket simulation |

### `predict.py`
| Function | Purpose |
|---|---|
| `predict_match_prob()` | Win probability for a single match |
| `win_probability_over_overs()` | Live in-game probability curve |
| `prediction_with_ci()` | Bootstrap 95% confidence interval |

---

## 🖥️ Streamlit Pages

| Page | What it shows |
|---|---|
| 🏠 Overview | Win leaderboard, season trends, model accuracy |
| 🔮 Match Predictor | Pick teams → win % with confidence rating |
| 📊 Team Analytics | Season trend, toss impact, venue win rates |
| 🆚 Head-to-Head | H2H pie chart + season-by-season results |
| 🏆 Tournament Simulator | Monte Carlo bracket with 1000 iterations |
| 📈 Live Probability | Real-time chase win curve |
| ⚙️ Data Updater | One-click ESPN Cricinfo scrape + auto-merge |

---

## 📊 Feature Engineering

| Feature | Description |
|---|---|
| `team1_win_rate` | Historical win % across all seasons |
| `team2_win_rate` | Same for opponent |
| `h2h_ratio` | Head-to-head win ratio between these two teams |
| `team1_venue_rate` | Win % at selected venue |
| `team2_venue_rate` | Same for opponent |
| `team1_recent_form` | Win rate over last 10 matches |
| `team2_recent_form` | Same for opponent |
| `bat_first` | 1 if toss winner bats, 0 if fields |
| `team1_enc` | Label-encoded team ID |
| `venue_enc` | Label-encoded venue ID |


---

## Image
<img width="1919" height="740" alt="Screenshot 2026-04-06 140440" src="https://github.com/user-attachments/assets/ec9fcc60-217f-4928-887c-18314ed5c92c" />
<img width="1508" height="597" alt="Screenshot 2026-04-06 140455" src="https://github.com/user-attachments/assets/8a8a28f3-16fa-483d-950d-d436db0e223d" />
<img width="1488" height="752" alt="Screenshot 2026-04-06 140509" src="https://github.com/user-attachments/assets/531ffd7a-8fc3-4c4b-9ef8-a095a613a074" />

---

## ⚠️ Disclaimer
All predictions are **based on historical patterns and model estimation**.  
Cricket is inherently unpredictable — these are statistical estimates, not guarantees.

---

## 📄 License
MIT
