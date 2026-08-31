# NFL Win Probability Model

A machine learning pipeline that predicts NFL game outcomes as a win probability, trained on real historical data (2015–2025) and capable of generating live predictions for upcoming games. Includes a Streamlit web app for browsing predictions with a Vegas odds comparison.

**Model accuracy:** ~68% on held-out 2024–2025 test data, versus ~69% for Vegas closing lines on the same games. See [`FEATURES.md`](./FEATURES.md) for the full technical breakdown of every feature, what was tried and rejected, and known limitations. See [`HOW_IT_WORKS.md`](./HOW_IT_WORKS.md) for a plain-English, non-technical explanation.

---

## What's in this repo

```
sports_betting/
├── data/               # SQLite database (created by fetch_data.py)
├── models/             # Saved trained model (created by train_model.py)
├── notebooks/          # Exploration and testing notebooks
├── src/
│   ├── fetch_data.py       # Pulls raw NFL schedules into SQLite
│   ├── features.py         # All feature-engineering functions
│   ├── build_features.py   # Runs the full feature pipeline, saves games_with_features
│   ├── train_model.py      # Trains and saves the production model
│   └── predict.py          # Generates live predictions for upcoming games
├── app.py              # Streamlit web app
├── requirements.txt
├── FEATURES.md          # Full technical feature documentation
└── HOW_IT_WORKS.md       # Plain-English project summary
```

---

## Setup

**1. Clone the repo and create a virtual environment**

```bash
git clone <your-repo-url>
cd sports_betting
python -m venv venv
```

Activate it:
- Windows (Git Bash): `source venv/Scripts/activate`
- Mac/Linux: `source venv/bin/activate`

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Running the pipeline

Run these in order. Each step builds on the one before it.

**1. Pull raw schedule data**

```bash
python src/fetch_data.py
```

Creates `data/nfl.db` with a `games` table (2015–2025 seasons). Takes a minute or two.

**2. Build all engineered features**

```bash
python src/build_features.py
```

Pulls player stats, team stats, injuries, coaching data, and more from [`nflreadpy`](https://github.com/nflverse/nflreadpy), then runs every feature function in `features.py`. Saves a `games_with_features` table back into `data/nfl.db`. This is the slowest step — expect several minutes, since it pulls multiple seasons of detailed player-level data.

**3. Train the model**

```bash
python src/train_model.py
```

Trains a logistic regression model on all available games and saves it to `models/win_probability_model.joblib`, bundled together with the imputer used to fill missing values and the exact list of features the model expects.

**4. Generate live predictions (optional, for testing)**

```bash
python src/predict.py
```

Builds "current state" snapshots for every team (recent form, Elo, current starters via depth charts, etc.) and prints win probabilities for Week 1 of the upcoming season. This is what powers the app — running it directly is mainly useful for testing or debugging.

**5. Run the app**

```bash
streamlit run app.py
```

Opens a browser tab showing win probabilities for each upcoming week, with team colors, a Vegas odds comparison, and a "favorite" call-out for each game.

---

## Updating for a new season or week

Once the season progresses, refresh the data and retrain periodically:

```bash
python src/fetch_data.py
python src/build_features.py
python src/train_model.py
```

The Streamlit app automatically re-generates live snapshots every 6 hours (cached), so it will pick up new game results and updated rosters without needing a manual restart — but rerunning the three commands above ensures the underlying model itself is retrained on the latest completed games, not just refreshed snapshots.

---

## A few honest notes

- **This is not a betting system.** The model has not been shown to beat Vegas — when they disagree, Vegas is right more often than not. Treat predictions as a well-reasoned estimate, not a guarantee.
- **Live predictions use each team's most recent identified starters** (via official depth charts, not just "whoever played last"), but real game-week injury news isn't incorporated until much closer to kickoff, since that data simply doesn't exist further in advance.
- **Weather is tracked but not used in the model** — it showed a real, sensible effect in specific cases (e.g., warm-climate teams struggling in extreme cold) but wasn't statistically powerful enough to improve overall accuracy.

See `FEATURES.md` for the complete list of what's included, what was tested and rejected, and why.
