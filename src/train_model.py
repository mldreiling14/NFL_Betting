import pandas as pd
import sqlite3
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

FEATURE_COLS = [
    'home_recent_form', 'away_recent_form',
    'home_recent_point_diff', 'away_recent_point_diff',
    'home_qb_recent_yards', 'away_qb_recent_yards',
    'home_qb_recent_tds', 'away_qb_recent_tds',
    'home_qb_recent_ints', 'away_qb_recent_ints',
    'home_qb_recent_epa', 'away_qb_recent_epa',
    'home_rb_recent_rush_yards', 'away_rb_recent_rush_yards',
    'home_rb_recent_rush_epa', 'away_rb_recent_rush_epa',
    'home_rb_recent_rec_yards', 'away_rb_recent_rec_yards',
    'home_wrte_recent_rec_yards', 'away_wrte_recent_rec_yards',
    'home_wrte_recent_rec_epa', 'away_wrte_recent_rec_epa',
    'home_wrte_recent_targets', 'away_wrte_recent_targets',
    'home_qb_injury_flag', 'away_qb_injury_flag',
    'home_rb_injury_flag', 'away_rb_injury_flag',
    'home_wrte_injury_flag', 'away_wrte_injury_flag',
    'home_epa_allowed_recent', 'away_epa_allowed_recent',
    'home_yards_allowed_recent', 'away_yards_allowed_recent',
    'home_takeaways_recent', 'away_takeaways_recent',
    'home_coach_h2h_wins', 'h2h_games_played',
    'home_elo_pre', 'away_elo_pre',
    'rest_advantage',
    'home_sack_rate_recent', 'away_sack_rate_recent',
    'home_pressure_pct_recent', 'away_pressure_pct_recent',
    'home_wr_height_advantage', 'away_wr_height_advantage',
    'home_wr_weight_advantage', 'away_wr_weight_advantage',
    'home_opp_cb_completion_allowed', 'away_opp_cb_completion_allowed',
    'home_opp_cb_rating_allowed', 'away_opp_cb_rating_allowed',
    'div_game'
]


def train_and_save_model(db_path="data/nfl.db", model_path="models/win_probability_model.joblib"):
    conn = sqlite3.connect(db_path)
    df_full = pd.read_sql_query("SELECT * FROM games_with_features", conn)
    conn.close()

    df_full['home_win'] = (df_full['home_score'] > df_full['away_score']).astype(int)

    X = df_full[FEATURE_COLS].copy()
    y = df_full['home_win']

    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=FEATURE_COLS, index=X.index)

    # Train on ALL available data - for a live prediction model, we want
    # every real game we have, not a held-out test set (testing already
    # happened during development; this is the final production model)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_imputed, y)

    # Save the model AND the imputer AND the feature list together -
    # all three are needed to make a real prediction later, and must
    # stay in sync with each other
    joblib.dump({
        'model': model,
        'imputer': imputer,
        'feature_cols': FEATURE_COLS
    }, model_path)

    print(f"Trained on {len(X_imputed)} games, {len(FEATURE_COLS)} features")
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_and_save_model()