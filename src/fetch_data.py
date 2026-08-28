import nflreadpy as nfl
import pandas as pd
import sqlite3

def fetch_and_save_schedules(seasons, db_path="data/nfl.db"):
    df = nfl.load_schedules(seasons=seasons)
    df = df.to_pandas()

    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)

    conn = sqlite3.connect(db_path)
    df.to_sql("games", conn, if_exists="replace", index=False)
    conn.close()

    print(f"Saved {len(df)} games to {db_path}")

if __name__ == "__main__":
    fetch_and_save_schedules(seasons=list(range(2015, 2026)))