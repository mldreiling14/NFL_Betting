import pandas as pd
import sqlite3
import nflreadpy as nfl

from features import (
    add_recent_form_features,
    add_qb_features,
    add_rb_features,
    add_wrte_features,
    add_injury_features,
    add_defense_features,
    add_coach_features
)


def build_all_features(seasons=range(2015, 2024), db_path="data/nfl.db"):
    # Load games
    conn = sqlite3.connect(db_path)
    df_full = pd.read_sql_query("SELECT * FROM games", conn)
    conn.close()

    # Pull player-level data
    player_stats = nfl.load_player_stats(seasons=list(seasons)).to_pandas()
    snaps = nfl.load_snap_counts(seasons=list(seasons)).to_pandas()
    snaps = snaps[['pfr_player_id', 'game_id', 'offense_pct']]

    players = nfl.load_players().to_pandas()
    crosswalk = players[['gsis_id', 'pfr_id']].dropna().rename(
        columns={'gsis_id': 'player_id', 'pfr_id': 'pfr_player_id'})

    # QB stats
    qb_stats = player_stats[(player_stats['position'] == 'QB') & (player_stats['attempts'] > 0)].copy()
    qb_stats = qb_stats[['player_id', 'player_name', 'game_id', 'season', 'week', 'team',
                          'attempts', 'passing_yards', 'passing_tds', 'passing_interceptions', 'passing_epa']]

    # RB stats
    rb_stats = player_stats[player_stats['position'] == 'RB'].copy()
    rb_stats = rb_stats[['player_id', 'player_name', 'game_id', 'season', 'week', 'team',
                          'carries', 'rushing_yards', 'rushing_tds', 'rushing_epa',
                          'receptions', 'receiving_yards', 'receiving_tds']]
    rb_stats = rb_stats.merge(crosswalk, on='player_id', how='left')
    rb_stats = rb_stats.merge(snaps, on=['pfr_player_id', 'game_id'], how='left')

    # WR/TE stats
    wrte_stats = player_stats[player_stats['position'].isin(['WR', 'TE'])].copy()
    wrte_stats = wrte_stats[['player_id', 'player_name', 'game_id', 'season', 'week', 'team',
                              'targets', 'receptions', 'receiving_yards', 'receiving_tds', 'receiving_epa']]
    wrte_stats = wrte_stats.merge(crosswalk, on='player_id', how='left')
    wrte_stats = wrte_stats.merge(snaps, on=['pfr_player_id', 'game_id'], how='left')

    # Injury reports
    injuries = nfl.load_injuries(seasons=list(seasons)).to_pandas()
    injuries = injuries[['season', 'week', 'team', 'gsis_id', 'position', 'report_status']]

    # Apply all feature functions
    df_full = add_recent_form_features(df_full)
    df_full = add_qb_features(df_full, qb_stats)
    df_full = add_rb_features(df_full, rb_stats)
    df_full = add_wrte_features(df_full, wrte_stats)
    df_full = add_injury_features(df_full, injuries, rb_stats, wrte_stats)
    df_full = add_defense_features(df_full, player_stats)
    df_full = add_coach_features(df_full)

    # Save the finished, feature-complete table
    conn = sqlite3.connect(db_path)
    df_full.to_sql("games_with_features", conn, if_exists="replace", index=False)
    conn.close()

    print(f"Saved {len(df_full)} games with {len(df_full.columns)} columns to games_with_features")
    return df_full


if __name__ == "__main__":
    build_all_features()