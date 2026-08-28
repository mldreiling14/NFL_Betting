import pandas as pd
import sqlite3
import nflreadpy as nfl

from features import (
    add_recent_form_features,
    add_qb_features,
    add_rb_features,
    add_wrte_features,
    add_injury_features,
    add_defense_allowed_features,
    add_coach_features,
    add_elo_features,
    add_rest_advantage,
    add_oline_features,
    add_db_wr_matchup_features,
    add_weather_features
)


def build_all_features(seasons=range(2015, 2026), db_path="data/nfl.db"):
    # Load games
    conn = sqlite3.connect(db_path)
    df_full = pd.read_sql_query("SELECT * FROM games", conn)
    conn.close()

    # Pull player-level data
    player_stats = nfl.load_player_stats(seasons=list(seasons)).to_pandas()
    snaps = nfl.load_snap_counts(seasons=list(seasons)).to_pandas()
    snaps_full = snaps.copy()  # keep the full version with position/defense_pct for DB/WR matchup
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

    # Team-level stats (for defense-allowed features)
    team_stats = nfl.load_team_stats(seasons=list(seasons)).to_pandas()
    oline_team_stats = nfl.load_team_stats(seasons=list(seasons)).to_pandas()
    oline_team_stats = oline_team_stats[['game_id', 'season', 'week', 'team', 'attempts', 'sacks_suffered']]

    pfr_seasons = [s for s in seasons if s >= 2018]  # PFR advanced stats only available 2018+
    pressure_stats = nfl.load_pfr_advstats(seasons=pfr_seasons, stat_type='pass').to_pandas()
    pressure_stats = pressure_stats[['game_id', 'season', 'week', 'team', 'times_pressured_pct']]

    pfr_seasons_def = [s for s in seasons if s >= 2018]
    adv_def_full = nfl.load_pfr_advstats(seasons=pfr_seasons_def, stat_type='def').to_pandas()
    adv_def_full = adv_def_full[['game_id', 'season', 'week', 'team', 'pfr_player_id',
                                  'def_completion_pct', 'def_passer_rating_allowed']]

    physical = players[['pfr_id', 'height', 'weight']].dropna().rename(columns={'pfr_id': 'pfr_player_id'})

    # Apply all feature functions
    df_full = add_recent_form_features(df_full)
    df_full = add_qb_features(df_full, qb_stats)
    df_full = add_rb_features(df_full, rb_stats)
    df_full = add_wrte_features(df_full, wrte_stats)
    df_full = add_injury_features(df_full, injuries, rb_stats, wrte_stats)
    df_full = add_defense_allowed_features(df_full, team_stats)
    df_full = add_coach_features(df_full)
    df_full = add_elo_features(df_full)
    df_full = add_rest_advantage(df_full)
    df_full = add_oline_features(df_full, oline_team_stats, pressure_stats)
    df_full = add_db_wr_matchup_features(df_full, player_stats, snaps_full, crosswalk, physical, adv_def_full)
    df_full = add_weather_features(df_full)
    # Save the finished, feature-complete table
    conn = sqlite3.connect(db_path)
    df_full.to_sql("games_with_features", conn, if_exists="replace", index=False)
    conn.close()

    print(f"Saved {len(df_full)} games with {len(df_full.columns)} columns to games_with_features")
    return df_full


if __name__ == "__main__":
    build_all_features()