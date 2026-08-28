import pandas as pd
import sqlite3
import joblib
import nflreadpy as nfl


def load_model(model_path="models/win_probability_model.joblib"):
    return joblib.load(model_path)


def build_snapshots(db_path="data/nfl.db", seasons=range(2015, 2026)):
    """
    Builds every 'current state' table needed for live predictions:
    each team's current form, Elo, QB/RB/WR-TE stats, coach, defense-
    allowed, O-line, and primary WR/CB for DB matchup purposes.

    Unlike the training features (which use shift(1) to exclude each
    row's own game), these use every real completed game INCLUDING
    the most recent one, since that IS each team's current state
    entering their next (not-yet-played) game.

    Returns a dict of DataFrames, one per snapshot type.
    """
    conn = sqlite3.connect(db_path)
    df_full = pd.read_sql_query("SELECT * FROM games_with_features", conn)
    conn.close()
    df_full['home_win'] = (df_full['home_score'] > df_full['away_score']).astype(int)

    game_dates = df_full[['game_id', 'gameday']].drop_duplicates()

    player_stats = nfl.load_player_stats(seasons=list(seasons)).to_pandas()
    snaps_full = nfl.load_snap_counts(seasons=list(seasons)).to_pandas()
    players = nfl.load_players().to_pandas()
    crosswalk = players[['gsis_id', 'pfr_id']].dropna().rename(
        columns={'gsis_id': 'player_id', 'pfr_id': 'pfr_player_id'})
    physical = players[['pfr_id', 'height', 'weight']].dropna().rename(columns={'pfr_id': 'pfr_player_id'})

    pfr_seasons = [s for s in seasons if s >= 2018]
    adv_def_full = nfl.load_pfr_advstats(seasons=pfr_seasons, stat_type='def').to_pandas()
    adv_def_full = adv_def_full[['game_id', 'season', 'week', 'team', 'pfr_player_id',
                                  'def_completion_pct', 'def_passer_rating_allowed']]

    # --- Team form + point differential ---
    home = df_full[['game_id', 'gameday', 'home_team_std', 'home_win', 'home_score', 'away_score']].rename(
        columns={'home_team_std': 'team'})
    home['win'] = home['home_win']
    home['point_diff'] = home['home_score'] - home['away_score']
    away = df_full[['game_id', 'gameday', 'away_team_std', 'home_win', 'home_score', 'away_score']].rename(
        columns={'away_team_std': 'team'})
    away['win'] = 1 - away['home_win']
    away['point_diff'] = away['away_score'] - away['home_score']
    team_games = pd.concat([home, away], ignore_index=True).sort_values(['team', 'gameday']).reset_index(drop=True)

    current_form = (
        team_games.groupby('team').apply(lambda g: g.tail(5)[['win', 'point_diff']].mean())
        .reset_index().rename(columns={'win': 'current_recent_form', 'point_diff': 'current_recent_point_diff'})
    )

    # --- Elo (full recompute, then apply season-boundary revert for the upcoming season) ---
    elo_games = df_full[['game_id', 'season', 'gameday', 'home_team_std', 'away_team_std', 'home_win']].copy()
    elo_games = elo_games.sort_values(['season', 'gameday']).reset_index(drop=True)
    all_teams = set(elo_games['home_team_std']).union(set(elo_games['away_team_std']))
    elo = {team: 1500 for team in all_teams}
    k_factor, home_advantage, revert_fraction = 20, 65, 1/3
    current_season = None
    for _, row in elo_games.iterrows():
        if current_season is not None and row['season'] != current_season:
            for team in elo:
                elo[team] = elo[team] * (1 - revert_fraction) + 1500 * revert_fraction
        current_season = row['season']
        h, a = row['home_team_std'], row['away_team_std']
        elo_diff = (elo[h] + home_advantage) - elo[a]
        expected_home = 1 / (1 + 10 ** (-elo_diff / 400))
        actual_home = row['home_win']
        elo[h] += k_factor * (actual_home - expected_home)
        elo[a] += k_factor * ((1 - actual_home) - (1 - expected_home))
    # Apply one more revert for the upcoming season boundary
    for team in elo:
        elo[team] = elo[team] * (1 - revert_fraction) + 1500 * revert_fraction
    current_elo = pd.DataFrame(list(elo.items()), columns=['team', 'current_elo'])

    # --- QB ---
    qb_stats = player_stats[(player_stats['position'] == 'QB') & (player_stats['attempts'] > 0)].copy()
    qb_stats = qb_stats[['player_id', 'game_id', 'team', 'passing_yards', 'passing_tds',
                         'passing_interceptions', 'passing_epa']]
    qb_stats = qb_stats.merge(game_dates, on='game_id', how='left').sort_values(['team', 'gameday']).reset_index(drop=True)
    for col in ['passing_yards', 'passing_tds', 'passing_interceptions', 'passing_epa']:
        qb_stats[f'recent_{col}'] = qb_stats.groupby('player_id')[col].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean())
    current_qb = qb_stats.loc[qb_stats.groupby('team')['gameday'].idxmax()][
        ['team', 'recent_passing_yards', 'recent_passing_tds', 'recent_passing_interceptions', 'recent_passing_epa']]

    # --- RB (aggregate to team-game BEFORE rolling - see bug history) ---
    rb_stats = player_stats[player_stats['position'] == 'RB'].copy()
    rb_stats = rb_stats[['player_id', 'game_id', 'team', 'rushing_yards', 'rushing_epa', 'receiving_yards']]
    rb_stats = rb_stats.merge(crosswalk, on='player_id', how='left')
    rb_stats = rb_stats.merge(
        snaps_full[snaps_full['position'] == 'RB'][['pfr_player_id', 'game_id', 'offense_pct']],
        on=['pfr_player_id', 'game_id'], how='left')
    rb_stats = rb_stats.merge(game_dates, on='game_id', how='left').sort_values(['player_id', 'gameday']).reset_index(drop=True)
    rb_stats['offense_pct'] = rb_stats.groupby('player_id')['offense_pct'].ffill()
    rb_stats['w_rush_yards'] = rb_stats['rushing_yards'] * rb_stats['offense_pct']
    rb_stats['w_rush_epa'] = rb_stats['rushing_epa'] * rb_stats['offense_pct']
    rb_stats['w_rec_yards'] = rb_stats['receiving_yards'] * rb_stats['offense_pct']

    rb_team_game = rb_stats.groupby(['team', 'game_id', 'gameday'], as_index=False).agg(
        t_rush_yards=('w_rush_yards', 'sum'), t_rush_epa=('w_rush_epa', 'sum'), t_rec_yards=('w_rec_yards', 'sum'))
    rb_team_game = rb_team_game.sort_values(['team', 'gameday']).reset_index(drop=True)
    for col in ['t_rush_yards', 't_rush_epa', 't_rec_yards']:
        rb_team_game[f'recent_{col}'] = rb_team_game.groupby('team')[col].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean())
    current_rb = rb_team_game.loc[rb_team_game.groupby('team')['gameday'].idxmax()][
        ['team', 'recent_t_rush_yards', 'recent_t_rush_epa', 'recent_t_rec_yards']]

    # --- WR/TE (aggregate to team-game BEFORE rolling) ---
    wrte_stats = player_stats[player_stats['position'].isin(['WR', 'TE'])].copy()
    wrte_stats = wrte_stats[['player_id', 'game_id', 'team', 'targets', 'receiving_yards', 'receiving_epa']]
    wrte_stats = wrte_stats.merge(crosswalk, on='player_id', how='left')
    wrte_stats = wrte_stats.merge(
        snaps_full[snaps_full['position'].isin(['WR', 'TE'])][['pfr_player_id', 'game_id', 'offense_pct']],
        on=['pfr_player_id', 'game_id'], how='left')
    wrte_stats = wrte_stats.merge(game_dates, on='game_id', how='left').sort_values(['player_id', 'gameday']).reset_index(drop=True)
    wrte_stats['offense_pct'] = wrte_stats.groupby('player_id')['offense_pct'].ffill()
    wrte_stats['w_rec_yards'] = wrte_stats['receiving_yards'] * wrte_stats['offense_pct']
    wrte_stats['w_rec_epa'] = wrte_stats['receiving_epa'] * wrte_stats['offense_pct']
    wrte_stats['w_targets'] = wrte_stats['targets'] * wrte_stats['offense_pct']

    wrte_team_game = wrte_stats.groupby(['team', 'game_id', 'gameday'], as_index=False).agg(
        t_rec_yards=('w_rec_yards', 'sum'), t_rec_epa=('w_rec_epa', 'sum'), t_targets=('w_targets', 'sum'))
    wrte_team_game = wrte_team_game.sort_values(['team', 'gameday']).reset_index(drop=True)
    for col in ['t_rec_yards', 't_rec_epa', 't_targets']:
        wrte_team_game[f'recent_{col}'] = wrte_team_game.groupby('team')[col].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean())
    current_wrte = wrte_team_game.loc[wrte_team_game.groupby('team')['gameday'].idxmax()][
        ['team', 'recent_t_rec_yards', 'recent_t_rec_epa', 'recent_t_targets']]

    # --- Coach (current identity only; h2h computed per-matchup at prediction time) ---
    hc = df_full[['home_team_std', 'home_coach', 'gameday']].rename(columns={'home_team_std': 'team', 'home_coach': 'coach'})
    ac = df_full[['away_team_std', 'away_coach', 'gameday']].rename(columns={'away_team_std': 'team', 'away_coach': 'coach'})
    all_coach = pd.concat([hc, ac], ignore_index=True)
    current_coach = all_coach.loc[all_coach.groupby('team')['gameday'].idxmax()][['team', 'coach']]

    # --- Defense-allowed + sack rate ---
    ts = nfl.load_team_stats(seasons=list(seasons)).to_pandas()
    ts = ts[['game_id', 'team', 'opponent_team', 'passing_epa', 'rushing_epa', 'passing_yards',
             'rushing_yards', 'def_interceptions', 'fumble_recovery_opp', 'attempts', 'sacks_suffered']]
    opp = ts[['game_id', 'team', 'passing_epa', 'rushing_epa', 'passing_yards', 'rushing_yards']].rename(
        columns={'team': 'opponent_team', 'passing_epa': 'opp_pass_epa', 'rushing_epa': 'opp_rush_epa',
                 'passing_yards': 'opp_pass_yds', 'rushing_yards': 'opp_rush_yds'})
    ts = ts.merge(opp, on=['game_id', 'opponent_team'], how='left')
    ts['epa_allowed'] = ts['opp_pass_epa'] + ts['opp_rush_epa']
    ts['yards_allowed'] = ts['opp_pass_yds'] + ts['opp_rush_yds']
    ts['takeaways'] = ts['def_interceptions'] + ts['fumble_recovery_opp']
    ts['sack_rate'] = ts['sacks_suffered'] / (ts['attempts'] + ts['sacks_suffered'])
    ts = ts.merge(game_dates, on='game_id', how='left').sort_values(['team', 'gameday']).reset_index(drop=True)
    for col in ['epa_allowed', 'yards_allowed', 'takeaways', 'sack_rate']:
        ts[f'recent_{col}'] = ts.groupby('team')[col].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    current_team_stats = ts.loc[ts.groupby('team')['gameday'].idxmax()][
        ['team', 'recent_epa_allowed', 'recent_yards_allowed', 'recent_takeaways', 'recent_sack_rate']]

    # --- Pressure rate ---
    press = nfl.load_pfr_advstats(seasons=pfr_seasons, stat_type='pass').to_pandas()
    press = press[['game_id', 'team', 'times_pressured_pct']]
    press = press.merge(game_dates, on='game_id', how='left').sort_values(['team', 'gameday']).reset_index(drop=True)
    press['recent_pressure_pct'] = press.groupby('team')['times_pressured_pct'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean())
    current_pressure = press.loc[press.groupby('team')['gameday'].idxmax()][['team', 'recent_pressure_pct']]

    # --- Primary WR (current roster's top by own recent snap share) ---
    wr = player_stats[player_stats['position'] == 'WR'][['player_id', 'game_id', 'team']].copy()
    wr = wr.merge(crosswalk, on='player_id', how='left')
    wr = wr.merge(snaps_full[snaps_full['position'] == 'WR'][['pfr_player_id', 'game_id', 'offense_pct']],
                  on=['pfr_player_id', 'game_id'], how='left')
    wr = wr.merge(game_dates, on='game_id', how='left').sort_values(['player_id', 'gameday']).reset_index(drop=True)
    wr['offense_pct'] = wr.groupby('player_id')['offense_pct'].ffill()
    wr['recent_offense_pct'] = wr.groupby('player_id')['offense_pct'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean())
    each_wr = wr.loc[wr.groupby('player_id')['gameday'].idxmax()][['player_id', 'pfr_player_id', 'team', 'recent_offense_pct']]
    current_primary_wr = each_wr.loc[each_wr.groupby('team')['recent_offense_pct'].idxmax()][['team', 'pfr_player_id']]
    current_primary_wr = current_primary_wr.merge(physical, on='pfr_player_id', how='left')

    # --- Primary CB (INNER merge - critical bug fix, see docstring history in features.py) ---
    cb_only = snaps_full[snaps_full['position'] == 'CB'][['pfr_player_id', 'game_id', 'defense_pct']]
    cb = adv_def_full.merge(cb_only, on=['pfr_player_id', 'game_id'], how='inner')
    cb = cb.merge(game_dates, on='game_id', how='left').sort_values(['pfr_player_id', 'gameday']).reset_index(drop=True)
    for col in ['defense_pct', 'def_completion_pct', 'def_passer_rating_allowed']:
        cb[f'recent_{col}'] = cb.groupby('pfr_player_id')[col].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean())
    each_cb = cb.loc[cb.groupby('pfr_player_id')['gameday'].idxmax()][
        ['team', 'pfr_player_id', 'recent_defense_pct', 'recent_def_completion_pct', 'recent_def_passer_rating_allowed']]
    current_primary_cb = each_cb.loc[each_cb.groupby('team')['recent_defense_pct'].idxmax()][
        ['team', 'pfr_player_id', 'recent_def_completion_pct', 'recent_def_passer_rating_allowed']]
    current_primary_cb = current_primary_cb.merge(physical, on='pfr_player_id', how='left')

    return {
        'df_full': df_full,
        'current_form': current_form,
        'current_elo': current_elo,
        'current_qb': current_qb,
        'current_rb': current_rb,
        'current_wrte': current_wrte,
        'current_coach': current_coach,
        'current_team_stats': current_team_stats,
        'current_pressure': current_pressure,
        'current_primary_wr': current_primary_wr,
        'current_primary_cb': current_primary_cb,
    }


def _coach_h2h(df_full, home_coach, away_coach):
    meetings = df_full[
        ((df_full['home_coach'] == home_coach) & (df_full['away_coach'] == away_coach)) |
        ((df_full['home_coach'] == away_coach) & (df_full['away_coach'] == home_coach))
    ]
    if len(meetings) == 0:
        return 0.5, 0
    wins = (
        ((meetings['home_coach'] == home_coach) & (meetings['home_win'] == 1)) |
        ((meetings['away_coach'] == home_coach) & (meetings['home_win'] == 0))
    ).sum()
    return wins / len(meetings), len(meetings)


def build_matchup_features(home_team, away_team, snapshots, home_rest, away_rest, div_game):
    """Assembles one feature row for a specific upcoming matchup from the snapshot tables."""
    s = snapshots
    row = {}

    row['home_recent_form'] = s['current_form'].loc[s['current_form']['team'] == home_team, 'current_recent_form'].values[0]
    row['away_recent_form'] = s['current_form'].loc[s['current_form']['team'] == away_team, 'current_recent_form'].values[0]
    row['home_recent_point_diff'] = s['current_form'].loc[s['current_form']['team'] == home_team, 'current_recent_point_diff'].values[0]
    row['away_recent_point_diff'] = s['current_form'].loc[s['current_form']['team'] == away_team, 'current_recent_point_diff'].values[0]

    for side, team in [('home', home_team), ('away', away_team)]:
        qb = s['current_qb'][s['current_qb']['team'] == team]
        row[f'{side}_qb_recent_yards'] = qb['recent_passing_yards'].values[0]
        row[f'{side}_qb_recent_tds'] = qb['recent_passing_tds'].values[0]
        row[f'{side}_qb_recent_ints'] = qb['recent_passing_interceptions'].values[0]
        row[f'{side}_qb_recent_epa'] = qb['recent_passing_epa'].values[0]

        rb = s['current_rb'][s['current_rb']['team'] == team]
        row[f'{side}_rb_recent_rush_yards'] = rb['recent_t_rush_yards'].values[0]
        row[f'{side}_rb_recent_rush_epa'] = rb['recent_t_rush_epa'].values[0]
        row[f'{side}_rb_recent_rec_yards'] = rb['recent_t_rec_yards'].values[0]

        wrte = s['current_wrte'][s['current_wrte']['team'] == team]
        row[f'{side}_wrte_recent_rec_yards'] = wrte['recent_t_rec_yards'].values[0]
        row[f'{side}_wrte_recent_rec_epa'] = wrte['recent_t_rec_epa'].values[0]
        row[f'{side}_wrte_recent_targets'] = wrte['recent_t_targets'].values[0]

        row[f'{side}_qb_injury_flag'] = 0
        row[f'{side}_rb_injury_flag'] = 0
        row[f'{side}_wrte_injury_flag'] = 0

        ts = s['current_team_stats'][s['current_team_stats']['team'] == team]
        row[f'{side}_epa_allowed_recent'] = ts['recent_epa_allowed'].values[0]
        row[f'{side}_yards_allowed_recent'] = ts['recent_yards_allowed'].values[0]
        row[f'{side}_takeaways_recent'] = ts['recent_takeaways'].values[0]
        row[f'{side}_sack_rate_recent'] = ts['recent_sack_rate'].values[0]

        p = s['current_pressure'][s['current_pressure']['team'] == team]
        row[f'{side}_pressure_pct_recent'] = p['recent_pressure_pct'].values[0] if len(p) > 0 else None

    home_coach = s['current_coach'].loc[s['current_coach']['team'] == home_team, 'coach'].values[0]
    away_coach = s['current_coach'].loc[s['current_coach']['team'] == away_team, 'coach'].values[0]
    row['home_coach_h2h_wins'], row['h2h_games_played'] = _coach_h2h(s['df_full'], home_coach, away_coach)

    row['home_elo_pre'] = s['current_elo'].loc[s['current_elo']['team'] == home_team, 'current_elo'].values[0]
    row['away_elo_pre'] = s['current_elo'].loc[s['current_elo']['team'] == away_team, 'current_elo'].values[0]

    row['rest_advantage'] = home_rest - away_rest

    home_wr = s['current_primary_wr'][s['current_primary_wr']['team'] == home_team]
    away_cb = s['current_primary_cb'][s['current_primary_cb']['team'] == away_team]
    row['home_wr_height_advantage'] = home_wr['height'].values[0] - away_cb['height'].values[0]
    row['home_wr_weight_advantage'] = home_wr['weight'].values[0] - away_cb['weight'].values[0]
    row['home_opp_cb_completion_allowed'] = away_cb['recent_def_completion_pct'].values[0]
    row['home_opp_cb_rating_allowed'] = away_cb['recent_def_passer_rating_allowed'].values[0]

    away_wr = s['current_primary_wr'][s['current_primary_wr']['team'] == away_team]
    home_cb = s['current_primary_cb'][s['current_primary_cb']['team'] == home_team]
    row['away_wr_height_advantage'] = away_wr['height'].values[0] - home_cb['height'].values[0]
    row['away_wr_weight_advantage'] = away_wr['weight'].values[0] - home_cb['weight'].values[0]
    row['away_opp_cb_completion_allowed'] = home_cb['recent_def_completion_pct'].values[0]
    row['away_opp_cb_rating_allowed'] = home_cb['recent_def_passer_rating_allowed'].values[0]

    row['div_game'] = div_game

    return pd.DataFrame([row])


def predict_matchup(home_team, away_team, snapshots, model_bundle, home_rest, away_rest, div_game):
    """Returns home team win probability (float 0-1) for one matchup."""
    feature_row = build_matchup_features(home_team, away_team, snapshots, home_rest, away_rest, div_game)
    feature_cols = model_bundle['feature_cols']
    X = feature_row[feature_cols]
    X_imputed = pd.DataFrame(model_bundle['imputer'].transform(X), columns=feature_cols)
    return model_bundle['model'].predict_proba(X_imputed)[0][1]


def predict_week(season, week, snapshots, model_bundle):
    """Returns a DataFrame of predictions for every game in a given season/week."""
    sched = nfl.load_schedules(seasons=[season]).to_pandas()
    week_games = sched[sched['week'] == week]

    results = []
    for _, g in week_games.iterrows():
        try:
            prob = predict_matchup(
                g['home_team'], g['away_team'], snapshots, model_bundle,
                g['home_rest'], g['away_rest'], g['div_game']
            )
            results.append({
                'game_id': g['game_id'], 'home_team': g['home_team'], 'away_team': g['away_team'],
                'gameday': g['gameday'], 'home_win_prob': prob, 'away_win_prob': 1 - prob
            })
        except (IndexError, KeyError) as e:
            print(f"Could not predict {g['game_id']}: missing data ({e})")

    return pd.DataFrame(results)


if __name__ == "__main__":
    model_bundle = load_model()
    snapshots = build_snapshots()
    predictions = predict_week(2026, 1, snapshots, model_bundle)
    print(predictions)