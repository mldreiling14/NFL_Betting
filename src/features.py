import pandas as pd


def add_recent_form_features(df_full, window=5):
    """
    Adds rolling recent-form features (win %, point differential)
    for both home and away teams, using only games prior to each
    game (no data leakage), reset at each season boundary.
    """

    # Reshape: one row per team per game
    home = df_full[['game_id', 'season', 'week', 'gameday',
                     'home_team', 'away_team', 'home_win',
                     'home_score', 'away_score']].copy()
    home = home.rename(columns={'home_team': 'team', 'away_team': 'opponent'})
    home['win'] = home['home_win']
    home['is_home'] = 1
    home['points_for'] = home['home_score']
    home['points_against'] = home['away_score']

    away = df_full[['game_id', 'season', 'week', 'gameday',
                     'home_team', 'away_team', 'home_win',
                     'home_score', 'away_score']].copy()
    away = away.rename(columns={'away_team': 'team', 'home_team': 'opponent'})
    away['win'] = 1 - away['home_win']
    away['is_home'] = 0
    away['points_for'] = away['away_score']
    away['points_against'] = away['home_score']

    team_games = pd.concat([home, away], ignore_index=True)
    team_games = team_games.sort_values(['team', 'season', 'gameday']).reset_index(drop=True)
    team_games['point_diff'] = team_games['points_for'] - team_games['points_against']

    # Rolling calculations, season-aware, no leakage (shift before rolling)
    team_games['recent_form'] = (
        team_games.groupby(['team', 'season'])['win']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )

    team_games['recent_point_diff'] = (
        team_games.groupby(['team', 'season'])['point_diff']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )

    # Split back into home/away perspectives and merge into df_full
    home_form = team_games[team_games['is_home'] == 1][['game_id', 'recent_form']].rename(
        columns={'recent_form': 'home_recent_form'})
    away_form = team_games[team_games['is_home'] == 0][['game_id', 'recent_form']].rename(
        columns={'recent_form': 'away_recent_form'})

    home_pd = team_games[team_games['is_home'] == 1][['game_id', 'recent_point_diff']].rename(
        columns={'recent_point_diff': 'home_recent_point_diff'})
    away_pd = team_games[team_games['is_home'] == 0][['game_id', 'recent_point_diff']].rename(
        columns={'recent_point_diff': 'away_recent_point_diff'})

    df_full = df_full.drop(columns=[
        'home_recent_form', 'away_recent_form',
        'home_recent_point_diff', 'away_recent_point_diff'
    ], errors='ignore')

    df_full = df_full.merge(home_form, on='game_id', how='left')
    df_full = df_full.merge(away_form, on='game_id', how='left')
    df_full = df_full.merge(home_pd, on='game_id', how='left')
    df_full = df_full.merge(away_pd, on='game_id', how='left')

    return df_full

def add_qb_features(df_full, qb_stats, window=5):
    """
    Adds rolling QB performance features (yards, TDs, INTs, EPA)
    for both home and away starting QBs, using only that QB's
    prior starts (no data leakage). Does NOT reset at season
    boundaries, since a QB's own performance carries across
    the offseason (unlike team-level roster turnover).
    """

    qb_stats = qb_stats.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)

    for col in ['passing_yards', 'passing_tds', 'passing_interceptions', 'passing_epa']:
        qb_stats[f'recent_{col}'] = (
            qb_stats.groupby('player_id')[col]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        )

    qb_recent_cols = ['player_id', 'game_id', 'recent_passing_yards', 'recent_passing_tds',
                       'recent_passing_interceptions', 'recent_passing_epa']
    qb_recent = qb_stats[qb_recent_cols]

    home_qb = qb_recent.rename(columns={
        'player_id': 'home_qb_id',
        'recent_passing_yards': 'home_qb_recent_yards',
        'recent_passing_tds': 'home_qb_recent_tds',
        'recent_passing_interceptions': 'home_qb_recent_ints',
        'recent_passing_epa': 'home_qb_recent_epa'
    })

    away_qb = qb_recent.rename(columns={
        'player_id': 'away_qb_id',
        'recent_passing_yards': 'away_qb_recent_yards',
        'recent_passing_tds': 'away_qb_recent_tds',
        'recent_passing_interceptions': 'away_qb_recent_ints',
        'recent_passing_epa': 'away_qb_recent_epa'
    })

    df_full = df_full.drop(columns=[c for c in df_full.columns if 'qb_recent' in c], errors='ignore')

    df_full = df_full.merge(home_qb, on=['home_qb_id', 'game_id'], how='left')
    df_full = df_full.merge(away_qb, on=['away_qb_id', 'game_id'], how='left')

    return df_full


# Franchises that relocated during this dataset's date range —
# schedule data uses the old code, but player-stats data uses the new one.
TEAM_CODE_MAP = {
    'SD': 'LAC',   # San Diego -> LA Chargers
    'STL': 'LA',   # St. Louis -> LA Rams
    'OAK': 'LV'    # Oakland -> Las Vegas
}


def standardize_team_codes(df_full):
    """
    Adds home_team_std / away_team_std columns that translate old,
    relocated-franchise team codes into the codes used by player-level
    stats tables, without altering the original home_team/away_team
    columns used for display.
    """
    df_full = df_full.copy()
    df_full['home_team_std'] = df_full['home_team'].replace(TEAM_CODE_MAP)
    df_full['away_team_std'] = df_full['away_team'].replace(TEAM_CODE_MAP)
    return df_full


def add_rb_features(df_full, rb_stats, window=5):
    """
    Adds rolling team-level RB-group performance features (snap-share-
    weighted rushing yards, rushing EPA, receiving yards) for both home
    and away teams. Aggregates all RBs on a team into one team-level
    signal per game, then computes a rolling average over that team's
    prior games (no leakage), reset at each season boundary.

    rb_stats must already include: team, game_id, season, week,
    carries, rushing_yards, rushing_epa, receiving_yards, offense_pct
    """

    rb_stats = rb_stats.copy()
    rb_stats['weighted_rush_yards'] = rb_stats['rushing_yards'] * rb_stats['offense_pct']
    rb_stats['weighted_rush_epa'] = rb_stats['rushing_epa'] * rb_stats['offense_pct']
    rb_stats['weighted_rec_yards'] = rb_stats['receiving_yards'] * rb_stats['offense_pct']

    # Aggregate to one row per team per game
    rb_team_game = rb_stats.groupby(['team', 'game_id', 'season', 'week'], as_index=False).agg(
        team_weighted_rush_yards=('weighted_rush_yards', 'sum'),
        team_weighted_rush_epa=('weighted_rush_epa', 'sum'),
        team_weighted_rec_yards=('weighted_rec_yards', 'sum'),
        team_rb_carries=('carries', 'sum')
    )

    # Rolling calculations, season-aware, no leakage
    rb_team_game = rb_team_game.sort_values(['team', 'season', 'week']).reset_index(drop=True)
    for col in ['team_weighted_rush_yards', 'team_weighted_rush_epa', 'team_weighted_rec_yards']:
        rb_team_game[f'recent_{col}'] = (
            rb_team_game.groupby(['team', 'season'])[col]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        )

    # Standardize team codes in df_full before merging (fixes relocation mismatch)
    df_full = standardize_team_codes(df_full)

    rb_recent_cols = ['team', 'game_id', 'recent_team_weighted_rush_yards',
                       'recent_team_weighted_rush_epa', 'recent_team_weighted_rec_yards']
    rb_recent = rb_team_game[rb_recent_cols]

    home_rb = rb_recent.rename(columns={
        'team': 'home_team_std',
        'recent_team_weighted_rush_yards': 'home_rb_recent_rush_yards',
        'recent_team_weighted_rush_epa': 'home_rb_recent_rush_epa',
        'recent_team_weighted_rec_yards': 'home_rb_recent_rec_yards'
    })

    away_rb = rb_recent.rename(columns={
        'team': 'away_team_std',
        'recent_team_weighted_rush_yards': 'away_rb_recent_rush_yards',
        'recent_team_weighted_rush_epa': 'away_rb_recent_rush_epa',
        'recent_team_weighted_rec_yards': 'away_rb_recent_rec_yards'
    })

    df_full = df_full.drop(columns=[c for c in df_full.columns if 'rb_recent' in c], errors='ignore')

    df_full = df_full.merge(home_rb, on=['home_team_std', 'game_id'], how='left')
    df_full = df_full.merge(away_rb, on=['away_team_std', 'game_id'], how='left')

    return df_full


def add_wrte_features(df_full, wrte_stats, window=5):
    """
    Adds rolling team-level WR+TE receiving-corps performance features
    (snap-share-weighted receiving yards, EPA, targets) for both home
    and away teams. Aggregates all WRs and TEs on a team into one
    team-level signal per game, then computes a rolling average over
    that team's prior games (no leakage), reset at each season boundary.

    wrte_stats must already include: team, game_id, season, week,
    targets, receiving_yards, receiving_epa, offense_pct
    """

    wrte_stats = wrte_stats.copy()
    wrte_stats['weighted_rec_yards'] = wrte_stats['receiving_yards'] * wrte_stats['offense_pct']
    wrte_stats['weighted_rec_epa'] = wrte_stats['receiving_epa'] * wrte_stats['offense_pct']
    wrte_stats['weighted_targets'] = wrte_stats['targets'] * wrte_stats['offense_pct']

    wrte_team_game = wrte_stats.groupby(['team', 'game_id', 'season', 'week'], as_index=False).agg(
        team_weighted_rec_yards=('weighted_rec_yards', 'sum'),
        team_weighted_rec_epa=('weighted_rec_epa', 'sum'),
        team_weighted_targets=('weighted_targets', 'sum')
    )

    wrte_team_game = wrte_team_game.sort_values(['team', 'season', 'week']).reset_index(drop=True)
    for col in ['team_weighted_rec_yards', 'team_weighted_rec_epa', 'team_weighted_targets']:
        wrte_team_game[f'recent_{col}'] = (
            wrte_team_game.groupby(['team', 'season'])[col]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        )

    df_full = standardize_team_codes(df_full)

    wrte_recent_cols = ['team', 'game_id', 'recent_team_weighted_rec_yards',
                         'recent_team_weighted_rec_epa', 'recent_team_weighted_targets']
    wrte_recent = wrte_team_game[wrte_recent_cols]

    home_wrte = wrte_recent.rename(columns={
        'team': 'home_team_std',
        'recent_team_weighted_rec_yards': 'home_wrte_recent_rec_yards',
        'recent_team_weighted_rec_epa': 'home_wrte_recent_rec_epa',
        'recent_team_weighted_targets': 'home_wrte_recent_targets'
    })

    away_wrte = wrte_recent.rename(columns={
        'team': 'away_team_std',
        'recent_team_weighted_rec_yards': 'away_wrte_recent_rec_yards',
        'recent_team_weighted_rec_epa': 'away_wrte_recent_rec_epa',
        'recent_team_weighted_targets': 'away_wrte_recent_targets'
    })

    df_full = df_full.drop(columns=[c for c in df_full.columns if 'wrte_recent' in c], errors='ignore')

    df_full = df_full.merge(home_wrte, on=['home_team_std', 'game_id'], how='left')
    df_full = df_full.merge(away_wrte, on=['away_team_std', 'game_id'], how='left')

    return df_full

def add_injury_features(df_full, injuries, rb_stats, wrte_stats, window=5):
    """
    Adds simple injury-designation flags (Out/Doubtful/Questionable)
    for the home and away starting QB, primary RB, and top-2 WR/TE
    (by recent snap share), for each game. A missing injury-report
    entry is treated as healthy (flag = 0).

    injuries must include: season, week, team, gsis_id, position, report_status
    rb_stats / wrte_stats must already include offense_pct (snap share)
    """

    df_full = standardize_team_codes(df_full)

    # --- QB injury flag (matches schedule's identified starter directly) ---
    qb_injuries = injuries[injuries['position'] == 'QB'].copy()
    qb_injuries['qb_injury_flag'] = qb_injuries['report_status'].isin(
        ['Out', 'Doubtful', 'Questionable']).astype(int)

    home_qb_inj = qb_injuries[['season', 'week', 'team', 'gsis_id', 'qb_injury_flag']].rename(
        columns={'team': 'home_team', 'gsis_id': 'home_qb_id', 'qb_injury_flag': 'home_qb_injury_flag'})
    away_qb_inj = qb_injuries[['season', 'week', 'team', 'gsis_id', 'qb_injury_flag']].rename(
        columns={'team': 'away_team', 'gsis_id': 'away_qb_id', 'qb_injury_flag': 'away_qb_injury_flag'})

    df_full = df_full.merge(home_qb_inj, on=['season', 'week', 'home_team', 'home_qb_id'], how='left')
    df_full = df_full.merge(away_qb_inj, on=['season', 'week', 'away_team', 'away_qb_id'], how='left')
    df_full['home_qb_injury_flag'] = df_full['home_qb_injury_flag'].fillna(0).astype(int)
    df_full['away_qb_injury_flag'] = df_full['away_qb_injury_flag'].fillna(0).astype(int)

    # --- RB injury flag (primary RB by recent snap share) ---
    rb_sorted = rb_stats.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)
    rb_sorted['recent_snap_pct'] = (
        rb_sorted.groupby(['player_id', 'season'])['offense_pct']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )
    primary_rb = (
        rb_sorted.sort_values('recent_snap_pct', ascending=False)
        .groupby(['team', 'game_id'], as_index=False)
        .first()[['team', 'game_id', 'season', 'week', 'player_id']]
    )

    rb_injuries = injuries[injuries['position'] == 'RB'].copy()
    rb_injuries['injury_flag'] = rb_injuries['report_status'].isin(
        ['Out', 'Doubtful', 'Questionable']).astype(int)
    rb_injuries = rb_injuries.rename(columns={'gsis_id': 'player_id'})

    primary_rb = primary_rb.merge(
        rb_injuries[['season', 'week', 'team', 'player_id', 'injury_flag']],
        on=['season', 'week', 'team', 'player_id'], how='left')
    primary_rb['injury_flag'] = primary_rb['injury_flag'].fillna(0).astype(int)
    primary_rb = primary_rb.rename(columns={'injury_flag': 'rb_injury_flag'})

    home_rb_inj = primary_rb[['season', 'week', 'team', 'rb_injury_flag']].rename(
        columns={'team': 'home_team_std', 'rb_injury_flag': 'home_rb_injury_flag'})
    away_rb_inj = primary_rb[['season', 'week', 'team', 'rb_injury_flag']].rename(
        columns={'team': 'away_team_std', 'rb_injury_flag': 'away_rb_injury_flag'})

    df_full = df_full.merge(home_rb_inj, on=['season', 'week', 'home_team_std'], how='left')
    df_full = df_full.merge(away_rb_inj, on=['season', 'week', 'away_team_std'], how='left')
    df_full['home_rb_injury_flag'] = df_full['home_rb_injury_flag'].fillna(0).astype(int)
    df_full['away_rb_injury_flag'] = df_full['away_rb_injury_flag'].fillna(0).astype(int)

    # --- WR/TE injury flag (top-2 by recent snap share, flagged if either is banged up) ---
    wrte_sorted = wrte_stats.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)
    wrte_sorted['recent_snap_pct'] = (
        wrte_sorted.groupby(['player_id', 'season'])['offense_pct']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )
    top2_wrte = (
        wrte_sorted.sort_values('recent_snap_pct', ascending=False)
        .groupby(['team', 'game_id'])
        .head(2)[['team', 'game_id', 'season', 'week', 'player_id']]
    )

    wrte_injuries = injuries[injuries['position'].isin(['WR', 'TE'])].copy()
    wrte_injuries['injury_flag'] = wrte_injuries['report_status'].isin(
        ['Out', 'Doubtful', 'Questionable']).astype(int)
    wrte_injuries = wrte_injuries.rename(columns={'gsis_id': 'player_id'})

    top2_wrte = top2_wrte.merge(
        wrte_injuries[['season', 'week', 'team', 'player_id', 'injury_flag']],
        on=['season', 'week', 'team', 'player_id'], how='left')
    top2_wrte['injury_flag'] = top2_wrte['injury_flag'].fillna(0).astype(int)

    wrte_injury_flag = top2_wrte.groupby(
        ['team', 'game_id', 'season', 'week'], as_index=False)['injury_flag'].max()
    wrte_injury_flag = wrte_injury_flag.rename(columns={'injury_flag': 'wrte_injury_flag'})

    home_wrte_inj = wrte_injury_flag[['season', 'week', 'team', 'wrte_injury_flag']].rename(
        columns={'team': 'home_team_std', 'wrte_injury_flag': 'home_wrte_injury_flag'})
    away_wrte_inj = wrte_injury_flag[['season', 'week', 'team', 'wrte_injury_flag']].rename(
        columns={'team': 'away_team_std', 'wrte_injury_flag': 'away_wrte_injury_flag'})

    df_full = df_full.merge(home_wrte_inj, on=['season', 'week', 'home_team_std'], how='left')
    df_full = df_full.merge(away_wrte_inj, on=['season', 'week', 'away_team_std'], how='left')
    df_full['home_wrte_injury_flag'] = df_full['home_wrte_injury_flag'].fillna(0).astype(int)
    df_full['away_wrte_injury_flag'] = df_full['away_wrte_injury_flag'].fillna(0).astype(int)

    return df_full


def add_defense_allowed_features(df_full, team_stats, window=5):
    """
    Adds rolling team-level defensive features based on what a team's
    defense ALLOWED (opponent's own offensive EPA/yards that game) and
    takeaways forced (INTs + fumble recoveries combined). This is a
    richer signal than raw counting stats like sacks, since it reflects
    what the defense actually gave up rather than isolated events.

    team_stats must include: game_id, season, week, team, opponent_team,
    passing_epa, rushing_epa, passing_yards, rushing_yards,
    def_interceptions, fumble_recovery_opp

    Note: team_stats already uses current franchise codes (LAC/LA/LV),
    so it merges directly against df_full's standardized team columns.
    """

    team_stats = team_stats[['game_id', 'season', 'week', 'team', 'opponent_team',
                              'passing_epa', 'rushing_epa', 'passing_yards', 'rushing_yards',
                              'def_interceptions', 'fumble_recovery_opp']].copy()

    opponent_offense = team_stats[['game_id', 'team', 'passing_epa', 'rushing_epa',
                                    'passing_yards', 'rushing_yards']].rename(
        columns={'team': 'opponent_team', 'passing_epa': 'opp_passing_epa',
                 'rushing_epa': 'opp_rushing_epa', 'passing_yards': 'opp_passing_yards',
                 'rushing_yards': 'opp_rushing_yards'}
    )

    team_stats = team_stats.merge(opponent_offense, on=['game_id', 'opponent_team'], how='left')

    team_stats['epa_allowed'] = team_stats['opp_passing_epa'] + team_stats['opp_rushing_epa']
    team_stats['yards_allowed'] = team_stats['opp_passing_yards'] + team_stats['opp_rushing_yards']
    team_stats['takeaways'] = team_stats['def_interceptions'] + team_stats['fumble_recovery_opp']

    team_stats = team_stats.sort_values(['team', 'season', 'week']).reset_index(drop=True)
    for col in ['epa_allowed', 'yards_allowed', 'takeaways']:
        team_stats[f'recent_{col}'] = (
            team_stats.groupby(['team', 'season'])[col]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        )

    df_full = standardize_team_codes(df_full)

    def_allowed_recent = team_stats[['team', 'game_id', 'recent_epa_allowed',
                                       'recent_yards_allowed', 'recent_takeaways']]

    home_def = def_allowed_recent.rename(columns={
        'team': 'home_team_std',
        'recent_epa_allowed': 'home_epa_allowed_recent',
        'recent_yards_allowed': 'home_yards_allowed_recent',
        'recent_takeaways': 'home_takeaways_recent'
    })
    away_def = def_allowed_recent.rename(columns={
        'team': 'away_team_std',
        'recent_epa_allowed': 'away_epa_allowed_recent',
        'recent_yards_allowed': 'away_yards_allowed_recent',
        'recent_takeaways': 'away_takeaways_recent'
    })

    df_full = df_full.drop(columns=[c for c in df_full.columns if
                                     'epa_allowed_recent' in c or 'yards_allowed_recent' in c or 'takeaways_recent' in c],
                            errors='ignore')
    df_full = df_full.merge(home_def, on=['home_team_std', 'game_id'], how='left')
    df_full = df_full.merge(away_def, on=['away_team_std', 'game_id'], how='left')

    return df_full

def add_coach_features(df_full, window=5):
    """
    Adds two coach-related features for both home and away sides:
    1) Coach recent form (rolling win %, carries across seasons since
       it tracks the person, not the roster).
    2) Head-to-head record between the two coaches in this specific
       matchup, using only games played before this one. Games with
       no prior history get a neutral 0.5 (no known edge).
    """

    # --- Coach recent form ---
    home_coach_games = df_full[['game_id', 'season', 'week', 'gameday', 'home_coach', 'home_win']].copy()
    home_coach_games = home_coach_games.rename(columns={'home_coach': 'coach'})
    home_coach_games['win'] = home_coach_games['home_win']

    away_coach_games = df_full[['game_id', 'season', 'week', 'gameday', 'away_coach', 'home_win']].copy()
    away_coach_games = away_coach_games.rename(columns={'away_coach': 'coach'})
    away_coach_games['win'] = 1 - away_coach_games['home_win']

    coach_games = pd.concat([home_coach_games, away_coach_games], ignore_index=True)
    coach_games = coach_games.sort_values(['coach', 'gameday']).reset_index(drop=True)

    coach_games['recent_coach_form'] = (
        coach_games.groupby('coach')['win']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )

    home_rows = coach_games.merge(df_full[['game_id', 'home_coach']],
                                   left_on=['game_id', 'coach'], right_on=['game_id', 'home_coach'])
    away_rows = coach_games.merge(df_full[['game_id', 'away_coach']],
                                   left_on=['game_id', 'coach'], right_on=['game_id', 'away_coach'])

    home_coach_recent = home_rows[['game_id', 'recent_coach_form']].rename(
        columns={'recent_coach_form': 'home_coach_recent_form'})
    away_coach_recent = away_rows[['game_id', 'recent_coach_form']].rename(
        columns={'recent_coach_form': 'away_coach_recent_form'})

    df_full = df_full.drop(columns=['home_coach_recent_form', 'away_coach_recent_form'], errors='ignore')
    df_full = df_full.merge(home_coach_recent, on='game_id', how='left')
    df_full = df_full.merge(away_coach_recent, on='game_id', how='left')

    # --- Coach head-to-head record ---
    coach_matchups = df_full[['game_id', 'season', 'week', 'gameday', 'home_coach', 'away_coach', 'home_win']].copy()
    coach_matchups['matchup_key'] = coach_matchups.apply(
        lambda r: tuple(sorted([r['home_coach'], r['away_coach']])), axis=1
    )
    coach_matchups = coach_matchups.sort_values('gameday').reset_index(drop=True)

    results = []
    for key, group in coach_matchups.groupby('matchup_key'):
        group = group.sort_values('gameday').reset_index(drop=True)
        for i in range(len(group)):
            row = group.iloc[i]
            prior = group.iloc[:i]
            if len(prior) == 0:
                results.append({'game_id': row['game_id'], 'home_coach_h2h_wins': None, 'h2h_games_played': 0})
                continue

            home_coach_wins = (
                ((prior['home_coach'] == row['home_coach']) & (prior['home_win'] == 1)) |
                ((prior['away_coach'] == row['home_coach']) & (prior['home_win'] == 0))
            ).sum()

            results.append({
                'game_id': row['game_id'],
                'home_coach_h2h_wins': home_coach_wins / len(prior),
                'h2h_games_played': len(prior)
            })

    h2h_results = pd.DataFrame(results)

    df_full = df_full.drop(columns=['home_coach_h2h_wins', 'h2h_games_played'], errors='ignore')
    df_full = df_full.merge(h2h_results, on='game_id', how='left')
    df_full['home_coach_h2h_wins'] = df_full['home_coach_h2h_wins'].fillna(0.5)
    df_full['h2h_games_played'] = df_full['h2h_games_played'].fillna(0)

    return df_full


def add_elo_features(df_full, k_factor=20, home_advantage=65, revert_fraction=1/3, initial_rating=1500):
    """
    Adds Elo ratings for both home and away teams, reflecting each
    team's overall strength entering this specific game (accounts
    for strength of opponent, unlike simple win %). Ratings partially
    revert toward league average (1500) at the start of each season
    to account for roster turnover, then evolve game-by-game based
    on actual vs. expected outcomes.
    """

    df_full = standardize_team_codes(df_full)

    games = df_full[['game_id', 'season', 'week', 'gameday', 'home_team_std', 'away_team_std', 'home_win']].copy()
    games = games.sort_values(['season', 'gameday']).reset_index(drop=True)

    all_teams = set(games['home_team_std']).union(set(games['away_team_std']))
    elo = {team: initial_rating for team in all_teams}

    home_elo_pre = []
    away_elo_pre = []
    current_season = None

    for idx, row in games.iterrows():
        if current_season is not None and row['season'] != current_season:
            for team in elo:
                elo[team] = elo[team] * (1 - revert_fraction) + initial_rating * revert_fraction
        current_season = row['season']

        home_team = row['home_team_std']
        away_team = row['away_team_std']

        home_elo_pre.append(elo[home_team])
        away_elo_pre.append(elo[away_team])

        elo_diff = (elo[home_team] + home_advantage) - elo[away_team]
        expected_home = 1 / (1 + 10 ** (-elo_diff / 400))
        actual_home = row['home_win']

        elo[home_team] += k_factor * (actual_home - expected_home)
        elo[away_team] += k_factor * ((1 - actual_home) - (1 - expected_home))

    games['home_elo_pre'] = home_elo_pre
    games['away_elo_pre'] = away_elo_pre

    elo_merge = games[['game_id', 'home_elo_pre', 'away_elo_pre']]
    df_full = df_full.drop(columns=['home_elo_pre', 'away_elo_pre'], errors='ignore')
    df_full = df_full.merge(elo_merge, on='game_id', how='left')

    return df_full

def add_rest_advantage(df_full):
    """
    Adds rest_advantage: the difference in days of rest between the
    home and away team (positive = home team had more rest). Derived
    directly from home_rest/away_rest already present in the schedule
    data — a scheduling/fatigue signal distinct from team performance.
    """
    df_full = df_full.copy()
    df_full['rest_advantage'] = df_full['home_rest'] - df_full['away_rest']

    return df_full

def add_oline_features(df_full, oline_stats, pressure_stats):
    """
    Adds O-line/pass-protection features for both home and away teams:
    1) Sack rate (sacks per dropback, from load_team_stats — available
       full 2015+ range).
    2) QB pressure rate (from PFR advanced stats, load_pfr_advstats —
       only available 2018+; earlier seasons will show NaN and should
       be mean-imputed at modeling time).

    IMPORTANT: pressure_stats uses PFR's team codes, which reflect the
    ACTUAL code at the time (e.g. OAK through 2019, LV from 2020) —
    the opposite convention from load_team_stats, which backfills to
    CURRENT codes for all seasons. This function merges oline_stats
    against home_team_std/away_team_std (standardized), but merges
    pressure_stats against the raw home_team/away_team. Do not "fix"
    this to be consistent without re-verifying against real data.

    oline_stats must include: game_id, season, week, team, attempts,
    sacks_suffered
    pressure_stats must include: game_id, season, week, team,
    times_pressured_pct
    """

    # --- Sack rate (season-aware rolling, standardized team codes) ---
    oline_stats = oline_stats.copy()
    oline_stats['sack_rate'] = oline_stats['sacks_suffered'] / (oline_stats['attempts'] + oline_stats['sacks_suffered'])
    oline_stats = oline_stats.sort_values(['team', 'season', 'week']).reset_index(drop=True)
    oline_stats['recent_sack_rate'] = (
        oline_stats.groupby(['team', 'season'])['sack_rate']
        .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    )

    df_full = standardize_team_codes(df_full)

    sack_rate_recent = oline_stats[['team', 'game_id', 'recent_sack_rate']]
    home_sr = sack_rate_recent.rename(columns={'team': 'home_team_std', 'recent_sack_rate': 'home_sack_rate_recent'})
    away_sr = sack_rate_recent.rename(columns={'team': 'away_team_std', 'recent_sack_rate': 'away_sack_rate_recent'})

    df_full = df_full.drop(columns=['home_sack_rate_recent', 'away_sack_rate_recent'], errors='ignore')
    df_full = df_full.merge(home_sr, on=['home_team_std', 'game_id'], how='left')
    df_full = df_full.merge(away_sr, on=['away_team_std', 'game_id'], how='left')

    # --- Pressure rate (season-aware rolling, RAW team codes - see docstring) ---
    pressure_stats = pressure_stats.copy()
    pressure_team_game = pressure_stats.groupby(['team', 'game_id', 'season', 'week'], as_index=False).mean(numeric_only=True)
    pressure_team_game = pressure_team_game.sort_values(['team', 'season', 'week']).reset_index(drop=True)
    pressure_team_game['recent_pressure_pct'] = (
        pressure_team_game.groupby(['team', 'season'])['times_pressured_pct']
        .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    )

    pressure_recent = pressure_team_game[['team', 'game_id', 'recent_pressure_pct']]
    home_press = pressure_recent.rename(columns={'team': 'home_team', 'recent_pressure_pct': 'home_pressure_pct_recent'})
    away_press = pressure_recent.rename(columns={'team': 'away_team', 'recent_pressure_pct': 'away_pressure_pct_recent'})

    df_full = df_full.drop(columns=['home_pressure_pct_recent', 'away_pressure_pct_recent'], errors='ignore')
    df_full = df_full.merge(home_press, on=['home_team', 'game_id'], how='left')
    df_full = df_full.merge(away_press, on=['away_team', 'game_id'], how='left')

    return df_full

def add_db_wr_matchup_features(df_full, player_stats, snaps_full, crosswalk, physical, adv_def_full, window=5):
    """
    Adds an approximated WR-vs-CB size and coverage matchup feature:
    for each team's primary WR (highest recent snap share among WRs),
    compares height/weight against the OPPOSING team's primary CB
    (highest recent snap share among CBs that game), plus that CB's
    real recent coverage performance (completion %, passer rating
    allowed).

    IMPORTANT CAVEAT: NFL data does not publish which specific CB
    covers which specific WR on any given play. This is an
    approximation ("team's best WR" vs "opponent's most-used CB"),
    not a guaranteed real matchup.

    IMPORTANT BUG HISTORY: the CB identification step MUST use an
    INNER merge (not left) between adv_def_full and cb_snaps. A left
    merge lets non-CB defenders (who happened to record some pass-
    defense stat that game, e.g. a DE or LB) enter the "primary CB"
    candidate pool, and they can incorrectly win the selection when
    multiple players tie at NaN (e.g. every Week 1, before any rolling
    history exists). This produced a real bug (a 300lb "cornerback"
    who was actually a defensive end) that was caught via a physical-
    plausibility sanity check, not by the merge logic itself failing.

    Only available for 2018+ (PFR advanced stats limitation) — earlier
    seasons will show NaN and should be mean-imputed at modeling time.

    player_stats must include: player_id, game_id, season, week, team,
    position, targets
    snaps_full must include: pfr_player_id, game_id, team, position,
    offense_pct, defense_pct
    crosswalk must include: player_id, pfr_player_id
    physical must include: pfr_player_id, height, weight
    adv_def_full must include: game_id, season, week, team,
    pfr_player_id, def_completion_pct, def_passer_rating_allowed
    (team column follows PFR's actual-code-at-the-time convention,
    NOT the standardized/current-code convention — merge against
    df_full's raw home_team/away_team, not home_team_std/away_team_std)
    """

    # --- Primary CB per team/game (INNER merge is the critical fix) ---
    cb_snaps = snaps_full[snaps_full['position'] == 'CB'][['pfr_player_id', 'game_id', 'team', 'defense_pct']]

    cb_stats = adv_def_full.merge(
        cb_snaps[['pfr_player_id', 'game_id', 'defense_pct']],
        on=['pfr_player_id', 'game_id'], how='inner'  # INNER, not left - see docstring
    )

    cb_stats = cb_stats.sort_values(['pfr_player_id', 'season', 'week']).reset_index(drop=True)
    cb_stats['recent_defense_pct'] = (
        cb_stats.groupby(['pfr_player_id', 'season'])['defense_pct']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )
    for col in ['def_completion_pct', 'def_passer_rating_allowed']:
        cb_stats[f'recent_{col}'] = (
            cb_stats.groupby(['pfr_player_id', 'season'])[col]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        )

    primary_cb = (
        cb_stats.sort_values('recent_defense_pct', ascending=False)
        .groupby(['team', 'game_id'], as_index=False)
        .first()[['team', 'game_id', 'season', 'week', 'pfr_player_id',
                  'recent_def_completion_pct', 'recent_def_passer_rating_allowed']]
    )
    primary_cb = primary_cb.merge(physical, on='pfr_player_id', how='left')

    # --- Primary WR per team/game ---
    wr_snaps = snaps_full[snaps_full['position'] == 'WR'][['pfr_player_id', 'game_id', 'team', 'offense_pct']]

    wr_targets = player_stats[player_stats['position'] == 'WR'][
        ['player_id', 'game_id', 'season', 'week', 'team', 'targets']].copy()
    wr_targets = wr_targets.merge(crosswalk, on='player_id', how='left')
    wr_targets = wr_targets.merge(
        wr_snaps[['pfr_player_id', 'game_id', 'offense_pct']], on=['pfr_player_id', 'game_id'], how='left')

    wr_targets = wr_targets.sort_values(['pfr_player_id', 'season', 'week']).reset_index(drop=True)
    wr_targets['recent_offense_pct'] = (
        wr_targets.groupby(['pfr_player_id', 'season'])['offense_pct']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )

    primary_wr = (
        wr_targets.sort_values('recent_offense_pct', ascending=False)
        .groupby(['team', 'game_id'], as_index=False)
        .first()[['team', 'game_id', 'season', 'week', 'pfr_player_id']]
    )
    primary_wr = primary_wr.merge(physical, on='pfr_player_id', how='left')

    # --- Match each WR against the OPPOSING team's primary CB ---
    wr_for_matchup = primary_wr[['team', 'game_id', 'season', 'height', 'weight']].rename(
        columns={'height': 'wr_height', 'weight': 'wr_weight'})

    cb_for_matchup = primary_cb[['team', 'game_id', 'height', 'weight',
                                   'recent_def_completion_pct', 'recent_def_passer_rating_allowed']].rename(
        columns={'team': 'opponent_team', 'height': 'cb_height', 'weight': 'cb_weight'})

    game_opponents = df_full[['game_id', 'home_team', 'away_team']].copy()

    matchup = wr_for_matchup.merge(game_opponents, on='game_id', how='left')
    matchup['opponent_team'] = matchup.apply(
        lambda r: r['away_team'] if r['team'] == r['home_team'] else r['home_team'], axis=1)

    matchup = matchup.merge(cb_for_matchup, on=['game_id', 'opponent_team'], how='left')
    matchup['height_advantage'] = matchup['wr_height'] - matchup['cb_height']
    matchup['weight_advantage'] = matchup['wr_weight'] - matchup['cb_weight']

    matchup_final = matchup[['team', 'game_id', 'height_advantage', 'weight_advantage',
                              'recent_def_completion_pct', 'recent_def_passer_rating_allowed']].rename(
        columns={'recent_def_completion_pct': 'opp_cb_completion_pct_allowed',
                 'recent_def_passer_rating_allowed': 'opp_cb_rating_allowed'})

    # NOTE: merges on RAW home_team/away_team, not standardized - see docstring
    home_matchup = matchup_final.rename(columns={
        'team': 'home_team',
        'height_advantage': 'home_wr_height_advantage',
        'weight_advantage': 'home_wr_weight_advantage',
        'opp_cb_completion_pct_allowed': 'home_opp_cb_completion_allowed',
        'opp_cb_rating_allowed': 'home_opp_cb_rating_allowed'
    })
    away_matchup = matchup_final.rename(columns={
        'team': 'away_team',
        'height_advantage': 'away_wr_height_advantage',
        'weight_advantage': 'away_wr_weight_advantage',
        'opp_cb_completion_pct_allowed': 'away_opp_cb_completion_allowed',
        'opp_cb_rating_allowed': 'away_opp_cb_rating_allowed'
    })

    df_full = df_full.drop(columns=[c for c in df_full.columns if
                                     'wr_height_advantage' in c or 'wr_weight_advantage' in c or 'opp_cb' in c],
                            errors='ignore')
    df_full = df_full.merge(home_matchup, on=['game_id', 'home_team'], how='left')
    df_full = df_full.merge(away_matchup, on=['game_id', 'away_team'], how='left')

    return df_full

def add_weather_features(df_full):
    """
    Adds weather-related features:
    1) Raw temp/wind, with a neutral fill (70F, 0 wind) for dome/closed
       games — this is the actual real condition for domes, not an
       estimate, so it's distinct from mean-imputation.
    2) Simple threshold flags for cold (<=32F) and high wind (>=15mph)
       games, only ever flagged for genuinely outdoor games.
    3) Climate shock: how much colder today's game is than the visiting
       team's own typical home outdoor temperature, and a flag for
       extreme cases (25+ degree swing) - e.g. a warm-climate team
       traveling into unfamiliar cold. Real effect confirmed (61.6% vs
       54.8% baseline home win rate in cold-shock games), though it did
       not net a measurable overall accuracy improvement in testing -
       kept for completeness/interpretability per deliberate choice.

    Requires temp, wind, roof, home_team_std, away_team_std, home_win
    already present in df_full.
    """
    df_full = df_full.copy()

    df_full['is_outdoor'] = df_full['roof'].isin(['outdoors', 'open']).astype(int)
    df_full['temp_adj'] = df_full['temp'].where(df_full['is_outdoor'] == 1, 70)
    df_full['wind_adj'] = df_full['wind'].where(df_full['is_outdoor'] == 1, 0)

    df_full['cold_game'] = ((df_full['is_outdoor'] == 1) & (df_full['temp_adj'] <= 32)).astype(int)
    df_full['high_wind_game'] = ((df_full['is_outdoor'] == 1) & (df_full['wind_adj'] >= 15)).astype(int)

    team_climate = df_full[df_full['is_outdoor'] == 1].groupby('home_team_std')['temp_adj'].mean()
    df_full['away_home_climate'] = df_full['away_team_std'].map(team_climate)
    df_full['away_home_climate'] = df_full['away_home_climate'].fillna(70)  # dome/no-data teams treated as neutral

    df_full['climate_shock'] = None
    outdoor_mask = df_full['is_outdoor'] == 1
    df_full.loc[outdoor_mask, 'climate_shock'] = (
        df_full.loc[outdoor_mask, 'away_home_climate'] - df_full.loc[outdoor_mask, 'temp_adj']
    )
    df_full['climate_shock'] = df_full['climate_shock'].fillna(0).astype(float)

    df_full['cold_shock_game'] = ((df_full['climate_shock'] >= 25) & (df_full['is_outdoor'] == 1)).astype(int)

    return df_full

def add_star_rb_injury_feature(df_full, player_stats, snaps_full, crosswalk, injuries, window=5):
    """
    Adds a flag for whether a team's primary RB (by recent snap share)
    is BOTH a genuine star performer AND carrying an injury designation
    that week. "Star" = top 10% league-wide among RBs by rolling
    fantasy points (PPR) over their last 5 games, recalculated fresh
    for every game date.

    Distinct from the general primary-RB injury flag in
    add_injury_features, which fires for ANY primary RB injury
    regardless of talent level - this only fires for a genuine
    difference-maker.

    Real-world check: home win rate drops from ~54.8% (overall) to
    ~48.7% when the home team's star RB is out (n=76). Real, sensible
    effect; did not net a measurable overall accuracy improvement in
    isolated testing (kept anyway - see FEATURES.md).
    """

    # Rolling fantasy production, all RBs league-wide
    rb_fantasy = player_stats[player_stats['position'] == 'RB'][
        ['player_id', 'game_id', 'season', 'week', 'fantasy_points_ppr']].copy()
    rb_fantasy = rb_fantasy.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)
    rb_fantasy['recent_fantasy_ppr'] = (
        rb_fantasy.groupby('player_id')['fantasy_points_ppr']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )
    rb_fantasy['position_percentile'] = (
        rb_fantasy.groupby('game_id')['recent_fantasy_ppr'].rank(pct=True)
    )
    rb_fantasy['is_star'] = (rb_fantasy['position_percentile'] >= 0.90).astype(int)
    star_lookup = rb_fantasy[['player_id', 'game_id', 'is_star']]

    # Identify primary RB per team/game (same pattern as add_injury_features)
    rb_stats_local = player_stats[player_stats['position'] == 'RB'][
        ['player_id', 'game_id', 'season', 'week', 'team']].copy()
    rb_stats_local = rb_stats_local.merge(crosswalk, on='player_id', how='left')
    rb_stats_local = rb_stats_local.merge(
        snaps_full[snaps_full['position'] == 'RB'][['pfr_player_id', 'game_id', 'offense_pct']],
        on=['pfr_player_id', 'game_id'], how='left')

    rb_sorted = rb_stats_local.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)
    rb_sorted['recent_snap_pct'] = (
        rb_sorted.groupby(['player_id', 'season'])['offense_pct']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )
    primary_rb = (
        rb_sorted.sort_values('recent_snap_pct', ascending=False)
        .groupby(['team', 'game_id'], as_index=False)
        .first()[['team', 'game_id', 'season', 'week', 'player_id']]
    )

    primary_rb = primary_rb.merge(star_lookup, on=['player_id', 'game_id'], how='left')
    primary_rb['is_star'] = primary_rb['is_star'].fillna(0)

    rb_injuries = injuries[injuries['position'] == 'RB'].copy()
    rb_injuries['injury_flag'] = rb_injuries['report_status'].isin(
        ['Out', 'Doubtful', 'Questionable']).astype(int)
    rb_injuries = rb_injuries.rename(columns={'gsis_id': 'player_id'})

    primary_rb = primary_rb.merge(
        rb_injuries[['season', 'week', 'team', 'player_id', 'injury_flag']],
        on=['season', 'week', 'team', 'player_id'], how='left')
    primary_rb['injury_flag'] = primary_rb['injury_flag'].fillna(0).astype(int)
    primary_rb['star_rb_injured'] = (
        (primary_rb['is_star'] == 1) & (primary_rb['injury_flag'] == 1)
    ).astype(int)

    df_full = standardize_team_codes(df_full)

    home_star_rb = primary_rb[['season', 'week', 'team', 'star_rb_injured']].rename(
        columns={'team': 'home_team_std', 'star_rb_injured': 'home_star_rb_injured'})
    away_star_rb = primary_rb[['season', 'week', 'team', 'star_rb_injured']].rename(
        columns={'team': 'away_team_std', 'star_rb_injured': 'away_star_rb_injured'})

    df_full = df_full.drop(columns=['home_star_rb_injured', 'away_star_rb_injured'], errors='ignore')
    df_full = df_full.merge(home_star_rb, on=['season', 'week', 'home_team_std'], how='left')
    df_full = df_full.merge(away_star_rb, on=['season', 'week', 'away_team_std'], how='left')
    df_full['home_star_rb_injured'] = df_full['home_star_rb_injured'].fillna(0).astype(int)
    df_full['away_star_rb_injured'] = df_full['away_star_rb_injured'].fillna(0).astype(int)

    return df_full

def add_star_wr_injury_feature(df_full, player_stats, snaps_full, crosswalk, injuries, window=5):
    """
    Adds a flag for whether a team's primary WR (by recent snap share)
    is BOTH a genuine star performer (top 5% league-wide by rolling
    targets over their last 5 games) AND carrying a serious injury
    designation (Out or Doubtful - "Questionable" is deliberately
    excluded, since questionable players very often still play close
    to their normal role, which would dilute/reverse the signal).

    BUG HISTORY: an earlier version of this feature identified the
    "primary WR" and their "star" status using ONLY games where that
    player has a real stat row. But a player who is genuinely injured
    generates NO stat row for the game they miss - so they became
    structurally invisible to their own "primary WR" and "star" checks
    at exactly the moment they were actually out. This produced a
    counterintuitive result (higher win rate when the "star" WR was
    "injured") because the feature was actually measuring something
    close to the opposite of its intent. Fixed using pd.merge_asof to
    project each player's last known snap share AND star status
    forward across every game their team plays (direction='backward'),
    so an injured star's real, established role and talent level still
    "shows up" for that game, right up until the injury check runs.

    This fix is what turned a nonsensical result into a real, sensible
    one: home win rate drops to ~38% (vs ~54.8% overall) when a
    properly-identified star WR is out/doubtful (n=21, small sample -
    treat the exact number as an estimate). Confirmed to add real
    accuracy value in isolated model testing (67.4% -> 67.7%).

    player_stats must include: player_id, game_id, season, week,
    position, targets
    snaps_full must include: pfr_player_id, game_id, position, offense_pct
    """

    game_dates = df_full[['game_id', 'gameday']].drop_duplicates()
    game_dates['gameday'] = pd.to_datetime(game_dates['gameday'])

    df_full = standardize_team_codes(df_full)

    team_game_dates = pd.concat([
        df_full[['home_team_std', 'game_id', 'gameday']].rename(columns={'home_team_std': 'team'}),
        df_full[['away_team_std', 'game_id', 'gameday']].rename(columns={'away_team_std': 'team'})
    ], ignore_index=True)
    team_game_dates['gameday'] = pd.to_datetime(team_game_dates['gameday'])
    team_game_dates = team_game_dates.sort_values('gameday').reset_index(drop=True)

    # --- WR snap share history + rolling target-based star status ---
    wr_stats = player_stats[player_stats['position'] == 'WR'][
        ['player_id', 'game_id', 'season', 'week', 'team', 'targets']].copy()
    wr_stats = wr_stats.merge(crosswalk, on='player_id', how='left')
    wr_stats = wr_stats.merge(
        snaps_full[snaps_full['position'] == 'WR'][['pfr_player_id', 'game_id', 'offense_pct']],
        on=['pfr_player_id', 'game_id'], how='left')
    wr_stats = wr_stats.merge(game_dates, on='game_id', how='left')
    wr_stats = wr_stats.sort_values(['player_id', 'gameday']).reset_index(drop=True)

    wr_stats['recent_snap_pct'] = (
        wr_stats.groupby('player_id')['offense_pct']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )
    wr_stats['recent_targets'] = (
        wr_stats.groupby('player_id')['targets']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )
    wr_stats['target_percentile'] = wr_stats.groupby('game_id')['recent_targets'].rank(pct=True)
    wr_stats['is_star_wr'] = (wr_stats['target_percentile'] >= 0.95).astype(int)

    snap_history = wr_stats[['player_id', 'team', 'gameday', 'recent_snap_pct']].dropna(subset=['gameday', 'recent_snap_pct'])
    star_history = wr_stats[['player_id', 'gameday', 'is_star_wr']].dropna(subset=['gameday'])

    # --- Project snap share forward across every game the player's team played (fixes the bug) ---
    snap_projected_frames = []
    for player_id, player_df in snap_history.sort_values('gameday').groupby('player_id'):
        team = player_df['team'].iloc[-1]
        team_games = team_game_dates[team_game_dates['team'] == team].sort_values('gameday')
        projected = pd.merge_asof(
            team_games, player_df[['gameday', 'recent_snap_pct']].sort_values('gameday'),
            on='gameday', direction='backward'
        )
        projected['player_id'] = player_id
        snap_projected_frames.append(projected)
    wr_snap_projected = pd.concat(snap_projected_frames, ignore_index=True).dropna(subset=['recent_snap_pct'])

    # --- Project star status forward the same way ---
    star_projected_frames = []
    for player_id, player_df in star_history.sort_values('gameday').groupby('player_id'):
        player_teams = snap_history[snap_history['player_id'] == player_id]['team']
        if len(player_teams) == 0:
            continue
        team = player_teams.iloc[-1]
        team_games = team_game_dates[team_game_dates['team'] == team].sort_values('gameday')
        projected = pd.merge_asof(
            team_games, player_df[['gameday', 'is_star_wr']].sort_values('gameday'),
            on='gameday', direction='backward'
        )
        projected['player_id'] = player_id
        star_projected_frames.append(projected)
    wr_star_projected = pd.concat(star_projected_frames, ignore_index=True).dropna(subset=['is_star_wr'])
    wr_star_projected = wr_star_projected[['player_id', 'game_id', 'is_star_wr']]

    # --- Identify primary WR per team/game using the GAP-FREE projected snap share ---
    primary_wr = (
        wr_snap_projected.sort_values('recent_snap_pct', ascending=False)
        .groupby(['team', 'game_id'], as_index=False)
        .first()[['team', 'game_id', 'player_id']]
    )
    primary_wr = primary_wr.merge(wr_star_projected, on=['player_id', 'game_id'], how='left')
    primary_wr['is_star_wr'] = primary_wr['is_star_wr'].fillna(0)

    # --- Injury check: Out or Doubtful ONLY (Questionable players often still play) ---
    wr_injuries = injuries[injuries['position'] == 'WR'].copy()
    wr_injuries['injury_flag_tight'] = wr_injuries['report_status'].isin(['Out', 'Doubtful']).astype(int)
    wr_injuries = wr_injuries.rename(columns={'gsis_id': 'player_id'})

    primary_wr = primary_wr.merge(df_full[['game_id', 'season', 'week']].drop_duplicates(), on='game_id', how='left')
    primary_wr = primary_wr.merge(
        wr_injuries[['season', 'week', 'team', 'player_id', 'injury_flag_tight']],
        on=['season', 'week', 'team', 'player_id'], how='left')
    primary_wr['injury_flag_tight'] = primary_wr['injury_flag_tight'].fillna(0).astype(int)

    primary_wr['star_wr_injured'] = (
        (primary_wr['is_star_wr'] == 1) & (primary_wr['injury_flag_tight'] == 1)
    ).astype(int)

    home_star_wr = primary_wr[['season', 'week', 'team', 'star_wr_injured']].rename(
        columns={'team': 'home_team_std', 'star_wr_injured': 'home_star_wr_injured'})
    away_star_wr = primary_wr[['season', 'week', 'team', 'star_wr_injured']].rename(
        columns={'team': 'away_team_std', 'star_wr_injured': 'away_star_wr_injured'})

    df_full = df_full.drop(columns=['home_star_wr_injured', 'away_star_wr_injured'], errors='ignore')
    df_full = df_full.merge(home_star_wr, on=['season', 'week', 'home_team_std'], how='left')
    df_full = df_full.merge(away_star_wr, on=['season', 'week', 'away_team_std'], how='left')
    df_full['home_star_wr_injured'] = df_full['home_star_wr_injured'].fillna(0).astype(int)
    df_full['away_star_wr_injured'] = df_full['away_star_wr_injured'].fillna(0).astype(int)

    return df_full