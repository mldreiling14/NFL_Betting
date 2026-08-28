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


def add_defense_features(df_full, player_stats, window=5):
    """
    Adds rolling team-level defensive performance features (sacks,
    interceptions, tackles for loss, QB hits, forced fumbles, passes
    defended) for both home and away teams. Aggregates all defensive
    players on a team into one team-level signal per game, then
    computes a rolling average over that team's prior games (no
    leakage), reset at each season boundary.

    player_stats must include: game_id, season, week, team, position,
    def_sacks, def_interceptions, def_tackles_for_loss, def_qb_hits,
    def_fumbles_forced, def_pass_defended
    """

    def_positions = ['DE', 'DT', 'DL', 'NT', 'LB', 'ILB', 'OLB', 'MLB', 'CB', 'DB', 'S', 'SAF', 'FS']
    stat_cols = ['def_sacks', 'def_interceptions', 'def_tackles_for_loss',
                 'def_qb_hits', 'def_fumbles_forced', 'def_pass_defended']

    def_stats = player_stats[player_stats['position'].isin(def_positions)].copy()
    def_stats = def_stats[['game_id', 'season', 'week', 'team'] + stat_cols]

    def_team_game = def_stats.groupby(['team', 'game_id', 'season', 'week'], as_index=False).sum()
    def_team_game = def_team_game.sort_values(['team', 'season', 'week']).reset_index(drop=True)

    for col in stat_cols:
        def_team_game[f'recent_{col}'] = (
            def_team_game.groupby(['team', 'season'])[col]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        )

    df_full = standardize_team_codes(df_full)

    def_recent_cols = ['team', 'game_id'] + [f'recent_{c}' for c in stat_cols]
    def_recent = def_team_game[def_recent_cols]

    home_def = def_recent.rename(columns={
        'team': 'home_team_std',
        **{f'recent_{c}': f'home_def_recent_{c}' for c in stat_cols}
    })
    away_def = def_recent.rename(columns={
        'team': 'away_team_std',
        **{f'recent_{c}': f'away_def_recent_{c}' for c in stat_cols}
    })

    df_full = df_full.drop(columns=[c for c in df_full.columns if 'def_recent' in c], errors='ignore')
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