# NFL_Betting

# NFL Win Probability Model — Feature Reference

This document describes every variable in the `games_with_features` table, produced by running:

```bash
python src/build_features.py
```

Source data comes from [`nflreadpy`](https://github.com/nflverse/nflreadpy), covering the 2015–2023 seasons (2458 games).

---

## Core / Identifier Columns

These come directly from the raw schedule data (`fetch_data.py`) and are not engineered features — they identify the game itself.

| Column | Description |
|---|---|
| `game_id` | Unique ID for the game (format: `season_week_awayteam_hometeam`) |
| `season` | Season year |
| `week` | Week number within the season |
| `gameday` | Date the game was played |
| `home_team` / `away_team` | Team abbreviations (original codes, e.g. `SD`, `STL`, `OAK` for pre-relocation seasons) |
| `home_team_std` / `away_team_std` | Standardized team codes — relocated franchises mapped to their current code (`SD`→`LAC`, `STL`→`LA`, `OAK`→`LV`) so they merge correctly against player-stats tables |
| `home_score` / `away_score` | Final score |
| `home_win` | **Target variable.** `1` if the home team won, `0` otherwise |
| `home_coach` / `away_coach` | Head coach name for each team that game |
| `div_game` | `1` if this was a divisional matchup, `0` otherwise |
| `spread_line`, `away_moneyline`, `home_moneyline`, `total_line` | Vegas market lines, useful as a benchmark to compare model performance against |
| `roof`, `surface`, `temp`, `wind` | Stadium/weather conditions (weather feature engineering not yet built as of this doc) |

---

## Engineered Features

All rolling features use only games **before** the current one (no data leakage — verified via `shift(1)` before any rolling calculation). Team-level features reset at each season boundary (roster turnover); QB and coach features carry across seasons (tracking the individual, not the roster).

### Team Recent Form
*Built by: `add_recent_form_features`*

| Column | Description |
|---|---|
| `home_recent_form` / `away_recent_form` | Win % over the team's last 5 games (resets each season) |
| `home_recent_point_diff` / `away_recent_point_diff` | Average scoring margin over the team's last 5 games (resets each season) |

### QB Performance
*Built by: `add_qb_features`*

Tracks the actual starting QB (`home_qb_id` / `away_qb_id`) by player identity — carries across trades and seasons, since it reflects the player, not the team.

| Column | Description |
|---|---|
| `home_qb_recent_yards` / `away_qb_recent_yards` | Avg passing yards over QB's last 5 starts |
| `home_qb_recent_tds` / `away_qb_recent_tds` | Avg passing TDs over QB's last 5 starts |
| `home_qb_recent_ints` / `away_qb_recent_ints` | Avg interceptions thrown over QB's last 5 starts |
| `home_qb_recent_epa` / `away_qb_recent_epa` | Avg Expected Points Added (EPA) per game over QB's last 5 starts — best single efficiency metric |

### RB Group Performance
*Built by: `add_rb_features`*

Aggregates **all RBs on a team** into one team-level signal (not just the lead back), weighted by each player's offensive snap share so bench-player performances count less than starter performances.

| Column | Description |
|---|---|
| `home_rb_recent_rush_yards` / `away_rb_recent_rush_yards` | Snap-share-weighted rushing yards, team RB group, avg over last 5 games |
| `home_rb_recent_rush_epa` / `away_rb_recent_rush_epa` | Snap-share-weighted rushing EPA, team RB group, avg over last 5 games |
| `home_rb_recent_rec_yards` / `away_rb_recent_rec_yards` | Snap-share-weighted receiving yards (RBs catching passes), avg over last 5 games |

### WR/TE Receiving-Corps Performance
*Built by: `add_wrte_features`*

Same approach as RB — combines all WRs and TEs into one team-level receiving signal, snap-share weighted.

| Column | Description |
|---|---|
| `home_wrte_recent_rec_yards` / `away_wrte_recent_rec_yards` | Snap-share-weighted receiving yards, avg over last 5 games |
| `home_wrte_recent_rec_epa` / `away_wrte_recent_rec_epa` | Snap-share-weighted receiving EPA, avg over last 5 games |
| `home_wrte_recent_targets` / `away_wrte_recent_targets` | Snap-share-weighted targets, avg over last 5 games |

### Injuries
*Built by: `add_injury_features`*

Simple binary flags: `1` if the relevant player carried an Out/Doubtful/Questionable designation that week, `0` if healthy or not on the report at all.

| Column | Description |
|---|---|
| `home_qb_injury_flag` / `away_qb_injury_flag` | Starting QB (matched via `home_qb_id`/`away_qb_id`) injury designation |
| `home_rb_injury_flag` / `away_rb_injury_flag` | Primary RB (identified by highest recent snap share) injury designation |
| `home_wrte_injury_flag` / `away_wrte_injury_flag` | Flag if **either** of the team's top-2 recent-snap-share WR/TE is banged up |

### Team Defense
*Built by: `add_defense_features`*

Aggregates all defensive players on a team into one team-level signal per game (sacks, turnovers, pressure), rolling average over last 5 games, resets each season.

| Column | Description |
|---|---|
| `home_def_recent_def_sacks` / `away_def_recent_def_sacks` | Avg team sacks over last 5 games |
| `home_def_recent_def_interceptions` / `away_def_recent_def_interceptions` | Avg team interceptions over last 5 games |
| `home_def_recent_def_tackles_for_loss` / `away_def_recent_def_tackles_for_loss` | Avg tackles for loss over last 5 games |
| `home_def_recent_def_qb_hits` / `away_def_recent_def_qb_hits` | Avg QB hits over last 5 games |
| `home_def_recent_def_fumbles_forced` / `away_def_recent_def_fumbles_forced` | Avg forced fumbles over last 5 games |
| `home_def_recent_def_pass_defended` / `away_def_recent_def_pass_defended` | Avg passes defended over last 5 games |

### Coaching
*Built by: `add_coach_features`*

| Column | Description |
|---|---|
| `home_coach_recent_form` / `away_coach_recent_form` | Coach's own win % over last 5 games (carries across teams/seasons — tracks the person) |
| `home_coach_h2h_wins` | Win rate of the **home team's current coach** in this specific coach-vs-coach matchup, using only meetings before this game. `0.5` (neutral) if the two coaches have never faced each other |
| `h2h_games_played` | Number of prior meetings between these two specific coaches |

---

## Known Data Quality Notes

- **Relocated franchises**: `SD`→`LAC`, `STL`→`LA`, `OAK`→`LV`. Schedule data (`fetch_data.py`) uses the historical code; player-stats tables (`nflreadpy`) use the current code. Always merge player-level data using `home_team_std`/`away_team_std`, never the raw `home_team`/`away_team`.
- **Missing values**: Week 1 of each season has `NaN` for team-level rolling features (no prior-season history, by design — resets each season). QB/coach features may show `NaN` for a player's/coach's first-ever appearance in the dataset. Current modeling approach fills these with column means (`SimpleImputer`, `strategy='mean'`), except `home_coach_h2h_wins`, which fills with a neutral `0.5`.
- **home_win** is baked directly into the `games` table (via `fetch_data.py`) — do not recreate it manually after loading from SQLite.

---

## Not Yet Built (as of this document)

- Weather-adjusted features (dome vs. outdoor handling)
- Special teams performance