# NFL Win Probability Model — Feature Reference

This document describes every variable in the `games_with_features` table, produced by running:

```bash
python src/fetch_data.py
python src/build_features.py
```

Source data comes from [`nflreadpy`](https://github.com/nflverse/nflreadpy), covering the 2015–2025 seasons (3028 games, 112 columns as of this revision). The 2026 schedule is also available for live predictions via `src/predict.py`.

---

## Plain-Language Glossary

A few terms used throughout this document, explained simply:

- **EPA (Expected Points Added)**: a single number that captures how much a play (or a player's average play) helped or hurt a team's chances of scoring, compared to what's typically expected in that situation. A short run on 3rd-and-1 that gets the first down is worth more EPA than the same run on 3rd-and-10 that doesn't. Positive EPA = the play/player helped; negative = it hurt. It's considered a better measure of "how well did this really go" than simpler stats like yards, because it accounts for context (down, distance, field position).
- **Elo rating**: a single number representing how strong a team currently is. Every team starts at 1500. Beating a strong team raises your rating more than beating a weak one; losing to a weak team lowers it more than losing to a strong one. So Elo naturally reflects *quality of opponent*, not just a team's raw win/loss record.
- **Rolling average (e.g., "last 5 games")**: instead of using a team or player's full-season or career stats, this looks only at their most recent handful of games — a way of asking "how are they playing lately," not "how have they done ever."
- **Snap share**: what percentage of a team's offensive (or defensive) plays a specific player was on the field for. A player at 90% snap share is essentially always playing; a player at 20% is a rotational/backup role.
- **Correlation**: a number between -1 and +1 that measures how strongly two things move together. A value near 0 means "no real relationship." A value near +1 means "when this goes up, so does the other thing" (e.g., a higher Elo rating tends to go with the home team winning). A value near -1 means "when this goes up, the other thing tends to go down" (e.g., a higher *away* team Elo tends to go with the home team *losing*). None of the individual correlations in this project are very close to ±1 — the biggest is about 0.19 — which reflects a simple truth about the NFL: no single stat predicts outcomes strongly on its own, because the sport has a lot of inherent unpredictability.
- **Leakage-free / no data leakage**: a rule followed throughout this project that a feature describing "how a team has been playing" must only use games that happened *before* the game being predicted — never the game's own result. This is what makes the model's back-tested accuracy trustworthy rather than artificially inflated.

---

## Model Performance Summary

Evaluated with a chronological train/test split to avoid lookahead bias, using logistic regression.

| Version | Test window | Test Accuracy | Vegas (same window) |
|---|---|---|---|
| Naive baseline (always guess home team wins) | — | 55.0% | — |
| Original feature set (team form, QB, RB, WR/TE, injuries) | 2022–2023 | 60.1% | 66.3% |
| + Elo ratings | 2022–2023 | 62.7% | 66.3% |
| + Rest advantage | 2022–2023 | 62.9% | 66.3% |
| Same feature set, extended training data (2015–2023 train) | 2024–2025 | 67.2% | 68.9% |
| + Sack rate + QB pressure rate | 2024–2025 | 67.7% | 68.9% |
| + DB/WR size & coverage matchup | 2024–2025 | **68.4%** | 68.9% |
| + Weather + climate shock | 2024–2025 | 68.1% | 68.9% |

**Current best model (68.4%, without weather) is within 0.5 points of Vegas** on the most recent test window. Weather/climate-shock features are included in the pipeline (`games_with_features`) but excluded from the production model — see the Weather section below.

Note the 2022–2023 vs. 2024–2025 accuracy jump reflects that those seasons were genuinely more predictable overall (Vegas's own accuracy rose too, from 66.3% to 68.9%), not solely a model improvement — always compare your model against Vegas on the *same* test window.

When your model and Vegas disagree (~24% of games, measured on the 2022–2023 window), Vegas is correct roughly 63% of the time vs. your model's 37%. The model has not been found to identify a systematic edge the market misses — treat it as a legitimate, well-validated probability estimate, not a beat-the-market tool.

**A live, real-world example of this limitation:** comparing the model's Week 1 2026 predictions against already-published Vegas lines for the same games showed most predictions within about 13 points of Vegas, but two (KC vs. DEN, LV vs. MIA) differed by 30+ points. Investigating KC's case traced the gap to a real weakness in the live-prediction pipeline: it had picked up Kansas City's emergency third-string QB (who finished out a late-2025 game after the real starter was sidetracked) as the team's "current" quarterback, rather than their actual intended starter. This is being addressed by switching the live pipeline to use official team depth charts (identifying the *intended* starter) rather than "whoever played most recently" — a good illustration of why this project treats live predictions as a reasoned estimate, not a guarantee.

---

## Correlation with Winning (`home_win`)

Simple correlation between each feature and the outcome, calculated across the full dataset (excluding the outcome itself and anything directly derived from it, like the final score). This measures each variable's *individual*, standalone relationship with winning — it does not account for overlap between features (e.g., Elo and recent form partly capture similar things, so both can show meaningful correlation without both being equally "new" information once combined in the model).

| Feature | Correlation with home team winning |
|---|---|
| `home_elo_pre` | **+0.19** (strongest in the dataset) |
| `away_elo_pre` | -0.19 |
| `home_recent_point_diff` | +0.18 |
| `home_recent_form` | +0.16 |
| `away_qb_recent_epa` | -0.15 |
| `home_qb_recent_epa` | +0.15 |
| `away_recent_point_diff` | -0.14 |
| `away_recent_form` | -0.14 |
| `home_qb_recent_tds` | +0.12 |
| `home_qb_recent_yards` | +0.11 |
| `home_rb_recent_rush_epa` | +0.09 |
| `home_coach_h2h_wins` | +0.08 |
| `home_takeaways_recent` | +0.08 |
| `home_epa_allowed_recent` | -0.08 |
| `home_yards_allowed_recent` | -0.08 |
| `home_rb_recent_rush_yards` | +0.06 |
| `home_qb_recent_ints` | -0.06 |

**How to read this table honestly:** every value here is fairly modest — the strongest (Elo) is under 0.2, and most are well under 0.15. This isn't a weakness of the analysis; it's an accurate reflection of how unpredictable NFL outcomes genuinely are. No individual pre-game stat comes close to reliably determining who wins — that's exactly why even Vegas, with vastly more data and resources, only reaches about 66-69% accuracy rather than something close to certainty.

---

## Core / Identifier Columns

From the raw schedule data (`fetch_data.py`) — not engineered, just identify the game itself.

| Column | Description |
|---|---|
| `game_id` | Unique ID for the game (format: `season_week_awayteam_hometeam`) |
| `season` / `week` / `gameday` | Season year, week number, game date |
| `home_team` / `away_team` | Team abbreviations, **original/historical codes** (e.g. `SD`, `STL`, `OAK` pre-relocation) |
| `home_team_std` / `away_team_std` | Standardized codes — relocated franchises mapped to current code (`SD`→`LAC`, `STL`→`LA`, `OAK`→`LV`). Used for merging against `load_team_stats`-sourced data. **Not** used for PFR-sourced data — see the team-code convention warning below |
| `home_score` / `away_score` | Final score |
| `home_win` | **Target variable.** `1` if home team won. Baked into `games` table by `fetch_data.py` — never recreate manually after loading from SQLite |
| `home_coach` / `away_coach` | Head coach names |
| `home_rest` / `away_rest` | Days of rest since each team's previous game |
| `div_game` | `1` if divisional matchup |
| `spread_line`, `away_moneyline`, `home_moneyline`, `total_line` | Vegas market lines — benchmark only, not fed into the model |
| `roof`, `surface`, `temp`, `wind` | Raw stadium/weather conditions — see Weather section for the engineered versions |

### ⚠️ Team-code convention warning

Different `nflreadpy` tables use **inconsistent conventions** for relocated franchises (`OAK`/`LV`, `SD`/`LAC`, `STL`/`LA`). Always verify which convention a new table uses before merging — do not assume:

- **`load_team_stats`, `load_player_stats`, `load_snap_counts`**: backfill to the **current** code for all historical seasons (always `LV`, even for a 2019 game). Merge using `home_team_std`/`away_team_std`.
- **`load_pfr_advstats`** (both `stat_type='pass'` and `'def'`): uses the **actual code at the time** (`OAK` through 2019, `LV` from 2020 on). Merge using the raw `home_team`/`away_team`.

This was caught the hard way twice — first via a relocation-era `NaN` check (RB feature), second via the opposite failure mode (pressure rate initially showed `NaN` for a 2019 OAK game *because* `standardize_team_codes` had been applied where it shouldn't have been).

---

## Engineered Features (Currently in the Model)

All rolling features use only games **before** the current one (`shift(1)` before any rolling calculation — verified leakage-free on every feature). Team-level features reset each season boundary; QB, coach-h2h, and Elo carry across seasons.

### Team Recent Form — `add_recent_form_features`
| Column | Description | Correlation |
|---|---|---|
| `home_recent_form` / `away_recent_form` | Win % over last 5 games (resets each season) | +0.16 / -0.14 |
| `home_recent_point_diff` / `away_recent_point_diff` | Avg scoring margin over last 5 games (resets each season) | +0.18 / -0.14 |

### QB Performance — `add_qb_features`
Tracks the individual starting QB by `player_id` — carries across trades/seasons (verified: Case Keenum's rating followed him across 6 teams).
| Column | Description | Correlation |
|---|---|---|
| `home_qb_recent_yards` / `away_qb_recent_yards` | Avg passing yards, last 5 starts | +0.11 / -0.11 |
| `home_qb_recent_tds` / `away_qb_recent_tds` | Avg passing TDs, last 5 starts | +0.12 / -0.13 |
| `home_qb_recent_ints` / `away_qb_recent_ints` | Avg INTs thrown, last 5 starts | -0.06 / +0.03 |
| `home_qb_recent_epa` / `away_qb_recent_epa` | Avg EPA (see glossary), last 5 starts — strongest single QB metric by correlation | +0.15 / -0.15 |

### RB Group Performance — `add_rb_features`
Aggregates all RBs on a team, snap-share weighted (see glossary).
| Column | Description | Correlation |
|---|---|---|
| `home_rb_recent_rush_yards` / `away_rb_recent_rush_yards` | Weighted rushing yards, avg last 5 games | +0.06 / — |
| `home_rb_recent_rush_epa` / `away_rb_recent_rush_epa` | Weighted rushing EPA, avg last 5 games | +0.09 / — |
| `home_rb_recent_rec_yards` / `away_rb_recent_rec_yards` | Weighted receiving yards, avg last 5 games | — |

### WR/TE Receiving-Corps — `add_wrte_features`
Combines all WRs and TEs, snap-share weighted.
| Column | Description | Correlation |
|---|---|---|
| `home_wrte_recent_rec_yards` / `away_wrte_recent_rec_yards` | Weighted receiving yards, avg last 5 games | +0.08 / -0.09 |
| `home_wrte_recent_rec_epa` / `away_wrte_recent_rec_epa` | Weighted receiving EPA, avg last 5 games | +0.13 / -0.13 |
| `home_wrte_recent_targets` / `away_wrte_recent_targets` | Weighted targets, avg last 5 games | — |

### Injuries — `add_injury_features`
Binary flags: `1` = Out/Doubtful/Questionable that week, `0` = healthy/not listed.
| Column | Description |
|---|---|
| `home_qb_injury_flag` / `away_qb_injury_flag` | Actual starting QB's designation (~2.6% of games) |
| `home_rb_injury_flag` / `away_rb_injury_flag` | Primary RB (by recent snap share) designation (~5.6%) |
| `home_wrte_injury_flag` / `away_wrte_injury_flag` | Either of top-2 WR/TE (by recent snap share) designation (~12%) |

### Team Defense — EPA/Yards Allowed & Takeaways — `add_defense_allowed_features`
Measures defense by opponent's own offensive output that game (self-join on `game_id`/`opponent_team`).
| Column | Description | Correlation |
|---|---|---|
| `home_epa_allowed_recent` / `away_epa_allowed_recent` | Avg opponent passing+rushing EPA allowed, last 5 games | -0.08 / — |
| `home_yards_allowed_recent` / `away_yards_allowed_recent` | Avg opponent yards allowed, last 5 games | -0.08 / — |
| `home_takeaways_recent` / `away_takeaways_recent` | Avg takeaways forced (INTs + fumble recoveries), last 5 games | +0.08 / — |

### Coaching — `add_coach_features`
| Column | Description | Correlation |
|---|---|---|
| `home_coach_h2h_wins` | Home team's current coach's win rate in this specific coach-vs-coach matchup, prior meetings only. `0.5` if never met | +0.08 |
| `h2h_games_played` | Number of prior meetings between these two coaches | — |

*(`home_coach_recent_form`/`away_coach_recent_form` are computed internally but excluded from the model — see Retired Features)*

### Elo Ratings — `add_elo_features`
**Strongest single addition; highest individual correlation with `home_win` of any feature.** See glossary above for what Elo means in plain terms. Accounts for opponent strength, unlike simple win %. K-factor 20, home-field worth 65 Elo points, reverts 1/3 toward 1500 each season.
| Column | Description | Correlation |
|---|---|---|
| `home_elo_pre` / `away_elo_pre` | Each team's Elo rating entering this game | **+0.19** / -0.19 |

Verified: 2016 Cleveland (1-15) settles near 1364; 2018 New England (Super Bowl champs) settles near 1600.

### Rest Advantage — `add_rest_advantage`
| Column | Description |
|---|---|
| `rest_advantage` | `home_rest - away_rest`. Small, real, consistent contribution (+0.2pts isolated in model testing) |

### O-Line / Pass Protection — `add_oline_features`
Two metrics, two different source-table conventions (see warning above) and two different date ranges.
| Column | Description |
|---|---|
| `home_sack_rate_recent` / `away_sack_rate_recent` | Sacks per dropback, avg last 5 games. From `load_team_stats` (full 2015+ range, standardized codes) |
| `home_pressure_pct_recent` / `away_pressure_pct_recent` | QB pressure rate, avg last 5 games. From `load_pfr_advstats` (2018+ only, **raw** team codes) |

Together, +0.5pts on top of Elo/rest-advantage baseline in model testing. Kept even though the accuracy gain is modest, per deliberate choice for conceptual completeness.

### DB/WR Size & Coverage Matchup — `add_db_wr_matchup_features`
**Second-strongest addition tested** (+0.7pts in model testing). Approximates each team's primary WR (highest recent snap share) against the **opposing team's** primary CB (highest recent snap share), comparing height/weight and the CB's real recent coverage stats. 2018+ only (PFR limitation).

⚠️ **Real limitation:** NFL data does not publish actual man-coverage assignments — this is "team's best WR vs. opponent's most-used CB," an approximation, not a confirmed real matchup.

⚠️ **Bug history:** the CB-identification step originally used a `left` merge between `adv_def_full` and CB-only snap data, which let non-CB defenders (anyone who recorded any pass-defense stat that game) into the "primary CB" candidate pool. On ties (every Week 1, before rolling history exists), a non-CB could win the selection — this produced a 300lb "cornerback" who was actually a defensive end (Jonathan Allen), caught via a physical-plausibility check (real CB weights run ~170–225 lbs). **Fixed with an `inner` merge.**

| Column | Description |
|---|---|
| `home_wr_height_advantage` / `away_wr_height_advantage` | Team's primary WR height minus opposing primary CB height (inches) |
| `home_wr_weight_advantage` / `away_wr_weight_advantage` | Same, weight (lbs) |
| `home_opp_cb_completion_allowed` / `away_opp_cb_completion_allowed` | Opposing primary CB's recent completion % allowed, last 5 games |
| `home_opp_cb_rating_allowed` / `away_opp_cb_rating_allowed` | Opposing primary CB's recent passer rating allowed, last 5 games |

### Weather & Climate Shock — `add_weather_features`
**Built and tested, but excluded from the production model** (small net accuracy cost, -0.3pts in testing) — see reasoning below.

| Column | Description |
|---|---|
| `is_outdoor` | `1` if roof is `outdoors` or `open`, else `0` |
| `temp_adj` / `wind_adj` | Real temp/wind for outdoor games; **neutral fill (70°F, 0 wind) for dome/closed games** — the actual real condition for a climate-controlled dome, not an estimate |
| `cold_game` | `1` if outdoor and `temp_adj` ≤ 32°F |
| `high_wind_game` | `1` if outdoor and `wind_adj` ≥ 15mph |
| `away_home_climate` | Visiting team's typical home outdoor temperature (their "normal") |
| `climate_shock` | `away_home_climate − temp_adj`, only for outdoor games. Large positive = visiting team facing much colder conditions than they're used to |
| `cold_shock_game` | `1` if `climate_shock` ≥ 25°F and outdoor — e.g. Miami/Tampa/New Orleans traveling to Buffalo/KC/Cleveland/Green Bay in winter |

**Why kept in the pipeline despite the accuracy cost:** `cold_shock_game` shows a real, meaningful split — home teams win **61.6%** of cold-shock games (n=219) vs. **54.8%** overall — directionally exactly as expected. However, this affects too small a slice of all games (~7%) to reliably move net logistic-regression accuracy once combined with 25+ other features. A genuine example of "real effect, not enough statistical power to net out as a model improvement." Plan: surface as an informational badge in the app for the ~7% of games it applies to, without using it as a model input.

---

## Retired / Rejected Features

Built, verified correct, tested against the model — did not improve accuracy. Documented so this ground isn't re-covered.

| Feature | Result |
|---|---|
| Raw defense counting stats (sacks, tackles for loss, etc.) — old `add_defense_features` | Near-zero coefficients; replaced by EPA/yards allowed |
| `home_coach_recent_form` / `away_coach_recent_form` | 91% correlated with `home_recent_form` — redundant |
| Turnover margin / giveaways alone | +0.2pts, within noise |
| Third-down conversion rate (from play-by-play) | -0.7pts (likely noise) |
| `elo_diff` (combined home−away Elo column) | 0.0pts — logistic regression already derives this from the two separate inputs |

---

## Known Data Quality Notes

- **Relocated franchises**: see the team-code convention warning above.
- **Missing values**: Week 1 of each season → `NaN` for team-level rolling features (by design). Pre-2018 → `NaN` for all PFR-sourced features — a real, permanent data-availability gap, not a bug. Filled with column means (`SimpleImputer`), except `home_coach_h2h_wins` (neutral `0.5`).
- **"Primary player" selection ties**: any "pick the top-N by recent metric" selection can have ties, most commonly in each team's first game of a data range. Always filter the candidate pool to the *correct position/group* with an `inner` merge before selecting.
- **Live snapshot aggregation order matters**: for team-level features built from individual players (RB, WR/TE), the live prediction pipeline must aggregate to one row per team-game *before* computing a rolling average — rolling directly on individual player rows produces silently wrong, non-error-raising results. Caught and fixed in both the RB and WR/TE live snapshots.
- **"Most recent starter" can be misleading**: a real, live example — Kansas City's 2025 season ended with backup/third-string QBs finishing games after the actual starter was sidelined, causing a naive "who started most recently" lookup to badly misidentify KC's real starting QB for 2026 projections. Being addressed by using official team depth charts (`load_depth_charts`, rank 1) instead.
- **Vegas comparison caveat**: implied probabilities from moneylines aren't de-vigged/normalized — directionally correct, not a perfectly calibrated "true" probability.
- **"Real effect, no net accuracy gain" is a distinct category**: don't conflate "didn't improve accuracy" with "wasn't a real football phenomenon" (see weather/climate-shock).

---

## Not Yet Built (as of this revision)

- Depth-chart-based QB (and RB) identification for live predictions (in progress)
- Cold-shock badge in the app UI
- Special teams performance
- Live/in-game win probability (a fundamentally different model, using play-by-play data)
- Deployment for public sharing