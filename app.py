import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from predict import load_model, build_snapshots, predict_week
import nflreadpy as nfl

st.set_page_config(page_title="NFL Win Probability", page_icon="🏈", layout="centered")

st.title("🏈 NFL Win Probability")
st.caption("Pre-game win probabilities generated from a model trained on 2015–2025 data.")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data(ttl=3600 * 6)
def get_snapshots():
    return build_snapshots()


@st.cache_data(ttl=3600 * 6)
def get_available_weeks():
    sched = nfl.load_schedules(seasons=[2026]).to_pandas()
    upcoming = sched[sched['home_score'].isna()]
    return sorted(upcoming['week'].unique().tolist())


@st.cache_data(ttl=3600 * 24)
def get_team_info():
    teams = nfl.load_teams().to_pandas()
    colors = dict(zip(teams['team_abbr'], teams['team_color']))
    logos = dict(zip(teams['team_abbr'], teams['team_logo_espn']))
    return colors, logos


team_colors, team_logos = get_team_info()

model_bundle = get_model()
snapshots = get_snapshots()
weeks = get_available_weeks()

if not weeks:
    st.warning("No upcoming games found for the 2026 season.")
else:
    selected_week = st.selectbox("Select week", weeks, index=0)

    with st.spinner(f"Generating predictions for Week {selected_week}..."):
        predictions = predict_week(2026, selected_week, snapshots, model_bundle)

    # Attach Vegas odds for display purposes only - not part of the model itself
    def moneyline_to_prob(ml):
        if pd.isna(ml):
            return None
        if ml < 0:
            return -ml / (-ml + 100)
        else:
            return 100 / (ml + 100)

    sched_odds = nfl.load_schedules(seasons=[2026]).to_pandas()
    sched_odds = sched_odds[sched_odds['week'] == selected_week][
        ['game_id', 'home_moneyline', 'away_moneyline']].copy()
    sched_odds['vegas_home_prob'] = sched_odds['home_moneyline'].apply(moneyline_to_prob)
    sched_odds['vegas_away_prob'] = sched_odds['vegas_home_prob'].apply(lambda x: 1 - x if x is not None else None)

    predictions = predictions.merge(
        sched_odds[['game_id', 'home_moneyline', 'away_moneyline', 'vegas_home_prob', 'vegas_away_prob']],
        on='game_id', how='left'
    )

    if predictions.empty:
        st.warning("No predictions available for this week.")
    else:
        # ============ DISPLAY BLOCK STARTS HERE ============
        for _, game in predictions.iterrows():
            with st.container(border=True):
                away_pct = game['away_win_prob']
                home_pct = game['home_win_prob']
                away_color = team_colors.get(game['away_team'], '#888888')
                home_color = team_colors.get(game['home_team'], '#888888')
                away_logo = team_logos.get(game['away_team'])
                home_logo = team_logos.get(game['home_team'])
                favorite = game['home_team'] if home_pct > away_pct else game['away_team']
                favorite_color = home_color if home_pct > away_pct else away_color

                def format_odds(ml):
                    if pd.isna(ml):
                        return None
                    ml = int(ml)
                    return f"+{ml}" if ml > 0 else str(ml)

                has_vegas = pd.notna(game.get('vegas_home_prob'))
                away_odds = format_odds(game['away_moneyline']) if has_vegas else None
                home_odds = format_odds(game['home_moneyline']) if has_vegas else None
                vegas_away = game['vegas_away_prob'] if has_vegas else None
                vegas_home = game['vegas_home_prob'] if has_vegas else None

                col_away, col_div, col_home = st.columns([5, 1, 5])

                with col_away:
                    st.markdown(f"<p style='font-size:12px; color:var(--text-muted); margin:0 0 4px;'>Away</p>", unsafe_allow_html=True)
                    if away_logo:
                        st.image(away_logo, width=40)
                    st.markdown(f"<p style='font-size:20px; font-weight:500; margin:4px 0;'>{game['away_team']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-family:var(--font-mono); font-size:14px; margin:0;'>{away_pct:.0%} win</p>", unsafe_allow_html=True)
                    if has_vegas:
                        st.markdown(f"<p style='font-family:var(--font-mono); font-size:12px; color:var(--text-secondary); margin:0;'>vegas {away_odds} · {vegas_away:.0%}</p>", unsafe_allow_html=True)

                with col_div:
                    st.markdown("<div style='border-left:1px dashed var(--border-strong); height:100px; margin:0 auto;'></div>", unsafe_allow_html=True)

                with col_home:
                    st.markdown(f"<p style='font-size:12px; color:var(--text-muted); margin:0 0 4px; text-align:right;'>Home</p>", unsafe_allow_html=True)
                    if home_logo:
                        st.image(home_logo, width=40)
                    st.markdown(f"<p style='font-size:20px; font-weight:500; margin:4px 0; text-align:right;'>{game['home_team']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-family:var(--font-mono); font-size:14px; margin:0; text-align:right;'>{home_pct:.0%} win</p>", unsafe_allow_html=True)
                    if has_vegas:
                        st.markdown(f"<p style='font-family:var(--font-mono); font-size:12px; color:var(--text-secondary); margin:0; text-align:right;'>vegas {home_odds} · {vegas_home:.0%}</p>", unsafe_allow_html=True)

                bar_html = f"""
                <div style="display:flex; width:100%; height:20px; border-radius:6px; overflow:hidden; margin-top:12px;">
                    <div style="width:{away_pct*100}%; background:{away_color};"></div>
                    <div style="width:{home_pct*100}%; background:{home_color};"></div>
                </div>
                """
                st.markdown(bar_html, unsafe_allow_html=True)

                st.markdown(
                    f"<div style='text-align:center; margin-top:10px;'>"
                    f"<span style='display:inline-block; background:{favorite_color}22; color:{favorite_color}; "
                    f"font-size:12px; padding:3px 12px; border-radius:6px;'>⭐ Favorite: {favorite}</span></div>",
                    unsafe_allow_html=True
                )

                if has_vegas:
                    diff = abs(home_pct - vegas_home)
                    if diff >= 0.10:
                        st.caption(f"⚠️ Model differs from Vegas by {diff:.0%}")
                else:
                    st.caption("Vegas odds not yet published for this game")

                st.caption(f"📅 {pd.to_datetime(game['gameday']).strftime('%a, %b %d, %Y')}")

                               # Click-to-expand game detail
                with st.expander("📊 More details"):
                    home_coach = snapshots['current_coach'].loc[
                        snapshots['current_coach']['team'] == game['home_team'], 'coach']
                    away_coach = snapshots['current_coach'].loc[
                        snapshots['current_coach']['team'] == game['away_team'], 'coach']
                    home_coach = home_coach.values[0] if len(home_coach) > 0 else "Unknown"
                    away_coach = away_coach.values[0] if len(away_coach) > 0 else "Unknown"

                    d1, d2 = st.columns(2)

                    with d1:
                        st.markdown(f"**{game['away_team']}**")
                        st.caption(f"Recent form: {game.get('away_recent_form', 0):.0%} win rate")
                        st.caption(f"Offense rating: {game.get('away_off_rating_pre', 0):+.1f} vs league avg")
                        st.caption(f"Defense rating: {game.get('away_def_rating_pre', 0):+.1f} vs league avg")
                        st.caption(f"Coach: {away_coach}")

                    with d2:
                        st.markdown(f"**{game['home_team']}**")
                        st.caption(f"Recent form: {game.get('home_recent_form', 0):.0%} win rate")
                        st.caption(f"Offense rating: {game.get('home_off_rating_pre', 0):+.1f} vs league avg")
                        st.caption(f"Defense rating: {game.get('home_def_rating_pre', 0):+.1f} vs league avg")
                        st.caption(f"Coach: {home_coach}")

                    st.divider()

                    h2h_games = int(game.get('h2h_games_played', 0))
                    if h2h_games > 0:
                        h2h_wins = game.get('home_coach_h2h_wins', 0.5)
                        st.caption(f"🤝 Coach head-to-head: {home_coach} is {h2h_wins:.0%} vs {away_coach} over {h2h_games} meeting(s)")
                    else:
                        st.caption(f"🤝 Coach head-to-head: {home_coach} and {away_coach} haven't faced each other")

                    rest_diff = game.get('rest_advantage', 0)
                    if rest_diff > 0:
                        st.caption(f"😴 {game['home_team']} has {rest_diff:.0f} more day(s) of rest")
                    elif rest_diff < 0:
                        st.caption(f"😴 {game['away_team']} has {abs(rest_diff):.0f} more day(s) of rest")
                    else:
                        st.caption("😴 Equal rest for both teams")

                    injury_flags_present = any([
                        game.get('home_qb_injury_flag'), game.get('away_qb_injury_flag'),
                        game.get('home_rb_injury_flag'), game.get('away_rb_injury_flag'),
                        game.get('home_wrte_injury_flag'), game.get('away_wrte_injury_flag')
                    ])
                    if injury_flags_present:
                        st.caption("🏥 Injury designations reported for this game")
                    else:
                        st.caption("🏥 No injury designations yet (reports are typically published closer to game day)")
        # ============ DISPLAY BLOCK ENDS HERE ============

st.divider()
st.caption(
    "Model accuracy: ~68.4% on 2024–2025 test data, vs. ~68.9% for Vegas closing lines on the same "
    "games. Predictions use each team's most recent known starters — real, game-week injury news is "
    "not yet incorporated. Weather effects are not currently included."
)