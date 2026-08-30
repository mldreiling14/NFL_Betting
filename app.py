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
def get_team_colors():
    teams = nfl.load_teams().to_pandas()
    return dict(zip(teams['team_abbr'], teams['team_color']))


team_colors = get_team_colors()

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
                col1, col2, col3 = st.columns([2, 3, 2])

                away_color = team_colors.get(game['away_team'], '#888888')
                home_color = team_colors.get(game['home_team'], '#888888')
                away_pct = game['away_win_prob']
                home_pct = game['home_win_prob']

                with col1:
                    st.markdown(f"**{game['away_team']}**")
                    st.caption("Away")

                with col2:
                    favorite = game['home_team'] if home_pct > away_pct else game['away_team']

                    bar_html = f"""
                    <div style="display:flex; width:100%; height:24px; border-radius:6px; overflow:hidden;">
                        <div style="width:{away_pct*100}%; background:{away_color};"></div>
                        <div style="width:{home_pct*100}%; background:{home_color};"></div>
                    </div>
                    """
                    st.markdown(bar_html, unsafe_allow_html=True)
                    st.caption(f"{game['away_team']} {away_pct:.0%} — {home_pct:.0%} {game['home_team']}")
                    st.markdown(f"Favorite: **{favorite}**")

                    # Vegas comparison, if odds are already published for this game
                    if pd.notna(game.get('vegas_home_prob')):
                        vegas_away = game['vegas_away_prob']
                        vegas_home = game['vegas_home_prob']

                        def format_odds(ml):
                            if pd.isna(ml):
                                return ""
                            ml = int(ml)
                            return f"+{ml}" if ml > 0 else str(ml)

                        away_odds = format_odds(game['away_moneyline'])
                        home_odds = format_odds(game['home_moneyline'])

                        diff = abs(home_pct - vegas_home)
                        st.caption(f"🎰 Vegas: {game['away_team']} {vegas_away:.0%} ({away_odds}) — {vegas_home:.0%} ({home_odds}) {game['home_team']}")
                        if diff >= 0.10:
                            st.caption(f"⚠️ Model differs from Vegas by {diff:.0%}")
                    else:
                        st.caption("🎰 Vegas odds not yet published for this game")

                with col3:
                    st.markdown(f"**{game['home_team']}**")
                    st.caption("Home")

                st.caption(f"📅 {pd.to_datetime(game['gameday']).strftime('%a, %b %d, %Y')}")
        # ============ DISPLAY BLOCK ENDS HERE ============

st.divider()
st.caption(
    "Model accuracy: ~68.4% on 2024–2025 test data, vs. ~68.9% for Vegas closing lines on the same "
    "games. Predictions use each team's most recent known starters — real, game-week injury news is "
    "not yet incorporated. Weather effects are not currently included."
)