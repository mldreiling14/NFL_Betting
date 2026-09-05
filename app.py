import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from predict import load_model, build_snapshots, predict_week, get_injury_report
import nflreadpy as nfl

st.set_page_config(page_title="NFL Win Probability", page_icon="🏈", layout="centered")

st.markdown("""
<link rel="apple-touch-icon" href="https://em-content.zobj.net/source/apple/391/american-football_1f3c8.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="NFL Win Probability">
""", unsafe_allow_html=True)


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
    colors2 = dict(zip(teams['team_abbr'], teams['team_color2']))
    logos = dict(zip(teams['team_abbr'], teams['team_logo_espn']))
    return colors, colors2, logos


def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _color_distance(hex1, hex2):
    r1, g1, b1 = _hex_to_rgb(hex1)
    r2, g2, b2 = _hex_to_rgb(hex2)
    return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5


def get_display_colors(home_team, away_team, colors, colors2):
    home_opts = [colors.get(home_team, '#888888'), colors2.get(home_team, '#888888')]
    away_opts = [colors.get(away_team, '#888888'), colors2.get(away_team, '#888888')]
    best_pair, best_score = (home_opts[0], away_opts[0]), -1
    for h in home_opts:
        for a in away_opts:
            dist = _color_distance(h, a)
            if dist > best_score:
                best_score, best_pair = dist, (h, a)
    return best_pair


def format_odds(ml):
    if pd.isna(ml):
        return None
    ml = int(ml)
    return f"+{ml}" if ml > 0 else str(ml)


def safe_name(table, team_col, name_col='display_name'):
    """Looks up a player's display name from a snapshot table, handling missing rows/names gracefully."""
    def lookup(team):
        row = table[table[team_col] == team]
        if len(row) == 0 or pd.isna(row[name_col].values[0]):
            return "Unknown"
        return row[name_col].values[0]
    return lookup


team_colors, team_colors2, team_logos = get_team_info()
model_bundle = get_model()
snapshots = get_snapshots()
weeks = get_available_weeks()


def show_list_view():
    st.title("🏈 NFL Win Probability")
    st.caption("Pre-game win probabilities generated from a model trained on 2015–2025 data.")

    if not weeks:
        st.warning("No upcoming games found for the 2026 season.")
        return

    selected_week = st.selectbox("Select week", weeks, index=0)

    with st.spinner(f"Generating predictions for Week {selected_week}..."):
        predictions = predict_week(2026, selected_week, snapshots, model_bundle)

    sched_odds = nfl.load_schedules(seasons=[2026]).to_pandas()
    sched_odds = sched_odds[sched_odds['week'] == selected_week][['game_id', 'home_moneyline', 'away_moneyline']].copy()
    predictions = predictions.merge(sched_odds, on='game_id', how='left')

    if predictions.empty:
        st.warning("No predictions available for this week.")
        return

    for _, game in predictions.iterrows():
        home_color, away_color = get_display_colors(game['home_team'], game['away_team'], team_colors, team_colors2)
        away_pct, home_pct = game['away_win_prob'], game['home_win_prob']

        row_html = f"""
        <div style="background:var(--surface-1); border:1px solid var(--border); border-radius:10px; padding:12px 16px; margin-bottom:8px;">
            <div style="display:flex; align-items:center; justify-content:space-between;">
                <div style="display:flex; align-items:center; gap:8px; flex:1;">
                    <img src="{team_logos.get(game['away_team'],'')}" style="width:24px; height:24px; object-fit:contain;">
                    <span style="font-weight:500;">{game['away_team']}</span>
                    <span style="font-family:var(--font-mono); color:{away_color};">{away_pct:.0%}</span>
                </div>
                <span style="color:var(--text-muted); font-size:12px;">@</span>
                <div style="display:flex; align-items:center; gap:8px; flex:1; justify-content:flex-end;">
                    <span style="font-family:var(--font-mono); color:{home_color};">{home_pct:.0%}</span>
                    <span style="font-weight:500;">{game['home_team']}</span>
                    <img src="{team_logos.get(game['home_team'],'')}" style="width:24px; height:24px; object-fit:contain;">
                </div>
            </div>
            <div style="display:flex; width:100%; height:8px; border-radius:4px; overflow:hidden; margin-top:10px;">
                <div style="width:{away_pct*100}%; background:{away_color}; border-right:1px solid var(--surface-1);"></div>
                <div style="width:{home_pct*100}%; background:{home_color};"></div>
            </div>
        </div>
        """
        st.markdown(row_html, unsafe_allow_html=True)

        if st.button(f"View report: {game['away_team']} @ {game['home_team']}", key=game['game_id'], use_container_width=True):
            st.query_params['game_id'] = game['game_id']
            st.query_params['week'] = str(selected_week)
            st.rerun()


def show_detail_view(game_id, week):
    predictions = predict_week(2026, int(week), snapshots, model_bundle)
    game = predictions[predictions['game_id'] == game_id]

    if game.empty:
        st.warning("Game not found.")
        if st.button("← Back"):
            st.query_params.clear()
            st.rerun()
        return

    game = game.iloc[0]

    if st.button("← Back to all games"):
        st.query_params.clear()
        st.rerun()

    home_color, away_color = get_display_colors(game['home_team'], game['away_team'], team_colors, team_colors2)
    away_pct, home_pct = game['away_win_prob'], game['home_win_prob']
    favorite = game['home_team'] if home_pct > away_pct else game['away_team']
    favorite_color = home_color if favorite == game['home_team'] else away_color

    st.markdown(f"## {game['away_team']} @ {game['home_team']}")
    st.caption(pd.to_datetime(game['gameday']).strftime('%A, %B %d, %Y'))

    col1, col2 = st.columns(2)
    with col1:
        st.image(team_logos.get(game['away_team'], ''), width=60)
        st.metric(game['away_team'], f"{away_pct:.0%}")
    with col2:
        st.image(team_logos.get(game['home_team'], ''), width=60)
        st.metric(game['home_team'], f"{home_pct:.0%}")

    bar_html = f"""
    <div style="display:flex; width:100%; height:20px; border-radius:6px; overflow:hidden; margin:12px 0;">
        <div style="width:{away_pct*100}%; background:{away_color}; border-right:2px solid var(--surface-1);"></div>
        <div style="width:{home_pct*100}%; background:{home_color};"></div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)
    st.markdown(
        f"<div style='text-align:center;'><span style='background:var(--surface-2); border-left:3px solid "
        f"{favorite_color}; padding:3px 12px; border-radius:6px;'>⭐ Favorite: {favorite}</span></div>",
        unsafe_allow_html=True
    )

    has_vegas = pd.notna(game.get('vegas_home_prob'))
    if has_vegas:
        st.caption(f"🎰 Vegas: {game['away_team']} {format_odds(game['away_moneyline'])} ({game['vegas_away_prob']:.0%}) — "
                   f"{game['home_team']} {format_odds(game['home_moneyline'])} ({game['vegas_home_prob']:.0%})")
    else:
        st.caption("🎰 Vegas odds not yet published for this game")

    st.divider()
    st.subheader("🔑 Key Players")

    from predict import get_qb_game_log

    qb_name = safe_name(snapshots['current_qb'], 'team')
    rb_name = safe_name(snapshots['current_rb_starter'], 'team')
    wr_name = safe_name(snapshots['current_primary_wr'], 'team')

    for team in [game['away_team'], game['home_team']]:
        st.markdown(f"**{team}**")
        st.caption(f"QB: {qb_name(team)} · RB1: {rb_name(team)} · WR1: {wr_name(team)}")

        qb_row = snapshots['current_qb'][snapshots['current_qb']['team'] == team]
        if len(qb_row) > 0 and 'player_id' in qb_row.columns and pd.notna(qb_row.iloc[0].get('player_id')):
            qb_player_id = qb_row.iloc[0]['player_id']
            log = get_qb_game_log(qb_player_id, snapshots['qb_stats_raw'], snapshots['df_full'])
            if not log.empty:
                with st.expander(f"📋 {qb_name(team)} — last {len(log)} games"):
                    st.dataframe(log, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("🛡️ Key Defensive Players")

    for team in [game['away_team'], game['home_team']]:
        st.markdown(f"**{team}**")

        rusher_row = snapshots['current_pass_rusher'][snapshots['current_pass_rusher']['team'] == team]
        cb_row = snapshots['current_primary_cb'][snapshots['current_primary_cb']['team'] == team]

        if len(rusher_row) > 0 and pd.notna(rusher_row['display_name'].values[0]):
            star_tag = " ⭐" if rusher_row['is_star_rusher'].values[0] == 1 else ""
            st.caption(f"Top pass rusher: {rusher_row['display_name'].values[0]}{star_tag}")
        else:
            st.caption("Top pass rusher: Unknown")

        if len(cb_row) > 0 and pd.notna(cb_row['display_name'].values[0]):
            completion_allowed = cb_row['recent_def_completion_pct'].values[0]
            st.caption(f"Top corner: {cb_row['display_name'].values[0]} ({completion_allowed:.0%} completion allowed)")
        else:
            st.caption("Top corner: Unknown")

    st.divider()
    st.subheader("🏥 Injury Report")

    for team in [game['away_team'], game['home_team']]:
        report = get_injury_report(team, 2026, int(week))
        st.markdown(f"**{team}**")
        if report is None:
            st.caption("Injury reports not yet published for the 2026 season.")
        elif len(report) == 0:
            st.caption("No injury designations reported for this game.")
        else:
            for entry in report:
                st.caption(f"{entry['full_name']} ({entry['position']}) — {entry['report_status']}")


if 'game_id' in st.query_params:
    show_detail_view(st.query_params['game_id'], st.query_params.get('week', weeks[0] if weeks else 1))
else:
    show_list_view()

st.divider()
st.caption(
    "Model accuracy: ~68% on 2024–2025 test data, vs. ~69% for Vegas closing lines on the same games. "
    "Predictions use each team's current depth-chart starters. Injury data reflects official reports when published."
)