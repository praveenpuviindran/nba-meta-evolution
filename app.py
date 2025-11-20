import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Configuration
st.set_page_config(page_title="NBA Meta Evolution", layout="wide")
st.title("🏀 Moneyball 2.0: The Evolution of NBA Archetypes")
st.markdown("### Quantifying the shift from 'Positions' to 'Playstyles' (2014-2025)")

# 2. Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('data/gmm_archetypes.csv')
    return df

df = load_data()

# 3. Define Archetype Labels (Based on your GMM Output)
archetype_map = {
    6: "Heliocentric Stars (Luka/SGA)",
    0: "Movement Snipers (Klay/Joe)",
    7: "Scoring Wings (Powell/Vassell)",
    5: "Versatile Bigs (Jokic/Draymond)",
    8: "Dominant Bigs (Giannis/Embiid)",
    3: "Defensive Wings (Vanderbilt)",
    2: "Rim Runners (Capela/Allen)",
    4: "Raw Athletic Bigs (Ayton)",
    9: "Veteran Guards (Conley/Mills)",
    1: "Traditional Facilitators (CP3)"
}

df['Archetype Label'] = df['ARCHETYPE_ID'].map(archetype_map)

# 4. Sidebar Controls
st.sidebar.header("Analysis Controls")
view_mode = st.sidebar.radio("Choose View:", ["Player Evolution", "Market Trends (The Meta)"])

# --- VIEW 1: PLAYER EVOLUTION ---
if view_mode == "Player Evolution":
    st.subheader("🔍 Player Career Arc")
    
    # Search Box
    all_players = sorted(df['PLAYER_NAME'].unique())
    selected_player = st.selectbox("Select a Player:", all_players, index=all_players.index("LeBron James") if "LeBron James" in all_players else 0)
    
    # Filter Data & Force Sort
    player_data = df[df['PLAYER_NAME'] == selected_player].sort_values('SEASON_LABEL')
    
    # Get list of all seasons in correct order for the X-axis
    all_seasons_sorted = sorted(df['SEASON_LABEL'].unique())

    # Plot 1: Archetype Probability over Time
    fig = px.scatter(
        player_data, 
        x='SEASON_LABEL', 
        y='ARCHETYPE_ID',
        color='Archetype Label',
        size='ARCHETYPE_CONFIDENCE',
        hover_data=['OFF_SPEED', 'PTS_PER_TOUCH'],
        title=f"The Evolution of {selected_player}",
        height=500
    )
    
    # FORCE THE X-AXIS ORDER
    fig.update_xaxes(categoryorder='array', categoryarray=all_seasons_sorted)
    
    # Update Y-axis to show names instead of numbers
    fig.update_yaxes(tickvals=list(archetype_map.keys()), ticktext=list(archetype_map.values()))
    st.plotly_chart(fig, use_container_width=True)
    
    # Data Table
    st.write("### Underlying Metrics")
    st.dataframe(player_data[['SEASON_LABEL', 'Archetype Label', 'PTS_PER_TOUCH', 'OFF_SPEED', 'DRIBBLES_PER_TOUCH']])

# --- VIEW 2: MARKET TRENDS ---
elif view_mode == "Market Trends (The Meta)":
    st.subheader("📈 The Stock Market of Playstyles")
    st.write("Which archetypes are becoming more valuable to winning?")
    
    # Calculate "Value" (Impact Score)
    df['IMPACT_SCORE'] = (df['PTS_PER_TOUCH'] * 100) + (df['OFF_SPEED'] * 10)
    
    # Group by Year and Archetype
    trends = df.groupby(['SEASON_LABEL', 'Archetype Label'])['IMPACT_SCORE'].mean().reset_index()
    
    # Plot Line Chart
    fig = px.line(
        trends, 
        x='SEASON_LABEL', 
        y='IMPACT_SCORE', 
        color='Archetype Label',
        markers=True,
        title="Projected Value of NBA Playstyles (2014-2025)",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("Observation: Note how 'Heliocentric Stars' (Blue) and 'Movement Snipers' (Green) are trending up, while 'Traditional Facilitators' are flatlining.")