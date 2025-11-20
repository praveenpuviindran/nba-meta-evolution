import pandas as pd

INPUT_FILE = 'data/cleaned_nba_logs.csv'
OUTPUT_FILE = 'data/player_season_stats.csv'

print("--- STARTING FEATURE ENGINEERING ---")

# 1. Load Cleaned Data
df = pd.read_csv(INPUT_FILE)

# 2. Group by Player and Season to get Averages
# We sum the totals first, then divide by games played manually for precision
aggregations = {
    'GAME_ID': 'count',      # This becomes Games Played (GP)
    'MIN': 'mean',           # Minutes Per Game
    'PTS': 'mean',
    'REB': 'mean',
    'AST': 'mean',
    'STL': 'mean',
    'BLK': 'mean',
    'TOV': 'mean',
    'FGM': 'sum',
    'FGA': 'sum',
    'FG3M': 'sum',
    'FG3A': 'sum',
    'FTM': 'sum',
    'FTA': 'sum',
    'OREB': 'sum',
    'DREB': 'sum'
}

season_stats = df.groupby(['PLAYER_ID', 'PLAYER_NAME', 'SEASON_LABEL']).agg(aggregations).reset_index()

# Rename GAME_ID to GP (Games Played)
season_stats.rename(columns={'GAME_ID': 'GP'}, inplace=True)

# 3. Filter: Remove "Cup of Coffee" Players
# If a player played fewer than 10 games, they are noise. Remove them.
season_stats = season_stats[season_stats['GP'] >= 10]
print(f"Players remaining after filtering for GP >= 10: {len(season_stats)}")

# 4. Calculate Advanced "Style" Metrics
# 3-Point Attempt Rate (How much of their offense is 3s?)
season_stats['3P_RATE'] = season_stats['FG3A'] / season_stats['FGA']
season_stats['3P_RATE'] = season_stats['3P_RATE'].fillna(0) # Handle 0 attempts

# Free Throw Rate (Do they drive and draw fouls?)
season_stats['FT_RATE'] = season_stats['FTA'] / season_stats['FGA']
season_stats['FT_RATE'] = season_stats['FT_RATE'].fillna(0)

# True Shooting % (Efficiency approximation)
# Formula: PTS / (2 * (FGA + 0.44 * FTA))
season_stats['TS_PCT'] = season_stats['PTS'] / (2 * (season_stats['FGA'] + 0.44 * season_stats['FTA']))

# Assist to Turnover Ratio
season_stats['AST_TOV'] = season_stats['AST'] / season_stats['TOV']
season_stats['AST_TOV'] = season_stats['AST_TOV'].fillna(0)

# 5. Save
season_stats.to_csv(OUTPUT_FILE, index=False)
print("--- ENGINEERING COMPLETE ---")
print(f"Season Aggregates Saved to: {OUTPUT_FILE}")
print(f"Total Player-Seasons: {len(season_stats)}")