import pandas as pd
import numpy as np
from datetime import datetime

def load_and_preprocess_data():
    """Load and preprocess the data"""
    print("Loading preprocessed data...")
    data = pd.read_csv("preprocessed_h2h.csv", low_memory=False)
    
    # convert dates to datetime
    data['gameDate'] = pd.to_datetime(data['gameDate'])
    
    # cutoff - early 2019 to have enough historical data for calculations
    initial_cutoff = pd.to_datetime('2019-01-01')
    data = data[data['gameDate'] >= initial_cutoff]
    print(f"Initial filter: keeping games from {initial_cutoff.date()} onwards for calculation purposes")
    
    # Sort by date
    data = data.sort_values('gameDate')
    
    return data

def calculate_basic_rolling_stats(df, window=5):
    """Calculate basic rolling statistics for each team"""
    # define columns to calculate rolling stats for
    cols = [
        'teamScore', 'fieldGoalsPercentage', 'threePointersPercentage',
        'freeThrowsPercentage', 'reboundsTotal', 'assists', 'turnovers',
        'steals', 'blocks', 'plusMinusPoints', 'win'
    ]

    # Calculate rolling stats with shift to avoid data leakage
    # In other words, we want to ensure that the rolling stats are calculated based on games that have already occurred,
    # without including information from the current game in the calculation which has not yet occurred
    rolled = (
        df
        .sort_values(['teamId', 'gameDate'])
        .groupby('teamId')[cols + ['gameDate']]
        # First shift to exclude current game, then calculate rolling mean
        .apply(lambda x: pd.DataFrame({
            'gameDate': x['gameDate'],
            **{col: x[col].shift(1).rolling(window=window, min_periods=1).mean() 
               for col in cols}
        }))
        .reset_index()
    )

    # Fix column names
    rolled.columns = ['teamId', 'level_1', 'gameDate'] + [f'last_{window}_{c}' for c in cols]
    # Drop the level_1 column
    rolled = rolled.drop('level_1', axis=1)
    
    print(f"Calculated rolling stats with {window}-game window (shifted to avoid leakage)")
    return rolled

def create_feature_dataset(data):
    """Create the feature dataset with basic information and rolling stats"""
    print("Creating feature dataset...")
    
    # Calculate rolling stats first
    print("Calculating rolling statistics...")
    rolling_stats = calculate_basic_rolling_stats(data, window=5)
    
    # Get only home games
    home_games = data[data['home'] == 1].copy()
    print(f"Number of home games found: {len(home_games)}")
    
    # Get away games (for h2h rates)
    away_games = data[data['home'] == 0].copy()
    
    # Create base feature dataset
    features = pd.DataFrame()
    
    # Basic game information
    features['gameId'] = home_games['gameId']
    features['gameDate'] = home_games['gameDate']
    features['home_win'] = home_games['win'].astype(int)
    
    # Team information
    features['home_teamId'] = home_games['teamId']
    features['home_teamCity'] = home_games['teamCity']
    features['home_teamName'] = home_games['teamName']
    features['away_teamId'] = home_games['opponentTeamId']
    features['away_teamCity'] = home_games['opponentTeamCity']
    features['away_teamName'] = home_games['opponentTeamName']
    
    # Merge home team rolling stats
    print("Merging home team statistics...")
    home_stats = rolling_stats.copy()
    home_stats.columns = ['teamId', 'gameDate'] + [f'home_{col}' for col in home_stats.columns[2:]]
    features = features.merge(
        home_stats,
        left_on=['home_teamId', 'gameDate'],
        right_on=['teamId', 'gameDate'],
        how='left'
    )
    
    # Merge away team rolling stats
    print("Merging away team statistics...")
    away_stats = rolling_stats.copy()
    away_stats.columns = ['teamId', 'gameDate'] + [f'away_{col}' for col in away_stats.columns[2:]]
    features = features.merge(
        away_stats,
        left_on=['away_teamId', 'gameDate'],
        right_on=['teamId', 'gameDate'],
        how='left'
    )
    
    # Add head-to-head win rates from preprocessed data
    print("Adding head-to-head win rates...")
    features['h2h_winrate_last_3_home'] = home_games['h2h_winrate_last_3']  # Home team's perspective
    
    # Get away team's h2h rate by matching game IDs
    away_h2h = away_games[['gameId', 'h2h_winrate_last_3']].copy()
    features = features.merge(
        away_h2h,
        on='gameId',
        how='left'
    )
    # Rename the merged h2h column for away team
    features = features.rename(columns={'h2h_winrate_last_3': 'h2h_winrate_last_3_away'})
    
    # Calculate rest days before any filtering
    print("Calculating rest days...")
    rest_days = {}
    
    # Calculate rest days for all teams
    for team_id in data['teamId'].unique():
        # Get all games for this team (both home and away)
        team_games = data[
            (data['teamId'] == team_id) | 
            (data['opponentTeamId'] == team_id)
        ].copy()
        
        # Sort by date
        team_games = team_games.sort_values('gameDate')
        
        # Calculate days between games
        team_games['days_rest'] = team_games['gameDate'].diff().dt.days
        
        # Store in dictionary with gameId and whether team was home/away
        for _, game in team_games.iterrows():
            if game['teamId'] == team_id:  # Team was listed first (their perspective)
                is_home = game['home']
                rest_key = 'home_days_rest' if is_home else 'away_days_rest'
                if game['gameId'] not in rest_days:
                    rest_days[game['gameId']] = {}
                rest_days[game['gameId']][rest_key] = game['days_rest']
    
    # Add rest days to features
    for game_id in features['gameId']:
        if game_id in rest_days:
            if 'home_days_rest' in rest_days[game_id]:
                features.loc[features['gameId'] == game_id, 'home_days_rest'] = rest_days[game_id]['home_days_rest']
            if 'away_days_rest' in rest_days[game_id]:
                features.loc[features['gameId'] == game_id, 'away_days_rest'] = rest_days[game_id]['away_days_rest']
    
    # Fill NaN rest days with a reasonable value (like 3 days) for season openers
    features['home_days_rest'] = features['home_days_rest'].fillna(3)
    features['away_days_rest'] = features['away_days_rest'].fillna(3)
    
    # Drop temporary columns
    features = features.drop(columns=['teamId_x', 'teamId_y'])
    
    # Final date filter - keep only games from July 30, 2020 onwards
    # I found that using earlier data hurt accuracy due to the drastic change in playstyle in the NBA
    final_cutoff = pd.to_datetime('2020-07-30')
    features = features[features['gameDate'] >= final_cutoff]
    print(f"\nFinal filter: keeping games from {final_cutoff.date()} onwards")
    print(f"Final dataset contains {len(features)} games")
    
    return features

def main():
    # Load and preprocess data
    data = load_and_preprocess_data()
    
    # Create feature dataset
    features = create_feature_dataset(data)
    
    # Save to CSV
    output_file = "engineered_features.csv"
    print(f"\nSaving engineered features to {output_file}...")
    features.to_csv(output_file, index=False)
    
    # Print dataset information
    print("\nFeature Dataset Information:")
    print(f"Date range: {features['gameDate'].min().date()} to {features['gameDate'].max().date()}")
    print(f"Number of games: {len(features)}")
    print("\nColumns:")
    for col in features.columns:
        print(f"- {col}")

if __name__ == "__main__":
    main() 