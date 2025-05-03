import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# the_very_best_model = None
# dataset = None

best_accuracy = 0
best_model = None
best_scaler = None
best_test_df = None

def load_and_validate_data():
    features = pd.read_csv("engineered_features.csv", low_memory=False)
    # convert dates to datetime
    features['gameDate'] = pd.to_datetime(features['gameDate'])

    feature_columns = get_feature_columns()

    # Fill in missing values (days rest)
    for col in feature_columns:
        if 'days_rest' in col:
            features[col] = features[col].fillna(3)
        else:
            features[col] = features[col].fillna(features[col].mean())

    return features

def print_dataset_info(features):
    # print dataset information
    print("\nDataset Information:")
    print(f"Date range: {features['gameDate'].min().date()} to {features['gameDate'].max().date()}")
    print(f"Total games: {len(features)}")
    print(f"Games with valid outcomes: {features['home_win'].notna().sum()}")

def get_feature_columns():
# input features for prediction
    feature_columns = [
        'home_last_5_teamScore',
        'home_last_5_fieldGoalsPercentage',
        'home_last_5_threePointersPercentage',
        'home_last_5_freeThrowsPercentage',
        'home_last_5_reboundsTotal',
        'home_last_5_assists',
        'home_last_5_turnovers',
        'home_last_5_steals',
        'home_last_5_blocks',
        'home_last_5_plusMinusPoints',
        'home_last_5_win',
        'away_last_5_teamScore',
        'away_last_5_fieldGoalsPercentage',
        'away_last_5_threePointersPercentage',
        'away_last_5_freeThrowsPercentage',
        'away_last_5_reboundsTotal',
        'away_last_5_assists',
        'away_last_5_turnovers',
        'away_last_5_steals',
        'away_last_5_blocks',
        'away_last_5_plusMinusPoints',
        'away_last_5_win',
        'home_days_rest',
        'away_days_rest'
    ]
    return feature_columns

# developer printouts
def devPrintOuts(features):
    print("\nFeature Statistics:")
    null_stats = {}
    feature_columns = get_feature_columns()
    
    for col in feature_columns:
        null_count = features[col].isnull().sum()
        null_stats[col] = null_count
        if null_count > 0:
            print(f"{col}: {null_count} null values ({null_count/len(features):.1%})")


def split_dataset(features):
    test_size = 0.2
    cutoff_date = features['gameDate'].quantile(1 - test_size)
    train_df = features[features['gameDate'] < cutoff_date]
    test_df = features[features['gameDate'] >= cutoff_date]

    return train_df, test_df

def print_dataset_split_info(train_df, test_df):
    print(f"\nDataset Split Information:")
    print(f"Training set: {len(train_df)} games ({train_df['gameDate'].min().date()} to {train_df['gameDate'].max().date()})")
    print(f"Test set: {len(test_df)} games ({test_df['gameDate'].min().date()} to {test_df['gameDate'].max().date()})")


def train_and_evaluate(train_df, test_df):
    n_runs = 10
    best_accuracy = 0
    # best_model = None
    # best_scaler = None
    # best_test_df = None

    print(f"\nRunning {n_runs} iterations to find best model...")

    feature_columns = get_feature_columns()

    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs}")
        
        # prepare training data
        X_train = train_df[feature_columns].values
        y_train = train_df['home_win'].values
        
        # prepare test data
        X_test = test_df[feature_columns].values
        y_test = test_df['home_win'].values
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # hyperparameters for xgboost model
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=2,
            subsample=0.7,
            colsample_bytree=0.7,
            gamma=0.2,
            reg_alpha=0.5,
            reg_lambda=2,
            random_state=run,
            objective='binary:logistic',
            eval_metric='logloss'
        )
        
        # train model
        xgb_model.fit(X_train_scaled, y_train)
        
        # evaluate model
        y_pred = xgb_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        # update best model if necessary
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = xgb_model

            best_scaler = scaler
            best_test_df = test_df.copy()


    # post-training printouts
    print("\nBest Model Results:")
    print(f"Accuracy: {best_accuracy:.4f}")
    # the_very_best_model = best_model
    # return best_model


def plotFeatureImportance():
    
    feature_columns = get_feature_columns()
    best_model = train_and_evaluate()

    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=True)

    print("\nAll Feature Importances:")
    print(feature_importance)

    plt.figure(figsize=(10, 12))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance Ranking')
    plt.tight_layout()
    plt.show()


def predict_latest(best_test_df, feature_columns, best_scaler, best_model):

    latest_games = best_test_df.sort_values('gameDate', ascending=False).head(10)
    X_latest = latest_games[feature_columns].values
    X_latest_scaled = best_scaler.transform(X_latest)
    y_pred = best_model.predict(X_latest_scaled)
    y_pred_proba = best_model.predict_proba(X_latest_scaled)


    # print out the latest games predictions
    print("\nLatest Game Predictions:")
    for i, (_, game) in enumerate(latest_games.iterrows()):
        print("\n" + "="*80)
        print(f"Game Date: {game['gameDate'].strftime('%Y-%m-%d')}")
        print(f"Matchup: {game['home_teamCity']} {game['home_teamName']} vs {game['away_teamCity']} {game['away_teamName']}")
        
        print("\nInput Features:")
        print("\nHome Team Stats (Last 5 Games Average):")
        print(f"Scoring: {game['home_last_5_teamScore']:.1f} points")
        print(f"Field Goal %: {game['home_last_5_fieldGoalsPercentage']:.1%}")
        print(f"Three Point %: {game['home_last_5_threePointersPercentage']:.1%}")
        print(f"Free Throw %: {game['home_last_5_freeThrowsPercentage']:.1%}")
        print(f"Rebounds: {game['home_last_5_reboundsTotal']:.1f}")
        print(f"Assists: {game['home_last_5_assists']:.1f}")
        print(f"Turnovers: {game['home_last_5_turnovers']:.1f}")
        print(f"Steals: {game['home_last_5_steals']:.1f}")
        print(f"Blocks: {game['home_last_5_blocks']:.1f}")
        print(f"Plus/Minus: {game['home_last_5_plusMinusPoints']:.1f}")
        print(f"Win Rate: {game['home_last_5_win']:.3f}")
        print(f"Days Rest: {game['home_days_rest']:.0f}")
        
        print("\nAway Team Stats (Last 5 Games Average):")
        print(f"Scoring: {game['away_last_5_teamScore']:.1f} points")
        print(f"Field Goal %: {game['away_last_5_fieldGoalsPercentage']:.1%}")
        print(f"Three Point %: {game['away_last_5_threePointersPercentage']:.1%}")
        print(f"Free Throw %: {game['away_last_5_freeThrowsPercentage']:.1%}")
        print(f"Rebounds: {game['away_last_5_reboundsTotal']:.1f}")
        print(f"Assists: {game['away_last_5_assists']:.1f}")
        print(f"Turnovers: {game['away_last_5_turnovers']:.1f}")
        print(f"Steals: {game['away_last_5_steals']:.1f}")
        print(f"Blocks: {game['away_last_5_blocks']:.1f}")
        print(f"Plus/Minus: {game['away_last_5_plusMinusPoints']:.1f}")
        print(f"Win Rate: {game['away_last_5_win']:.3f}")
        print(f"Days Rest: {game['away_days_rest']:.0f}")
        
        home_win_prob = y_pred_proba[i][1]
        predicted_winner = f"{game['home_teamCity']} {game['home_teamName']}" if y_pred[i] == 1 else f"{game['away_teamCity']} {game['away_teamName']}"
        actual_winner = f"{game['home_teamCity']} {game['home_teamName']}" if game['home_win'] == 1 else f"{game['away_teamCity']} {game['away_teamName']}"
        
        print("\nModel Prediction:")
        print(f"Predicted Winner: {predicted_winner}")
        print(f"Home Team Win Probability: {home_win_prob:.1%}")
        print(f"Away Team Win Probability: {(1-home_win_prob):.1%}")
        print(f"Actual Winner: {actual_winner}")
        print(f"Prediction was: {'CORRECT' if y_pred[i] == game['home_win'] else 'INCORRECT'}")
        print(i)


# Flask Function
@app.route('/predict', methods=['POST'])
def predict_matchup():
    data = request.get_json()
    home_team = data.get('homeTeam', '')
    away_team = data.get('awayTeam', '')
    df = load_and_validate_data()
    # home_team_stats = df[df['home_teamId'] == home_team.id or df['away_teamId' == home_team.id]]
    # away_team_stats = df[df['away_teamId'] == away_team.id or df['home_teamId' == away_team.id]]

    df = df[(df['home_teamId'] == home_team.id and df['away_teamId'] == away_team.id) or (df['home_teamId'] == away_team.id and df['away_teamId'] == away_team.id)]

    feature_columns = get_feature_columns()

    latest_matchups = df.sort_values('gameDate', ascending=False).head(5)
    X_latest = latest_matchups[feature_columns].values
    X_latest_scaled = best_scaler.transform(X_latest)
    y_pred = best_model.predict(X_latest_scaled)
    y_pred_proba = best_model.predict_proba(X_latest_scaled)

    return jsonify({
        'winner': y_pred,
        'winner_probability': y_pred_proba
    })

def main(): 
    features = load_and_validate_data()
    train_df, test_df = split_dataset(features)
    train_and_evaluate(train_df, test_df)
    

if __name__ == "__main__":
    main()
    app.run(debug=True, port=5001) 