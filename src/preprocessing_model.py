import os
import sys
import json
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import shap

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# Project root resolution
if os.path.exists('data') and os.path.exists('figures'):
    PROJECT_ROOT = os.getcwd()
elif os.path.exists(os.path.join('..', 'data')) and os.path.exists(os.path.join('..', 'figures')):
    PROJECT_ROOT = os.path.abspath('..')
else:
    PROJECT_ROOT = os.getcwd()

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
FIG_DIR = os.path.join(PROJECT_ROOT, 'figures')

def run_member2_pipeline():
    print("="*80)
    print("MEMBER 2: ALL-POSITION PREPROCESSING, TARGET FORMULATION & RANDOM FOREST + SHAP")
    print("="*80)
    
    csv_path = os.path.join(DATA_DIR, 'sample_processed_events_with_360.csv')
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} events across all 23 positions from {csv_path}.")
    
    # 1. Action Value Formulation (Socceraction / VAEP foundation + 360 geometric features)
    def calculate_xt(x, y):
        dx = np.clip(x / 120.0, 0, 1)
        dy = 1.0 - np.abs(y - 40.0) / 40.0
        return 0.005 + 0.25 * (dx**3) * (0.3 + 0.7 * dy)

    df_360 = df[df['has_360'] == True].copy()
    print(f"All-position 360 events count: {len(df_360):,}")

    df_360['xt_start'] = calculate_xt(df_360['location_x'].fillna(60), df_360['location_y'].fillna(40))
    df_360['xt_next'] = df_360.groupby(['match_id', 'possession'])['xt_start'].shift(-1).fillna(df_360['xt_start'])
    spatial_delta = (df_360['xt_next'] - df_360['xt_start'])

    # 360 Off-Ball & Positional Context
    line_break_value = df_360['opponents_eliminated'].fillna(0) * 0.02
    pass_option_value = (df_360['forward_passing_options'].fillna(0) / 5.0) * 0.015
    press_resist_value = np.where(df_360['closest_opp_dist'].fillna(10) < 5.0, 0.015, 0.0)
    shot_value = df_360['shot_statsbomb_xg'].fillna(0.0)

    df_360['instant_value'] = (
        spatial_delta * 0.5 +
        line_break_value +
        pass_option_value +
        press_resist_value +
        shot_value
    )

    # 2. Cumulative Future Target across n in {1, 3, 5, 7, 10}
    horizons = [1, 3, 5, 7, 10]
    for n in horizons:
        col = f'target_vaep_n{n}'
        df_360[col] = 0.0
        for k in range(1, n+1):
            df_360[col] += df_360.groupby(['match_id', 'possession'])['instant_value'].shift(-k).fillna(0.0)

    # 3. Features across ALL positions
    feature_cols = [
        'location_x', 'location_y', 'dist_to_goal', 'angle_to_goal',
        'opponents_eliminated', 'forward_passing_options', 'closest_opp_dist',
        'opponents_within_3m', 'opponents_within_5m', 'defenders_in_goal_cone',
        'teammates_visible', 'opponents_visible', 'duration'
    ]

    for c in feature_cols:
        df_360[c] = df_360[c].fillna(df_360[c].median())

    play_patterns = pd.get_dummies(df_360['play_pattern'], prefix='pattern', drop_first=True)
    X = pd.concat([df_360[feature_cols], play_patterns], axis=1)

    unique_matches = df_360['match_id'].unique()
    train_matches, test_matches = train_test_split(unique_matches, test_size=0.2, random_state=42)
    train_matches, val_matches = train_test_split(train_matches, test_size=0.25, random_state=42)

    train_idx = df_360['match_id'].isin(train_matches)
    val_idx = df_360['match_id'].isin(val_matches)
    test_idx = df_360['match_id'].isin(test_matches)

    optimal_n = 5
    X_train, y_train = X[train_idx], df_360.loc[train_idx, f'target_vaep_n{optimal_n}']
    X_val, y_val = X[val_idx], df_360.loc[val_idx, f'target_vaep_n{optimal_n}']
    X_test, y_test = X[test_idx], df_360.loc[test_idx, f'target_vaep_n{optimal_n}']

    print(f"Dataset Splits: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")

    rf_model = RandomForestRegressor(
        n_estimators=120, max_depth=12, min_samples_split=8, min_samples_leaf=4,
        max_features='sqrt', random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    y_pred_train = rf_model.predict(X_train)
    y_pred_val = rf_model.predict(X_val)
    y_pred_test = rf_model.predict(X_test)

    metrics_summary = [
        {
            'Split': 'Training Set (60%)',
            'Samples': len(X_train),
            'MAE': round(mean_absolute_error(y_train, y_pred_train), 4),
            'RMSE': round(np.sqrt(mean_squared_error(y_train, y_pred_train)), 4),
            'R2 Score': round(r2_score(y_train, y_pred_train), 4)
        },
        {
            'Split': 'Validation Set (20%)',
            'Samples': len(X_val),
            'MAE': round(mean_absolute_error(y_val, y_pred_val), 4),
            'RMSE': round(np.sqrt(mean_squared_error(y_val, y_pred_val)), 4),
            'R2 Score': round(r2_score(y_val, y_pred_val), 4)
        },
        {
            'Split': 'Held-Out Test Set (20%)',
            'Samples': len(X_test),
            'MAE': round(mean_absolute_error(y_test, y_pred_test), 4),
            'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 4),
            'R2 Score': round(r2_score(y_test, y_pred_test), 4)
        }
    ]
    pd.DataFrame(metrics_summary).to_csv(os.path.join(DATA_DIR, 'model_evaluation_metrics.csv'), index=False)
    print()
    print("Model Evaluation Metrics:")
    print(pd.DataFrame(metrics_summary).to_string(index=False))

    # TreeSHAP Explainer
    explainer = shap.TreeExplainer(rf_model)
    shap_sample = X_val.sample(min(800, len(X_val)), random_state=42)
    shap_values = explainer.shap_values(shap_sample)

    shap_df = pd.DataFrame({
        'Feature': X.columns,
        'Mean_|SHAP|_Value': np.round(np.abs(shap_values).mean(axis=0), 5)
    }).sort_values(by='Mean_|SHAP|_Value', ascending=False)
    shap_df.to_csv(os.path.join(DATA_DIR, 'shap_feature_importance.csv'), index=False)
    print()
    print("Top Features by SHAP Importance:")
    print(shap_df.head(8))

    # All-Position Predictions & Leaderboard
    df_360['aura_score_pred'] = rf_model.predict(X)
    print()
    print("=== AVERAGE PREDICTED AURA SCORE ACROSS ALL 23 POSITIONS (NO FILTERING) ===")
    pos_summary = df_360.groupby('position').agg(
        mean_score=('aura_score_pred', 'mean'),
        std_score=('aura_score_pred', 'std'),
        total_actions=('event_id', 'count')
    ).sort_values(by='mean_score', ascending=False)
    print(pos_summary.head(15))

    print()
    print("=== TOP 15 PLAYERS OVERALL ACROSS ALL POSITIONS (MIN 50 ACTIONS) ===")
    player_summary = df_360.groupby('player').agg(
        aura_score=('aura_score_pred', 'mean'),
        actions=('event_id', 'count'),
        position=('position', 'first'),
        team=('team', 'first')
    ).sort_values(by='aura_score', ascending=False)
    print(player_summary[player_summary['actions'] >= 50].head(15))
    print("="*80)

if __name__ == '__main__':
    run_member2_pipeline()
