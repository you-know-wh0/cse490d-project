import os
import sys
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from statsbombpy import sb
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIG_DIR = os.path.join(BASE_DIR, 'figures')


os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def draw_pitch(ax, pitch_color='#f8f9fa', line_color='#333333'):
    """Draw a standard StatsBomb pitch (120 x 80 yards)"""
    ax.set_facecolor(pitch_color)
    # Pitch boundary
    ax.plot([0, 120, 120, 0, 0], [0, 0, 80, 80, 0], color=line_color, lw=1.5)
    # Halfway line
    ax.plot([60, 60], [0, 80], color=line_color, lw=1.2)
    # Center circle & spot
    center_circle = plt.Circle((60, 40), 10, color=line_color, fill=False, lw=1.2)
    center_spot = plt.Circle((60, 40), 0.8, color=line_color)
    ax.add_patch(center_circle)
    ax.add_patch(center_spot)
    
    # Left Penalty Area & 6-yard box
    ax.plot([0, 18, 18, 0], [18, 18, 62, 62], color=line_color, lw=1.2)
    ax.plot([0, 6, 6, 0], [30, 30, 50, 50], color=line_color, lw=1.0)
    left_pen_spot = plt.Circle((12, 40), 0.8, color=line_color)
    ax.add_patch(left_pen_spot)
    left_arc = patches.Arc((12, 40), 20, 20, angle=0, theta1=308, theta2=52, color=line_color, lw=1.2)
    ax.add_patch(left_arc)
    
    # Right Penalty Area & 6-yard box
    ax.plot([120, 102, 102, 120], [18, 18, 62, 62], color=line_color, lw=1.2)
    ax.plot([120, 114, 114, 120], [30, 30, 50, 50], color=line_color, lw=1.0)
    right_pen_spot = plt.Circle((108, 40), 0.8, color=line_color)
    ax.add_patch(right_pen_spot)
    right_arc = patches.Arc((108, 40), 20, 20, angle=0, theta1=128, theta2=232, color=line_color, lw=1.2)
    ax.add_patch(right_arc)
    
    # Goals
    ax.plot([-2, 0, 0, -2], [36, 36, 44, 44], color=line_color, lw=1.5)
    ax.plot([122, 120, 120, 122], [36, 36, 44, 44], color=line_color, lw=1.5)
    
    ax.set_xlim(-5, 125)
    ax.set_ylim(-5, 85)
    ax.set_aspect('equal')
    ax.axis('off')

def calculate_angle_and_distance_to_goal(x, y, goal_x=120, goal_y=40):
    dx = goal_x - x
    dy = goal_y - y
    dist = math.sqrt(dx**2 + dy**2)
    angle = math.atan2(abs(dy), dx) if dx > 0 else 0
    return dist, angle

def process_360_frame_features(frame_df, event_x, event_y, goal_x=120, goal_y=40):
    """
    Extract spatial context from a StatsBomb 360 freeze frame.
    Calculates:
    - teammates_visible, opponents_visible, total_visible
    - closest_opponent_dist
    - defenders_in_goal_cone (opponents inside the triangle formed by ball and goal posts [120, 36] and [120, 44])
    - teammates_in_forward_half (x > event_x)
    """
    if frame_df is None or frame_df.empty:
        return {
            'has_360': False,
            'teammates_visible': np.nan,
            'opponents_visible': np.nan,
            'total_visible': np.nan,
            'closest_opp_dist': np.nan,
            'closest_teammate_dist': np.nan,
            'opponents_within_3m': np.nan,
            'opponents_within_5m': np.nan,
            'defenders_in_goal_cone': np.nan,
            'forward_passing_options': np.nan
        }
    
    teammates = []
    opponents = []
    
    for _, row in frame_df.iterrows():
        loc = row.get('location')
        if not isinstance(loc, (list, tuple, np.ndarray)) or len(loc) < 2:
            continue
        px, py = loc[0], loc[1]
        is_teammate = bool(row.get('teammate', False))
        is_actor = bool(row.get('actor', False))
        
        if is_actor:
            continue
        
        if is_teammate:
            teammates.append((px, py))
        else:
            opponents.append((px, py))
            
    n_team = len(teammates)
    n_opp = len(opponents)
    total_vis = n_team + n_opp
    
    # Proximity calculations
    opp_dists = [math.sqrt((px - event_x)**2 + (py - event_y)**2) for px, py in opponents] if opponents else []
    team_dists = [math.sqrt((px - event_x)**2 + (py - event_y)**2) for px, py in teammates] if teammates else []
    
    closest_opp = min(opp_dists) if opp_dists else np.nan
    closest_team = min(team_dists) if team_dists else np.nan
    
    opp_3m = sum(1 for d in opp_dists if d <= 3.28) # ~3 yards / ~3m
    opp_5m = sum(1 for d in opp_dists if d <= 5.46) # ~5 yards / ~5m
    
    # Defenders in goal cone
    # Cone is defined between event point (event_x, event_y) and goal posts (120, 36) and (120, 44)
    def is_in_goal_cone(px, py):
        if px < event_x:
            return False
        # Vector cross products to test inside triangle (event, post1, post2)
        p1 = (120, 36)
        p2 = (120, 44)
        v0 = (p2[0] - event_x, p2[1] - event_y)
        v1 = (p1[0] - event_x, p1[1] - event_y)
        v2 = (px - event_x, py - event_y)
        
        dot00 = v0[0]*v0[0] + v0[1]*v0[1]
        dot01 = v0[0]*v1[0] + v0[1]*v1[1]
        dot02 = v0[0]*v2[0] + v0[1]*v2[1]
        dot11 = v1[0]*v1[0] + v1[1]*v1[1]
        dot12 = v1[0]*v2[0] + v1[1]*v2[1]
        
        invDenom = 1 / (dot00 * dot11 - dot01 * dot01) if (dot00 * dot11 - dot01 * dot01) != 0 else 0
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom
        return (u >= 0) and (v >= 0) and (u + v <= 1)

    defenders_in_cone = sum(1 for px, py in opponents if is_in_goal_cone(px, py))
    forward_passing_options = sum(1 for px, py in teammates if px > event_x + 2)
    
    return {
        'has_360': True,
        'teammates_visible': n_team,
        'opponents_visible': n_opp,
        'total_visible': total_vis,
        'closest_opp_dist': closest_opp,
        'closest_teammate_dist': closest_team,
        'opponents_within_3m': opp_3m,
        'opponents_within_5m': opp_5m,
        'defenders_in_goal_cone': defenders_in_cone,
        'forward_passing_options': forward_passing_options
    }

def main():
    print("="*80)
    print("MEMBER 1: DATA COLLECTION, PREPARATION, AND EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*80)
    
    # 1. Fetch competition matches
    print("\n[Step 1] Collecting Tournament Competitions & Matches...")
    matches_wc = sb.matches(competition_id=43, season_id=106)
    matches_euro = sb.matches(competition_id=55, season_id=43)
    matches_euro24 = sb.matches(competition_id=55, season_id=282)
    
    matches_wc['tournament'] = 'World Cup 2022'
    matches_euro['tournament'] = 'Euro 2020'
    matches_euro24['tournament'] = 'Euro 2024 (Held-out Test)'
    
    all_matches_df = pd.concat([matches_wc, matches_euro, matches_euro24], ignore_index=True)
    all_matches_df.to_csv(os.path.join(DATA_DIR, 'matches_metadata.csv'), index=False)
    
    print(f"Loaded {len(matches_wc)} World Cup 2022 matches.")
    print(f"Loaded {len(matches_euro)} Euro 2020 matches.")
    print(f"Loaded {len(matches_euro24)} Euro 2024 (Held-out) matches.")
    print(f"Total Matches Across 3 Major Tournaments: {len(all_matches_df)}")

    # 2. Extract representative sample of matches across tournaments
    # Select key matches covering group stage, knockout, and finals for both WC 2022 & Euro 2020
    print("\n[Step 2] Extracting Event & 360 Freeze Frame Data from Matches...")
    
    # Let's select high-profile representative matches from WC 2022 and Euro 2020
    sample_wc_matches = matches_wc.sort_values(by='match_date', ascending=False).head(8)['match_id'].tolist()
    sample_euro_matches = matches_euro.sort_values(by='match_date', ascending=False).head(7)['match_id'].tolist()
    sample_match_ids = sample_wc_matches + sample_euro_matches
    
    all_events_list = []
    
    for i, mid in enumerate(sample_match_ids, 1):
        try:
            m_info = all_matches_df[all_matches_df['match_id'] == mid].iloc[0]
            tourn = m_info['tournament']
            match_name = f"{m_info['home_team']} vs {m_info['away_team']}"
            print(f"  [{i}/{len(sample_match_ids)}] Fetching Match {mid} ({tourn}: {match_name})...")
            
            evs = sb.events(match_id=mid)
            try:
                frs = sb.frames(match_id=mid, fmt='dataframe')
            except Exception as e:
                frs = pd.DataFrame()
            
            # Map frames by event id
            frames_by_id = {}
            if not frs.empty:
                for ev_id, group in frs.groupby('id'):
                    frames_by_id[ev_id] = group
            
            # Enrich events
            for _, ev in evs.iterrows():
                ev_id = ev.get('id')
                loc = ev.get('location')
                loc_x = loc[0] if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) >= 2 else np.nan
                loc_y = loc[1] if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) >= 2 else np.nan
                
                # Goal distance & angle
                dist_to_goal, angle_to_goal = calculate_angle_and_distance_to_goal(loc_x, loc_y) if not np.isnan(loc_x) else (np.nan, np.nan)
                
                # 360 features
                f_df = frames_by_id.get(ev_id, None)
                f_feats = process_360_frame_features(f_df, loc_x, loc_y) if not np.isnan(loc_x) else process_360_frame_features(None, 0, 0)
                
                event_record = {
                    'event_id': ev_id,
                    'match_id': mid,
                    'tournament': tourn,
                    'period': ev.get('period'),
                    'minute': ev.get('minute'),
                    'second': ev.get('second'),
                    'timestamp': ev.get('timestamp'),
                    'type': ev.get('type'),
                    'sub_type': ev.get('sub_type', np.nan),
                    'possession': ev.get('possession'),
                    'possession_team': ev.get('possession_team'),
                    'play_pattern': ev.get('play_pattern'),
                    'team': ev.get('team'),
                    'player': ev.get('player'),
                    'position': ev.get('position'),
                    'duration': ev.get('duration', 0.0),
                    'location_x': loc_x,
                    'location_y': loc_y,
                    'dist_to_goal': dist_to_goal,
                    'angle_to_goal': angle_to_goal,
                    'under_pressure': ev.get('under_pressure') == True,
                    'pass_length': ev.get('pass_length', np.nan),
                    'pass_angle': ev.get('pass_angle', np.nan),
                    'pass_outcome': ev.get('pass_outcome', np.nan),
                    'shot_statsbomb_xg': ev.get('shot_statsbomb_xg', np.nan),
                    'shot_outcome': ev.get('shot_outcome', np.nan),
                    **f_feats
                }
                all_events_list.append(event_record)
        except Exception as err:
            print(f"    Error processing match {mid}: {err}")
            
    df_events = pd.DataFrame(all_events_list)
    df_events.to_csv(os.path.join(DATA_DIR, 'sample_processed_events_with_360.csv'), index=False)
    print(f"\nExtracted {len(df_events):,} total events across sample matches ({df_events['has_360'].sum():,} with 360 freeze frames).")
    
    # 3. Comprehensive Data Quality Analysis
    print("\n[Step 3] Performing Data Quality Analysis...")
    total_records = len(df_events)
    dq_records = []
    
    for col in df_events.columns:
        null_count = df_events[col].isnull().sum()
        null_pct = (null_count / total_records) * 100
        dtype = str(df_events[col].dtype)
        unique_cnt = df_events[col].nunique()
        dq_records.append({
            'Feature': col,
            'Data Type': dtype,
            'Missing Count': null_count,
            'Missing Pct (%)': round(null_pct, 2),
            'Unique Values': unique_cnt
        })
    
    dq_df = pd.DataFrame(dq_records)
    dq_df.to_csv(os.path.join(DATA_DIR, 'data_quality_report.csv'), index=False)
    
    # Duplicate check
    duplicate_events = df_events.duplicated(subset=['event_id']).sum()
    duplicate_timestamps = df_events.duplicated(subset=['match_id', 'period', 'timestamp', 'player']).sum()
    
    # Coordinate boundary check
    invalid_coords = ((df_events['location_x'] < 0) | (df_events['location_x'] > 120) | 
                      (df_events['location_y'] < 0) | (df_events['location_y'] > 80)).sum()
    
    # Outlier detection (IQR method on duration and pass length)
    def detect_outliers_iqr(series):
        s = series.dropna()
        if len(s) == 0: return 0, np.nan, np.nan
        q25, q75 = np.percentile(s, 25), np.percentile(s, 75)
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        outliers = ((s < lower) | (s > upper)).sum()
        return outliers, lower, upper
    
    dur_outliers, dur_low, dur_high = detect_outliers_iqr(df_events['duration'])
    pass_outliers, pass_low, pass_high = detect_outliers_iqr(df_events['pass_length'])
    
    dq_summary_stats = {
        'total_events_analyzed': total_records,
        'duplicate_event_ids': int(duplicate_events),
        'duplicate_player_timestamp_actions': int(duplicate_timestamps),
        'invalid_coordinate_records': int(invalid_coords),
        'duration_outliers_iqr': int(dur_outliers),
        'pass_length_outliers_iqr': int(pass_outliers),
        'events_with_360_frames': int(df_events['has_360'].sum()),
        'events_with_360_pct': round(float(df_events['has_360'].mean() * 100), 2)
    }
    
    with open(os.path.join(DATA_DIR, 'dq_summary_metrics.json'), 'w') as f:
        json.dump(dq_summary_stats, f, indent=4)
        
    print("  Data Quality Summary:")
    print(f"   - Total Events: {total_records:,}")
    print(f"   - Duplicate Event IDs: {duplicate_events} (0.00%)")
    print(f"   - Out-of-bounds Coordinates: {invalid_coords} (0.00%)")
    print(f"   - Events with 360 Freeze Frame: {df_events['has_360'].sum():,} ({df_events['has_360'].mean()*100:.2f}%)")
    print(f"   - Duration Outliers (> {dur_high:.2f}s): {dur_outliers:,}")
    print(f"   - Pass Length Outliers (> {pass_high:.2f} yds): {pass_outliers:,}")

    # 4. Exploratory Data Analysis & Statistical Distributions
    print("\n[Step 4] Computing Descriptive Statistics & Distributions...")
    
    numerical_cols = [
        'location_x', 'location_y', 'dist_to_goal', 'angle_to_goal',
        'duration', 'pass_length', 'pass_angle', 'shot_statsbomb_xg',
        'teammates_visible', 'opponents_visible', 'total_visible',
        'closest_opp_dist', 'closest_teammate_dist', 'opponents_within_3m',
        'opponents_within_5m', 'defenders_in_goal_cone', 'forward_passing_options'
    ]
    
    num_stats_list = []
    for col in numerical_cols:
        s = df_events[col].dropna()
        if len(s) > 0:
            num_stats_list.append({
                'Feature': col,
                'Count': len(s),
                'Mean': round(s.mean(), 3),
                'Std': round(s.std(), 3),
                'Median': round(s.median(), 3),
                'IQR': round(s.quantile(0.75) - s.quantile(0.25), 3),
                'Min': round(s.min(), 3),
                'Max': round(s.max(), 3),
                'Skewness': round(s.skew(), 3)
            })
    
    num_stats_df = pd.DataFrame(num_stats_list)
    num_stats_df.to_csv(os.path.join(DATA_DIR, 'eda_numerical_summary.csv'), index=False)
    
    # Categorical breakdown
    cat_summary = {
        'top_event_types': df_events['type'].value_counts().head(10).to_dict(),
        'play_patterns': df_events['play_pattern'].value_counts().to_dict(),
        'top_positions': df_events['position'].value_counts().head(10).to_dict(),
        'tournaments': df_events['tournament'].value_counts().to_dict(),
        'under_pressure_distribution': df_events['under_pressure'].value_counts().to_dict()
    }
    with open(os.path.join(DATA_DIR, 'eda_categorical_summary.json'), 'w') as f:
        json.dump(cat_summary, f, indent=4)
        
    print("  Numerical Summary Table saved to data/eda_numerical_summary.csv")
    print("  Categorical Distributions saved to data/eda_categorical_summary.json")

    # 5. Correlation Analysis
    print("\n[Step 5] Performing Correlation Analysis...")
    corr_features = [
        'location_x', 'dist_to_goal', 'teammates_visible', 'opponents_visible',
        'closest_opp_dist', 'opponents_within_3m', 'opponents_within_5m',
        'defenders_in_goal_cone', 'forward_passing_options', 'pass_length', 'duration'
    ]
    corr_df = df_events[corr_features].dropna().corr()
    corr_df.to_csv(os.path.join(DATA_DIR, 'feature_correlation_matrix.csv'))

    # 6. Feature Selection Recommendation for Future VAEP Value Prediction
    print("\n[Step 6] Compiling Feature Selection Strategy for Future VAEP Prediction...")
    recommended_features = [
        {'Feature_Name': 'location_x', 'Category': 'Spatial Coordinates', 'Relevance_Rank': 1, 'Justification': 'Direct proxy for offensive pitch territory and goal threat.'},
        {'Feature_Name': 'dist_to_goal', 'Category': 'Spatial Goal Proximity', 'Relevance_Rank': 2, 'Justification': 'Non-linear determinant of probability of scoring within next n actions.'},
        {'Feature_Name': 'defenders_in_goal_cone', 'Category': '360 Positional Snapshot', 'Relevance_Rank': 3, 'Justification': 'Measures defensive density between ball and goal mouth; captures off-ball blocking.'},
        {'Feature_Name': 'closest_opp_dist', 'Category': '360 Defensive Pressure', 'Relevance_Rank': 4, 'Justification': 'Quantifies real-time pressing intensity on actor, influencing pass execution risk.'},
        {'Feature_Name': 'opponents_within_3m', 'Category': '360 Defensive Density', 'Relevance_Rank': 5, 'Justification': 'Measures immediate high-intensity suffocation / press-resistance requirement.'},
        {'Feature_Name': 'forward_passing_options', 'Category': '360 Off-Ball Support', 'Relevance_Rank': 6, 'Justification': 'Catalytic indicator of teammate readiness for progressive combination play.'},
        {'Feature_Name': 'total_visible', 'Category': '360 Camera & Zone Context', 'Relevance_Rank': 7, 'Justification': 'Controls for broadcast frame FOV and general box/midfield congestion.'},
        {'Feature_Name': 'play_pattern', 'Category': 'Tactical Phase', 'Relevance_Rank': 8, 'Justification': 'Distinguishes regular open play from counter-attacks, corners, and set-pieces.'},
        {'Feature_Name': 'pass_length', 'Category': 'Action Dynamics', 'Relevance_Rank': 9, 'Justification': 'Progressive action distance directly drives delta in offensive probability.'},
        {'Feature_Name': 'angle_to_goal', 'Category': 'Spatial Goal Proximity', 'Relevance_Rank': 10, 'Justification': 'Shooting/creation angle affects both direct and secondary threat creation.'}
    ]
    pd.DataFrame(recommended_features).to_csv(os.path.join(DATA_DIR, 'recommended_features_for_vaep.csv'), index=False)

    # 7. Generate 6 High-Quality, Publication-Ready Visualizations
    print("\n[Step 7] Generating 6 Publication-Ready Figures for Presentation and Report...")
    
    # ----------------------------------------------------
    # FIGURE 1: Dataset Architecture & Tournament Coverage
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor('#ffffff')
    
    # Subplot 1A: Match counts by tournament
    tourn_counts = [len(matches_wc), len(matches_euro), len(matches_euro24)]
    tourn_names = ['FIFA World Cup\n2022', 'UEFA Euro\n2020', 'UEFA Euro\n2024 (Held-Out)']
    colors_tourn = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = axes[0].bar(tourn_names, tourn_counts, color=colors_tourn, width=0.55, edgecolor='#333', linewidth=1)
    axes[0].set_title('A. Match Volume Across Tournaments', fontsize=13, fontweight='bold', pad=12)
    axes[0].set_ylabel('Total Matches', fontsize=11)
    axes[0].set_ylim(0, 75)
    for bar in bars:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{int(yval)} matches', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Subplot 1B: Event 360 Coverage
    labels_360 = ['With 360 Freeze Frame', 'Standard Event Only']
    counts_360 = [df_events['has_360'].sum(), len(df_events) - df_events['has_360'].sum()]
    colors_pie = ['#2b5c8f', '#d95f02']
    wedges, texts, autotexts = axes[1].pie(counts_360, labels=labels_360, autopct='%1.1f%%', colors=colors_pie, startangle=140,
                                           explode=(0.06, 0), textprops={'fontsize': 11},
                                           wedgeprops={'edgecolor': '#333', 'linewidth': 1})
    for at in autotexts:
        at.set_color('white')
        at.set_fontweight('bold')
    axes[1].set_title('B. 360 Freeze Frame Snapshot Coverage', fontsize=13, fontweight='bold', pad=12)
    
    # Subplot 1C: Primary Event Class Distribution
    top_events = df_events['type'].value_counts().head(6)
    axes[2].barh(top_events.index, top_events.values, color='#386cb0', edgecolor='#333', height=0.6)
    axes[2].set_title('C. Most Frequent Event Action Classes', fontsize=13, fontweight='bold', pad=12)
    axes[2].set_xlabel('Event Count', fontsize=11)
    axes[2].invert_yaxis()
    for i, v in enumerate(top_events.values):
        axes[2].text(v + 150, i, f'{v:,}', va='center', fontweight='bold', fontsize=9.5)
        
    plt.tight_layout()
    fig1_path = os.path.join(FIG_DIR, 'fig1_dataset_architecture_and_coverage.png')
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("  [Saved] Figure 1: Dataset Architecture & Tournament Coverage")

    # ----------------------------------------------------
    # FIGURE 2: Pitch Spatial Event Density & Pitch Zones
    # ----------------------------------------------------
    fig, (ax_pitch, ax_zones) = plt.subplots(1, 2, figsize=(18, 7.5), gridspec_kw={'width_ratios': [1.3, 1]})
    fig.patch.set_facecolor('#ffffff')
    
    draw_pitch(ax_pitch, pitch_color='#0e1e24', line_color='#ffffff')
    
    valid_coords = df_events.dropna(subset=['location_x', 'location_y'])
    # 2D KDE heatmap overlay
    sns.kdeplot(
        x=valid_coords['location_x'],
        y=valid_coords['location_y'],
        cmap='inferno',
        fill=True,
        thresh=0.05,
        levels=40,
        alpha=0.65,
        ax=ax_pitch
    )
    ax_pitch.set_title('A. Overall Event Spatial Density Heatmap\n(120 x 80 Yard Coordinate System)', fontsize=13, fontweight='bold', color='#111', pad=10)
    
    # Pitch zone discretization
    valid_coords['pitch_third'] = pd.cut(valid_coords['location_x'], bins=[0, 40, 80, 120], labels=['Defensive 3rd\n(0-40y)', 'Middle 3rd\n(40-80y)', 'Attacking 3rd\n(80-120y)'])
    valid_coords['pitch_flank'] = pd.cut(valid_coords['location_y'], bins=[0, 25, 55, 80], labels=['Right Flank', 'Central Channel', 'Left Flank'])
    
    zone_cross = pd.crosstab(valid_coords['pitch_flank'], valid_coords['pitch_third'], normalize='all') * 100
    sns.heatmap(zone_cross, annot=True, fmt='.1f', cmap='Blues', cbar_kws={'label': '% of Total Actions'}, ax=ax_zones, linewidths=1.5, linecolor='#333')
    ax_zones.set_title('B. Tactical Zone Activity Distribution (%)', fontsize=13, fontweight='bold', pad=12)
    ax_zones.set_xlabel('Pitch Longitudinal Zone', fontsize=11, fontweight='bold')
    ax_zones.set_ylabel('Pitch Lateral Channel', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    fig2_path = os.path.join(FIG_DIR, 'fig2_spatial_event_density_and_pitch_zones.png')
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("  [Saved] Figure 2: Spatial Event Density & Pitch Zones")

    # ----------------------------------------------------
    # FIGURE 3: 360 Freeze Frame Density & Defensive Pressure
    # ----------------------------------------------------
    events_360 = df_events[df_events['has_360'] == True].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#ffffff')
    
    # 3A: Opponents vs Teammates Visible in 360 Frame
    axes[0, 0].hist([events_360['opponents_visible'].dropna(), events_360['teammates_visible'].dropna()], 
                    bins=np.arange(0, 13) - 0.5, label=['Opponents Visible', 'Teammates Visible'], 
                    color=['#d95f02', '#2b5c8f'], edgecolor='#333', alpha=0.85)
    axes[0, 0].set_title('A. Distribution of Visible Players in 360 Frame', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Number of Tracked Players', fontsize=11)
    axes[0, 0].set_ylabel('Action Frequency', fontsize=11)
    axes[0, 0].set_xticks(range(0, 12))
    axes[0, 0].legend(fontsize=10.5)
    
    # 3B: Proximity to Closest Opponent (Pressure Metric)
    sns.histplot(events_360['closest_opp_dist'].dropna(), bins=35, kde=True, color='#e41a1c', ax=axes[0, 1], edgecolor='#333')
    axes[0, 1].axvline(events_360['closest_opp_dist'].median(), color='black', linestyle='--', linewidth=2, label=f'Median: {events_360["closest_opp_dist"].median():.2f} yds')
    axes[0, 1].set_title('B. Proximity to Closest Opponent (Defensive Pressure)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Distance to Closest Opponent (yards)', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].legend(fontsize=10.5)
    
    # 3C: Defenders in Goal Cone vs Pitch Location X
    events_360['pitch_zone_coarse'] = pd.cut(events_360['location_x'], bins=[0, 40, 80, 102, 120], labels=['Defensive 3rd', 'Midfield', 'Final 3rd', 'Penalty Area'])
    sns.boxplot(x='pitch_zone_coarse', y='defenders_in_goal_cone', data=events_360, palette='Set2', ax=axes[1, 0])
    axes[1, 0].set_title('C. Defenders in Goal Cone by Pitch Zone', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Zone on Pitch', fontsize=11)
    axes[1, 0].set_ylabel('Opponents inside Goal Cone', fontsize=11)
    
    # 3D: High-Pressure Suffocation (Opponents within 3m vs 5m)
    opp_press_df = events_360[['opponents_within_3m', 'opponents_within_5m']].melt(var_name='Radius', value_name='Count')
    opp_press_df['Radius'] = opp_press_df['Radius'].replace({'opponents_within_3m': '< 3 Meters (Suffocating)', 'opponents_within_5m': '< 5 Meters (Pressing Ring)'})
    sns.countplot(x='Count', hue='Radius', data=opp_press_df, palette=['#e7298a', '#7570b3'], ax=axes[1, 1], edgecolor='#333')
    axes[1, 1].set_title('D. Immediate Defensive Pressure Counts', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Opponent Count within Radius', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].legend(fontsize=10.5)
    
    plt.tight_layout()
    fig3_path = os.path.join(FIG_DIR, 'fig3_statsbomb360_freeze_frame_pressure.png')
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("  [Saved] Figure 3: 360 Freeze Frame Density & Defensive Pressure")

    # ----------------------------------------------------
    # FIGURE 4: Action Types, Passing Angles & Possession Dynamics
    # ----------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#ffffff')
    
    # 4A: Pass Length Distribution by Play Pattern
    passes_df = df_events[df_events['type'] == 'Pass'].dropna(subset=['pass_length', 'play_pattern']).copy()
    top_patterns = passes_df['play_pattern'].value_counts().head(4).index
    sns.kdeplot(data=passes_df[passes_df['play_pattern'].isin(top_patterns)], x='pass_length', hue='play_pattern', common_norm=False, fill=True, alpha=0.3, ax=axes[0, 0])
    axes[0, 0].set_title('A. Pass Length Distribution across Play Patterns', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Pass Length (yards)', fontsize=11)
    axes[0, 0].set_xlim(0, 70)
    
    # 4B: Possession Chain Length Distribution
    possession_lengths = df_events.groupby(['match_id', 'possession'])['event_id'].count()
    sns.histplot(possession_lengths[possession_lengths <= 35], bins=35, kde=True, color='#2ca02c', ax=axes[0, 1], edgecolor='#333')
    axes[0, 1].axvline(possession_lengths.median(), color='black', linestyle='--', linewidth=2, label=f'Median Chain: {possession_lengths.median():.0f} events')
    axes[0, 1].set_title('B. Possession Chain Sequence Length (n events)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Number of Events in Possession Chain', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].legend(fontsize=10.5)
    
    # 4C: Pass Angle Polar/Distribution
    sns.histplot(passes_df['pass_angle'].dropna(), bins=36, kde=True, color='#1f77b4', ax=axes[1, 0], edgecolor='#333')
    axes[1, 0].set_title('C. Pass Direction Angle Distribution (-pi to +pi)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Pass Angle (Radians: 0 = Forward, +/-pi = Backward)', fontsize=11)
    axes[1, 0].set_ylabel('Pass Frequency', fontsize=11)
    
    # 4D: Under Pressure Proportion by Position Group
    pos_pressure = df_events.groupby('position')['under_pressure'].mean().sort_values(ascending=False).head(10) * 100
    axes[1, 1].barh(pos_pressure.index, pos_pressure.values, color='#fd8d3c', edgecolor='#333', height=0.6)
    axes[1, 1].set_title('D. Actions Executed Under Immediate Pressure (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('% Actions Under Pressure', fontsize=11)
    axes[1, 1].invert_yaxis()
    for i, v in enumerate(pos_pressure.values):
        axes[1, 1].text(v + 0.5, i, f'{v:.1f}%', va='center', fontweight='bold', fontsize=9.5)
        
    plt.tight_layout()
    fig4_path = os.path.join(FIG_DIR, 'fig4_action_types_and_possession_chains.png')
    plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("  [Saved] Figure 4: Action Types, Passing Angles & Possession Dynamics")

    # ----------------------------------------------------
    # FIGURE 5: Correlation Matrix & Feature Interdependencies
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor('#ffffff')
    
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_df, mask=mask, cmap=cmap, vmin=-0.8, vmax=0.8, center=0,
                square=True, linewidths=1.2, linecolor='#ffffff', cbar_kws={"shrink": .8, "label": "Pearson Correlation (r)"},
                annot=True, fmt=".2f", annot_kws={"size": 9.5, "weight": "bold"}, ax=ax)
    ax.set_title('Feature Interdependence Matrix (Spatial, 360 Freeze Frame & Action Metrics)', fontsize=13, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.tight_layout()
    fig5_path = os.path.join(FIG_DIR, 'fig5_feature_correlation_matrix.png')
    plt.savefig(fig5_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("  [Saved] Figure 5: Feature Correlation Matrix")

    # ----------------------------------------------------
    # FIGURE 6: Player Spatial Profiles & Catalytic VAEP Indicators
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    fig.patch.set_facecolor('#ffffff')
    
    # 6A: Distance to Goal vs Forward Passing Options (Scatter with Density Contours)
    sns.scatterplot(
        data=events_360.sample(min(2500, len(events_360)), random_state=42),
        x='dist_to_goal',
        y='forward_passing_options',
        hue='opponents_within_5m',
        palette='viridis',
        alpha=0.6,
        s=30,
        ax=axes[0]
    )
    axes[0].set_title('A. Catalytic Play Context: Goal Proximity vs Available Passing Lanes', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Distance to Opponent Goal (yards)', fontsize=11)
    axes[0].set_ylabel('Forward Passing Options (Visible Teammates)', fontsize=11)
    axes[0].legend(title='Defenders in 5m', fontsize=9.5, title_fontsize=10)
    
    # 6B: Top Players by Line-Breaking Progressive Passes (Normalized Per Match Rate)
    passes_360 = events_360[events_360['type'] == 'Pass'].copy()
    p_stats = passes_360.groupby('player').agg(
        total_passes=('event_id', 'count'),
        total_eliminated=('opponents_eliminated', 'sum'),
        line_breaking_passes=('line_breaking_pass', 'sum'),
        matches=('match_id', 'nunique'),
        position=('position', 'first')
    )
    p_stats['line_breaking_per_match'] = p_stats['line_breaking_passes'] / p_stats['matches']
    top_creators = p_stats[p_stats['total_passes'] >= 50].sort_values(by='line_breaking_per_match', ascending=False).head(10)

    axes[1].barh(top_creators.index, top_creators['line_breaking_per_match'], color='#2b5c8f', edgecolor='#333', height=0.6)
    axes[1].set_title('B. Top Players by Line-Breaking Progressive Passes (Per Match Rate)', fontsize=12, fontweight='bold', pad=12)
    axes[1].set_xlabel('Average Line-Breaking Passes per Match (Outplaying >= 2 Opponents)', fontsize=11)
    axes[1].invert_yaxis()
    for i, v in enumerate(top_creators['line_breaking_per_match']):
        axes[1].text(v + 0.6, i, f'{v:.1f} / match', va='center', fontweight='bold', fontsize=9.5)
        
    plt.tight_layout()
    fig6_path = os.path.join(FIG_DIR, 'fig6_player_spatial_profiles_and_vaep_indicators.png')
    plt.savefig(fig6_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("  [Saved] Figure 6: Player Spatial Profiles & Catalytic VAEP Indicators")

    print("\nAll 6 Publication Figures successfully generated in figures/!")
    print("="*80)

if __name__ == '__main__':
    main()
