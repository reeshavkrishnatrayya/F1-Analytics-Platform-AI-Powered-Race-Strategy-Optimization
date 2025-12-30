# ============================================================================
# F1 MULTI-RACE AI ANALYTICS PLATFORM (2022-2023)
# ============================================================================

print("Initializing F1 Multi-Race AI Platform...")

# Create cache directory
import os
if not os.path.exists('./f1_cache'):
    os.makedirs('./f1_cache')
    print("Created cache directory")

# Install packages
!pip install fastf1 pandas numpy matplotlib seaborn scikit-learn -q
!pip install --upgrade websocket-client -q

import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

print("Packages installed and imported successfully!")

# ============================================================================
# 1. MULTI-RACE DATA COLLECTION WITH CLEANING
# ============================================================================

print("Loading race data...")

# Try to load Monaco 2023 as sample
try:
    fastf1.Cache.enable_cache('./f1_cache')
    session = fastf1.get_session(2023, 'Monaco', 'R')
    session.load()
    laps_data = session.laps.copy()
    
    # Keep only necessary columns
    laps_data = laps_data[['Driver', 'LapNumber', 'LapTime', 'Compound', 'TyreLife', 'Stint']]
    laps_data['LapTimeSeconds'] = laps_data['LapTime'].dt.total_seconds()
    
    # Clean the data
    laps_data = laps_data.dropna(subset=['Driver', 'LapNumber', 'LapTimeSeconds'])
    
    # Fill missing values
    if 'Compound' in laps_data.columns:
        laps_data['Compound'] = laps_data['Compound'].fillna('UNKNOWN')
    if 'TyreLife' in laps_data.columns:
        laps_data['TyreLife'] = laps_data['TyreLife'].fillna(1)
    if 'Stint' in laps_data.columns:
        laps_data['Stint'] = laps_data['Stint'].fillna(1)
    
    print(f"Loaded {len(laps_data)} laps from Monaco 2023")
    
except Exception as e:
    print(f"Failed to load real data: {e}")
    print("Creating sample data...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 500
    
    # Create realistic lap data
    drivers = ['VER', 'HAM', 'LEC', 'PER', 'SAI', 'RUS', 'NOR', 'ALO', 'GAS', 'TSU']
    
    all_laps = []
    for driver in drivers:
        # Each driver has 50-60 laps
        n_laps = np.random.randint(50, 61)
        for lap in range(1, n_laps + 1):
            # Base time per driver
            base_time = 78.0 + np.random.normal(0, 1)
            
            # Add tire degradation
            degradation = lap * 0.03
            
            # Add randomness
            lap_time = base_time + degradation + np.random.normal(0, 0.5)
            
            # Determine stint and compound
            if lap <= 25:
                stint = 1
                compound = 'SOFT'
                tyre_life = lap
            elif lap <= 50:
                stint = 2
                compound = 'MEDIUM'
                tyre_life = lap - 25
            else:
                stint = 3
                compound = 'HARD'
                tyre_life = lap - 50
            
            all_laps.append({
                'Driver': driver,
                'LapNumber': lap,
                'LapTimeSeconds': lap_time,
                'Compound': compound,
                'TyreLife': tyre_life,
                'Stint': stint
            })
    
    laps_data = pd.DataFrame(all_laps)
    print(f"Created sample data with {len(laps_data)} laps")

print(f"\nData Summary:")
print(f"   Total laps: {len(laps_data)}")
print(f"   Drivers: {laps_data['Driver'].nunique()}")
print(f"   Average lap time: {laps_data['LapTimeSeconds'].mean():.3f}s")
print(f"   Fastest lap: {laps_data['LapTimeSeconds'].min():.3f}s")

# Show first few rows
print("\nSample data:")
print(laps_data.head())

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

print("\nFeature Engineering...")

# Encode categorical variables
le_driver = LabelEncoder()
laps_data['DriverCode'] = le_driver.fit_transform(laps_data['Driver'])

compound_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4, 'UNKNOWN': 1}
laps_data['CompoundCode'] = laps_data['Compound'].map(lambda x: compound_map.get(x, 1))

# Prepare features and target
feature_cols = ['LapNumber', 'TyreLife', 'DriverCode', 'CompoundCode', 'Stint']
X = laps_data[feature_cols].values
y = laps_data['LapTimeSeconds'].values

# Check for NaN
nan_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
X_clean = X[nan_mask]
y_clean = y[nan_mask]

print(f"Features created:")
print(f"   Samples: {len(X_clean)}")
print(f"   Features: {feature_cols}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# ============================================================================
# 3. LAP TIME PREDICTION (Random Forest)
# ============================================================================

if len(X_scaled) > 100:
    print("\nTraining Lap Time Prediction Model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_clean, test_size=0.2, random_state=42)
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Train Random Forest
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Training Complete:")
    print(f"   MSE: {mse:.3f}")
    print(f"   R² Score: {r2:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\nFeature Importance:")
    for _, row in feature_importance.iterrows():
        print(f"   {row['Feature']}: {row['Importance']:.4f}")
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6, s=20)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Lap Time (seconds)')
    plt.ylabel('Predicted Lap Time (seconds)')
    plt.title(f'Lap Time Prediction - R² = {r2:.3f}')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.barh(range(len(feature_importance)), feature_importance['Importance'], color='green')
    plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
else:
    print("Not enough data for model training")
    model = None

# ============================================================================
# 4. DRIVER DASHBOARD
# ============================================================================

print("\nCreating Driver Dashboard...")

# Calculate driver statistics
driver_stats = laps_data.groupby('Driver').agg(
    AvgLap=('LapTimeSeconds', 'mean'),
    BestLap=('LapTimeSeconds', 'min'),
    WorstLap=('LapTimeSeconds', 'max'),
    Consistency=('LapTimeSeconds', 'std'),
    TotalLaps=('LapTimeSeconds', 'count')
).reset_index().round(3)

# Sort by average lap time
driver_stats = driver_stats.sort_values('AvgLap')

print(f"\nDriver Statistics:")
print(driver_stats.to_string(index=False))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Average lap times
axes[0, 0].barh(range(len(driver_stats)), driver_stats['AvgLap'], color='steelblue')
axes[0, 0].set_yticks(range(len(driver_stats)))
axes[0, 0].set_yticklabels(driver_stats['Driver'])
axes[0, 0].set_xlabel('Average Lap Time (seconds)')
axes[0, 0].set_title('Driver Performance Ranking')
axes[0, 0].invert_yaxis()
axes[0, 0].grid(True, alpha=0.3, axis='x')

for i, time in enumerate(driver_stats['AvgLap']):
    axes[0, 0].text(time + 0.1, i, f'{time:.3f}s', va='center')

# 2. Consistency
axes[0, 1].barh(range(len(driver_stats)), driver_stats['Consistency'], color='orange')
axes[0, 1].set_yticks(range(len(driver_stats)))
axes[0, 1].set_yticklabels(driver_stats['Driver'])
axes[0, 1].set_xlabel('Standard Deviation (seconds)')
axes[0, 1].set_title('Driver Consistency')
axes[0, 1].invert_yaxis()
axes[0, 1].grid(True, alpha=0.3, axis='x')

# 3. Best lap times
top_5_best = driver_stats.nsmallest(5, 'BestLap')
axes[1, 0].barh(range(len(top_5_best)), top_5_best['BestLap'], color='green')
axes[1, 0].set_yticks(range(len(top_5_best)))
axes[1, 0].set_yticklabels(top_5_best['Driver'])
axes[1, 0].set_xlabel('Best Lap Time (seconds)')
axes[1, 0].set_title('Top 5 Best Laps')
axes[1, 0].invert_yaxis()
axes[1, 0].grid(True, alpha=0.3, axis='x')

# 4. Performance scatter
axes[1, 1].scatter(driver_stats['AvgLap'], driver_stats['Consistency'], 
                  s=driver_stats['TotalLaps']*2, alpha=0.6)

for i, row in driver_stats.iterrows():
    axes[1, 1].text(row['AvgLap'] + 0.01, row['Consistency'], row['Driver'], 
                   fontsize=9, alpha=0.8)

axes[1, 1].set_xlabel('Average Lap Time (seconds)')
axes[1, 1].set_ylabel('Consistency (std dev)')
axes[1, 1].set_title('Performance vs Consistency')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# 5. RACE STRATEGY OPTIMIZER
# ============================================================================

print("\nRace Strategy Optimizer...")

def optimize_strategy(driver):
    """Analyze race strategy for a specific driver"""
    if driver not in laps_data['Driver'].values:
        print(f"Driver {driver} not found in data")
        return
    
    driver_data = laps_data[laps_data['Driver'] == driver].copy()
    
    if len(driver_data) == 0:
        print(f"No data for driver {driver}")
        return
    
    # Group by stint
    strategy_summary = driver_data.groupby('Stint').agg(
        StartLap=('LapNumber', 'min'),
        EndLap=('LapNumber', 'max'),
        Laps=('LapNumber', 'count'),
        AvgLap=('LapTimeSeconds', 'mean'),
        BestLap=('LapTimeSeconds', 'min'),
        Compound=('Compound', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown')
    ).reset_index().round(3)
    
    print(f"\nRace Strategy for {driver}:")
    print(strategy_summary.to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Stint lengths
    axes[0].bar(strategy_summary['Stint'], strategy_summary['Laps'], 
               color=['red', 'green', 'blue'][:len(strategy_summary)])
    axes[0].set_xlabel('Stint')
    axes[0].set_ylabel('Number of Laps')
    axes[0].set_title(f'Stint Strategy - {driver}')
    axes[0].grid(True, alpha=0.3)
    
    # Add compound labels
    for i, row in strategy_summary.iterrows():
        axes[0].text(row['Stint'], row['Laps'] + 0.5, row['Compound'][0], 
                    ha='center', fontweight='bold')
    
    # Lap time progression
    for stint in strategy_summary['Stint']:
        stint_data = driver_data[driver_data['Stint'] == stint].sort_values('LapNumber')
        if len(stint_data) > 0:
            axes[1].plot(stint_data['LapNumber'], stint_data['LapTimeSeconds'], 
                        'o-', label=f'Stint {stint}', markersize=3)
    
    axes[1].set_xlabel('Lap Number')
    axes[1].set_ylabel('Lap Time (seconds)')
    axes[1].set_title(f'Lap Time Progression - {driver}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Analyze a sample driver
sample_driver = laps_data['Driver'].iloc[0]
print(f"\nAnalyzing strategy for sample driver: {sample_driver}")
optimize_strategy(sample_driver)

# ============================================================================
# 6. TELEMETRY COMPARISON TOOL
# ============================================================================

print("\nTelemetry Comparison Tool...")

def compare_drivers_simple(driver1, driver2, max_laps=30):
    """Simple comparison without complex alignment issues"""
    
    if driver1 not in laps_data['Driver'].values or driver2 not in laps_data['Driver'].values:
        print(f"One or both drivers not found")
        return
    
    df1 = laps_data[laps_data['Driver'] == driver1].copy()
    df2 = laps_data[laps_data['Driver'] == driver2].copy()
    
    if len(df1) == 0 or len(df2) == 0:
        print(f"No data for one or both drivers")
        return
    
    # Take first N laps for each driver
    df1 = df1.sort_values('LapNumber').head(max_laps)
    df2 = df2.sort_values('LapNumber').head(max_laps)
    
    # Create a simple comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Individual lap times
    axes[0, 0].plot(df1['LapNumber'], df1['LapTimeSeconds'], 'ro-', label=driver1, markersize=4, linewidth=1)
    axes[0, 0].plot(df2['LapNumber'], df2['LapTimeSeconds'], 'bo-', label=driver2, markersize=4, linewidth=1)
    axes[0, 0].set_xlabel('Lap Number')
    axes[0, 0].set_ylabel('Lap Time (seconds)')
    axes[0, 0].set_title('Lap Time Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Moving average comparison (smoother)
    window = 3
    df1_smooth = df1['LapTimeSeconds'].rolling(window=window, center=True).mean()
    df2_smooth = df2['LapTimeSeconds'].rolling(window=window, center=True).mean()
    
    axes[0, 1].plot(df1['LapNumber'], df1_smooth, 'r-', label=f'{driver1} (smoothed)', linewidth=2)
    axes[0, 1].plot(df2['LapNumber'], df2_smooth, 'b-', label=f'{driver2} (smoothed)', linewidth=2)
    axes[0, 1].set_xlabel('Lap Number')
    axes[0, 1].set_ylabel('Smoothed Lap Time (seconds)')
    axes[0, 1].set_title('Smoothed Lap Time Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Histogram of lap times
    axes[1, 0].hist(df1['LapTimeSeconds'], bins=20, alpha=0.5, label=driver1, color='red')
    axes[1, 0].hist(df2['LapTimeSeconds'], bins=20, alpha=0.5, label=driver2, color='blue')
    axes[1, 0].set_xlabel('Lap Time (seconds)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Lap Time Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Statistics table
    axes[1, 1].axis('off')
    
    # Calculate statistics
    stats = {
        'Driver': [driver1, driver2],
        'Avg Time': [f"{df1['LapTimeSeconds'].mean():.3f}s", f"{df2['LapTimeSeconds'].mean():.3f}s"],
        'Best Time': [f"{df1['LapTimeSeconds'].min():.3f}s", f"{df2['LapTimeSeconds'].min():.3f}s"],
        'Worst Time': [f"{df1['LapTimeSeconds'].max():.3f}s", f"{df2['LapTimeSeconds'].max():.3f}s"],
        'Std Dev': [f"{df1['LapTimeSeconds'].std():.3f}s", f"{df2['LapTimeSeconds'].std():.3f}s"],
        'Laps': [len(df1), len(df2)]
    }
    
    stats_df = pd.DataFrame(stats)
    
    # Create table
    table_data = [stats_df.columns.tolist()] + stats_df.values.tolist()
    table = axes[1, 1].table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color the header
    for i in range(len(stats_df.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1, 1].set_title('Performance Statistics', pad=20)
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison summary
    print(f"\nComparison Summary: {driver1} vs {driver2}")
    print(f"   {driver1}: Avg {df1['LapTimeSeconds'].mean():.3f}s, Best {df1['LapTimeSeconds'].min():.3f}s")
    print(f"   {driver2}: Avg {df2['LapTimeSeconds'].mean():.3f}s, Best {df2['LapTimeSeconds'].min():.3f}s")
    
    if df1['LapTimeSeconds'].mean() < df2['LapTimeSeconds'].mean():
        advantage = df2['LapTimeSeconds'].mean() - df1['LapTimeSeconds'].mean()
        print(f"   {driver1} has advantage of {advantage:.3f}s on average")
    else:
        advantage = df1['LapTimeSeconds'].mean() - df2['LapTimeSeconds'].mean()
        print(f"   {driver2} has advantage of {advantage:.3f}s on average")

# Compare two sample drivers
drivers = laps_data['Driver'].unique()[:2]
print(f"\nComparing {drivers[0]} vs {drivers[1]}")
compare_drivers_simple(drivers[0], drivers[1])

# ============================================================================
# 7. RACE OUTCOME FORECASTING
# ============================================================================

print("\nRace Outcome Forecasting...")

# Calculate performance metrics
race_predictions = laps_data.groupby('Driver').agg(
    AvgLap=('LapTimeSeconds', 'mean'),
    BestLap=('LapTimeSeconds', 'min'),
    Consistency=('LapTimeSeconds', 'std'),
    TotalLaps=('LapTimeSeconds', 'count')
).reset_index()

# Calculate performance score (lower is better)
race_predictions['PerformanceScore'] = (
    race_predictions['AvgLap'] * 0.6 +
    race_predictions['BestLap'] * 0.2 +
    race_predictions['Consistency'] * 0.2
)

# Sort by performance score
race_predictions = race_predictions.sort_values('PerformanceScore')
race_predictions['PredictedPosition'] = range(1, len(race_predictions) + 1)

# Calculate win probabilities (simple model)
race_predictions['WinProbability'] = 100 / race_predictions['PredictedPosition']
race_predictions['WinProbability'] = (race_predictions['WinProbability'] / 
                                      race_predictions['WinProbability'].sum()) * 100

print("\nPredicted Race Outcomes:")
print(race_predictions[['Driver', 'PredictedPosition', 'WinProbability', 
                      'AvgLap', 'BestLap']].round(3).to_string(index=False))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Predicted positions
top_10 = race_predictions.head(10)
colors = ['gold', 'silver', '#cd7f32'] + ['steelblue'] * 7

bars = axes[0].barh(range(len(top_10)), top_10['PerformanceScore'], color=colors)
axes[0].set_yticks(range(len(top_10)))
axes[0].set_yticklabels(top_10['Driver'])
axes[0].set_xlabel('Performance Score (lower is better)')
axes[0].set_title('Predicted Race Order - Top 10')
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

for i, (bar, pos) in enumerate(zip(bars, top_10['PredictedPosition'])):
    axes[0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'P{pos}', va='center')

# Win probabilities
top_8 = race_predictions.head(8)
colors_win = plt.cm.Reds(np.linspace(0.3, 0.9, len(top_8)))
bars2 = axes[1].barh(range(len(top_8)), top_8['WinProbability'], color=colors_win)
axes[1].set_yticks(range(len(top_8)))
axes[1].set_yticklabels(top_8['Driver'])
axes[1].set_xlabel('Win Probability (%)')
axes[1].set_title('Win Probability - Top 8 Drivers')
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, axis='x')

for i, (bar, prob) in enumerate(zip(bars2, top_8['WinProbability'])):
    axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{prob:.1f}%', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Key insights
print(f"\nKey Insights:")
print(f"   Predicted winner: {race_predictions.iloc[0]['Driver']} "
      f"({race_predictions.iloc[0]['WinProbability']:.1f}% chance)")
print(f"   Best lap time: {race_predictions.loc[race_predictions['BestLap'].idxmin()]['Driver']} "
      f"({race_predictions['BestLap'].min():.3f}s)")
print(f"   Most consistent: {race_predictions.loc[race_predictions['Consistency'].idxmin()]['Driver']} "
      f"({race_predictions['Consistency'].min():.3f}s std dev)")

# ============================================================================
# 8. INTERACTIVE MENU
# ============================================================================

def main_menu():
    """Interactive menu system"""
    
    print("\n" + "="*60)
    print("F1 ANALYTICS PLATFORM - INTERACTIVE MENU")
    print("="*60)
    
    while True:
        print("\nMAIN MENU:")
        print("1. View Data Summary")
        print("2. Lap Time Prediction Results")
        print("3. Driver Dashboard")
        print("4. Race Strategy Optimizer")
        print("5. Telemetry Comparison")
        print("6. Race Outcome Forecast")
        print("7. Make Custom Prediction")
        print("0. Exit")
        
        choice = input("\nSelect an option (0-7): ").strip()
        
        if choice == '1':
            print(f"\nData Summary:")
            print(f"   Total laps: {len(laps_data)}")
            print(f"   Number of drivers: {laps_data['Driver'].nunique()}")
            print(f"   Average lap time: {laps_data['LapTimeSeconds'].mean():.3f}s")
            print(f"   Fastest lap: {laps_data['LapTimeSeconds'].min():.3f}s")
            print(f"   Drivers: {', '.join(sorted(laps_data['Driver'].unique()))}")
            
        elif choice == '2':
            if model is not None:
                print(f"\nLap Time Prediction Model:")
                print(f"   Model: Random Forest Regressor")
                print(f"   R² Score: {r2:.3f}")
                print(f"   MSE: {mse:.3f}")
                print(f"\nFeature Importance:")
                print(feature_importance.to_string(index=False))
            else:
                print("Model not trained")
                
        elif choice == '3':
            print("\nDriver Dashboard")
            print(driver_stats.to_string(index=False))
            
        elif choice == '4':
            print(f"\nAvailable drivers: {', '.join(sorted(laps_data['Driver'].unique()))}")
            driver = input("Enter driver code: ").upper()
            optimize_strategy(driver)
            
        elif choice == '5':
            print(f"\nAvailable drivers: {', '.join(sorted(laps_data['Driver'].unique()))}")
            driver1 = input("Enter first driver code: ").upper()
            driver2 = input("Enter second driver code: ").upper()
            compare_drivers_simple(driver1, driver2)
            
        elif choice == '6':
            print("\nRace Outcome Forecast")
            print(race_predictions[['Driver', 'PredictedPosition', 'WinProbability', 'AvgLap']].to_string(index=False))
            
        elif choice == '7':
            if model is not None:
                print("\nMake Custom Lap Time Prediction")
                print("Enter the following details:")
                
                try:
                    driver_code = input(f"Driver code (available: {', '.join(sorted(laps_data['Driver'].unique()))}): ").upper()
                    lap_num = int(input("Lap number: "))
                    tyre_life = int(input("Tyre life (laps): "))
                    compound = input("Compound (SOFT/MEDIUM/HARD): ").upper()
                    stint = int(input("Stint number: "))
                    
                    # Prepare input
                    driver_encoded = le_driver.transform([driver_code])[0] if driver_code in le_driver.classes_ else 0
                    compound_encoded = compound_map.get(compound, 1)
                    
                    input_features = np.array([[lap_num, tyre_life, driver_encoded, compound_encoded, stint]])
                    input_scaled = scaler.transform(input_features)
                    
                    prediction = model.predict(input_scaled)[0]
                    
                    print(f"\nPredicted Lap Time: {prediction:.3f} seconds")
                    print(f"   ({int(prediction//60):02d}:{prediction%60:.3f})")
                    
                except Exception as e:
                    print(f"Error: {e}. Using sample prediction.")
                    sample_pred = 78.5
                    print(f"Sample prediction: {sample_pred:.3f}s ({int(sample_pred//60):02d}:{sample_pred%60:.3f})")
            else:
                print("Model not trained yet")
                
        elif choice == '0':
            print("\nThank you for using F1 Analytics Platform!")
            print("See you at the next race!")
            break
            
        else:
            print("Invalid choice. Please try again.")

# Start the interactive menu
print("\n" + "="*60)
print("F1 ANALYTICS PLATFORM READY!")
print("="*60)
print(f"\nPlatform Statistics:")
print(f"   Laps analyzed: {len(laps_data)}")
print(f"   Drivers in database: {laps_data['Driver'].nunique()}")
print(f"   Models trained: {'1' if model is not None else '0'}")

print("\nRecommended features to try:")
print("   5 - Compare two drivers")
print("   4 - Analyze race strategy")
print("   7 - Make custom lap time prediction")

main_menu()
