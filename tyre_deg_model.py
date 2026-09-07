import pandas as pd
import numpy as np
import os, glob, warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(".", "data_fastf1_v1", "laps")
OUTPUT_DIR = os.path.join(".", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEASONS = [2022, 2023, 2024, 2025]
print("Setup complete. Output directory ready.")
all_schemas = {}
file_counts = {}
all_files = []

for season in SEASONS:
    season_dir = os.path.join(DATA_DIR, str(season))
    count = 0
    for csv_file in sorted(glob.glob(os.path.join(season_dir, "*.csv"))):
        fname = os.path.basename(csv_file)
        hdr = pd.read_csv(csv_file, nrows=0)
        all_schemas[f"{season}/{fname}"] = set(hdr.columns)
        all_files.append(csv_file)
        count += 1
    file_counts[season] = count
    print(f"  {season}: {count} files located.")

unique_sets = {}
for key, cols in all_schemas.items():
    frozen = frozenset(cols)
    unique_sets.setdefault(frozen, []).append(key)

print(f"\nTotal files: {len(all_files)}")
print(f"Distinct schemas found: {len(unique_sets)}")
if len(unique_sets) == 1:
    ref_cols = list(list(unique_sets.keys())[0])
    print(f"[OK] Schema is 100% consistent. All {len(all_files)} files share the exact same {len(ref_cols)}-column schema.")
else:
    print("[WARNING] Schema differences detected!")
demo_path = os.path.join(DATA_DIR, "2024", "Bahrain_Grand_Prix.csv")
if os.path.exists(demo_path):
    demo_df = pd.read_csv(demo_path)
    print(f"Bahrain 2024 Demo Dataset:")
    print(f"  Rows: {len(demo_df):,}")
    print(f"  Cols: {len(demo_df.columns)}")
    
    # Pick a random other file to compare
    other_path = os.path.join(DATA_DIR, "2023", "Monaco_Grand_Prix.csv")
    other_df = pd.read_csv(other_path)
    
    print(f"\nComparison with Monaco 2023:")
    print(f"  Columns match? {set(demo_df.columns) == set(other_df.columns)}")
    
    demo_types = demo_df.dtypes.to_dict()
    other_types = other_df.dtypes.to_dict()
    type_diff = {k: (demo_types[k], other_types[k]) for k in demo_types if demo_types[k] != other_types[k]}
    
    if not type_diff:
        print("  Data types match 100% between the files.")
    else:
        print(f"  Type differences detected: {type_diff}")
else:
    print("Bahrain 2024 Demo file not found.")
all_dfs = []
for f in all_files:
    # Read without assuming types to avoid inference errors, then we'll cast
    all_dfs.append(pd.read_csv(f))

df = pd.concat(all_dfs, ignore_index=True)
print(f"Combined dataset loaded: {len(df):,} laps x {len(df.columns)} columns")
print(f"Memory footprint: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
duplicates = df.duplicated().sum()
if duplicates == 0:
    print("[OK] Zero exact duplicate rows found across the entire dataset.")
else:
    print(f"[WARNING] Found {duplicates:,} duplicate rows!")
numeric_cols = [
    "LapNumber", "Stint", "TyreLife", "Position",
    "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
    "AirTemp", "Humidity", "Pressure", "TrackTemp", "WindSpeed",
    "LapTimeSeconds", "PitInTimeSeconds", "PitOutTimeSeconds",
    "Sector1TimeSeconds", "Sector2TimeSeconds", "Sector3TimeSeconds",
    "TimeSeconds", "LapStartTimeSeconds",
    "GapToLeaderSeconds", "IntervalToPositionAheadSeconds",
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

bool_cols = [
    "IsPersonalBest", "FreshTyre", "Deleted", "FastF1Generated",
    "IsAccurate", "Rainfall", "HasGreen", "HasYellow",
    "HasSafetyCar", "HasRedFlag", "HasVSC", "HasVSCEnding",
]
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(bool)

print("Type casting complete. Summary of Data Types:")
type_counts = df.dtypes.value_counts()
for dtype, count in type_counts.items():
    print(f"  {dtype}: {count} columns")
# 1. Drop dead column
dead_cols = [col for col in df.columns if df[col].isna().all()]
print(f"Fully null columns: {dead_cols}")
if "LapStartDate" in df.columns:
    df = df.drop(columns=["LapStartDate"])
    print("Dropped: LapStartDate")

# 2. Create RaceId
df["RaceId"] = df["Year"].astype(str) + "_" + df["GrandPrix"].str.replace(" ", "_")

# 3. Canonical Team Mapping
TEAM_MAPPING = {
    "Alfa Romeo": "Sauber", "Kick Sauber": "Sauber",
    "AlphaTauri": "VCARB", "RB": "VCARB", "Racing Bulls": "VCARB",
    "Alpine": "Alpine", "Aston Martin": "Aston Martin",
    "Ferrari": "Ferrari", "Haas F1 Team": "Haas",
    "McLaren": "McLaren", "Mercedes": "Mercedes",
    "Red Bull Racing": "Red Bull", "Williams": "Williams",
}
df["CanonicalTeam"] = df["Team"].map(TEAM_MAPPING)

print("\nStandardization complete. (Raw data preserved, mapping added).")
print("=" * 60)
print("GLOBAL DATASET SUMMARY")
print("=" * 60)
print(f"  Total laps:       {len(df):,}")
print(f"  Unique races:     {df['RaceId'].nunique()}")
print(f"  Seasons:          {sorted(df['Year'].unique())}")
print(f"  Unique drivers:   {df['Driver'].nunique()}")
print(f"  Canonical teams:  {df['CanonicalTeam'].nunique()} (from {df['Team'].nunique()} raw names)")
print(f"  Unique circuits:  {df['GrandPrix'].nunique()}")
print(f"  Compounds:        {sorted(df['Compound'].dropna().unique())}")

print(f"\nLaps per season:")
for season in SEASONS:
    s = df[df["Year"] == season]
    print(f"  {season}: {len(s):,} laps, {s['RaceId'].nunique()} races")
print(f"{'Column':<45} {'Nulls':>8}  {'%':>6}")
print("-" * 62)
for col in df.columns:
    null_n = df[col].isna().sum()
    if null_n > 0:
        print(f"  {col:<43} {null_n:>8,}  {100*null_n/len(df):>5.1f}%")
output_path = os.path.join(OUTPUT_DIR, "data", "combined_laps.parquet")
df.to_parquet(output_path, index=False)
fsize = os.path.getsize(output_path) / 1e6
print(f"Saved dataset to: {output_path}")
print(f"File size: {fsize:.1f} MB")
print(f"Shape: {df.shape}")
print("\n[OK] Notebook 01 Data Loading & Inspection complete.")
import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(".", "outputs")
input_path = os.path.join(OUTPUT_DIR, "data", "combined_laps.parquet")

print("Loading dataset...")
df = pd.read_parquet(input_path)
total_raw_laps = len(df)
print(f"Loaded {total_raw_laps:,} laps across {df['RaceId'].nunique()} races.")
# Drop string timings since we have _Seconds floats
string_cols = [
    "Time", "LapTime", "PitInTime", "PitOutTime", 
    "Sector1Time", "Sector2Time", "Sector3Time",
    "Sector1SessionTime", "Sector2SessionTime", "Sector3SessionTime",
    "LapStartTime"
]
# Drop target leakage columns (Gaps represent the END of the lap)
leakage_cols = [
    "GapToLeader", "IntervalToPositionAhead",
    "GapToLeaderSeconds", "IntervalToPositionAheadSeconds"
]

to_drop = [c for c in string_cols + leakage_cols if c in df.columns]
df = df.drop(columns=to_drop)
print(f"Dropped {len(to_drop)} redundant/leakage columns.")
# Using the specific flag names requested where appropriate, and mapping to our logic.

# 1. Missing target
df['flag_missing_target'] = df['LapTimeSeconds'].isna() | (df['LapTimeSeconds'] <= 0)

# 2. Missing crucial tyre data
df['flag_missing_tyre'] = df['Compound'].isna() | df['TyreLife'].isna() | df['Stint'].isna()

# 3. Pit lane interference
df['is_pit_in'] = df['PitInTimeSeconds'].notna()
df['is_pit_out'] = df['PitOutTimeSeconds'].notna()

# 4. Standing starts
df['flag_lap_1'] = df['LapNumber'] == 1

# 5. Non-Dry tyres
df['flag_wet_compound'] = df['Compound'].isin(['INTERMEDIATE', 'WET'])

# 6. Race Control (SC, VSC, Red Flag)
df['is_safety_car'] = df['HasSafetyCar'] == True
df['is_vsc'] = (df['HasVSC'] == True) | (df['HasVSCEnding'] == True)
df['is_red_flag'] = df['HasRedFlag'] == True

# 7. Local Yellow (without green clearing it in the same lap)
df['is_yellow_flag'] = df['HasYellow'] & ~df['HasGreen']

# 8. Unreliable Telemetry/Sensors
df['is_deleted'] = df['Deleted'] == True
df['is_accurate'] = df['IsAccurate'].fillna(True)

# 9. Light Rain on Dry Tyres
df['flag_rainfall'] = df['Rainfall'] == True

# 10. Extreme Outliers (115% of median race/compound pace)
median_pace = df.groupby(['RaceId', 'Compound'])['LapTimeSeconds'].transform('median')
df['flag_outlier'] = (df['LapTimeSeconds'] > (median_pace * 1.15)) & df['LapTimeSeconds'].notna()

# Collect all exclusionary flags (Note: is_accurate is positive, so we invert it for exclusion)
exclusion_flags = [
    'flag_missing_target', 'flag_missing_tyre', 'is_pit_in', 'is_pit_out',
    'flag_lap_1', 'flag_wet_compound', 'is_safety_car', 'is_vsc', 'is_red_flag',
    'is_yellow_flag', 'is_deleted', 'flag_rainfall', 'flag_outlier'
]

# Create inverted accurate flag for the sum
df['flag_inaccurate'] = ~df['is_accurate']
exclusion_flags.append('flag_inaccurate')

print("Flags generated successfully.")
# Overall loss by flag
report_data = []
print(f"{'Flag Reason':<25} {'Laps Excluded':>15} {'% of Total':>12}")
print("-" * 55)
for flag in exclusion_flags:
    count = df[flag].sum()
    pct = (count / total_raw_laps) * 100
    report_data.append({'Metric': flag, 'Category': 'Overall', 'Laps_Excluded': count, 'Percentage': pct})
    print(f"{flag:<25} {count:>15,} {pct:>11.1f}%")

df['is_clean_lap'] = ~df[exclusion_flags].any(axis=1)
clean_count = df['is_clean_lap'].sum()
overall_retention = (clean_count / total_raw_laps) * 100
report_data.append({'Metric': 'Total_Clean', 'Category': 'Overall', 'Laps_Excluded': total_raw_laps - clean_count, 'Percentage': overall_retention})

print(f"\nTotal Raw Laps:   {total_raw_laps:,}")
print(f"Total Clean Laps: {clean_count:,}")
print(f"Clean Retention:  {overall_retention:.1f}%")
def analyze_dimension(dim_col):
    res = df.groupby(dim_col).agg(
        Total_Laps=('LapNumber', 'count'),
        Clean_Laps=('is_clean_lap', 'sum')
    )
    res['Retention_Pct'] = (res['Clean_Laps'] / res['Total_Laps']) * 100
    res['Loss_Pct'] = 100 - res['Retention_Pct']
    return res

dimensions = ['Year', 'GrandPrix', 'CanonicalTeam', 'Compound']

for dim in dimensions:
    if dim in df.columns:
        res = analyze_dimension(dim)
        print(f"\n--- Retention by {dim} ---")
        print(res[['Total_Laps', 'Clean_Laps', 'Retention_Pct']].sort_values('Retention_Pct'))
        
        # Add to report
        for idx, row in res.iterrows():
            report_data.append({
                'Metric': str(idx),
                'Category': dim,
                'Laps_Excluded': row['Total_Laps'] - row['Clean_Laps'],
                'Percentage': row['Loss_Pct']
            })

report_df = pd.DataFrame(report_data)
report_out = os.path.join(OUTPUT_DIR, "reports", "cleaning_report.csv")
report_df.to_csv(report_out, index=False)
print(f"\nSaved extensive report to {report_out}")
# 1. Full processed dataset (with flags)
cleaned_out = os.path.join(OUTPUT_DIR, "data", "cleaned_laps.parquet")
df.to_parquet(cleaned_out, index=False)
print(f"Saved complete processed dataset to: {cleaned_out} ({(os.path.getsize(cleaned_out)/1e6):.1f} MB)")

# 2. Subset for tyre modelling (only clean laps)
clean_subset_out = os.path.join(OUTPUT_DIR, "clean_laps.parquet")
clean_df = df[df['is_clean_lap']].copy()
# Drop flags since they are all False in this subset to save space
cols_to_drop = exclusion_flags + ['is_clean_lap', 'is_accurate']
clean_df = clean_df.drop(columns=[c for c in cols_to_drop if c in clean_df.columns])
clean_df.to_parquet(clean_subset_out, index=False)
print(f"Saved clean-lap subset to: {clean_subset_out} ({(os.path.getsize(clean_subset_out)/1e6):.1f} MB)")

print("\n[OK] Notebook 02 Data Cleaning complete.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(".", "outputs")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
input_path = os.path.join(OUTPUT_DIR, "data", "cleaned_laps.parquet")

print("Loading dataset...")
df = pd.read_parquet(input_path)
print(f"Loaded {len(df):,} laps.")
# Sort strictly chronologically per driver
df = df.sort_values(['Year', 'Round', 'Driver', 'LapNumber'])

# Forward-fill Stint, TyreLife, Compound within Race & Driver to bridge 1-lap dropouts
cols_to_fill = ['Stint', 'TyreLife', 'Compound']
for col in cols_to_fill:
    df[col] = df.groupby(['RaceId', 'Driver'])[col].ffill(limit=1)

print(f"Missing values remaining after ffill:")
for col in cols_to_fill:
    print(f"  {col}: {df[col].isna().sum()}")
# Stint counter resets per race/driver, so we create a global ID
df['StintId'] = df['RaceId'].astype(str) + '_' + df['Driver'] + '_S' + df['Stint'].astype(str)
df.loc[df['Stint'].isna(), 'StintId'] = np.nan

# StintLap: Sequential counter within the StintId (phase of the stint)
df['StintLap'] = df.groupby('StintId').cumcount() + 1
df.loc[df['StintId'].isna(), 'StintLap'] = np.nan

# Preserve original TyreLife, but also assign it to TyreAge as requested
df['TyreAge'] = df['TyreLife']

# TyreChangeFlag
df['PrevStintId'] = df.groupby(['RaceId', 'Driver'])['StintId'].shift(1)
df['TyreChangeFlag'] = (df['StintId'] != df['PrevStintId']) & df['PrevStintId'].notna() & df['StintId'].notna()
df = df.drop(columns=['PrevStintId'])

print("Globally unique StintId, StintLap, TyreAge, and TyreChangeFlag created.")
# A stint needs at least 3 valid racing laps to establish a baseline.
clean_laps_per_stint = df.groupby('StintId')['is_clean_lap'].transform('sum')
# 1 = Good Stint, 0 = Bad Stint (e.g., short stints < 3 clean laps)
df['StintQualityFlag'] = np.where(clean_laps_per_stint >= 3, 1, 0)
df.loc[df['StintId'].isna(), 'StintQualityFlag'] = 0

good_stints = df[df['StintQualityFlag'] == 1]['StintId'].nunique()
bad_stints = df[df['StintQualityFlag'] == 0]['StintId'].nunique() - 1 # exclude NaN
print(f"Identified {good_stints:,} high-quality stints and {bad_stints:,} short/low-quality stints.")
# 1. Unique Compound per Stint
compound_check = df.groupby('StintId')['Compound'].nunique()
if compound_check.max() > 1:
    print("[WARNING] Found a StintId containing multiple compounds!")
else:
    print("[OK] All StintIds contain exactly 1 compound.")

# 2. StintLap starts at 1
min_stintlap = df.dropna(subset=['StintId']).groupby('StintId')['StintLap'].min()
if min_stintlap.max() > 1:
    print("[WARNING] StintLap does not start at 1 for all stints!")
else:
    print("[OK] StintLap correctly starts at 1.")

# 3. TyreLife Monotonicity
tyrelife_diff = df.groupby('StintId')['TyreLife'].diff()
non_monotonic = df[tyrelife_diff < 0]
if len(non_monotonic) > 0:
    print(f"[WARNING] Found {len(non_monotonic)} laps where TyreLife decreased within a stint!")
else:
    print("[OK] TyreLife is perfectly monotonic within all stints.")
report_data = []
def report_dimension(dim_col):
    stint_level = df.dropna(subset=['StintId']).groupby([dim_col, 'StintId'])['StintQualityFlag'].first().reset_index()
    res = stint_level.groupby(dim_col).agg(
        Total_Stints=('StintId', 'count'),
        Good_Stints=('StintQualityFlag', 'sum')
    )
    res['Quality_Pct'] = (res['Good_Stints'] / res['Total_Stints']) * 100
    return res

dimensions = ['Year', 'GrandPrix', 'CanonicalTeam', 'Compound']

for dim in dimensions:
    if dim in df.columns:
        res = report_dimension(dim)
        print(f"\n--- Stint Quality by {dim} ---")
        print(res.sort_values('Quality_Pct', ascending=False).head(10))
        for idx, row in res.iterrows():
            report_data.append({
                'Dimension': dim,
                'Value': str(idx),
                'Total_Stints': row['Total_Stints'],
                'Good_Stints': row['Good_Stints'],
                'Quality_Pct': row['Quality_Pct']
            })

report_df = pd.DataFrame(report_data)
report_out = os.path.join(OUTPUT_DIR, "reports", "stint_quality_report.csv")
report_df.to_csv(report_out, index=False)
print(f"\nSaved quality report to {report_out}")
plt.figure(figsize=(10, 6))
sns.histplot(data=df[df['StintQualityFlag']==1], x='TyreLife', hue='Compound', bins=30, multiple='stack')
plt.title('TyreLife Distribution in High-Quality Stints')
plt.xlabel('Physical Tyre Age (Laps)')
plt.ylabel('Lap Count')
plot_path = os.path.join(PLOT_DIR, 'tyre_life_distribution.png')
plt.savefig(plot_path)
plt.close()
print(f"Saved diagnostic plot to {plot_path}")
out_path = os.path.join(OUTPUT_DIR, "data", "tyre_stints.parquet")
df.to_parquet(out_path, index=False)
fsize = os.path.getsize(out_path) / 1e6
print(f"Saved stint dataset to: {out_path}")
print(f"File size: {fsize:.1f} MB")
print("\n[OK] Notebook 04 Tyre Stint Processing complete.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(".", "outputs")
input_path = os.path.join(OUTPUT_DIR, "data", "tyre_stints.parquet")

print("Loading dataset...")
df = pd.read_parquet(input_path)
# Ensure strictly chronologically sorted
df = df.sort_values(['Year', 'Round', 'Driver', 'LapNumber']).reset_index(drop=True)
print(f"Loaded {len(df):,} laps.")
# 1. Total Race Laps (Deterministic)
total_laps = df.groupby('RaceId')['LapNumber'].max().rename('Total_Race_Laps')
df = df.merge(total_laps, on='RaceId', how='left')

# 2. Fuel Weight Penalty
# ~0.065s per lap of fuel burned
df['Fuel_Weight_Penalty'] = (df['Total_Race_Laps'] - df['LapNumber']) * 0.065

# 3. Circuit Base Pace (Causal Historical Baseline)
races = df[['Year', 'Round', 'RaceId', 'GrandPrix']].drop_duplicates().sort_values(['Year', 'Round'])
historical_pace = {}

for i, row in races.iterrows():
    gp = row['GrandPrix']
    # Look ONLY at past races
    past_races = races[(races['GrandPrix'] == gp) & 
                       ((races['Year'] < row['Year']) | 
                        ((races['Year'] == row['Year']) & (races['Round'] < row['Round'])))]
    
    if len(past_races) > 0:
        past_ids = past_races['RaceId'].tolist()
        past_pace = df[(df['RaceId'].isin(past_ids)) & (df['is_clean_lap']) & (~df['flag_wet_compound'])]['LapTimeSeconds'].min()
        historical_pace[row['RaceId']] = past_pace if pd.notna(past_pace) else 90.0
    else:
        # First time at track in dataset: Bootstrap using only the first 10 laps to prevent full-race leakage
        first_10 = df[(df['RaceId'] == row['RaceId']) & (df['is_clean_lap']) & (~df['flag_wet_compound']) & (df['LapNumber'] <= 10)]
        if len(first_10) > 0:
            historical_pace[row['RaceId']] = first_10['LapTimeSeconds'].min()
        else:
            historical_pace[row['RaceId']] = df[(df['RaceId'] == row['RaceId']) & (df['is_clean_lap'])]['LapTimeSeconds'].min()

df['Circuit_Base_Pace'] = df['RaceId'].map(historical_pace)

# 4. Target Residual
df['Expected_Non_Tyre_Pace'] = df['Circuit_Base_Pace'] + df['Fuel_Weight_Penalty']
df['Target_Tyre_Degradation'] = df['LapTimeSeconds'] - df['Expected_Non_Tyre_Pace']

print("Performance normalization complete. Causal Target_Tyre_Degradation created.")
# 1. Non-linear Tyre Age (Cliff Capture)
df['Log_TyreAge'] = np.log1p(df['TyreAge'])

# 2. Race Progress
df['RaceProgressFraction'] = df['LapNumber'] / df['Total_Race_Laps']

# 3. Compound Ordinal Encoding (Soft=1, Medium=2, Hard=3)
compound_map = {'SOFT': 1, 'MEDIUM': 2, 'HARD': 3, 'INTERMEDIATE': -1, 'WET': -2, 'TEST_UNKNOWN': 0}
df['Compound_Encoded'] = df['Compound'].map(compound_map).fillna(0)

print("Tyre and Progress features engineered.")
def calc_slope(y):
    y_clean = y.dropna()
    if len(y_clean) < 3: return np.nan
    x = np.arange(len(y_clean))
    return np.polyfit(x, y_clean, 1)[0]

# 1. Lagged Residual (t-1)
# We only lag within the same StintId!
df['Lag1_Pace_Residual'] = df.groupby('StintId')['Target_Tyre_Degradation'].shift(1)

# 2. Rolling 3-Lap Trend (t-3 to t-1)
df['Rolling3_Degradation_Trend'] = df.groupby('StintId')['Lag1_Pace_Residual'].transform(lambda x: x.rolling(3).apply(calc_slope))

print("Causal rolling trends (strictly using t-1 lagged data) engineered.")
# Calculate the team's median pace residual in PAST races.
# First, get team median per race
team_race_pace = df[(df['is_clean_lap']) & (~df['flag_wet_compound'])].groupby(['Year', 'Round', 'RaceId', 'CanonicalTeam'])['Target_Tyre_Degradation'].median().reset_index()
team_race_pace = team_race_pace.sort_values(['Year', 'Round'])

# Calculate expanding historical median (shifted to prevent leakage of current race)
team_race_pace['Team_Median_Pace_Lag1'] = team_race_pace.groupby('CanonicalTeam')['Target_Tyre_Degradation'].transform(lambda x: x.expanding().median().shift(1))

# Merge back
df = df.merge(team_race_pace[['RaceId', 'CanonicalTeam', 'Team_Median_Pace_Lag1']], on=['RaceId', 'CanonicalTeam'], how='left')

# For the first race of the dataset where Lag1 is NaN, fill with the global median to prevent NaNs
global_median = df['Target_Tyre_Degradation'].median()
df['Team_Median_Pace_Lag1'] = df['Team_Median_Pace_Lag1'].fillna(global_median)

print("Team historical context (Team_Median_Pace_Lag1) engineered.")
feature_cols = [
    'TyreAge', 'Log_TyreAge', 'StintLap', 'Compound_Encoded', 
    'RaceProgressFraction', 'TrackTemp', 'AirTemp', 
    'is_safety_car', 'is_vsc', 'Lag1_Pace_Residual', 
    'Rolling3_Degradation_Trend', 'Team_Median_Pace_Lag1'
]

print("\nFeature Missing Value Check:")
for col in feature_cols + ['Target_Tyre_Degradation']:
    print(f"  {col}: {df[col].isna().sum():,} NaNs ({(df[col].isna().sum()/len(df))*100:.1f}%)")

out_path = os.path.join(OUTPUT_DIR, "data", "features_engineered.parquet")
df.to_parquet(out_path, index=False)
print(f"\nSaved feature-engineered dataset to: {out_path} ({(os.path.getsize(out_path)/1e6):.1f} MB)")
print("\n[OK] Notebook 05 Feature Engineering complete.")
import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(".", "outputs")
input_path = os.path.join(OUTPUT_DIR, "data", "features_engineered.parquet")

print("Loading engineered dataset...")
df = pd.read_parquet(input_path)
total_raw = len(df)
print(f"Loaded {total_raw:,} total laps.")
# Rolling and lagged features were already safely calculated using the full sequence.
# We can now safely drop non-representative laps (Safety Cars, Out-Laps, Outliers) 
# and short/bad stints so the XGBoost model learns pure tyre physics.

clean_df = df[(df['is_clean_lap']) & (df['StintQualityFlag'] == 1) & (~df['flag_wet_compound'])].copy()

# Drop rows where critical rolling/lag features are NaN (e.g. the first lap of a stint)
critical_features = ['Target_Tyre_Degradation', 'Lag1_Pace_Residual', 'Rolling3_Degradation_Trend', 'Circuit_Base_Pace']
clean_df = clean_df.dropna(subset=critical_features)

print(f"Filtered down to {len(clean_df):,} highly rigorous modelling laps ({(len(clean_df)/total_raw)*100:.1f}% retention).")
# Approved Chronological Split:
# Train: 2022, 2023, 2024 (Rounds 1-14)
# Val: 2024 (Rounds 15-24)
# Test: 2025

condition_train = (clean_df['Year'] < 2024) | ((clean_df['Year'] == 2024) & (clean_df['Round'] <= 14))
condition_val = (clean_df['Year'] == 2024) & (clean_df['Round'] > 14)
condition_test = (clean_df['Year'] == 2025)

train_df = clean_df[condition_train].copy()
val_df = clean_df[condition_val].copy()
test_df = clean_df[condition_test].copy()

print(f"Train set: {len(train_df):,} laps ({(len(train_df)/len(clean_df))*100:.1f}%)")
print(f"Val set:   {len(val_df):,} laps ({(len(val_df)/len(clean_df))*100:.1f}%)")
print(f"Test set:  {len(test_df):,} laps ({(len(test_df)/len(clean_df))*100:.1f}%)")
# 1. Date Overlap Check
train_max_date = train_df['LapStartTimeSeconds'].max()
val_min_date = val_df['LapStartTimeSeconds'].min()
val_max_date = val_df['LapStartTimeSeconds'].max()
test_min_date = test_df['LapStartTimeSeconds'].min()

print("Chronological Integrity Check:")
if train_max_date < val_min_date:
    print("  [OK] Train strictly precedes Validation.")
else:
    print("  [WARNING] Train and Val timestamps overlap!")

if pd.notna(test_min_date) and val_max_date < test_min_date:
    print("  [OK] Validation strictly precedes Test.")
elif pd.isna(test_min_date):
    print("  [INFO] Test set is currently empty (2025 data might not be present in the demo subset).")
else:
    print("  [WARNING] Val and Test timestamps overlap!")

# 2. Sequence Integrity Check
# Ensure no StintId is split across datasets
train_stints = set(train_df['StintId'].unique())
val_stints = set(val_df['StintId'].unique())
test_stints = set(test_df['StintId'].unique())

overlap_tv = train_stints.intersection(val_stints)
overlap_vt = val_stints.intersection(test_stints)

print("\nSequence Integrity Check:")
if len(overlap_tv) == 0 and len(overlap_vt) == 0:
    print("  [OK] Zero stints span across split boundaries.")
else:
    print(f"  [WARNING] Found {len(overlap_tv)} stints spanning Train/Val boundaries!")
train_out = os.path.join(OUTPUT_DIR, "data", "train_data.parquet")
val_out = os.path.join(OUTPUT_DIR, "data", "val_data.parquet")
test_out = os.path.join(OUTPUT_DIR, "data", "test_data.parquet")

train_df.to_parquet(train_out, index=False)
val_df.to_parquet(val_out, index=False)
test_df.to_parquet(test_out, index=False)

print("\nSuccessfully saved Train, Val, and Test datasets.")
print("[OK] Notebook 06 Temporal Split complete.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings
warnings.filterwarnings("ignore")

# Set aesthetics for premium plots
plt.style.use('dark_background')
sns.set_palette("husl")

OUTPUT_DIR = os.path.join(".", "outputs")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

input_path = os.path.join(OUTPUT_DIR, "data", "features_engineered.parquet")
print("Loading dataset...")
df = pd.read_parquet(input_path)

# Filter for clean modelling laps only to reduce noise in plots
plot_df = df[(df['is_clean_lap']) & (df['StintQualityFlag'] == 1) & (~df['flag_wet_compound'])].copy()
print(f"Loaded {len(plot_df):,} clean laps for validation.")
# 1. Target Distribution
plt.figure(figsize=(10, 6))
sns.histplot(plot_df['Target_Tyre_Degradation'], bins=100, kde=True, color='cyan')
plt.title('Target Distribution: Tyre Pace Residual')
plt.xlabel('Pace Deficit (Seconds)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'target_distribution.png'))
plt.close()

# 2. Tyre Age vs Target
plt.figure(figsize=(12, 6))
sns.lineplot(data=plot_df, x='TyreAge', y='Target_Tyre_Degradation', ci='sd', color='magenta')
plt.title('Average Degradation Curve: Tyre Age vs Pace Deficit')
plt.xlabel('Physical Tyre Age (Laps)')
plt.ylabel('Pace Deficit (Seconds)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'target_vs_tyreage.png'))
plt.close()

# 3. Target by Compound
plt.figure(figsize=(10, 6))
sns.boxplot(data=plot_df, x='Compound', y='Target_Tyre_Degradation', order=['SOFT', 'MEDIUM', 'HARD'])
plt.title('Pace Deficit by Compound')
plt.xlabel('Compound')
plt.ylabel('Pace Deficit (Seconds)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'target_by_compound.png'))
plt.close()

# 4. Target by Team
plt.figure(figsize=(14, 6))
sns.boxplot(data=plot_df, x='CanonicalTeam', y='Target_Tyre_Degradation')
plt.title('Pace Deficit by Team (Tyre Management)')
plt.xticks(rotation=45)
plt.xlabel('Team')
plt.ylabel('Pace Deficit (Seconds)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'target_by_team.png'))
plt.close()

# 5. Target by Circuit
plt.figure(figsize=(14, 6))
circuit_medians = plot_df.groupby('GrandPrix')['Target_Tyre_Degradation'].median().sort_values()
sns.boxplot(data=plot_df, x='GrandPrix', y='Target_Tyre_Degradation', order=circuit_medians.index)
plt.title('Pace Deficit by Circuit (Track Severity)')
plt.xticks(rotation=90)
plt.xlabel('Grand Prix')
plt.ylabel('Pace Deficit (Seconds)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'target_by_circuit.png'))
plt.close()

print("Diagnostic plots generated and saved to outputs/plots/")
# Generate aggregated metrics for the report
report_data = []

def analyze_target(dim_col):
    res = plot_df.groupby(dim_col)['Target_Tyre_Degradation'].agg(['count', 'mean', 'median', 'std']).reset_index()
    res['Dimension'] = dim_col
    res.rename(columns={dim_col: 'Value'}, inplace=True)
    return res

dimensions = ['Year', 'GrandPrix', 'Compound', 'CanonicalTeam']
for dim in dimensions:
    if dim in plot_df.columns:
        res = analyze_target(dim)
        report_data.append(res)

report_df = pd.concat(report_data, ignore_index=True)
report_df = report_df[['Dimension', 'Value', 'count', 'mean', 'median', 'std']]

out_report = os.path.join(OUTPUT_DIR, "reports", "target_validation_report.csv")
report_df.to_csv(out_report, index=False)

print(f"Validation report saved to {out_report}")
print(report_df.head(10))
# Save the fully featured, engineered, and normalized master dataset
# (Using the requested filename `tyre_model_dataset.parquet`)
out_path = os.path.join(OUTPUT_DIR, "tyre_model_dataset.parquet")
df.to_parquet(out_path, index=False)
fsize = os.path.getsize(out_path) / 1e6
print(f"Saved master dataset to: {out_path} ({fsize:.1f} MB)")
print("\n[OK] Notebook 07 Target Validation complete.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore")
plt.style.use('dark_background')
sns.set_palette("husl")

OUTPUT_DIR = os.path.join(".", "outputs")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

print("Loading Train and Validation datasets (quarantining Test)...")
train_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "data", "train_data.parquet"))
val_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "data", "val_data.parquet"))
# We load Test only to generate final predictions at the very end, but do NOT use it for any decisions.
test_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "data", "test_data.parquet"))

print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
# Define exactly the features we audited and approved
features = [
    'TyreAge', 'Log_TyreAge', 'StintLap', 'Compound_Encoded', 
    'RaceProgressFraction', 'TrackTemp', 'AirTemp', 
    'is_safety_car', 'is_vsc', 'Lag1_Pace_Residual', 
    'Rolling3_Degradation_Trend', 'Team_Median_Pace_Lag1'
]
target = 'Target_Tyre_Degradation'

X_train = train_df[features]
y_train = train_df[target]
X_val = val_df[features]
y_val = val_df[target]

print(f"Using {len(features)} causal features.")
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    return {'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Bias': bias}

metrics = []

# Baseline 1: Global Mean Prediction
mean_pred = np.full_like(y_val, y_train.mean())
metrics.append(evaluate_model('Global Mean', y_val, mean_pred))

# Baseline 2: Simple Tyre-Age Linear Regression
lr_age = LinearRegression()
lr_age.fit(X_train[['TyreAge']], y_train)
age_pred = lr_age.predict(X_val[['TyreAge']])
metrics.append(evaluate_model('Linear Reg (TyreAge Only)', y_val, age_pred))

# Baseline 3: Multiple Linear Regression
lr_multi = LinearRegression()
lr_multi.fit(X_train, y_train)
multi_pred = lr_multi.predict(X_val)
metrics.append(evaluate_model('Multiple Linear Reg', y_val, multi_pred))

baseline_results = pd.DataFrame(metrics)
print(baseline_results)
print("Training XGBoost Regressor (Baseline Hyperparameters)...")

xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist'
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=50
)

# Predict on Validation
val_df['XGB_Pred'] = xgb_model.predict(X_val)
xgb_metrics = evaluate_model('XGBoost Baseline', val_df[target], val_df['XGB_Pred'])

results_df = pd.concat([baseline_results, pd.DataFrame([xgb_metrics])], ignore_index=True)
print(results_df)
importance = pd.DataFrame({
    'Feature': features,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance, x='Importance', y='Feature')
plt.title('XGBoost Feature Importance (Gain)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'xgb_feature_importance.png'))
plt.show()
val_df['Residual'] = val_df['XGB_Pred'] - val_df[target]

# 1. Actual vs Predicted
plt.figure(figsize=(8, 8))
plt.scatter(val_df[target], val_df['XGB_Pred'], alpha=0.2, s=2, color='cyan')
plt.plot([-5, 5], [-5, 5], 'r--')
plt.title('Actual vs Predicted Pace Deficit (Validation)')
plt.xlabel('Actual Target')
plt.ylabel('Predicted Target')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'actual_vs_predicted.png'))
plt.close()

# 2. Residual Distribution
plt.figure(figsize=(10, 5))
sns.histplot(val_df['Residual'], bins=100, kde=True, color='magenta')
plt.title('Residual Distribution (Pred - Actual)')
plt.xlabel('Error (Seconds)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'residual_distribution.png'))
plt.close()

# 3. Residual vs Tyre Age
plt.figure(figsize=(12, 5))
sns.boxplot(data=val_df[val_df['TyreAge'] <= 30], x='TyreAge', y='Residual')
plt.axhline(0, color='r', linestyle='--')
plt.title('Error (Residuals) by Tyre Age')
plt.xlabel('Tyre Age')
plt.ylabel('Residual (Pred - Actual)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'residual_vs_tyreage.png'))
plt.close()

# 4. Degradation Trajectory by Stint (Sample)
sample_stints = val_df['StintId'].unique()[:4]
sample_df = val_df[val_df['StintId'].isin(sample_stints)]

plt.figure(figsize=(12, 6))
sns.lineplot(data=sample_df, x='StintLap', y=target, hue='StintId', marker='o', alpha=0.5, legend=False)
sns.lineplot(data=sample_df, x='StintLap', y='XGB_Pred', hue='StintId', linestyle='--', legend=False)
plt.title('Stint Trajectories: Solid=Actual, Dashed=Predicted')
plt.xlabel('Stint Lap')
plt.ylabel('Pace Deficit')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'trajectory_by_stint.png'))
plt.close()

print("Diagnostic plots generated and saved.")
# Final Evaluation on the untouched 2025 Test Set (ONLY the baseline model, to establish the benchmark)
X_test = test_df[features]
y_test = test_df[target]
test_df['XGB_Pred'] = xgb_model.predict(X_test)
test_metrics = evaluate_model('XGBoost Baseline (TEST SET 2025)', y_test, test_df['XGB_Pred'])

final_metrics_df = pd.concat([results_df, pd.DataFrame([test_metrics])], ignore_index=True)
print(final_metrics_df)

# Save metrics
metrics_path = os.path.join(OUTPUT_DIR, "reports", "xgboost_baseline_metrics.csv")
final_metrics_df.to_csv(metrics_path, index=False)

# Save predictions
val_df[['RaceId', 'Driver', 'LapNumber', 'StintId', target, 'XGB_Pred']].to_parquet(os.path.join(OUTPUT_DIR, "predictions", "val_predictions.parquet"), index=False)
test_df[['RaceId', 'Driver', 'LapNumber', 'StintId', target, 'XGB_Pred']].to_parquet(os.path.join(OUTPUT_DIR, "predictions", "test_predictions.parquet"), index=False)

# Save model
model_path = os.path.join(OUTPUT_DIR, "models", "xgboost_baseline.json")
xgb_model.save_model(model_path)

print(f"\nModel saved to: {model_path}")
print(f"Metrics saved to: {metrics_path}")
print("\n[OK] Notebook 08 XGBoost Baseline complete.")