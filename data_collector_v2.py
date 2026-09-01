import os
import glob
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, *args, **kwargs):
        return iterable

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_timedelta_to_seconds(td_str):
    """Parses a FastF1 timedelta string to float seconds."""
    if pd.isna(td_str):
        return np.nan
    try:
        # Expected format like '0 days 00:01:25.396000'
        td_str = str(td_str).strip()
        parts = td_str.split(' ')
        time_part = parts[-1]
        h, m, s = time_part.split(':')
        return float(h) * 3600 + float(m) * 60 + float(s)
    except Exception:
        return np.nan

def load_all_data(datasets_path: str = 'datasets') -> pd.DataFrame:
    """
    Scans all subdirectories in datasets_path, reads all *_Laps.csv files,
    extracts year and circuit from filenames/directories, and concatenates them.
    
    Args:
        datasets_path: Root directory containing circuit subdirectories.
        
    Returns:
        pd.DataFrame: Concatenated raw DataFrame.
    """
    datasets_dir = Path(datasets_path)
    if not datasets_dir.exists() or not datasets_dir.is_dir():
        logging.error(f"Directory {datasets_path} does not exist.")
        return pd.DataFrame()

    csv_files = list(datasets_dir.glob('*/*_Laps.csv'))
    if not csv_files:
        logging.warning(f"No CSV files found in {datasets_path} subdirectories.")
        return pd.DataFrame()

    df_list = []
    
    for file_path in tqdm(csv_files, desc="Loading CSV files"):
        try:
            # File name example: 2022_Bahrain_Laps.csv
            filename = file_path.name
            year_str = filename.split('_')[0]
            year = int(year_str)
            
            # Directory name is circuit name
            circuit = file_path.parent.name
            
            df = pd.read_csv(file_path, low_memory=False)
            df['Year'] = year
            df['Circuit'] = circuit
            
            df_list.append(df)
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
            
    if not df_list:
        return pd.DataFrame()
        
    return pd.concat(df_list, ignore_index=True)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw DataFrame according to requirements.
    
    Args:
        df: Raw DataFrame containing all lap data.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    if df.empty:
        return df
        
    df_clean = df.copy()
    
    # 1. Convert LapTime from timedelta string to float seconds
    if 'LapTime' in df_clean.columns:
        df_clean['LapTime'] = df_clean['LapTime'].apply(parse_timedelta_to_seconds)
        
    # 2. Drop rows where LapTime is NaN or <= 0
    df_clean = df_clean.dropna(subset=['LapTime'])
    df_clean = df_clean[df_clean['LapTime'] > 0]
    
    # 3. Filter: IsAccurate == True (handle both bool and string types)
    if 'IsAccurate' in df_clean.columns:
        if df_clean['IsAccurate'].dtype == bool:
            df_clean = df_clean[df_clean['IsAccurate'] == True]
        else:
            df_clean = df_clean[df_clean['IsAccurate'].astype(str).str.lower() == 'true']
        
    # 4. Filter: Remove pit-in laps (PitInTime has a real value, not NaT/NaN)
    if 'PitInTime' in df_clean.columns:
        # Handle both actual NaN and string 'NaT'
        pit_in_is_empty = df_clean['PitInTime'].isna() | (df_clean['PitInTime'].astype(str).str.strip() == 'NaT')
        df_clean = df_clean[pit_in_is_empty]
        
    # 5. Filter: Remove pit-out laps (PitOutTime has a real value, not NaT/NaN)
    if 'PitOutTime' in df_clean.columns:
        pit_out_is_empty = df_clean['PitOutTime'].isna() | (df_clean['PitOutTime'].astype(str).str.strip() == 'NaT')
        df_clean = df_clean[pit_out_is_empty]
        
    # 6. Filter: Remove lap 1 (formation/standing start lap): LapNumber > 1
    if 'LapNumber' in df_clean.columns:
        df_clean = df_clean[pd.to_numeric(df_clean['LapNumber'], errors='coerce') > 1]
        
    # 7. Filter: Green flag only: TrackStatus == '1'
    if 'TrackStatus' in df_clean.columns:
        # TrackStatus can be int or float or string; normalize to string, strip '.0'
        ts = df_clean['TrackStatus'].astype(str).str.replace(r'\.0$', '', regex=True)
        df_clean = df_clean[ts == '1']
        
    # 8. Filter: Dry compounds only: Compound in ['SOFT', 'MEDIUM', 'HARD']
    if 'Compound' in df_clean.columns:
        df_clean = df_clean[df_clean['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])]
        
    return df_clean

def engineer_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers base features and selects required columns.
    
    Args:
        df: Cleaned DataFrame.
        
    Returns:
        pd.DataFrame: Feature-engineered DataFrame.
    """
    if df.empty:
        return df
        
    df_eng = df.copy()
    
    # fuel_load & race_progress_pct
    if 'Year' in df_eng.columns and 'Circuit' in df_eng.columns and 'LapNumber' in df_eng.columns:
        df_eng['LapNumber'] = pd.to_numeric(df_eng['LapNumber'], errors='coerce')
        total_laps = df_eng.groupby(['Year', 'Circuit'])['LapNumber'].transform('max')
        
        df_eng['fuel_load'] = total_laps - df_eng['LapNumber']
        df_eng['race_progress_pct'] = (df_eng['LapNumber'] / total_laps) * 100
    else:
        df_eng['fuel_load'] = np.nan
        df_eng['race_progress_pct'] = np.nan
        
    # stint_number
    if 'Stint' in df_eng.columns:
        df_eng['stint_number'] = pd.to_numeric(df_eng['Stint'], errors='coerce')
    else:
        df_eng['stint_number'] = np.nan
        
    # is_fresh_tyre
    if 'FreshTyre' in df_eng.columns:
        def parse_fresh_tyre(val):
            if pd.isna(val):
                return 0
            val_str = str(val).lower()
            return 1 if val_str == 'true' or val_str == '1' else 0
        df_eng['is_fresh_tyre'] = df_eng['FreshTyre'].apply(parse_fresh_tyre)
    else:
        df_eng['is_fresh_tyre'] = 0
        
    # compound_encoded
    if 'Compound' in df_eng.columns:
        compound_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}
        df_eng['compound_encoded'] = df_eng['Compound'].map(compound_map)
    else:
        df_eng['compound_encoded'] = np.nan
        
    # Sector time conversions
    for sector in ['Sector1Time', 'Sector2Time', 'Sector3Time']:
        if sector in df_eng.columns:
            df_eng[sector] = df_eng[sector].apply(parse_timedelta_to_seconds)
            
    # Keep specific columns
    keep_cols = [
        'LapTime', 'LapNumber', 'TyreLife', 'Compound', 'compound_encoded', 
        'fuel_load', 'race_progress_pct', 'stint_number', 'is_fresh_tyre', 
        'Circuit', 'Year', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 
        'Sector1Time', 'Sector2Time', 'Sector3Time', 'Position'
    ]
    
    existing_cols = [col for col in keep_cols if col in df_eng.columns]
    
    return df_eng[existing_cols]

def save_consolidated_data(df: pd.DataFrame, output_path: str = 'datasets/f1_consolidated_data.csv'):
    """
    Saves DataFrame to CSV and prints summary stats.
    
    Args:
        df: Feature-engineered DataFrame.
        output_path: Path to save the consolidated CSV.
    """
    if df.empty:
        logging.warning("DataFrame is empty. Nothing to save.")
        return
        
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_file, index=False)
    
    logging.info(f"Consolidated data saved to {output_path}")
    logging.info("--- Data Summary ---")
    logging.info(f"Total Rows: {len(df)}")
    
    if 'Circuit' in df.columns:
        logging.info(f"Unique Circuits: {df['Circuit'].nunique()}")
    if 'Year' in df.columns:
        logging.info(f"Years: {sorted(df['Year'].unique().tolist())}")

def main():
    datasets_path = 'datasets'
    output_path = 'datasets/f1_consolidated_data.csv'
    
    print("Starting data consolidation...")
    
    # 1. Load
    print(f"Loading data from {datasets_path}...")
    raw_df = load_all_data(datasets_path)
    if raw_df.empty:
        print("No data loaded. Exiting.")
        return
    raw_count = len(raw_df)
    print(f"Loaded {raw_count} raw records.")
    
    # 2. Clean
    print("Cleaning data...")
    clean_df = clean_data(raw_df)
    clean_count = len(clean_df)
    print(f"Cleaned data: kept {clean_count} records (dropped {raw_count - clean_count}).")
    
    # 3. Engineer
    print("Engineering base features...")
    features_df = engineer_base_features(clean_df)
    
    # Data Quality Report
    print("\n================ DATA QUALITY REPORT ================")
    print(f"Original Raw Rows      : {raw_count}")
    print(f"Cleaned Features Rows  : {len(features_df)}")
    print(f"Dropped Rows           : {raw_count - len(features_df)}")
    print("\n--- Per-Circuit Record Counts ---")
    if 'Circuit' in features_df.columns:
        print(features_df['Circuit'].value_counts().to_string())
    print("=====================================================\n")
    
    # 4. Save
    save_consolidated_data(features_df, output_path)
    print("Process completed successfully.")

if __name__ == '__main__':
    main()
