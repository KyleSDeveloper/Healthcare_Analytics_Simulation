#!/usr/bin/env python3
"""
Fix for Gender One-Hot Encoding Issue
Shows the correct way to handle one-hot encoding when some expected columns don't exist.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_and_clean_data():
    """Load and perform basic cleaning on the diabetes dataset."""
    # Load data
    data_path = Path("../notebooks/diabetic_data.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Basic cleaning
    print("\n=== Basic Cleaning ===")
    
    # Drop ID columns
    id_cols = ['encounter_id', 'patient_nbr']
    df = df.drop(columns=[col for col in id_cols if col in df.columns])
    
    # Replace "?" with NaN
    df = df.replace('?', np.nan)
    
    # Drop rows with missing values in critical columns
    critical_cols = ['race', 'gender', 'age']
    initial_rows = len(df)
    df = df.dropna(subset=critical_cols)
    dropped_rows = initial_rows - len(df)
    print(f"Dropped {dropped_rows} rows with missing values in critical columns")
    
    return df

def fix_gender_encoding(df):
    """Fix the gender encoding issue by checking what columns actually exist."""
    print("\n=== Fixing Gender Encoding ===")
    
    # Check current gender values
    print("Current gender values:")
    gender_counts = df['gender'].value_counts()
    print(gender_counts)
    
    # One-hot encode gender column
    x = pd.get_dummies(df, columns=['gender'])
    
    # Check what gender columns were actually created
    gender_columns = [col for col in x.columns if col.startswith('gender_')]
    print(f"\nGender columns actually created: {gender_columns}")
    
    # Convert the gender columns to integers
    x[gender_columns] = x[gender_columns].astype(int)
    print("✅ Gender columns converted to integers")
    
    # Show the first few rows to verify
    print("\nFirst few rows of gender encoding:")
    print(x[gender_columns].head())
    
    return x

def one_hot_encode_race(df):
    """Apply one-hot encoding to race column."""
    print("\n=== One-Hot Encoding Race Column ===")
    
    # Check current race values
    print("Current race values:")
    race_counts = df['race'].value_counts()
    print(race_counts)
    
    # Create one-hot encoded columns for race
    race_dummies = pd.get_dummies(df['race'], prefix='race', dummy_na=False)
    print(f"\nCreated {len(race_dummies.columns)} race dummy columns:")
    print(race_dummies.columns.tolist())
    
    # Drop original race column and add dummy columns
    df_encoded = df.drop(columns=['race'])
    df_encoded = pd.concat([df_encoded, race_dummies], axis=1)
    
    # Convert race columns to integers
    race_columns = [col for col in df_encoded.columns if col.startswith('race_')]
    df_encoded[race_columns] = df_encoded[race_columns].astype(int)
    
    print(f"\n✅ Successfully encoded race column!")
    print(f"Dataset shape after encoding: {df_encoded.shape}")
    
    return df_encoded

def handle_target_variable(df):
    """Handle the target variable (readmitted)."""
    print("\n=== Handling Target Variable ===")
    
    if 'readmitted' in df.columns:
        print("Converting readmitted to binary...")
        df['readmitted_binary'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
        df = df.drop(columns=['readmitted'])
        print("Created binary target: readmitted_binary")
        print(f"Class distribution: {df['readmitted_binary'].value_counts().to_dict()}")
    
    return df

def main():
    """Main pipeline to fix the encoding issue."""
    print("🔧 Fixing Gender Encoding Issue")
    print("=" * 40)
    
    try:
        # Load and clean data
        df = load_and_clean_data()
        
        # Fix gender encoding
        df = fix_gender_encoding(df)
        
        # One-hot encode race
        df = one_hot_encode_race(df)
        
        # Handle target variable
        df = handle_target_variable(df)
        
        # Show final info
        print("\n=== Final Data Info ===")
        print(f"Final shape: {df.shape}")
        print(f"Data types: {df.dtypes.value_counts()}")
        
        # Show gender and race columns
        gender_cols = [col for col in df.columns if col.startswith('gender_')]
        race_cols = [col for col in df.columns if col.startswith('race_')]
        print(f"\nGender columns: {gender_cols}")
        print(f"Race columns: {race_cols}")
        
        print("\n🎉 Gender encoding issue fixed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()