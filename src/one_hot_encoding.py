#!/usr/bin/env python3
"""
One-Hot Encoding Script for Diabetes Dataset
Converts categorical columns like race to one-hot encoded format.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the diabetes dataset from the notebooks directory."""
    data_path = Path("../notebooks/diabetic_data.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def initial_cleaning(df):
    """Perform initial cleaning steps."""
    print("\n=== Initial Cleaning ===")
    
    # Drop irrelevant ID columns
    id_cols = ['encounter_id', 'patient_nbr']
    df = df.drop(columns=[col for col in id_cols if col in df.columns])
    print(f"Dropped ID columns: {id_cols}")
    
    # Replace "?" with NaN
    df = df.replace('?', np.nan)
    print("Replaced '?' with NaN")
    
    # Drop columns with too many missing values or unclear relevance
    cols_to_drop = ['weight', 'payer_code', 'medical_specialty']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    print(f"Dropped columns: {cols_to_drop}")
    
    # Drop rows with missing values in critical columns
    critical_cols = ['race', 'gender', 'age']
    initial_rows = len(df)
    df = df.dropna(subset=critical_cols)
    dropped_rows = initial_rows - len(df)
    print(f"Dropped {dropped_rows} rows with missing values in critical columns")
    
    return df

def one_hot_encode_categorical(df, categorical_cols):
    """Apply one-hot encoding to categorical columns."""
    print(f"\n=== One-Hot Encoding Categorical Columns ===")
    print(f"Categorical columns to encode: {categorical_cols}")
    
    # Create a copy to avoid modifying original
    df_encoded = df.copy()
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            print(f"\nEncoding column: {col}")
            
            # Get unique values
            unique_vals = df_encoded[col].dropna().unique()
            print(f"  Unique values: {unique_vals}")
            
            # Create one-hot encoded columns
            dummies = pd.get_dummies(df_encoded[col], prefix=col, dummy_na=False)
            print(f"  Created {len(dummies.columns)} dummy columns: {dummies.columns.tolist()}")
            
            # Drop original column and add dummy columns
            df_encoded = df_encoded.drop(columns=[col])
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            
            print(f"  Successfully encoded {col}")
    
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

def convert_remaining_objects(df):
    """Convert remaining object columns to integers."""
    print("\n=== Converting Remaining Object Columns ===")
    
    # Get remaining object columns
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if len(object_cols) > 0:
        print(f"Remaining object columns: {object_cols}")
        
        for col in object_cols:
            if col in df.columns:
                print(f"Converting {col} to integer...")
                
                # Get unique values
                unique_vals = df[col].dropna().unique()
                
                # Create mapping for categorical encoding
                unique_vals_sorted = sorted(unique_vals)
                mapping = {val: idx for idx, val in enumerate(unique_vals_sorted)}
                
                # Apply mapping
                df[col] = df[col].map(mapping)
                
                # Convert to int, handling NaN
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                
                print(f"  Converted to integers (0-{len(unique_vals)-1})")
    else:
        print("No remaining object columns to convert!")
    
    return df

def handle_missing_values(df):
    """Handle remaining missing values."""
    print("\n=== Handling Missing Values ===")
    
    # Check missing values
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    
    if len(missing_cols) > 0:
        print(f"Columns with missing values: {missing_cols.index.tolist()}")
        
        # For integer columns, fill with median
        for col in missing_cols.index:
            if df[col].dtype in ['int64', 'Int64']:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"Filled {col} with median: {median_val}")
            else:
                # For other types, fill with mode
                mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 0
                df[col] = df[col].fillna(mode_val)
                print(f"Filled {col} with mode: {mode_val}")
    else:
        print("No missing values found!")
    
    return df

def save_cleaned_data(df):
    """Save the cleaned dataset."""
    print("\n=== Saving Cleaned Data ===")
    
    # Create data directory if it doesn't exist
    data_dir = Path("../data")
    data_dir.mkdir(exist_ok=True)
    
    # Save cleaned data
    output_path = data_dir / "diabetic_data_onehot.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data with one-hot encoding saved to {output_path}")
    
    # Save data info
    info_path = data_dir / "onehot_data_info.txt"
    with open(info_path, 'w') as f:
        f.write("Diabetes Dataset - One-Hot Encoded Data Info\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Shape: {df.shape}\n\n")
        f.write("Data Types:\n")
        f.write(str(df.dtypes.value_counts()) + "\n\n")
        f.write("Column Information:\n")
        for col in df.columns:
            f.write(f"{col}: {df[col].dtype}\n")
    
    print(f"✅ Data info saved to {info_path}")

def main():
    """Main cleaning pipeline with one-hot encoding."""
    print("🧼 Diabetes Dataset Cleaning Pipeline with One-Hot Encoding")
    print("=" * 60)
    
    try:
        # Load data
        df = load_data()
        
        # Initial cleaning
        df = initial_cleaning(df)
        
        # Define categorical columns for one-hot encoding
        categorical_cols = ['race', 'gender', 'age', 'admission_type_id', 
                           'discharge_disposition_id', 'admission_source_id']
        
        # Apply one-hot encoding to categorical columns
        df = one_hot_encode_categorical(df, categorical_cols)
        
        # Handle target variable
        df = handle_target_variable(df)
        
        # Convert remaining object columns to integers
        df = convert_remaining_objects(df)
        
        # Handle missing values
        df = handle_missing_values(df)
        
        # Final data info
        print("\n=== Final Data Info ===")
        print(f"Final shape: {df.shape}")
        print(f"Data types: {df.dtypes.value_counts()}")
        
        # Show some example columns
        print(f"\nExample columns: {df.columns[:10].tolist()}")
        
        # Save cleaned data
        save_cleaned_data(df)
        
        print("\n🎉 Data cleaning with one-hot encoding completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during cleaning: {e}")
        raise

if __name__ == "__main__":
    main() 