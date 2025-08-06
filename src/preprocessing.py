import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

def map_diagnosis(code):
    try:
        code = float(code)
        if 390 <= code <= 459 or code == 785:
            return 0  # Circulatory
        elif 460 <= code <= 519 or code == 786:
            return 1  # Respiratory
        elif 520 <= code <= 579 or code == 787:
            return 2  # Digestive
        elif 250 <= code < 251:
            return 3  # Diabetes
        elif 800 <= code <= 999:
            return 4  # Injury
        elif 710 <= code <= 739:
            return 5  # Musculoskeletal
        elif 140 <= code <= 239:
            return 6  # Neoplasms
        else:
            return 7  # Other
    except:
        return 8  # Unknown / non-numeric

def clip_outliers_iqr(df):
    df_clipped = df.copy()
    for col in df_clipped.select_dtypes(include=['int64', 'float64']):
        Q1 = df_clipped[col].quantile(0.25)
        Q3 = df_clipped[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clipped[col] = df_clipped[col].clip(lower=lower, upper=upper)
    return df_clipped

def load_and_preprocess_data(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    data_path = Path(data_path)
    df = pd.read_csv(data_path)

    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # Drop irrelevant/missing columns
    df.drop(columns=[
        'encounter_id', 'patient_nbr', 'weight', 'payer_code',
        'medical_specialty', 'examide', 'citoglipton',
        'metformin-rosiglitazone', 'metformin-pioglitazone'
    ], inplace=True)

    # Fill missing race and one-hot encode
    df['race'] = df['race'].fillna(df['race'].mode()[0])
    df = pd.get_dummies(df, columns=['race'], prefix='race')

    # Encode age bins
    age_map = {
        '[0-10)'   : 0, '[10-20)'  : 1, '[20-30)'  : 2, '[30-40)'  : 3,
        '[40-50)'  : 4, '[50-60)'  : 5, '[60-70)'  : 6, '[70-80)'  : 7,
        '[80-90)'  : 8, '[90-100)' : 9
    }
    df['age'] = df['age'].map(age_map).astype(int)

    # Binary encode drug features (only if they exist)
    drug_columns = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'insulin',
        'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone'
    ]
    for col in drug_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 0 if x == 'No' else 1).astype(int)

    # Encode change and diabetesMed
    df['change'] = df['change'].apply(lambda x: 1 if x == 'Ch' else 0).astype(int)
    df['diabetesMed'] = df['diabetesMed'].apply(lambda x: 1 if x == 'Yes' else 0).astype(int)

    # Encode gender
    df = pd.get_dummies(df, columns=['gender'])
    for col in ['gender_Female', 'gender_Male', 'gender_Unknown/Invalid']:
        if col not in df.columns:
            df[col] = 0
    df[['gender_Female', 'gender_Male', 'gender_Unknown/Invalid']] = df[
        ['gender_Female', 'gender_Male', 'gender_Unknown/Invalid']
    ].astype(int)

    # Map diag_1–3 to broad categories
    for col in ['diag_1', 'diag_2', 'diag_3']:
        df[col] = df[col].fillna('Unknown')
        df[col] = df[col].apply(map_diagnosis).astype(int)

    # Encode readmitted as multi-class target
    df['readmitted_binary'] = df['readmitted'].apply(
        lambda x: 1 if x == '<30' else (2 if x == '>30' else 0)
    )
    df.drop(columns=['readmitted'], inplace=True)

    # Clip outliers
    df = clip_outliers_iqr(df)

    # Cast remaining object columns to category
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category')

    # Final one-hot encode all categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # Separate target and features
    y = df['readmitted_binary']
    X = df.drop(columns=['readmitted_binary'])

    return X, y


