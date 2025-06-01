import pandas as pd

# Load the preprocessed dataset
df_large = pd.concat([pd.read_csv('diabetic_data_preprocessed.csv')] * 5, ignore_index=True)  # Or 'Data/diabetic_data.csv'
df_large.to_csv('diabetic_data_large.csv', index=False)  # Or 'data/diabetic_data_large.csv'
print("Large dataset saved with", len(df_large), "rows")