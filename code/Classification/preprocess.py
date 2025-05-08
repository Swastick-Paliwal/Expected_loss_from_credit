import pandas as pd
from config import features_to_keep

#keeping only important entries as per mentor and chatgpt, this list is made from values with less than 20% of missing values 
dataset = pd.read_csv('data/accepted_2007_to_2018Q4.csv', usecols=features_to_keep)
# Calculate percentage of NaN values for each column
nan_percentages = dataset.isna().mean() * 100

# Keep columns with less than 20% NaN values
columns_to_keep = nan_percentages[nan_percentages <= 20].index
dataset = dataset[columns_to_keep]
# Remove rows where debt_settlement_flag is NaN
big_dataset = dataset.dropna(subset=['debt_settlement_flag'])

big_dataset.to_csv('data/big_dataset.csv', index=False)
