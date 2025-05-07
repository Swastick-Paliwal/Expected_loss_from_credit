import pandas as pd
from config import features_to_keep

#keeping only important entries as per mentor and chatgpt, this list is made from values with less than 20% of missing values 
dataset = pd.read_csv('accepted_2007_to_2018Q4.csv', usecols=features_to_keep)
dataset.to_csv('dataset.csv', index=False)

