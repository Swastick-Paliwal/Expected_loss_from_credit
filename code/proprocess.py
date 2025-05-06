import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

accepted=pd.read_csv('accepted_2007_to_2018Q4.csv')

#looking at missing entries
miss=accepted.isna().sum()
missing_entries_sorted = miss.sort_values(ascending=False)
print(missing_entries_sorted)

