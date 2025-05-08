import pandas as pd

# Remove or comment out the following line to avoid memory error
# df = pd.read_csv('data/accepted_2007_to_2018Q4.csv')

# Create an empty DataFrame to store defaulters
defaulters = pd.DataFrame()

# Process the data in chunks
chunk_size = 10000
for chunk in pd.read_csv('data/accepted_2007_to_2018Q4.csv', chunksize=chunk_size, low_memory=False):
    # Filter rows where debt_settlement_flag is 'Y'
    chunk_defaulters = chunk[chunk['debt_settlement_flag'] == 'Y']
    
    # Append to defaulters DataFrame
    defaulters = pd.concat([defaulters, chunk_defaulters], ignore_index=True)

# Save the filtered data to a new CSV file
defaulters.to_csv('data/defaulters.csv', index=False)