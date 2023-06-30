import pandas as pd
from scipy.stats import gmean
import numpy as np
# Load the csv file
df = pd.read_csv('/Users/jack/GitHub/MLB/dk_data/ss_proj.csv')  # Replace with your file path

df = df[df['fpts'] >= 4]

# Filter out pitchers
df_hitters = df[df['pos'] != 'P']

# Group by team and calculate the average ownership
team_avg_ownership = df_hitters.groupby('team')['own%'].apply(lambda x: gmean(x) if len(x) > 0 else np.nan)
# Convert Series to DataFrame
team_avg_ownership = team_avg_ownership.reset_index()

# Rename columns
team_avg_ownership.columns = ['team', 'own%']

# Calculate total sum of ownership
total_ownership = team_avg_ownership['own%'].sum()

# Normalize the 'own%' column
team_avg_ownership['own%'] = (team_avg_ownership['own%'] / total_ownership) * 100

# Save the DataFrame to a CSV file
team_avg_ownership.to_csv('/Users/jack/GitHub/MLB/dk_data/team_stacks.csv', index=False)  # Replace with your desired file path
