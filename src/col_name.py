import pandas as pd

# Read the CSV file
df = pd.read_csv('/Users/jack/GitHub/MLB/dk_data/topbatters.csv')

# Rename the columns
df.rename(columns={
    'Tm': 'team',
    'Sal': 'salary',
    'Pos': 'pos'

}, inplace=True)


# Create a new column coloum caled "ceiling" and set it equal to FP + StdDev
df['ceiling'] = df['FP'] + df['StdDev']


# Save the final dataframe to a new CSV file
df.to_csv('/Users/jack/GitHub/MLB/dk_data/topbatters.csv', index=False)
