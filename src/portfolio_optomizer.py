import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
from collections import Counter

lineups = 150
overlap = 3

def count_common_athletes(lineup, all_lineups):
    if not all_lineups:
        return 0
    common_athletes = []
    for existing_lineup in all_lineups:
        common_athletes.append(len(set(lineup) & set(existing_lineup)))
    return max(common_athletes) if common_athletes else 0

def portfolio_optimizer(filepath, max_lineups, max_overlap):
    df = pd.read_csv(filepath)

    # Remove the '$' sign from 'Avg. Return' and convert the column to float
    df['Avg. Return'] = df['Avg. Return'].replace({'\$': '', ',': ''}, regex=True).astype(float)

    df_sorted = df.sort_values(by='Avg. Return', ascending=False)

    selected_lineups = []
    selected_indices = []
    player_exposure_counter = Counter()
    for index, row in tqdm(df_sorted.iterrows(), total=df_sorted.shape[0], desc="Processing lineups"):
        lineup = row[:6].tolist()
        common_athletes = count_common_athletes(lineup, selected_lineups)

        if common_athletes <= max_overlap and lineup not in selected_lineups:
            selected_lineups.append(lineup)
            selected_indices.append(index)

            # Add players to player exposure counter
            player_exposure_counter.update(lineup)

            if len(selected_lineups) == max_lineups:
                break

    # Calculate player exposure as a percentage
    player_exposure = pd.DataFrame.from_dict(player_exposure_counter, orient='index', columns=['Exposure']).reset_index().rename(columns={'index': 'Player'})
    player_exposure['Exposure'] = (player_exposure['Exposure'] / max_lineups) * 100

    # Extract id from player name
    player_exposure['dk_id'] = player_exposure['Player'].str.extract(r'\((.*?)\)', expand=False).astype(int)

    return df_sorted.loc[selected_indices], player_exposure

optimized_portfolio, player_exposure = portfolio_optimizer('/Users/jack/GitHub/PGA/output/dk_gpp_sim_lineups_30000_50000.csv', lineups, overlap)

# Load the main projections file
main_projections = pd.read_csv('/Users/jack/GitHub/PGA/dk_data/draftkings_main_projections.csv')

# Merge with the main projections dataframe to get the projected ownership
player_exposure = player_exposure.merge(main_projections[['dk_id', 'Ownership']], on='dk_id', how='left')

# Drop 'dk_id' column
player_exposure = player_exposure.drop('dk_id', axis=1)

optimized_portfolio.to_csv('/Users/jack/GitHub/PGA/output/optimized_portfolio.csv', index=False)
player_exposure.to_csv('/Users/jack/GitHub/PGA/output/player_exposure.csv', index=False)
