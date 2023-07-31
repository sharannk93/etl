import os
import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler

def load_data(path):
    # Construct file paths based on the provided path
    epl_path = os.path.join(path, 'epl.csv')
    ll_path = os.path.join(path, 'laliga.csv')
    it_path = os.path.join(path, 'seriea.csv')
    lp_path = os.path.join(path, 'liganos.csv')
    nl_path = os.path.join(path, 'eredivisie.csv')
    fr_path = os.path.join(path, 'ligue1.csv')
    de_path = os.path.join(path, 'bundesliga.csv')
    fifa_data_path = os.path.join(path, 'fifa_data.csv')

    # Load data from the specified file paths
    epl = pd.read_csv(epl_path)
    ll = pd.read_csv(ll_path)
    it = pd.read_csv(it_path)
    lp = pd.read_csv(lp_path)
    nl = pd.read_csv(nl_path)
    fr = pd.read_csv(fr_path)
    de = pd.read_csv(de_path)

    # Concatenate all dataframes
    raw_df = pd.concat([epl, ll, it, lp, nl, fr, de])
    raw_df = raw_df.reset_index(drop=True)

    # Load the second dataframe
    fif = pd.read_csv(fifa_data_path)
    fif = fif[['long_name', 'player_positions', 'height_cm', 'weight_kg', 'preferred_foot', 'pace', 'player_face_url', 'join', 'club_name', 'league_name', 'dob']]

    return raw_df, fif

def create_join_column(df, dob_column, club_name_column):
    df['join'] = df[dob_column].astype(str).str.lower() + df[club_name_column].astype(str).str.lower()
    return df

def replace_slash_with_dash(df, column_name):
    df[column_name] = df[column_name].str.replace('/', '-')
    return df

def fix_nj(df):
    if isinstance(df, (pd.DatetimeIndex, pd.MultiIndex)):
        df = df.to_frame(index=False)

    # remove any pre-existing indices for ease of use in the D-Tale code, but this is not required
    df = df.reset_index().drop('index', axis=1, errors='ignore')
    df.columns = [str(c) for c in df.columns]  # update columns to strings in case they are numbers

    df.at[1193, 'passes_per_90_overall'] = 17.9
    df.at[1193, 'passes_completed_per_90_overall'] = 13.7
    df.at[1193, 'tackles_per_90_overall'] = 0.80
    df.at[1193, 'shots_per_90_overall'] = 2.1
    df.at[1193, 'interceptions_per_90_overall'] = 0.2
    df.at[1193, 'key_passes_per_90_overall'] = 1.3
    df.at[1193, 'dribbles_successful_per_90_overall'] = 1.3
    df.at[1193, 'aerial_duels_won_total_overall'] = 13
    df.at[1193, 'accurate_crosses_per_90_overall'] = 0.2
    return df 

def drop_duplicate_join_by_minutes_played(raw_df):
    # Sort the DataFrame by 'join' and 'minutes_played_overall' in descending order
    raw_df.sort_values(by=['join', 'minutes_played_overall'], ascending=[True, False], inplace=True)
    
    # Find duplicate values in 'join' column
    duplicate_joins = raw_df['join'].duplicated(keep='first')
    
    # Drop the rows with duplicate 'join' values and lower 'minutes_played_overall'
    raw_df.drop(raw_df[duplicate_joins].index, inplace=True)
    
    return raw_df

def add_stats(df):
    df['clinical'] = df['goals_per_90_overall'] / df['shots_per_90_overall']
    df['clinical'] = df['clinical'].fillna(0)
    df['shot/goal conversion'] = df['clinical'].replace([np.inf, -np.inf], 0)
    df = df.drop(['clinical'], axis=1)
    df['Shooting'] = df['goals_per_90_overall'] + df['shot/goal conversion']
    df['Vision'] = df['key_passes_per_90_overall']  
    df['Crossing'] = df['accurate_crosses_per_90_overall'] 
    df['Dribbling'] = df['dribbles_successful_per_90_overall'] 
    df['Possession'] = df['passes_completed_per_90_overall'] / df['passes_per_90_overall']
    df['Interceptions'] = df['interceptions_per_90_overall']
    df['Tackling'] = df['tackles_per_90_overall']
    df['Aerials won'] = df['aerial_duels_won_total_overall'] / df['minutes_played_overall']
    df['Defense'] = df['Interceptions'] + df['Tackling'] + df['Aerials won'] + (df['Possession'] * 4.5)
    
    return df

def fill_nan_with_zero(dataframe, columns_to_fill):
    """
    Fills NaN values with 0 for specified columns in the DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        columns_to_fill (list): A list of column names where NaN values should be filled with 0.

    Returns:
        pd.DataFrame: A new DataFrame with NaN values filled with 0 for the specified columns.
    """
    # Make a copy of the original DataFrame to avoid modifying the input DataFrame directly
    df_filled = dataframe.copy()

    # Iterate through the columns and fill NaN values with 0
    for col in columns_to_fill:
        df_filled[col] = df_filled[col].fillna(0)

    return df_filled

def prepare_data(raw_df, fif):
    # Apply replacements and transformations on raw_df
    raw_df['birthday_GMT'] = raw_df['birthday_GMT'].astype(str)
    raw_df = replace_slash_with_dash(raw_df, 'birthday_GMT')
    raw_df['Current Club'] = raw_df['Current Club'].replace({
        'PSG': 'Paris Saint Germain',
        'Inter Milan': 'Inter',
        'Bayern München': 'FC Bayern München'
    })
    raw_df = create_join_column(raw_df, 'birthday_GMT', 'Current Club')
    raw_df = fix_nj(raw_df)
    raw_df1 = drop_duplicate_join_by_minutes_played(raw_df)
    raw_df1['join'].value_counts()
    
    
    # Then perform the left join
    # For example, if you want to keep only the first occurrence of each join value in fif:
    fif_unique = fif.drop_duplicates('join', keep='first')
    result_df = raw_df1.merge(fif_unique, on='join', how='left')
    m = result_df[result_df.minutes_played_overall > 1000]
    selected_df = add_stats(m)
    selected_df['age'] = selected_df['age'].astype(str)
    selected_df['height_cm'] = selected_df['height_cm'].astype(str)
    selected_df['weight_kg'] = selected_df['weight_kg'].astype(str)
    
    
    # Scale the numeric columns
    numeric_cols = selected_df.select_dtypes(include=[float, int]).columns
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaled_data = selected_df.copy()
    scaled_data[numeric_cols] = scaler.fit_transform(selected_df[numeric_cols])
    
    # Final analysis DataFrame
    analysis = scaled_data[['full_name', 'age', 'league', 'position', 'Current Club', 'Shooting', 'shot/goal conversion',
                            'Vision', 'Crossing', 'Dribbling', 'Possession', 'Interceptions', 'Tackling', 'Aerials won',
                            'pace', 'Defense', 'height_cm', 'weight_kg', 'preferred_foot', 'player_face_url',
                            'player_positions', 'club_name']]
    analysis['age'] = analysis['age'].astype(int)
    analysis['height_cm'] = analysis['height_cm'].astype(float)
    analysis['weight_kg'] = analysis['weight_kg'].astype(float)
    filled_data = fill_nan_with_zero(analysis, ['Shooting', 'Vision', 'Possession', 'Crossing', 'Dribbling',
                                                   'Interceptions', 'Tackling', 'Aerials won', 'Defense', 'pace',
                                                   'height_cm', 'weight_kg'])
    
    return filled_data

def store_result(filled_df, file_path):
    """
    Store the filled_df DataFrame as a CSV file in the given file_path.

    Parameters:
        filled_df (pd.DataFrame): The DataFrame to be stored.
        file_path (str): The file path where the CSV file should be saved.

    Returns:
        None
    """
    try:
        filled_df.to_csv(file_path, index=False)
        print(f"Data has been successfully stored in {file_path}")
    except Exception as e:
        print(f"An error occurred while storing the data: {e}")