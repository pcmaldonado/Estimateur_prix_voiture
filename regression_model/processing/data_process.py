# ===== LIBRARIES ======
# To handle data
import pandas as pd
import numpy as np

# To access configuration
from pathlib import Path
import sys; sys.path.append('.')
from regression_model.config.core import DATASET_DIR, config 


# ===== FUNCTIONS ======
def load_dataset():
    """Gets raw data filename, then loads it
    Arguments:
        None
    Returns:
        data: pandas DataFrame with raw data"""
    # Gets filename then loads and returns the raw data
    file_name = config.app_config.main_data_file
    data = pd.read_csv(Path(f'{DATASET_DIR/file_name}'))
    
    return data


def clean_data(data):
    """Removes data errors
    Arguments: 
        data: pandas DataFrame with raw data
    Returns:
        data: pandas DataFrame with clean data
    """
    # Lowering case for compatibility with input data
    data.apply(lambda x: x.astype(str).str.lower())
    
    # Removing duplicates
    data = data.drop_duplicates()
    
    # Removing unnecesary feature
    data = data.drop(['Model'], axis = 1) 
    
    # Removing rows containing errors & extreme values
    years_with_errors = data[(data.Years < 1800) | (data.Years > 2022)].index
    data = data.drop(years_with_errors)
    
    extreme_prices = data[(data['Prices'] < 500) | (data['Prices'] > 600000)].index
    data = data.drop(extreme_prices)

    seats_with_errors = data[data.Seats > 10]['Seats'].values
    data.Seats = data.Seats.replace(to_replace = seats_with_errors, value = np.nan)

    # Removing rows with missing prices
    data = data.dropna(subset = ['Prices'])
    
    return data


def save_clean_data(data):
    """Saves clean data into a csv file
    Arguments:
        data: pandas dataframe with data to save
    Returns:
        None
    """
    file_name = config.app_config.clean_data_file
    data.to_csv(f'{DATASET_DIR/file_name}')


def etl_process():
    data = load_dataset()
    print('Raw data Extracted')

    data = clean_data(data)
    print('Data Transformed')

    save_clean_data(data)
    print('Clean data Loaded into csv file')
