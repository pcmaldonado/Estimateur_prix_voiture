# ===== LIBRARIES ======
# To handle data
import pandas as pd
import numpy as np

# To export model/data
import joblib

# To access configuration
import sys; sys.path.append('.')
from regression_model.config.core import TRAINED_MODEL_DIR, DATASET_DIR, config 
from pathlib import Path


# ===== FUNCTIONS ======
# Get information to display on html
def get_info():
    """After loading the cleaned data, it creates pickle files 
    that contain information needed for the web application
    Arguments:
        None
    Returns:
        None
    """
    # Load clean data
    file_name = config.app_config.clean_data_file
    data = pd.read_csv(Path(f'{DATASET_DIR/file_name}'))

    # Get Brands
    brands = list(data['Brand'].sort_values().unique())
    brands.remove(np.nan)
    brands.insert(0, " ")
    save_file_name = config.app_config.brands_save_file
    save_path = TRAINED_MODEL_DIR / save_file_name
    joblib.dump(value = brands, filename = save_path, compress=3)

    # Get Fuels
    fuels = data['Fuel'].unique()
    fuels = [fuel for fuel in fuels if type(fuel) == str]
    fuels.insert(0, " ")
    save_file_name = config.app_config.fuels_save_file
    save_path = TRAINED_MODEL_DIR / save_file_name
    joblib.dump(value = fuels, filename  = save_path, compress=3)

    # Get Transmission
    trans = data['Transmission'].unique()
    trans = [tran for tran in trans if type(tran) == str]
    trans.insert(0, " ")
    save_file_name = config.app_config.trans_save_file
    save_path = TRAINED_MODEL_DIR / save_file_name
    joblib.dump(trans, filename = save_path, compress=3)
    
    # Get size dataset
    num_cars = len(data)
    save_file_name = config.app_config.data_size_save_file
    save_path = TRAINED_MODEL_DIR / save_file_name
    joblib.dump(value = num_cars, filename = save_path, compress=3)


# Load information to display on html
def load_brands():
    """Loads pickle file containing brands names
    Arguments:
        None
    Returns:
        Brands_: list"""
    file_name = config.app_config.brands_save_file
    file_path = TRAINED_MODEL_DIR / file_name
    brands_ = joblib.load(filename = file_path)
    return brands_    


def load_fuels():
    """Loads pickle file containing fuels labels
    Arguments:
        None
    Returns:
        Fuels_: list"""
    file_name = config.app_config.fuels_save_file
    file_path = TRAINED_MODEL_DIR / file_name
    fuels_ = joblib.load(filename = file_path)
    return fuels_    


def load_transmission():
    """Loads pickle file containing transmission labels 
    Arguments:
        None
    Returns:
        Trans_: list"""
    file_name = config.app_config.trans_save_file
    file_path = TRAINED_MODEL_DIR / file_name
    trans_ = joblib.load(filename = file_path)
    return trans_    


def load_data_size():
    """Loads pickle file containing data size information
    Arguments:
        None
    Returns:
        data_size: int"""
    file_name = config.app_config.data_size_save_file
    file_path = TRAINED_MODEL_DIR / file_name
    data_size = joblib.load(filename = file_path)
    return data_size    


if __name__ == '__main__':
    # Create pickle files with information
    # needed for web app display
    get_info()    