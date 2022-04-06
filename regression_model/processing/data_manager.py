# ===== LIBRARIES ======
import sys; sys.path.append('.')
from pathlib import Path

import pandas as pd
import numpy as np
import re

import joblib
from sklearn.pipeline import Pipeline

from regression_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config 


# ===== FUNCTIONS ======
# def load_dataset(*, file_name: str):
#     """Loads and quickly cleans errors from the data"""
#     # ===== LOADING TRAINING DATA ======
#     data = pd.read_csv(Path(f'{DATASET_DIR/file_name}'))
    
#     # ===== CLEANING TRAINING DATA ======
#     # Lowering case for compatability with input data
#     data.apply(lambda x: x.astype(str).str.lower())
    
#     # Removing duplicates
#     data = data.drop_duplicates()
    
#     # Removing unnecesary feature
#     data = data.drop(['Model'], axis = 1) 
    
#     # Removing rows containing errors & extreme values
#     years_with_errors = data[(data.Years < 1800) | (data.Years > 2022)].index
#     data = data.drop(years_with_errors)
    
#     extreme_prices = data[(data['Prices'] < 500) | (data['Prices'] > 600000)].index
#     data = data.drop(extreme_prices)

#     seats_with_errors = data[data.Seats > 10]['Seats'].values
#     data.Seats = data.Seats.replace(to_replace = seats_with_errors, value = np.nan)

#     # Removing rows with missing prices
#     data = data.dropna(subset = ['Prices'])
    
#     return data




# # Get information to display on html
# def get_info():
#     data = load_dataset(file_name = config.app_config.training_data_file)

#     # Get Brands
#     brands = list(data['Brand'].sort_values().unique())
#     brands.remove(np.nan)
#     brands.insert(0, " ")

#     save_file_name = f'{config.app_config.brands_save_file}.pkl'
#     save_path = TRAINED_MODEL_DIR / save_file_name
#     joblib.dump(value = brands, filename = save_path, compress=3)

#     # Get Fuels
#     fuels = data['Fuel'].unique()
#     fuels = [fuel for fuel in fuels if type(fuel) == str]
#     fuels.insert(0, " ")
#     save_file_name = f'{config.app_config.fuels_save_file}.pkl'
#     save_path = TRAINED_MODEL_DIR / save_file_name
#     joblib.dump(value = fuels, filename  = save_path, compress=3)

#     # Get Transmission
#     trans = data['Transmission'].unique()
#     trans = [tran for tran in trans if type(tran) == str]
#     trans.insert(0, " ")
#     save_file_name = f'{config.app_config.trans_save_file}.pkl'
#     save_path = TRAINED_MODEL_DIR / save_file_name
#     joblib.dump(trans, filename = save_path, compress=3)
    
#     # Get size dataset
#     num_cars = len(data)
#     save_file_name = f'{config.app_config.data_size_save_file}.pkl'
#     save_path = TRAINED_MODEL_DIR / save_file_name
#     joblib.dump(value = num_cars, filename = save_path, compress=3)


# def save_model(*, model_to_save: Pipeline):
#     """Saves the model"""
#     save_file_name = f'{config.app_config.model_save_file}.pkl'
#     save_path = TRAINED_MODEL_DIR / save_file_name
#     joblib.dump(value = model_to_save, filename= save_path, compress=3)


# def save_pipeline(*, pipeline_to_save: Pipeline):
#     """Saves the feature engineering pipeline"""
#     save_file_name = f'{config.app_config.feat_pipeline_save_file}.pkl'
#     save_path = TRAINED_MODEL_DIR / save_file_name
#     joblib.dump(pipeline_to_save, save_path, compress=3)



# def load_brands(file_name):
#     file_path = TRAINED_MODEL_DIR / file_name
#     brands_ = joblib.load(filename = file_path)
#     return brands_    

# def load_fuels(file_name):
#     file_path = TRAINED_MODEL_DIR / file_name
#     fuels_ = joblib.load(filename = file_path)
#     return fuels_    

# def load_trans(file_name):
#     file_path = TRAINED_MODEL_DIR / file_name
#     trans_ = joblib.load(filename = file_path)
#     return trans_    

# def load_data_size(file_name):
#     file_path = TRAINED_MODEL_DIR / file_name
#     data_size = joblib.load(filename = file_path)
#     return data_size    


# def load_pipeline(*, file_name: str):
#     """Loads the fitted pipeline"""
#     file_path = TRAINED_MODEL_DIR / file_name
#     trained_pipeline = joblib.load(filename = file_path)
#     return trained_pipeline
    

# def load_model(*, file_name: str):
#     """Loads the fitted model"""
#     file_path = TRAINED_MODEL_DIR / file_name
#     trained_model = joblib.load(filename = file_path)
#     return trained_model    
