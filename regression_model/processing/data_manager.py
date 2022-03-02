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
def load_dataset(*, file_name: str):
    """Loads and quickly cleans errors from the data"""
    # ===== LOADING TRAINING DATA ======
    data = pd.read_csv(Path(f'{DATASET_DIR/file_name}'))
    
    # ===== CLEANING TRAINING DATA ======
    # Lowering case for compatability with input data
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

def get_num_cars():
    data = load_dataset(file_name = config.app_config.training_data_file)
    num_cars = len(data)
    return num_cars

def save_model(*, model_to_save: Pipeline):
    """Saves the model"""
    save_file_name = f'{config.app_config.model_save_file}.pkl'
    save_path = TRAINED_MODEL_DIR / save_file_name
    joblib.dump(model_to_save, save_path)


def save_pipeline(*, pipeline_to_save: Pipeline):
    """Saves the feature engineering pipeline"""
    save_file_name = f'{config.app_config.feat_pipeline_save_file}.pkl'
    save_path = TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline_to_save, save_path)


def load_pipeline(*, file_name: str):
    """Loads the fitted pipeline"""
    file_path = TRAINED_MODEL_DIR / file_name
    trained_pipeline = joblib.load(filename = file_path)
    return trained_pipeline
    

def load_model(*, file_name: str):
    """Loads the fitted model"""
    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename = file_path)
    return trained_model    