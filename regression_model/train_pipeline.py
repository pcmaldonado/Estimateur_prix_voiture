import sys; sys.path.append('.')
# ===== LIBRARIES =====
# to handle datasets & plotting
import pandas as pd
import numpy as np

from xgboost import XGBRegressor

from regression_model.config.core import config
from regression_model.processing.data_manager import load_dataset, save_pipeline, save_model

from regression_model.pipeline import feature_engineering

def run_training():
    # load training data & separate target from features
    data = load_dataset(file_name = config.app_config.training_data_file)
    
    X_train = data[config.model_config.features]
    y_train = data[config.model_config.target]

    # apply log-transformation to target     
    y_train = np.log(y_train)

    # fit pipeline
    feature_engineering.fit(X_train)
    X_train = feature_engineering.transform(X_train)

    model = XGBRegressor(**config.model_config.model_params)
    print('model created')

    model.fit(X_train, y_train)
    print('model trained')

    save_pipeline(pipeline_to_save=feature_engineering)
    save_model(model_to_save=model)


def get_brands():
    data = load_dataset(file_name = config.app_config.training_data_file)
    brands = list(data['Brand'].unique())
    brands.insert(0, " ")
    return brands

def get_fuels():
    data = load_dataset(file_name = config.app_config.training_data_file)
    fuels = data['Fuel'].unique()
    fuels = [fuel for fuel in fuels if type(fuel) == str]
    fuels.insert(0, " ")
    return fuels

def get_trans():
    data = load_dataset(file_name = config.app_config.training_data_file)
    trans = data['Transmission'].unique()
    trans = [tran for tran in trans if type(tran) == str]
    trans.insert(0, " ")
    return trans