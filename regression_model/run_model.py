import sys; sys.path.append('.')
from pathlib import Path
# ===== LIBRARIES =====
# to handle datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ML model to use for regression
from xgboost import XGBRegressor

# to access configuration and custom functions
from regression_model.config.core import config, DATASET_DIR
from regression_model.processing.data_process import etl_process
from regression_model.processing.pipeline import feature_engineering
from regression_model.processing.data_manager import save_pipeline, save_model

# to measure model performance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# ===== FUNCTIONS =====
def prepare_data(file_name = config.app_config.clean_data_file):
    """Prepares data for modeling
    Arguments:
        file_name: 'str' with the file name containing cleaned data
    Returns:
        X_train, X_test, y_train, y_test: 'pandas DataFrame'
                containing train and test data split between
                features (X) and targets (y)"""
    data = pd.read_csv(Path(DATASET_DIR/file_name))

    # Split between train and test data
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
    
    # Split between target and features
    X_train = train_data[config.model_config.features]
    X_test = test_data[config.model_config.features]

    y_train = train_data[config.model_config.target]
    y_test = test_data[config.model_config.target]

    # apply log-transformation to target     
    y_train = np.log(y_train)
    y_test = np.log(y_test)
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Trains the XGBRegressor model
    Arguments:
        X_train, y_train: pandas DataFrame with training data
    Returns:
        model: xgboost trained model"""
     
    model = XGBRegressor(**config.model_config.model_params)
    
    print('Model training starting...')
    model.fit(X_train, y_train)
    print('Training complete')

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluates the model on test data and prints performance metrics
    Arguments:
        model: xgboost trained model
        X_test, y_test: pandas dataframes with testing data
    Reteurns:
        None"""
    # Getting predicted values
    print('Evaluating the model...')
    y_pred = model.predict(X_test)
    
    # Transforming back to original scale
    y_pred = np.exp(y_pred)
    y_test = np.exp(y_test)

    # Getting metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # determine rmse, mae, and r2
    print('\ntest mean price:', round(y_test.mean()))
    print('test std price:', round(y_test.std()))

    print('\ntest rmse:', round(rmse))
    print('test mae:',  round(mae))

    print('\ntest r2:', round(r2, 3))


# ===== MODEL =====
def run_training():
    """Runs all the necessary steps to creating, 
    training and saving the ML model
    Arguments:
        None
    Returns:
        None"""
    # Run ETL process
    etl_process()

    # Get training and test data
    X_train, X_test, y_train, y_test = prepare_data()

    # Fit and apply feature_engineering to training data
    print('Applying feature engineering')
    feature_engineering.fit(X_train)
    X_train = feature_engineering.transform(X_train)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate the model
    X_test = feature_engineering.transform(X_test)
    evaluate_model(model, X_test, y_test)

    # Save model
    print()
    save_pipeline(pipeline_to_save=feature_engineering)
    save_model(model_to_save=model)


if __name__ == '__main__':
    # run training
    run_training()