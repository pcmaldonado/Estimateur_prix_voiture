import sys; sys.path.append('.')
# ===== LIBRARIES =====
# to handle datasets & plotting
import pandas as pd
import numpy as np

from regression_model.config.core import config
from regression_model.processing.data_manager import load_pipeline, load_model

# from regression_model.pipeline import feature_engineering
from regression_model.train_pipeline import run_training

run_training()


pipeline_file_name = f'{config.app_config.feat_pipeline_save_file}.pkl'
_price_pipe = load_pipeline(file_name = pipeline_file_name)

model_file_name = f'{config.app_config.model_save_file}.pkl'
_model = load_model(file_name = model_file_name)


def estimate_price(input_data):
    # building test dataframe (from user input)
    X_test = pd.DataFrame([input_data])
    X_test.apply(lambda x: x.astype(str).str.lower())
    
    X_test = _price_pipe.transform(X_test)
    prediction = _model.predict(X_test)
 
    pred = round(np.exp(prediction[0]), 2)
    
    return pred