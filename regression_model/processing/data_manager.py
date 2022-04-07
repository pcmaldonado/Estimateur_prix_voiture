# ===== LIBRARIES ======
# To export model/data
import joblib

# To access configuration
from regression_model.config.core import TRAINED_MODEL_DIR, config 
import sys; sys.path.append('.')


# ===== FUNCTIONS ======
def save_model(model_to_save):
    """Saves the model
    Arguments:
        model_to_save: xgboost regressor model
    Returns:
        None
    """
    save_file_name = config.app_config.model_save_file
    save_path = TRAINED_MODEL_DIR / save_file_name
    joblib.dump(value = model_to_save, filename= save_path, compress=3)
    print('Model saved')


def save_pipeline(pipeline_to_save):
    """Saves the feature engineering pipeline
    Arguments:
        pipeline_to_save: sklearn pipeline
    Returns:
        None
    """
    save_file_name = config.app_config.feat_pipeline_save_file
    save_path = TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline_to_save, save_path, compress=3)
    print('Pipeline saved')


def load_pipeline():
    """Loads the fitted pipeline
    Arguments:
        None
    Returns:
        trained_pipeline: sklearn pipeline"""
    file_name = config.app_config.feat_pipeline_save_file
    file_path = TRAINED_MODEL_DIR / file_name
    trained_pipeline = joblib.load(filename = file_path)
    return trained_pipeline
    

def load_model():
    """Loads the fitted model
    Arguments:
        None
    Returns:
        trained_model: sklearn pipeline"""
    file_name = config.app_config.model_save_file
    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename = file_path)
    return trained_model    
