# ==== LIBRARIES ====
import sys; sys.path.append('.')
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Union
from strictyaml import YAML, load

import regression_model

# ==== DIRECTORIES ====
# ROOT = PACKAGE_ROOT.parent
CONFIG_DIR = Path(__file__).resolve().parent
CONFIG_FILE_PATH = CONFIG_DIR / "config.yml"

PACKAGE_ROOT = Path(regression_model.__file__).resolve().parent
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


# ==== CONFIGURATIONS ====
class AppConfig(BaseModel):
    """Application-level config"""

    package_name: str
    main_data_file: str
    clean_data_file: str
    cars_origin_file: str

    feat_pipeline_save_file: str
    model_save_file: str
    brands_save_file: str
    fuels_save_file: str
    trans_save_file: str
    data_size_save_file: str

class ModelConfig(BaseModel):
    """All configuration relevant to model training and feature engineering"""

    target: str
    features: List[str]
    luxury_cars_list: List[str]

    cat_vars_with_nan_frequent: List[str]
    freq_encode: List[str]
    rare_labels: List[str]
    num_vars_nan: List[str]
    num_log_vars: List[str]

    model_params: Dict[str, Union[int, float]]

class Config(BaseModel):
    """Master config object"""
    app_config: AppConfig
    model_config: ModelConfig
    

def find_config_file(): 
    """Locate the configuration file"""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None):
    """Parse YAML containing the package configuration"""
    if not cfg_path:
        cfg_path = find_config_file()
    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None):
    """Run validation on config values"""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()
    
    # specify the data distribution from the strityaml YAML type
    _config = Config(
        app_config = AppConfig(**parsed_config.data),
        model_config = ModelConfig(**parsed_config.data)
    )

    return _config

config = create_and_validate_config()