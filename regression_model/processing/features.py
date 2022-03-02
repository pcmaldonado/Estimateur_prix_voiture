import sys; sys.path.append('.')
# ===== LIBRARIES =====
import pandas as pd
from pathlib import Path

from regression_model.config.core import DATASET_DIR, config

# ===== FEATURE FUNCTIONS ======
def brand_country(X):
    # Load data
    df_car_origin = pd.read_csv(Path(f'{DATASET_DIR/config.app_config.cars_origin_file}'))

    # Creating a dict with brand & country
    dict_cars_origin = {}

    for brand in set(X.Brand):
        brand_isin = df_car_origin[df_car_origin.isin([f'{brand.lower()}'])].stack()

        if len(brand_isin) > 0:
            country = brand_isin.index[0][1]

            # Filtering for "France" & "Germany"
            if (country == 'france') or (country == 'germany'):
                dict_cars_origin[f'{brand}'] = country
            else:
                dict_cars_origin[f'{brand}'] = 'other'
    
    s1 = set(X.Brand)
    s2 = set(dict_cars_origin.keys())

    for brand in s1.difference(s2):
        dict_cars_origin[brand] = 'other'
    
    # Mapping results
    X['Country_brand'] = X['Brand'].map(dict_cars_origin)
    return X



def luxury_brand(X):
    # Load data
    luxury_cars_list = config.model_config.luxury_cars_list

    # Creating a dict with brand & luxury brands
    dict_lux_cars = {}
    for car in X.Brand:
        # if (car == 'Rolls-royce') or (car == 'Mercedes') or (car == 'Land-rover') or (car == 'Aston'):
        #     dict_lux_cars[car] = 1
        # else:
        if car.lower() in luxury_cars_list:
            dict_lux_cars[car] = 1
        else:
            dict_lux_cars[car] = 0

    # Mapping results 
    X['Luxury_brand'] = X.Brand.map(dict_lux_cars)

    return X