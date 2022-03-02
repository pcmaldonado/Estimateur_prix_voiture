import sys; sys.path.append('.')
# ===== LIBRARIES ======
# from sklearn
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer 
from sklearn.pipeline import Pipeline

# from feature_engine
from feature_engine.encoding import  RareLabelEncoder, CountFrequencyEncoder , OneHotEncoder
from feature_engine.imputation import AddMissingIndicator, MeanMedianImputer, CategoricalImputer, ArbitraryNumberImputer
from feature_engine.transformation import LogTransformer

# for modeling
from sklearn.ensemble import ExtraTreesRegressor

from regression_model.config.core import config
from regression_model.processing.features import brand_country, luxury_brand



# ===== PIPELINE =====
feature_engineering = Pipeline([   
    # ==== HANDLE MISSING VALUES ====
    # impute "mean" for features with less than 10% of missing values
    ('frequent_imputation', CategoricalImputer(
                            imputation_method = 'frequent', variables = config.model_config.cat_vars_with_nan_frequent)),
    
    #  # add missing indicator & impute "mean" to numerical varaibles 
        ('missing_indicator', AddMissingIndicator(variables=config.model_config.num_vars_nan)),
    
        ('mean_imputation', MeanMedianImputer(imputation_method='mean', variables=config.model_config.num_vars_nan)),

    
    # # ==== APPLY LOG TRANSFORMATION TO NUM VARIABLES ====
        # log transformation
        ('log', LogTransformer(variables=config.model_config.num_log_vars)),

    
    # ==== ADDING NEW FEATURES FROM BRAND ====
    # new features 
    ('add_brand_country', FunctionTransformer(brand_country)),
    ('add_luxury_brand', FunctionTransformer(luxury_brand)),
    
    
    # ==== ENCODING CATEGORICAL VARIABLES ====
    # Applying frequent encoder
        ('missing_brands', RareLabelEncoder(tol = 0.001, 
                                        variables = config.model_config.freq_encode ,
                                        replace_with = 0)), 
    ('frequency_encoder', CountFrequencyEncoder(variables = config.model_config.freq_encode)),


    
        # Applying rare label encoder
        ('rare_label_encoder', RareLabelEncoder(tol = 0.05, 
                                n_categories = 2, 
                                variables = config.model_config.rare_labels ,
                                replace_with = 'Other')),
    
    # Applying one-hot encoder
    ('one_hot_encoder', OneHotEncoder()),
    
    # Apply scaler
    ('apply_scaler', MinMaxScaler()),

])    