import sys; sys.path.append('.')
# ===== LIBRARIES =====
# to handle datasets & plotting
import pandas as pd
import numpy as np

from regression_model.processing.data_manager import load_pipeline, load_model


# ===== MODEL =====
# load the trained models
feature_engineering_pipeline = load_pipeline()
model = load_model()

def estimate_price(input_data):
    """Estimate price of new car given user input data
    Arguments:
        input_data: 'dict' containing data features as "keys" 
            and user input as "values"
    Returns:
        pred: 'float' prediction after having applied 
            feature engineering and trained model
    """
    # building test dataframe (from user input)
    X_test = pd.DataFrame([input_data])
    X_test.apply(lambda x: x.astype(str).str.lower())
    
    X_test = feature_engineering_pipeline.transform(X_test)
    prediction = model.predict(X_test)
 
    pred = round(np.exp(prediction[0]), 2)
    
    return pred