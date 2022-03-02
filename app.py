# ====== LIBRARIES ======
from flask import Flask, render_template, request
from regression_model import model
from datetime import datetime
import numpy as np

from regression_model.processing.input_validation import cast_num
from regression_model.processing.data_manager import get_num_cars
from regression_model.train_pipeline import get_brands, get_fuels, get_trans


# ====== SETUP ======
# Getting data to generate selections on index.html
years = list(np.arange(1930,2023))
years.insert(0, " ")
range10 = list(np.arange(1,11))
range10.insert(0, " ")
brands = get_brands()
fuels = get_fuels()
trans = get_trans()

# Getting number of cars to display in predict.hmtl
num_cars = get_num_cars()

# Translating features to french to display in table at predict.html
fr_features = ['Année', 'Marque', 'KMs', 'Energie', 'Emissions CO2', 'Consommation', 'Transmission', 'Portes', 'Puissance', 'Places']


# ====== FLASK ======
app = Flask(__name__, template_folder="./templates", static_folder="./static")

# Getting input from user
@app.route("/", methods = ["GET", "POST"])
def main():
    now = datetime.now() # get current date and time to generate new versions of .css to avoid "cache" problems
    date_time = now.strftime("%m%d%Y%H%M%S")

    # Return data needed for index.html
    return render_template('index.html', now = date_time, 
                            years = years, brands = brands, fuels = fuels, trans = trans, range10 = range10)


# Generating price estimation
@app.route("/predict", methods = ["Post"])
def home():
    now = datetime.now() 
    date_time = now.strftime("%m%d%Y%H%M%S")

    # Create dictionary with input data from user
    input_data = {}
    input_data['Years'] = cast_num(request.form['years'])
    input_data['Brand'] = request.form['brands']
    input_data['Kms'] = cast_num(request.form['kms'])
    input_data['Fuel'] = request.form['fuel']
    input_data['Emiss_CO2'] = cast_num(request.form['emiss_co2'])
    input_data['Cons_litres_100km'] = cast_num(request.form['cons_litre'])
    input_data['Transmission'] = request.form['transmission']
    input_data['Doors'] = cast_num(request.form['doors'])
    input_data['Power_CV'] = cast_num(request.form['power_cv'])
    input_data['Seats'] = cast_num(request.form['seats'])
    
    # Predict price with regression model
    pred = model.estimate_price(input_data)

    # Return data needed for predict.html
    return render_template("predict.html", prediction = pred, input_data = input_data, fr_features = fr_features,
                                            now = date_time, num_cars = num_cars)


if __name__=="__main__":
    app.run()