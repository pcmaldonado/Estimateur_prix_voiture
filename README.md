# Predicting Used Car Prices ("Estimateur prix voiture") - Web App 
Deployment of a machine learning algorithm trained on data collected from a used car sales site, available on Heroku: https://estimateur-prix-voiture.herokuapp.com/ (in french)

Details on the data wrangling & model selection process on: https://github.com/pcmaldonado/Predicting_used_cars_price.

------------

# Directories

## Regression Model
It contains a local package containing all the code needed to create the Machine Learning model deployed on Heroku.
Besides some subdirectories, it contains a yml file to handle configuration (see below) and three python scripts:
* *pipeline.py:* it contains the entire feature engineering pipeline
* *train_pipeline.py:* it applies the processing pipeline and trains the ML model
* *model.py:* it runs the trained model on new data (input data from users)


### Configuration
Configuration is handled by *config.yml* and *config/core.py*. These files contain mainly file names, expected features/target names and hyperparameters to use for modeling (found previously using GridSearchCV (fore more details check the [jupyter notebooks](https://github.com/pcmaldonado/Predicting_used_cars_price))).

### Datasets
It contains scraped data used to associate brands with countries.

### Processing
It contains three python scripts that handle data processing:
* *data_manager.py:* it contains functions to save/load data, model as well as helping functions to get data to display on web app
* *features.py:* it contains two functions that use additional data to better handle brands encoding ("luxury", "origin")
* *input_validation.py:* it contains a single function to cast numbers when taking input from user in the web app

### Trained Models
This folder contains pickle files regarding the trained feature engineering pipeline and modeling, as well as additional information to be displayed on the web app.

## Static & Templates
The **Static** folder contains all the CSS and images, and the **templates** folder contains the HTML code to run the web application.

## Additional files
* **Procfile** is needed to deploy the application on Heroku
* **requirements.txt** contains all the libraries (and their versions) needed to run the entire code
* **app.py** uses Flaks to render the Web Application and predict on new data