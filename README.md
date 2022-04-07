# Predicting Used Car Prices ("Estimateur prix voiture") - Web App 
Deployment of a machine learning algorithm trained on data collected from a used car sales site, available on Heroku: https://estimateur-prix-voiture.herokuapp.com/ (in french)

Details on the data wrangling & model selection process on: https://github.com/pcmaldonado/Predicting_used_cars_price.

------------

# Directories

## Regression Model
It contains a local package containing all the code needed to create the Machine Learning model deployed on Heroku.
Besides some subdirectories, it contains three python scripts:

<details>
<summary><b>run_model.py</b></summary> 
It runs the entire process to create, train and save the ML model.<br><br> To retrain the model:<br>
<img src="https://user-images.githubusercontent.com/84249222/162190556-344175a0-aada-461c-954b-bdaf63034563.png">
</details>

<details>
<summary><b>extract_information.py</b></summary> 
It saves information needed for the display of the web application onto pickle files.<br><br>  If a file update was needed, as before:<br>
<img src="https://user-images.githubusercontent.com/84249222/162190933-71c03dbd-ffe4-4917-aaf8-b13f63b428d1.png">
</details>


<details>
<summary><b>predict.py</b></summary> 
It loads the feature engineering pipeline and the trained model to predict on new data (used on deployment)
</details>

### Configuration
Configuration is handled by *config.yml* and *core.py*. These files contain mainly file names, expected features/target names and hyperparameters to use for modeling (previously found using GridSearchCV (for more details check the [jupyter notebooks](https://github.com/pcmaldonado/Predicting_used_cars_price))).

### Datasets
It contains csv files with scraped data used to associate brands with countries, as well as raw and cleaned data to run the model. 

### Processing
It contains five python scripts that handle data processing:
* *data_process.py:* it contains the entire ETL process (extract raw data, transforms it and loads the clean data to a csv file)
* *data_manager.py:* it contains functions to save and load the feature engineering pipeline and the trained model 
* *features.py:* it contains two functions that use additional data to better handle brands encoding ("luxury", "origin")
* *input_validation.py:* it contains a single function to cast numbers when taking input from user in the web app
* *pipeline.py:* it contains the feature engineering pipeline

### Trained Models
This folder contains pickle files regarding the trained feature engineering pipeline and modeling, as well as additional information to be displayed on the web app.

## Static & Templates
The **Static** folder contains all the CSS and images, and the **templates** folder contains the HTML code to run the web application.

## Additional files
* **Procfile** is needed to deploy the application on Heroku
* **.slugignore** excludes certain files for the deployment
* **requirements.txt** contains all the libraries (and their versions) needed to run the entire code
* **app.py** uses Flaks to render the Web Application and predict on new data, to run locally from terminal `python app.py`
