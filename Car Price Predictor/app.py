from flask import Flask, render_template, request
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'LinearRegressionModel.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'Cleaned_data.csv')

try:
    model = pickle.load(open(MODEL_PATH, 'rb'))
    car = pd.read_csv(DATA_PATH)
except FileNotFoundError as e:
    print(f"CRITICAL ERROR: {e}")
    print("Please ensure 'LinearRegressionModel.pkl' and 'Cleaned_data.csv' are in the same directory.")
    car = pd.DataFrame(columns=['company', 'name', 'year', 'fuel_type'])

@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()
    
    return render_template('index.html', 
                            companies=companies, 
                            car_models=car_models, 
                            years=years, 
                            fuel_types=fuel_types)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        company = request.form.get('company')
        car_model = request.form.get('car_models')
        year = int(request.form.get('year')) 
        fuel_type = request.form.get('fuel_type')
        driven = int(request.form.get('kilo_driven'))

        input_data = pd.DataFrame([[car_model, company, year, driven, fuel_type]], 
                                    columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
        
        prediction = model.predict(input_data)
        
        result = np.abs(np.round(prediction[0], 2))
        
        return str(result)

    except Exception as e:
        print(f"Prediction Error: {e}")
        return "Error processing request"

if __name__ == '__main__':
    app.run(debug=True)