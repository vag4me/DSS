import numpy as np
from flask import Flask, request, render_template 
import pickle
import pandas as pd
import joblib


app = Flask(__name__)
pipeline = joblib.load('models/random_forest_model.pkl')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    form_data = request.form
    
    # Convert form data to the correct data types
    data = {
        'age_group': [form_data['age_group']] if form_data['age_group'] else None,
        'travel_with': [form_data['travel_with']] if form_data['travel_with'] else None,
        'total_female': [float(form_data['total_female'])] if form_data['total_female'] else None,
        'total_male': [float(form_data['total_male'])] if form_data['total_male'] else None,
        'purpose': [form_data['purpose']] if form_data['purpose'] else None,
        'main_activity': [form_data['main_activity']] if form_data['main_activity'] else None,
        'info_source': [form_data['info_source']] if form_data['info_source'] else None,
        'tour_arrangement': [form_data['tour_arrangement']] if form_data['tour_arrangement'] else None,
        'package_transport_int': [form_data['package_transport_int']] if form_data['package_transport_int'] else None,
        'package_accomodation': [form_data['package_accomodation']] if form_data['package_accomodation'] else None,
        'package_food': [form_data['package_food']] if form_data['package_food'] else None,
        'package_transport_tz': [form_data['package_transport_tz']] if form_data['package_transport_tz'] else None,
        'package_sightseeing': [form_data['package_sightseeing']] if form_data['package_sightseeing'] else None,
        'package_guided_tour': [form_data['package_guided_tour']] if form_data['package_guided_tour'] else None,
        'package_insurance': [form_data['package_insurance']] if form_data['package_insurance'] else None,
        'night_mainland': [int(form_data['night_mainland'])] if form_data['night_mainland'] else None,
        'night_zanzibar': [int(form_data['night_zanzibar'])] if form_data['night_zanzibar'] else None,
        'first_trip_tz': [form_data['first_trip_tz']] if form_data['first_trip_tz'] else None
    }

    # Check if all form fields are empty
    if all(value is None for value in data.values()):
        return render_template('index.html', error_message='You have to fill all the data')

    dataframe = pd.DataFrame(data)
    missing = dataframe.isnull().sum()

    if missing.any() == 0:
        # If no missing values, make the prediction
        y_pred = pipeline.predict(dataframe)
        return render_template('index.html', prediction_text='The cost will be {}'.format(y_pred))
    else:
        # If missing values exist, render the error message
        return render_template('index.html', error_message='You have to fill all the data')

        

if __name__ == "__main__":
    app.debug = True
    app.run()
            
