import numpy as np
from flask import Flask, request, render_template 
import pickle
import pandas as pd
import joblib


app = Flask(__name__)
pipeline = joblib.load('Trained_pipeline.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict' , methods = ['POST'])
def predict():
    # Get form data and convert values to the correct data type
    data = {
        'Tour_ID': [request.form['Tour_ID']],
        'country': [request.form['country']]if request.form['country'] else None,
        'age_group': [request.form['age_group']]if request.form['age_group'] else None,
        'travel_with': [request.form['travel_with']]if request.form['travel_with'] else None,
        'total_female': [float(request.form['total_female'])]if request.form['total_female'] else None,
        'total_male': [float(request.form['total_male'])]if request.form['total_male'] else None,
        'purpose': [request.form['purpose']]if request.form['purpose'] else None,
        'main_activity': [request.form['main_activity']] if request.form['main_activity'] else None,
        'info_source': [request.form['info_source']] if request.form['info_source'] else None,
        'tour_arrangement': [request.form['tour_arrangement']]if request.form['tour_arrangement'] else None,
        'package_transport_int': [request.form['package_transport_int']]if request.form['package_transport_int'] else None,
        'package_accomodation': [request.form['package_accomodation']]if request.form['package_accomodation'] else None,
        'package_food': [request.form['package_food']]if request.form['package_food'] else None,
        'package_transport_tz': [request.form['package_transport_tz']]if request.form['package_transport_tz'] else None,
        'package_sightseeing': [request.form['package_sightseeing']]if request.form['package_sightseeing'] else None,
        'package_guided_tour': [request.form['package_guided_tour']]if request.form['package_guided_tour'] else None,
        'package_insurance': [request.form['package_insurance']]if request.form['package_insurance'] else None,
        'night_mainland': [int(request.form['night_mainland'])] if request.form['night_mainland'] else None,
        'night_zanzibar': [int(request.form['night_zanzibar'])] if request.form['night_zanzibar'] else None,
        'first_trip_tz': [request.form['first_trip_tz']]if request.form['first_trip_tz'] else None
    }

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
            
            
