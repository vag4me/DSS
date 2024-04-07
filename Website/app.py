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
        'country': [request.form['country']],
        'age_group': [request.form['age_group']],
        'travel_with': [request.form['travel_with']],
        'total_female': [float(request.form['total_female'])],
        'total_male': [float(request.form['total_male'])],
        'purpose': [request.form['purpose']],
        'main_activity': [request.form['main_activity']],
        'info_source': [request.form['info_source']],
        'tour_arrangement': [request.form['tour_arrangement']],
        'package_transport_int': [request.form['package_transport_int']],
        'package_accomodation': [request.form['package_accomodation']],
        'package_food': [request.form['package_food']],
        'package_transport_tz': [request.form['package_transport_tz']],
        'package_sightseeing': [request.form['package_sightseeing']],
        'package_guided_tour': [request.form['package_guided_tour']],
        'package_insurance': [request.form['package_insurance']],
        'night_mainland': [int(request.form['night_mainland'])],
        'night_zanzibar': [int(request.form['night_zanzibar'])],
        'first_trip_tz': [request.form['first_trip_tz']]
    }

    dataframe = pd.DataFrame(data)
    y_pred = pipeline.predict(dataframe)
    print("Prediction:", y_pred)  # Debugging statement

    # Render the prediction result in the HTML template
    return render_template('index.html', prediction_text='The cost will be {}'.format(y_pred))
        

if __name__ == "__main__":
    app.debug = True
    app.run()
            
            






