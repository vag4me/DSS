import pandas as pd
import joblib

# Load the trained pipeline
pipeline = joblib.load('Trained_pipeline.pkl')

# Define the data manually
data = {
    'Tour_ID': ['tour_idynufedne'],
    'country': ['KOREA'],
    'age_group': ['25-44'],
    'travel_with': ['Alone'],
    'total_female': [0.0],
    'total_male': [1.0],
    'purpose': ['Leisure and Holidays'],
    'main_activity': ['Widlife Tourism'],
    'info_source': ['Other'],
    'tour_arrangement': ['Independent'],
    'package_transport_int': ['No'],
    'package_accomodation': ['No'],
    'package_food': ['No'],
    'package_transport_tz': ['No'],
    'package_sightseeing': ['No'],
    'package_guided_tour': ['NO'],
    'package_insurance': ['NO'],
    'night_mainland': [7],
    'night_zanzibar': [4],
    'first_trip_tz': ['Yes']
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Make predictions on the manual data
y_pred = pipeline.predict(df)

# Print the predictions
print("Predictions:", y_pred)
