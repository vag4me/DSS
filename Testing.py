import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_and_predict(new_data, model_file):
    # Load the saved model
    model = joblib.load(model_file)
    
    # Convert the new data into a DataFrame
    test_data = pd.DataFrame(new_data)
    
    # Make predictions on the DataFrame
    predictions = model.predict(test_data)
    
    return predictions

if __name__ == "__main__":
    # Sample data
    new_data = {
        'age_group': ['30-39'],
        'travel_with': ['Family'],
        'total_female': [2.0],
        'total_male': [1.0],
        'purpose': ['Leisure'],
        'main_activity': ['Wildlife Tourism'],
        'info_source': ['Friends, Relatives'],
        'tour_arrangement': ['Package Tour'],
        'package_transport_int': ['No'],
        'package_accomodation': ['Yes'],
        'package_food': ['Yes'],
        'package_transport_tz': ['Yes'],
        'package_sightseeing': ['Yes'],
        'package_guided_tour': ['Yes'],
        'package_insurance': ['Yes'],
        'night_mainland': [5],
        'night_zanzibar': [0],
        'first_trip_tz': ['No']
    }
    
    # File path to the trained model
    model_file = 'random_forest_model.pkl'
    
    # Preprocess and make predictions
    predictions = preprocess_and_predict(new_data, model_file)
    
    print("Predictions:", predictions)

