import pandas as pd
import numpy as np 
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib


def train_model():
    trained_model_file = ("C:/Users/user/Desktop/Ionio/DSS/trained_rf_model.pkl")
    data = pd.read_csv("C:/Users/user/Desktop/Ionio/DSS/Preprocessed_Train.csv")

    # Separate features (X) and target variable (y)
    X = data[data.columns[:-1]] 
    y = data[data.columns[-1]] 

    
    # Create and train Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X, y)

    joblib.dump(rf_classifier, trained_model_file)
   


warnings.simplefilter(action='ignore', category=FutureWarning)
train_model()
