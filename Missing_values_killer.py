import pandas as pd
import numpy as np 
import warnings

def preprocess_and_check_missing_values(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file) 
    
    # Handle missing values
    df['travel_with'].fillna('Alone', inplace=True) 
    mean_total_female = df['total_female'].mean()  
    df['total_female'].fillna(mean_total_female, inplace=True) 
    mean_total_male = df['total_male'].mean()  
    df['total_male'].fillna(mean_total_male, inplace=True)
    
    # Perform one-hot encoding
    df = pd.get_dummies(df)
    
    # Save the preprocessed DataFrame back to a CSV file
    df.to_csv("C:/Users/user/Desktop/Ionio/DSS/Preprocessed_Train.csv", index=False)
    
    

warnings.simplefilter(action='ignore', category=FutureWarning)
preprocess_and_check_missing_values('C:/Users/user/Desktop/Ionio/DSS/Train.csv')
