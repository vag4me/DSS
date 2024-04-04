import pandas as pd
import numpy as np 
import warnings

def check_missing_values(csv_file):
    df = pd.read_csv(csv_file) #pernao to excel sto df
    missing_values = df.isnull().sum()
    print(missing_values)


warnings.simplefilter(action='ignore', category=FutureWarning)
check_missing_values('C:/Users/user/Desktop/Ionio/DSS/Train.csv')
