import pandas as pd
import numpy as np 
import warnings

def check_missing_values(csv_file):
    df = pd.read_csv(csv_file) #pernao to excel sto df
    df['travel_with'].fillna('Alone', inplace = True) #Βαζει alone στις χαμενες τιμες της μεταβλητης travel_with
    mean_total_female = df['total_female'].mean()  #Υπολογιζει τον μεσο ορο της μεταβλατης total_female
    df['total_female'].fillna('mean_total_female', inplace = True) #Γεμιζει τις missing_values με τον μερο ορο 
    mean_total_male = df.total_male.mean()  #Ιδια διαδικασια για τους αντρας ομως
    df['total_male'].fillna('mena_total_male' , inplace = True)
    df.to_csv("C:/Users/user/Desktop/Ionio/DSS/Train.csv", index=False)
    
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])


warnings.simplefilter(action='ignore', category=FutureWarning)
check_missing_values('C:/Users/user/Desktop/Ionio/DSS/Train.csv')
