import pandas as pd
import warnings

def preprocess_and_check_missing_values(csv_file):
    # Suppress future warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Handle missing values
    # Fill missing 'travel_with' with 'Alone'
    df['travel_with'].fillna('Alone', inplace=True)
    
    # Fill missing 'total_female' with the column mean
    mean_total_female = df['total_female'].mean()
    df['total_female'].fillna(mean_total_female, inplace=True)
    
    # Fill missing 'total_male' with the column mean
    mean_total_male = df['total_male'].mean()
    df['total_male'].fillna(mean_total_male, inplace=True)
    
    # Save the preprocessed DataFrame back to a CSV file
    output_file = "C:/Users/user/Desktop/Ionio/DSS/Preprocessed_Train.csv"
    df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

# Run the preprocessing function
preprocess_and_check_missing_values('C:/Users/user/Desktop/Ionio/DSS/Train.csv')
