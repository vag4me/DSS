import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(csv_file):
    # Load the preprocessed data
    df = pd.read_csv(csv_file)
    
    # Drop the 'Tour_ID' and 'country' columns
    df = df.drop(columns=['Tour_ID', 'country'])
    
    # Separate features and target variable
    X = df.drop(columns=["cost_category"])
    y = df["cost_category"]
    
    # Encode categorical features
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le
    
    # Encode the target variable if it is categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a RandomForestClassifier model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy:.2f}')
    
    return model

# Run the training function
model = train_model('C:/Users/user/Desktop/Ionio/DSS/Preprocessed_Train.csv')
