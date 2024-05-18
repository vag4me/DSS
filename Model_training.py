import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix

def train_and_save_model(train_csv_file, model_file):
    # Load the preprocessed training data
    train_df = pd.read_csv(train_csv_file)
    
    # Drop unnecessary columns
    train_df.drop(columns=['Tour_ID', 'country'], inplace=True)
    
    # Separate features and target variable
    X = train_df.drop(columns=["cost_category"])
    y = train_df["cost_category"]
    
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
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the trained model and label encoders
    with open(model_file, 'wb') as file:
        pickle.dump((model, label_encoders), file)
    
    return X_test, y_test, model

def evaluate_model(X_test, y_test, model):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy:.2f}')
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:')
    print(cm)

# File paths
train_csv_file = 'C:/Users/user/Desktop/Ionio/DSS/Preprocessed_Train.csv'
model_file = 'random_forest_model.pkl'

# Train the model and save it
X_test, y_test, model = train_and_save_model(train_csv_file, model_file)

# Evaluate the model
evaluate_model(X_test, y_test, model)
