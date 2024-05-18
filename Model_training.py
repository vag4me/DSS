import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import joblib

def train_model(train_csv_file, model_file):
    # Load the training data
    train_df = pd.read_csv(train_csv_file)
    
    # Drop unnecessary columns
    train_df.drop(columns=['Tour_ID', 'country'], inplace=True)
    
    # Separate features and target variable
    X = train_df.drop(columns=["cost_category"])
    y = train_df["cost_category"]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define categorical features for one-hot encoding
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Define preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown = 'ignore'), categorical_features)
        ])
    
    # Define the pipeline with preprocessing and model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(pipeline, model_file)
    
    # Evaluate the model
    evaluate_model(X_test, y_test, pipeline)

def evaluate_model(X_test, y_test, model):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:')
    print(cm)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    
    # Calculate TP, TN, FP, FN
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    
    # Print TP, TN, FP, FN
    print(f'True Positives (TP): {TP}')
    print(f'True Negatives (TN): {TN}')
    print(f'False Positives (FP): {FP}')
    print(f'False Negatives (FN): {FN}')

def predict(model_file, new_data_file):
    # Load the saved model
    model = joblib.load(model_file)
    
    # Load the new data
    new_data = pd.read_csv(new_data_file)
    
    # Drop unnecessary columns
    new_data.drop(columns=['Tour_ID', 'country'], inplace=True)
    
    # Make predictions on new data
    predictions = model.predict(new_data)
    
    return predictions

if __name__ == "__main__":
    # File paths
    train_csv_file = 'C:/Users/user/Desktop/Ionio/DSS/Preprocessed_Train.csv'
    model_file = 'random_forest_model.pkl'
    new_data_file = 'C:/Users/user/Desktop/Ionio/DSS/Preprocessed_Test.csv'

    # Train the model
    train_model(train_csv_file, model_file)

    # Predict using the trained model
    predictions = predict(model_file, new_data_file)
    
    print("Predictions:", predictions)
    print("Model trained, evaluated, and predictions made successfully.")
