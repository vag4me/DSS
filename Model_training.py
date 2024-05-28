import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

def print_unique_target_values_in_order(y):
    # Get the sorted unique values of the target variable
    unique_values = sorted(y.unique())
    print('Unique target values (cost_category) in confusion matrix order:', unique_values)

def train_model(train_csv_file, model_file):
    # Load the training data
    train_df = pd.read_csv(train_csv_file)
    
    # Drop unnecessary columns
    train_df.drop(columns=['Tour_ID', 'country'], inplace=True)
    
    # Separate features and target variable
    X = train_df.drop(columns=["cost_category"])
    y = train_df["cost_category"]
    
    # Print unique target values in the order they will appear in the confusion matrix
    print_unique_target_values_in_order(y)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define categorical features for one-hot encoding
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Define preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features)
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
    
    # Generate a classification report
    report = classification_report(y_test, y_pred)
    print('Classification Report:')
    print(report)
    
    # Get the unique classes
    unique_classes = sorted(y_test.unique())
    
    # Initialize dictionaries to store TP, FP, TN, FN for each class
    metrics = {cls: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for cls in unique_classes}
    
    # Calculate TP, FP, TN, FN for each class
    for i, cls in enumerate(unique_classes):
        # True Positives: Correctly predicted as the current class
        metrics[cls]['TP'] = cm[i, i]
        
        # False Positives: Incorrectly predicted as the current class
        metrics[cls]['FP'] = cm[:, i].sum() - cm[i, i]
        
        # False Negatives: Incorrectly predicted as other classes
        metrics[cls]['FN'] = cm[i, :].sum() - cm[i, i]
        
        # True Negatives: Correctly predicted as other classes
        metrics[cls]['TN'] = cm.sum() - (metrics[cls]['TP'] + metrics[cls]['FP'] + metrics[cls]['FN'])
    
    # Print TP, FP, TN, FN for each class
    for cls in unique_classes:
        print(f'Class: {cls}')
        print(f"  True Positives (TP): {metrics[cls]['TP']}")
        print(f"  False Positives (FP): {metrics[cls]['FP']}")
        print(f"  True Negatives (TN): {metrics[cls]['TN']}")
        print(f"  False Negatives (FN): {metrics[cls]['FN']}")
        print()
    
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
