import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Define the preprocessing steps
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values using the most frequent value
    ('onehot', OneHotEncoder(handle_unknown='ignore'))    # Perform one-hot encoding
])

# Define the model
model = RandomForestClassifier(n_estimators=10, random_state=42)

# Combine preprocessing and modeling into a single pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# Load the data
train_data = pd.read_csv("Train.csv")

# Drop unnecessary columns
train_data = train_data.drop(columns=['Tour_ID', 'country'])

# Separate features and target variable
X = train_data.drop(columns=["cost_category"])
y = train_data["cost_category"]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the pipeline (including preprocessing and modeling) on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = pipeline.predict(X_val)

# Calculate accuracy on the validation set
accuracy_val = accuracy_score(y_val, y_pred_val)
print("Validation Accuracy:", accuracy_val)

# Generate confusion matrix on the validation set
conf_matrix_val = confusion_matrix(y_val, y_pred_val)
print("Confusion Matrix (Validation):")
print(conf_matrix_val)

# Save the trained pipeline
joblib.dump(pipeline, 'Trained_pipeline.pkl')
