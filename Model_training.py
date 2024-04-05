import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Define the preprocessing steps
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values using the most frequent value
    ('onehot', OneHotEncoder(handle_unknown='ignore'))    # Perform one-hot encoding
])

# Define the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Combine preprocessing and modeling into a single pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# Load the data
train_data = pd.read_csv("Train.csv")
test_data = pd.read_csv("Test.csv")

# Separate features and target variable
X_train = train_data.drop(columns=["cost_category"])
y_train = train_data["cost_category"]

# Train the pipeline (including preprocessing and modeling)
pipeline.fit(X_train, y_train)

# Save the trained pipeline
joblib.dump(pipeline, 'Trained_pipeline.pkl')
