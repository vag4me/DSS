import pandas as pd
import joblib

# Load the trained pipeline
pipeline = joblib.load('Trained_pipeline.pkl')

# Load the test data
test_data = pd.read_csv("Test.csv")

# Drop unnecessary columns
test_data = test_data.drop(columns=['Tour_ID', 'country'])

# Make predictions on the test set
predictions = pipeline.predict(test_data)

# You can then analyze the predictions further based on your specific requirements
# For example, you can check the distribution of predicted labels
print("Predicted label distribution:")
print(pd.Series(predictions).value_counts())
