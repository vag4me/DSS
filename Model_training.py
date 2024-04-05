import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load the CSV file
data = pd.read_csv("C:/Users/user/Desktop/Ionio/DSS/Train.csv")

data = pd.get_dummies(data)

# Split the data into training and testing sets
X = data[data.columns[:-1]] # Features
y = data[data.columns[-1]] # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Choose a Model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 5: Train the Model
model.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

joblib.dump(model, 'Trained_model.pkl')


