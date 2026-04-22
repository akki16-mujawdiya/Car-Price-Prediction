import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
data = pd.read_csv("dataset.csv")

# Correct features (same structure as price model)
X = data[["fuel", "engine", "kms", "owner", "transmission"]]
y = data["mileage"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "mileage_model.pkl")

print("Mileage Model Saved")