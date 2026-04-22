import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

data = pd.read_csv("dataset.csv")

# सही features (5 only)
X = data[["fuel", "engine", "kms", "owner", "transmission"]]
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, "price_model.pkl")

print("Price Model Saved")