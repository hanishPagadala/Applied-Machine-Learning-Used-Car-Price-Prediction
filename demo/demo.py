import joblib
import pandas as pd

model = joblib.load("../models/random_forest.pkl")
data = pd.read_csv("../data/processed/cleaned_dataset.csv")

# Choose which row to use
row_index = 0

row = data.iloc[[row_index]]

actual_price = row["price"].iloc[0]

features = row.drop(columns=["price"])

predicted_price = model.predict(features)[0]
absolute_error = abs(actual_price - predicted_price)

print(f"Demo row number: {row_index + 1}")
print(f"Actual price: ${actual_price:,.0f}")
print(f"Predicted price: ${predicted_price:,.0f}")
print(f"Absolute error: ${absolute_error:,.0f}")