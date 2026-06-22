import joblib
import pandas as pd

model = joblib.load("../models/random_forest.pkl")

data = {
    "miles": [200000.0],
    "make": ["Honda"],
    "model": ["Civic"],
    "body_type": ["Sedan"],
    "vehicle_type": ["Car"],
    "drivetrain": ["FWD"],
    "color": ["Grey"],
    "transmission": ["Automatic"],
    "fuel_type": ["Gasoline"],
    "engine_size": [1.8],
    "engine_block": ["I"],
    "state": ["ON"],
    "carAge": [10.0],
}

df_civic = pd.DataFrame(data)

predicted_price = model.predict(df_civic)

print(f"Predicted price: ${predicted_price[0]:,.0f}")