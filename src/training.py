import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

def load_training_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, "..", "train_data.csv")
    val_path = os.path.join(script_dir, "..", "validation_data.csv")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    return train_df, val_df

def train():
    print("Loading training and validation data")
    train_df, val_df = load_training_data()

    X_train, y_train = train_df.drop("price", axis=1), train_df["price"]
    X_val, y_val = val_df.drop("price", axis=1), val_df["price"]

    print("Training Linear Regression")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_val_preds = lr_model.predict(X_val)

    print("Training Random Forest")
    
    depths = [20, 30, 40]
    best_rf_model = None
    best_rf_mae = 99999999
    best_rf_preds = None

    for depth in depths:
        rf_guess = RandomForestRegressor(n_estimators=100, max_depth=depth, random_state=42, n_jobs=-1)
        rf_guess.fit(X_train, y_train)
        
        rf_val_preds = rf_guess.predict(X_val)
        mae = mean_absolute_error(y_val, rf_val_preds)
        print(f"Testing max_depth={depth} | Validation MAE: ${mae}")
        
        if mae < best_rf_mae:
            best_rf_mae = mae
            best_rf_model = rf_guess
            best_rf_preds = rf_val_preds

    
    # Calculate linear regression errors
    lr_mae = mean_absolute_error(y_val, lr_val_preds)
    lr_mse = mean_squared_error(y_val, lr_val_preds)
    lr_rmse = root_mean_squared_error(y_val, lr_val_preds)


    # Calculate random forest errors
    rf_mae = mean_absolute_error(y_val, best_rf_preds)
    rf_mse = mean_squared_error(y_val, best_rf_preds)
    rf_rmse = root_mean_squared_error(y_val, best_rf_preds)

if __name__ == "__main__":
    train()