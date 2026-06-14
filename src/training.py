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

def plot_results(y_true, y_pred, model_name, mae, mse, rmse):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f'Validation Results: {model_name}')

    dollar_formatter = ticker.FuncFormatter(lambda x, pos: f'${x}')

    # Chart 1: Actual vs Predicted
    axes[0].scatter(y_true, y_pred, alpha=0.4)
    
    max_actual = y_true.max()
    axes[0].set_xlim(0, max_actual * 1.05)
    axes[0].plot([0, max_actual * 1.05], [0, max_actual * 1.05], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    
    axes[0].set_xlabel('Actual Real-World Prices')
    axes[0].set_ylabel('Model Predicted Prices',)
    axes[0].set_title('Prediction Accuracy',)
    axes[0].xaxis.set_major_formatter(dollar_formatter)
    axes[0].yaxis.set_major_formatter(dollar_formatter)
    axes[0].grid(True)

    # Chart 2: Error Distribution
    residuals = y_true - y_pred 
    axes[1].hist(residuals, bins=50)
    
    axes[1].axvline(x=0, color='red', linestyle='--', label='$0 Error')
    axes[1].set_xlabel('Prediction Error')
    axes[1].set_ylabel('Number of Cars')
    axes[1].set_title('Error Distribution')
    axes[1].xaxis.set_major_formatter(dollar_formatter)
    axes[1].grid(True)

    metrics_text = (
        f"Model Performance:\n"
        f"MAE:  ${mae:,.2f}\n"
        f"MSE:  ${mse:,.2f}\n"
        f"RMSE: ${rmse:,.2f}"
    )
    # Place text box in the top right corner of the second chart
    axes[1].text(0.95, 0.95, metrics_text, fontsize=12, verticalalignment='top', horizontalalignment='right',
                 transform=axes[1].transAxes, bbox=dict(facecolor='white', edgecolor='gray'))

    plot_path = os.path.join("..", f"{model_name}_Validation.png")
    plt.savefig(plot_path, dpi=500)
    plt.close()

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

    plot_results(y_val, lr_val_preds, "Linear_Regression", lr_mae, lr_mse, lr_rmse)

    # Calculate random forest errors
    rf_mae = mean_absolute_error(y_val, best_rf_preds)
    rf_mse = mean_squared_error(y_val, best_rf_preds)
    rf_rmse = root_mean_squared_error(y_val, best_rf_preds)

    plot_results(y_val, best_rf_preds, "Random_Forest", rf_mae, rf_mse, rf_rmse)

if __name__ == "__main__":
    train()