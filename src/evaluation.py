import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

def plot_results(y_true, y_pred, model_name, mae, mse, rmse, r2):
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
        f"RMSE: ${rmse:,.2f}\n"
        f"R^2: {r2:,.2f}\n"
    )
    # Place text box in the top right corner of the second chart
    axes[1].text(0.05, 0.95, metrics_text, fontsize=12, verticalalignment='top', horizontalalignment='left',
                 transform=axes[1].transAxes, bbox=dict(facecolor='white', edgecolor='gray'))

    model_name_formatted = model_name.casefold().replace(" ", "_") + "_plots.png"

    plot_path = os.path.join("../results/", model_name_formatted)
    plt.savefig(plot_path, dpi=500)
    plt.close()
    print(f"Saved {model_name} plot to {plot_path}")

def evaluate_model(model_name, model: Pipeline, X_test, y_test):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    plot_results(y_pred, y_test, model_name, mae=mae, rmse=rmse, mse=mse, r2=r2)

    return {
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "mse": mse,
        "r2": r2,
    }

if __name__ == "__main__":
    exit()