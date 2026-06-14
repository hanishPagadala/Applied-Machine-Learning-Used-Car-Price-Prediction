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

def test():
    plot_results(y_val, lr_val_preds, "Linear_Regression", lr_mae, lr_mse, lr_rmse)
    plot_results(y_val, best_rf_preds, "Random_Forest", rf_mae, rf_mse, rf_rmse)


if __name__ == "__main__":
    test()