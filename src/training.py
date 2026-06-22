import pandas as pd
import os
import joblib
import math

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from evaluation import evaluate_model

def load_training_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    df_path = os.path.join(script_dir, "../data/processed/", "cleaned_dataset.csv")
    df = pd.read_csv(df_path)
    
    return df

def train():
    print("Loading training and validation data")
    df = load_training_data()

    # Splitting requires at least 3 makes of certain car (e.g. there's only 1 Oldsmobile)
    make_counts = df['make'].value_counts()
    valid_makes = make_counts[make_counts >= 10].index
    df = df[df['make'].isin(valid_makes)]

    textColumns = df.select_dtypes(include=["str"]).columns.tolist();
    numericColumns = df.select_dtypes(include=["number"]).columns.tolist()

    # Convert to category to make train_test_split behave better
    df[textColumns] = df[textColumns].astype('category') 

    # Do not scale price because it is the target value
    if "price" in numericColumns:
        numericColumns.remove("price")

    X, y = df.drop("price", axis=1), df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size = 0.2, # Training 80%, Testing 20%
        stratify=X['make'], # Make sure manufacturers are distributed equally
        random_state=42
    )

    scaler = ("scaler", make_column_transformer((StandardScaler(), numericColumns), remainder="passthrough", verbose_feature_names_out=False))
    encoder = ("onehot", make_column_transformer((OneHotEncoder(handle_unknown="ignore", sparse_output=False), textColumns), remainder="passthrough", verbose_feature_names_out=False))
    scaler[1].set_output(transform="pandas")
    encoder[1].set_output(transform="pandas")

    models = {
        "Linear Regression": Pipeline([
            scaler, encoder,
            ("model", LinearRegression())
        ]),
        "Ridge Regression": Pipeline([
            scaler, encoder,
            ("model", Ridge())
        ]),
        "Random Forest": Pipeline([
            scaler, encoder,
            ("model", RandomForestRegressor(random_state=42))
        ]),
        "Decision Tree": Pipeline([
            scaler, encoder,
            ("model", DecisionTreeRegressor(random_state=42))
        ]),
        "Gradient Boosting": Pipeline([
            scaler, encoder,
            ("model", GradientBoostingRegressor(random_state=42))
        ]),
    }

    grids = {
        "Linear Regression": None,
        "Ridge Regression": {
            "model__alpha": [0.01, 0.1, 1.0, 10.0]
        },
        "Random Forest": {
            "model__n_estimators": [100], 
            "model__max_depth": [10, 20, 30]
        },
        "Decision Tree": {
            "model__max_depth": [10, 20], 
            "model__min_samples_split": [2, 5]
        },
        "Gradient Boosting": {
            "model__n_estimators": [50], 
            "model__learning_rate": [0.1],
            "model__max_depth": [2, 5]
        },
    }

    final_models = {};

    # Train Models
    for model_name, pipeline in models.items():
        grid = grids[model_name]

        if grid is None:
            print(f"\nTraining {model_name} without Grid Search...")
            pipeline.fit(X_train, y_train)
            print(f"\nTraining {model_name} complete")
            final_models[model_name] = pipeline

        else:
            print(f"\nTraining {model_name} with Grid Search with 5 folds for:")
            for hyperparameter_key, hyperparameter_value in grid.items():
                print(f"    {hyperparameter_key.split("model__")[1]} tuned for {hyperparameter_value}")
            
            gridsearch = GridSearchCV(estimator=pipeline, param_grid=grid, scoring="neg_mean_squared_error", cv=5, n_jobs=-1)
            gridsearch.fit(X_train, y_train)
            print(f"\nTraining {model_name} complete, saving best model")
            print(f"    Best Parameters: {gridsearch.best_params_}")
            print(f"    Best Score (RMSE): {math.sqrt(abs(gridsearch.best_score_)):,.2f}")
            final_models[model_name] = gridsearch.best_estimator_

    # Dump Models
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for model_name, pipeline in final_models.items():
        model_name_formatted = model_name.casefold().replace(" ", "_") + ".pkl"
        model_out_path = os.path.join(script_dir, "../models/", model_name_formatted)
        joblib.dump(pipeline, model_out_path)
        print(f"Dumped {model_name} to {model_out_path}")

    # Evaluate Models
    print("\n\nCalculating Final Performance Metrics")
    results = []
    for model_name, pipeline in final_models.items():
        result = evaluate_model(model_name, pipeline, X_test, y_test)
        print(f"\n{model_name} Performance:")
        print(f"    MAE: ${result["mae"]:,.2f}")
        print(f"    MSE: ${result["mse"]:,.2f}")
        print(f"    RMSE: ${result["rmse"]:,.2f}")
        print(f"    R^2: {result["r2"]:,.2f}")
        results.append(result)

    results_df = pd.DataFrame(results)
    results_out_path = os.path.join(script_dir, "../results/", "model_results.csv")
    results_df.to_csv(results_out_path)
    print(f"Saved results to {results_out_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.join(script_dir, "../models/")
    results_path = os.path.join(script_dir, "../results/")
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    train()