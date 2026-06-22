# Preprocess data for learning
import pandas as pd
import os

def removeOutliers(df, columns, factor = 1.5):
    cleanDF = df.copy()
    outlierMask = pd.Series(False, index = cleanDF.index)

    for column in columns:
    # Calculate the first and third quartiles
        q1 = cleanDF[column].quantile(0.25)
        q3 = cleanDF[column].quantile(0.75)

        iqr = q3 - q1

        lowerFence = q1 - (factor * iqr)
        upperFence = q3 + (factor * iqr)

        columnOutliers = (cleanDF[column] < lowerFence) | (cleanDF[column] > upperFence)
        outlierMask = outlierMask | columnOutliers

        print(f"{column} outliers removed: {columnOutliers.sum()}")
        
    return cleanDF[~outlierMask].copy()

def addFeatureEngineering(df):
    engineeringDF = df.copy()

    if "year" in engineeringDF.columns:
        # calculate car age based on current year and year of manufacture
        currentYear = 2026
        engineeringDF["carAge"] = currentYear - engineeringDF["year"]

        # clip lower bound to 0 to avoid negative ages
        engineeringDF["carAge"] = engineeringDF["carAge"].clip(lower=0)

        # remove year as it is now represented in carAge
        engineeringDF.drop("year", axis=1, inplace=True)
    
    return engineeringDF

def preprocessData():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "../data/raw/" ,"dataset.csv")
    df = pd.read_csv(file_path, encoding='cp1252')

    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)
    if "vin" in df.columns:
        df.drop("vin", axis=1, inplace=True)
    if "zip" in df.columns:
        df.drop("zip", axis=1, inplace=True)
    if "stock_no" in df.columns:
        df.drop("stock_no", axis=1, inplace=True)
    if "seller_name" in df.columns:
        df.drop("seller_name", axis=1, inplace=True)
    if "street" in df.columns:
        df.drop("street", axis=1, inplace=True)
    if "trim" in df.columns:
        df.drop("trim", axis=1, inplace=True)
    if "city" in df.columns:
        df.drop("city", axis=1, inplace=True)

    missingValueRows = df[df.isnull().any(axis=1)]
    print(f"Missing value rows: {len(missingValueRows)}")

    completeRows = df.dropna().copy()

    # Remove Duplicates
    beforeDuplicates = len(completeRows)
    completeRows = completeRows.drop_duplicates().copy()
    print(f"Duplicate rows removed: {beforeDuplicates - len(completeRows)}")

    # Remove impossible values
    if "price" in completeRows.columns:
        completeRows = completeRows[completeRows["price"] > 0]

    if "miles" in completeRows.columns: # Some vehicle had a enormous number of miles and messed up the linear regression
        completeRows = completeRows[completeRows["miles"] >= 0 & (completeRows["miles"] <= 1_000_000)]

    if "year" in completeRows.columns:
        completeRows = completeRows[(completeRows["year"] >= 1980) & (completeRows["year"] <= 2026)]

    if "engine_size" in completeRows.columns:
        completeRows = completeRows[completeRows["engine_size"] > 0]

    completeRows = completeRows.copy()
    
    # Add feature engineering
    completeRows = addFeatureEngineering(completeRows)
    print("Feature engineering added: carAge")

    # Remove outliers
    completeRows = removeOutliers(completeRows, ["price"])
    print(f"Rows after outlier removal: {len(completeRows)}")

    # Save readable cleaned file before one-hot encoding
    readableCSVPath = os.path.join(script_dir, "../data/processed", "cleaned_dataset.csv")
    completeRows.to_csv(readableCSVPath, index=False)
    print(f"Readable cleaned CSV saved to: {readableCSVPath}")

    return completeRows

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "../data/processed/")
    os.makedirs(file_path, exist_ok=True)
    preprocessData()