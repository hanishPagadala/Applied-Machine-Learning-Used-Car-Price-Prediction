# Preprocess data for learning
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
    readableCSVPath = os.path.join(script_dir, "..", "cleanedDatasheet.csv")
    completeRows.to_csv(readableCSVPath, index=False)
    print(f"Readable cleaned CSV saved to: {readableCSVPath}")

    textColumns = completeRows.select_dtypes(include=["str"]).columns.tolist();
    numericColumns = completeRows.select_dtypes(include=["number"]).columns.tolist()

    # Convert to category to make train_test_split behave better
    completeRows[textColumns] = completeRows[textColumns].astype('category') 

    # Do not scale price because it is the target value
    if "price" in numericColumns:
        numericColumns.remove("price")

    # Splitting requires at least 3 makes of certain car (e.g. there's only 1 Oldsmobile)
    make_counts = completeRows['make'].value_counts()
    valid_makes = make_counts[make_counts >= 10].index
    completeRows = completeRows[completeRows['make'].isin(valid_makes)]

    train_df, temp_df = train_test_split(
        completeRows,
        test_size = 0.2, # Will split again into 10% for testing, 10% for validation
        stratify=completeRows['make'] # Make sure manufacturers are distributed equally
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size = 0.5,
        stratify=temp_df['make'] # Make sure manufacturers are distributed equally
    )

    train_df = train_df.copy()
    validation_df = validation_df.copy()
    test_df = test_df.copy()

    train_df = pd.get_dummies(train_df, drop_first=True, dtype=int)
    validation_df = pd.get_dummies(validation_df, drop_first=True, dtype=int)
    test_df = pd.get_dummies(test_df, drop_first=True, dtype=int)

    scaler = StandardScaler()

    # Prevent data leakage
    train_df[numericColumns] = scaler.fit_transform(train_df[numericColumns])
    validation_df[numericColumns] = scaler.transform(validation_df[numericColumns])
    test_df[numericColumns] = scaler.transform(test_df[numericColumns])

    train_path = os.path.join(script_dir, "..", "train_data.csv")
    validation_path = os.path.join(script_dir, "..", "validation_data.csv")
    test_path = os.path.join(script_dir, "..", "test_data.csv")

    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(validation_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved learning CSV to: {train_path}")
    print(f"Saved validation CSV to: {validation_path}")
    print(f"Saved test CSV to: {test_path}")

    return train_df, validation_df, test_df

if __name__ == "__main__":
    preprocessData()