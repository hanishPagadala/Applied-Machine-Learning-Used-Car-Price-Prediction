# Preprocess data for learning
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

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

def preprocessData():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, ".." ,"datasheet.csv")
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

    if "miles" in completeRows.columns:
        completeRows = completeRows[completeRows["miles"] >= 0]

    if "year" in completeRows.columns:
        completeRows = completeRows[(completeRows["year"] >= 1980) & (completeRows["year"] <= 2026)]

    if "engine_size" in completeRows.columns:
        completeRows = completeRows[completeRows["engine_size"] > 0]

    completeRows = completeRows.copy()

    # Apply standardization to the cleaned data
    completeRows = removeOutliers(completeRows, ["price"])
    print(f"Rows after outlier removal: {len(completeRows)}")

    # Save readable cleaned file before one-hot encoding
    readableCSVPath = os.path.join(script_dir, "..", "cleanedReadableDatasheet.csv")
    completeRows.to_csv(readableCSVPath, index=False)
    print(f"Readable cleaned CSV saved to: {readableCSVPath}")

    # One-hot encode categorical values for ML
    completeRows = pd.get_dummies(completeRows, drop_first=True)

    # Select numeric columns after one-hot encoding
    numericColumns = completeRows.select_dtypes(include=["number", "bool"]).columns.tolist()

    # Do not scale price because it is the target value
    if "price" in numericColumns:
        numericColumns.remove("price")

    scaler = StandardScaler()
    completeRows[numericColumns] = scaler.fit_transform(completeRows[numericColumns])

    # Save final file
    machineLearningPath = os.path.join(script_dir, "..", "machineLearningDatasheet.csv")
    completeRows.to_csv(machineLearningPath, index=False)

    print(f"Model-ready scaled CSV saved to: {machineLearningPath}")

    return completeRows

preprocessData()