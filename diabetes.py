import tensorflow as tf
import numpy as np
import pandas as pd

# Ingesting the csv dataset
dataFrame = pd.read_csv('diabetes_prediction_dataset.csv')

print(dataFrame.head())
dataFrame.info()


# Dropping duplicates
duplicate_rows_data = dataFrame[dataFrame.duplicated()]
print("number of duplicate rows: ", duplicate_rows_data.shape)
dataFrame = dataFrame.drop_duplicates()


# Loop through each column and count the number of distinct values
for column in dataFrame.columns:
    num_distinct_values = len(dataFrame[column].unique())
    print(f"{column}: {num_distinct_values} distinct values")


# Checking null values
print(dataFrame.isnull().sum())


# Remove Unneccessary value [0.00195%]
df = dataFrame[dataFrame['gender'] != 'Other']