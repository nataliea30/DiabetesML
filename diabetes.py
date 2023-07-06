import tensorflow as tf
import numpy as np
import pandas as pd

import pandas as pd

# Ingesting the csv dataset
dataFrame = pd.read_csv('diabetes_prediction_dataset.csv')

# Printing the DataFrame head and information
print(dataFrame.head())
dataFrame.info()

# Dropping duplicates
dataFrame.drop_duplicates(inplace=True)

# Counting the number of distinct values in each column
num_distinct_values = dataFrame.nunique()
print(num_distinct_values)

# Checking null values
print(dataFrame.isnull().sum())

# Removing the 'Other' category from the 'gender' column
dataFrame = dataFrame[dataFrame['gender'] != 'Other']

# Define a function to map the existing categories to new ones in order to reduce to 3 main categories of smoking
def recategorize_smoking(smoking_status):
    if smoking_status in ['never', 'No Info']:
        return 'non-smoker'
    elif smoking_status == 'current':
        return 'current'
    elif smoking_status in ['ever', 'former', 'not current']:
        return 'past_smoker'

# Apply the function to the 'smoking_history' column
dataFrame.loc[:, 'smoking_history'] = dataFrame['smoking_history'].apply(recategorize_smoking)

# Check the new value counts
print(dataFrame['smoking_history'].value_counts())

data = dataFrame.copy()


#reaplcing the original coloumns of data to the recategorized ones
def perform_one_hot_encoding(df, column_name):
    # Perform one-hot encoding on the specified column
    dummies = pd.get_dummies(df[column_name], prefix=column_name)

    # Drop the original column and append the new dummy columns to the dataframe
    df = pd.concat([df.drop(column_name, axis=1), dummies], axis=1)

    return df

# Perform one-hot encoding on the gender variable
data = perform_one_hot_encoding(data, 'gender')

# Perform one-hot encoding on the smoking history variable
data = perform_one_hot_encoding(data, 'smoking_history')

