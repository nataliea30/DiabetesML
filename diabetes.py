import tensorflow as tf
import numpy as np
import pandas as pd

# Ingesting the csv dataset
dataFrame = pd.read_csv('diabetes_prediction_dataset.csv')

print(dataFrame.shape)
print(dataFrame.head())
