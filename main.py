import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from imblearn.over_sampling import SMOTE



# Ingesting the csv dataset
dataFrame = pd.read_csv('ds_salaries.csv')

# Removing the 'salary' and 'salary_currency' columns to focus on USD
    #Added them back in, it seems that when these two are removed the accuracy severely drops to around <50% 
# DataFrame.drop(dataFrame[['salary','salary_currency']], axis = 1, inplace = True)

print(dataFrame.shape)
print(dataFrame.head())



# Transform data using LabelEncoder to all be numerical
cols = ['experience_level', 'employment_type', 'job_title','employee_residence','company_location','company_size', 'salary_currency', 'salary']
dataFrame[cols]=dataFrame[cols].apply(LabelEncoder().fit_transform)

# Convert all data types to int64
dataFrame = dataFrame.astype('int64')

# Create X, y
X = dataFrame.drop(["salary_in_usd"], axis=1).values
y = dataFrame["salary_in_usd"]

# Build train and test sets
    # It is important to note that stratification is not possible, however oversampling the minority/undersampling the majority class can be done to make it possible
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

dataFrame.info()


def randomForestRegressor():
    # Create an instance of the Random Forest Regressor
        # Each decision tree has a maximun depth of 7, at each split only 3 features are considered, and there are 100 variants of decision trees
    rf_regressor = RandomForestRegressor(max_depth=3 , max_features=2,n_estimators= 100)

    # Train the model on the training data
    rf_regressor.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = rf_regressor.predict(X_test)

    # Evaluate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R-squared Score:", r2)
    print("Accuracy:", rf_regressor.score(X_train, y_train))

    # Displaying the results in a chart
    y_pred = rf_regressor.predict(X_test)
    df3 = pd.DataFrame({"Y_test": y_test , "Y_pred" : y_pred})
    df3 = df3.reset_index(drop=True)
    print(df3.head(20))

    # Plotting results on a graph
    plt.figure(figsize= (20,6))

    plt.plot(df3[:500])
    plt.legend(["Actual" , "Predicted"])
    plt.show()

randomForestRegressor()

def tensorFlowAdam(y_test):
    # Set seed incase I want to reproduce this
    tf.random.set_seed(42)

    # Create a model
    model = tf.keras.Sequential([
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1),
    ])

    # Compile the model
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

    # Fit the model
    history = model.fit(X_train, y_train, epochs=100, batch_size = 32)

    # Evaluate the model
    evaluation = model.evaluate(X_test, y_test)
    mae = evaluation[1]
    mse = evaluation[0]
    

    # Flatten the arrays to ensure they are 1-dimensional
    y_test = np.ravel(y_test)
    y_pred = np.ravel(model.predict(X_test))

    # Calculate R-squared score manually
    y_pred = model.predict(X_test)
    ssr = np.sum((y_test - y_pred) ** 2)
    sst = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ssr / sst)

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("R-squared Score:", r2)


    # Making prediction
    prediction = model.predict(X_test)

    # Flattening to a 1D array so it can be plotted
    prediction = np.ravel(prediction)

    # Prepare data for plot
    df_final = pd.DataFrame({"Y_test": y_test , "Prediction" : prediction})

    # Sort index before plot
    df_final = df_final.sort_index()

    # Plot the final result
    plt.figure(figsize= (10,5))
    plt.plot(df_final)
    plt.legend(["Actual" , "Prediction"])
    plt.show()


#tensorFlowAdam(y_test)

def tensorFlowAdagrad(y_test):
    # Set seed in case you want to reproduce the results
    tf.random.set_seed(42)

    # Create a model
    model = tf.keras.Sequential([
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1),
    ])

    # Compile the model with Adagrad optimizer
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adagrad(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

    # Fit the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32)

    # Evaluate the model
    evaluation = model.evaluate(X_test, y_test)
    mae = evaluation[1]
    mse = evaluation[0]

    # Flatten the arrays to ensure they are 1-dimensional
    y_test = np.ravel(y_test)
    y_pred = np.ravel(model.predict(X_test))

    # Calculate R-squared score manually
    ssr = np.sum((y_test - y_pred) ** 2)
    sst = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ssr / sst)

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("R-squared Score:", r2)

    # Making prediction
    prediction = model.predict(X_test)

    # Flattening to a 1D array so it can be plotted
    prediction = np.ravel(prediction)

    # Prepare data for plot
    df_final = pd.DataFrame({"Y_test": y_test , "Prediction" : prediction})

    # Sort index before plot
    df_final = df_final.sort_index()

    # Plot the final result
    plt.figure(figsize=(10, 5))
    plt.plot(df_final)
    plt.legend(["Actual" , "Prediction"])
    plt.show()

#tensorFlowAdagrad(y_test)
