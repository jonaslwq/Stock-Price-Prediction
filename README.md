# S&P 500 Stock Price Prediction
This repository explores the task of predicting whether the closing price of S&P 500 stocks will go up or down based on historical price and volume data. The prediction is performed using various machine learning models, including Random Forest, Logistic Regression, XGBoost, and LSTM (Long Short-Term Memory).

## Data Cleaning
The dataset underwent the following data cleaning steps:

Checked the data types of the price and volume columns to ensure they are correct.
Created a target column that compares yesterday's closing price to today's closing price.
Replaced zero values in the "Open" column with NaN values and filled the NaN values with the previous day's closing price.
Additional features were created from the data:

#### "Open Close Ratio": Ratio of closing price to opening price.
#### "Close Open Ratio": Ratio of the previous day's closing price to the opening price.
#### "Price Range": Difference between the high and low prices.
#### "Price Change": Difference between the opening and closing prices.
#### "Daily Returns": Percentage change in closing price.
#### "Volume Change": Percentage change in volume.
#### "Volume Weighted Average Price": Cumulative sum of (closing price * volume) divided by the cumulative sum of volume.

### Rolling averages and ratios were computed for different time horizons:

#### "Close_Ratio_{horizon}": Ratio of closing price to the rolling average of closing prices for the specified horizon.
#### "Volume_Ratio_{horizon}": Ratio of volume to the rolling average of volume for the specified horizon.
#### "Trend_{horizon}": Sum of the target column over a rolling window for the specified horizon.
#### The dataset was cleaned, removing NaN and infinite values.

## Training the Models
A rolling window approach was employed to train the models:

A rolling window of three years of data was used for training, followed by one year of testing.
The model was trained and tested at regular intervals, with step size 250.
Two functions were implemented:

predict: This function trains the model on the training data and predicts the probabilities for the test data. The predicted probabilities are then thresholded at 0.6 to classify them as positives and at 0.4 to classify them as negatives. The function returns a DataFrame combining the actual and predicted values.

backtest: This function implements the rolling window approach by iterating through the dataset and calling the predict function for each iteration. It concatenates all the predictions into a single DataFrame.

## Models and Hyperparameters
The following models were implemented with their respective hyperparameters:

### Random Forest:

Hyperparameters: n_estimators (100, 200, 50)
Best model: n_estimators=100, scaled data, 10 selected features using XGBoost
Logistic Regression:

Hyperparameters: solver (saga, liblinear), C (1, 0.1, 0.01), feature selection (XGBoost, Lasso)
Best model: solver=saga, C=0.1, scaled data, XGBoost with 15 selected features

### XGBoost:

Hyperparameters: max_depth (3, 4, 5, 6), subsample (1, 0.9, 0.8)
Best model: max_depth=6, subsample=0.9, scaled data, XGBoost with 15 selected features

### LSTM:

Hyperparameters: layers (3, 4, 2), dropout rate (0.2, 0.3, 0.4)
Best model: 2 layers of LSTM, dropout rate=0.2, scaled data, no feature selection

## Conclusion
Based on the evaluation metrics (accuracy and the difference between true positives and true negatives with false positives and false negatives), the most accurate model was Logistic Regression with solver=saga and C=0.01. However, the best model that achieved a good accuracy while having a notable difference for correct predictions was the LSTM model with 2 layers of LSTM and a dropout rate of 0.2, using scaled data without feature selection.
