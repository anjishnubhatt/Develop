Input data: sales_data
source: https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data

Notebook: Anomaly Detection

Output: anomalous_dates_df


Here's a brief explanation of the code:

The code begins by importing the necessary libraries and modules for anomaly detection.

The dataset is loaded from a CSV file and the relevant columns for anomaly detection (e.g., sales amount, store number, date) are selected.

The standard deviation of sales_amount is calculated for each store using the groupby() function.

The sales_amount is normalized using the store's standard deviation to bring all values on a common scale.

Anomaly detection models, including Isolation Forest, Local Outlier Factor, One-Class SVM, and Robust Covariance, are initialized.

Parameter grids are defined for each model to perform hyperparameter tuning.

An empty dataframe (anomalous_dates_df) is created to store the anomalous store numbers and dates.

The code iterates over each store in the dataset.

For each store, the data is filtered and preprocessed. Inf and large values are replaced with NaN, and rows with NaN values are dropped.

Cross-validation is performed to find the best parameters for each model. The model is fitted with the data and the anomaly scores are calculated.

The best parameters for each model are stored based on the mean anomaly score.

The models are fitted with the data using the best parameters obtained from cross-validation.

Anomaly scores are predicted for each model.

An ensemble score is calculated by averaging the anomaly scores from all models.

The dates where the ensemble score is below zero (indicating anomalies) are extracted and added to the anomalous_dates_df dataframe.

Finally, the resulting dataframe containing anomalous store numbers and dates is printed.

This code provides a framework for detecting anomalous behavior in store sales data using an ensemble of anomaly detection algorithms and parameter tuning. It helps identify stores that exhibit unusual sales patterns or behaviors.


