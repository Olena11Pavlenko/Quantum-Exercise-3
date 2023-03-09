This repository contains a regression model for analyzing the relationship between the feature '6' and a target variable in a dataset. The goal of the model is to predict the target variable based on the value of feature '6'.

Data

The original dataset contained 53 features, but after plotting the data, it was determined that only feature '6' was meaningful for our analysis, while all other features seemed random. Therefore, a cleaned dataset was created by selecting only feature '6' and the target variable.

Model

The regression model used is a second-degree polynomial regression model, implemented using the scikit-learn library in Python. The model takes the value of feature '6' as input and predicts the target variable as output. The model was trained using 80% of the cleaned dataset, while the remaining 20% was used for testing.

Usage

To use the model for making predictions, follow these steps:
1. Prepare a dataset with feature '6' values for which you want to make predictions. The dataset should be in CSV format, with each row containing a single value for feature '6'.
2. Load the model and its dependencies by running the following command:pip install -r requirements.txt
3. Load the dataset into a pandas DataFrame, and pass the values of feature '6' to the regression model to obtain predictions for the target variable.
4. The model predictions will be stored in the 'Internship_predictions.csv' file.

Results

The performance of the model was evaluated using the root mean squared error (RMSE) metric. The RMSE value for the test set was 0.2876. The model's predictions for new data can be found in the 'Internship_predictions.csv' file.

Contributions

Contributions to this project are welcome. Please submit bug reports, feature requests, or pull requests through the GitHub repository.