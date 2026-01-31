# Telecom Churn Prediction Model

A machine learning model that predicts customer churn in the telecom industry using ensemble learning methods.

## Overview

Built a binary classification model to identify customers likely to leave a telecom provider before they do. The goal was to give retention teams an early warning system so they could act on at-risk customers proactively. The model was trained on a dataset of 7,043 telecom customers and optimized specifically for recall — catching as many churners as possible matters more than avoiding false positives in this use case.

## Technologies

- Python
- scikit-learn
- pandas
- numpy
- matplotlib

## How It Works

The pipeline starts with loading and cleaning the telecom customer dataset, then moves through feature engineering where raw customer data like contract type, tenure, and monthly charges gets transformed into features the model can learn from. The dataset was imbalanced (more non-churners than churners), so class weighting was applied to prevent the model from just predicting the majority class.

Two ensemble models were trained and compared: Random Forest and Gradient Boosting. Both were tuned using GridSearchCV with 5-fold cross-validation, optimizing directly for recall. The final model was evaluated on a held-out test set to get an unbiased performance estimate.

## Model Performance

- **Algorithm:** Random Forest Classifier
- **Recall:** 76% on test set
- **Optimization:** GridSearchCV tuning hyperparameters including n_estimators, max_depth, min_samples_split, min_samples_leaf, and class_weight
- **Cross-validation:** 5-fold, scoring on recall

## Key Features Driving Churn

Feature importance analysis revealed which customer attributes most strongly predicted churn, allowing the model to surface the actual reasons customers leave rather than just flagging them.

## How to Run

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main script: `python churn_prediction.py`