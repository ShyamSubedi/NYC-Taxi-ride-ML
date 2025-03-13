🚖 NYC Taxi Trip Duration Analysis and Prediction

📌 Overview

This repository contains a complete Machine Learning project for predicting taxi trip durations in New York City, built using Python, Pandas, and XGBoost. It includes data analysis, feature engineering, model training, hyperparameter tuning, error analysis, and deployment steps.

📁 Dataset

The dataset consists of New York City taxi trips provided by Kaggle:

train_extracted.csv (used for training)

test_extracted.csv for predictions and submission

Due to file size constraints, large files are handled with Git LFS (Large File Storage).

🚀 Objective

Predict the trip duration (in seconds) for taxi trips in NYC, optimizing for the RMSLE (Root Mean Squared Logarithmic Error) metric.

⚙️ Project Workflow

Data Loading and Cleaning

Import datasets and handle missing values.

Exploratory Data Analysis (EDA)

Visualize and understand trip duration distributions.

Feature Engineering

Date-time features (hour, day_of_week, month)

Geospatial features (haversine distance, zone encoding)

Speed calculation (distance/time)

Rush hour, weekend flags

Outlier Detection & Removal

Trips less than 5 minutes or unrealistic speeds are analyzed and removed.

Model Training

XGBoost regression model trained with log-transformed target.

Hyperparameter Tuning

Utilized RandomizedSearchCV to optimize hyperparameters:

Learning rate, max_depth, subsample, regularization, etc.

Evaluation

Evaluated model performance using RMSLE and analyzed feature importance.

Deployment Preparation

Model serialized with pickle for further use and deployment.

🚀 How to Use

Step 1: Clone the repository

git clone https://github.com/ShyamSubedi/NYC-TAXI-DURATION-ANALYSIS.git
cd NYC-TAXI-DURATION-ANALYSIS

Step 2: Install dependencies

pip install -r requirements.txt

Step 2: Load the pre-trained model

import pickle

with open('updated_xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

Step 3: Make Predictions

import pandas as pd

# Sample input data
sample_data = pd.DataFrame({
    'hour': [10],
    'day_of_week': [2],
    'haversine_distance': [2.5],
    'passenger_distance': [2.9],
    'month': [5],
    'passenger_count': [2],
    'pickup_zone': [120],
    'dropoff_zone': [50]
})

# Predict
prediction = model.predict(sample_data)
print("Trip Duration (seconds):", prediction)

🧰 Tools & Libraries

Python

Pandas, NumPy

XGBoost

scikit-learn

Git & Git LFS

FastAPI (planned for API deployment)

✅ Improvements & Next Steps

Integrate weather data

Add holiday-based features

Implement a full FastAPI deployment

CI/CD setup for continuous deployment

📌 File Structure

NYC-TAXI-DURATION-ANALYSIS/
├── train_extracted.csv
├── test_extracted.csv
├── updated_xgb_model.pkl
├── README.md
└── notebooks/
    └── nyc-taxi-trip-duration-analysis.ipynb

📝 Kaggle Performance

Best Public Score: 0.49285

Private Leaderboard Score: 0.89365

🔖 Future Improvements

Integrate external data sources (weather, events, holidays)

Experiment with other ML algorithms and ensembles

Optimize feature extraction pipeline

✨ Author

Shyam Subedi

GitHub Profile

Connect on LinkedIn

📜 License

This project is licensed under the MIT License - see the LICENSE file for details.