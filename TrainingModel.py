import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np


def convert_units_to_kg(amount, unit):
    if unit.lower() == 'kg':
        return amount
    elif unit.lower() == 'units':
        # Assuming 1 unit is equivalent to 0.1 KG
        return amount * 0.1
    else:
        raise ValueError("Unknown unit")


def preprocess_data(data):
    # Preprocess the data
    data['Created at'] = pd.to_datetime(data['Created at'])
    data['Week'] = data['Created at'].dt.isocalendar().week
    data['Day_of_week'] = data['Created at'].dt.dayofweek
    data['Hour'] = data['Created at'].dt.hour  # New feature: hour of the day
    data['Month'] = data['Created at'].dt.month
    data['Amount_KG'] = data.apply(
    lambda row: convert_units_to_kg(row['Amount'], row['Units_of_measure']), 
    axis=1
)

    # Additional features
    # Historical donation trends: average daily donation
    avg_daily_donation = data.groupby(['Restaurant Name', 'Day_of_week'])['Amount_KG'].mean().reset_index()
    avg_daily_donation.rename(columns={'Amount_KG': 'Avg_Daily_Donation'}, inplace=True)
    data = pd.merge(data, avg_daily_donation, on=['Restaurant Name', 'Day_of_week'], how='left')

    return data


def train_model_and_save(data, save_path):
    # Preprocess the data
    data = preprocess_data(data)

    # Train-test split
    X = data[['Week', 'Day_of_week', 'Hour', 'Month', 'Avg_Daily_Donation']]
    y = data['Amount_KG']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # RandomizedSearchCV for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    rf_random.fit(X_train, y_train)

    # Best hyperparameters from RandomizedSearchCV
    best_params = rf_random.best_params_

    # Train the model with best hyperparameters
    model = RandomForestRegressor(**best_params)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("R-squared:", r2)

    # Save the trained model as a pickle file
    joblib.dump(model, save_path)


if __name__ == "__main__":
    # Example usage:
    # Step 1: Train the model and save it as a pickle file
    data = pd.read_excel("train_data.xlsx")
    save_path = "restaurant_donations_model.pkl"
    train_model_and_save(data, save_path)
