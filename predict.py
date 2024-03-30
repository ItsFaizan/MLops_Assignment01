import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify

# Function to preprocess data
def convert_units_to_kg(amount, unit):
    if unit.lower() == 'kg':
        return amount
    elif unit.lower() == 'units':
        return amount * 0.1  # Assuming 1 unit is equivalent to 0.1 KG
    else:
        raise ValueError("Unknown unit")

def preprocess_data(data):
    # Preprocess the data
    data['Created at'] = pd.to_datetime(data['Created at'])
    data['Week'] = data['Created at'].dt.isocalendar().week
    data['Day_of_week'] = data['Created at'].dt.dayofweek
    data['Hour'] = data['Created at'].dt.hour
    data['Month'] = data['Created at'].dt.month
    data['Amount_KG'] = data.apply(lambda row: convert_units_to_kg(row['Amount'], row['Units_of_measure']), axis=1)
    
    # Additional features
    avg_daily_donation = data.groupby(['Restaurant Name', 'Day_of_week'])['Amount_KG'].mean().reset_index()
    avg_daily_donation.rename(columns={'Amount_KG': 'Avg_Daily_Donation'}, inplace=True)
    data = pd.merge(data, avg_daily_donation, on=['Restaurant Name', 'Day_of_week'], how='left')
    
    return data

# Function to predict next week's amounts
def predict_next_week_amounts(input_data, model_path):
    model = joblib.load(model_path)
    input_data = preprocess_data(input_data)
    predictions = []

    for restaurant in input_data['Restaurant Name'].unique():
        restaurant_data = input_data[input_data['Restaurant Name'] == restaurant]
        next_week = restaurant_data['Week'].max() + 1
        next_day_of_week = (restaurant_data['Day_of_week'].max() + 1) % 7
        next_hour = restaurant_data['Hour'].max() + 1
        next_month = restaurant_data['Month'].max() + 1
        avg_daily_donation = restaurant_data['Avg_Daily_Donation'].mean()

        predicted_amount = model.predict([[next_week, next_day_of_week, next_hour, next_month, avg_daily_donation]])[0]
        predictions.append({'Restaurant Name': restaurant, 'Predicted_Amount_KG': predicted_amount})

    return jsonify(predictions)

# Initialize Flask app
app = Flask(__name__)

# Define endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = pd.DataFrame(data)
    predicted_amounts = predict_next_week_amounts(input_data, model_path="restaurant_donations_model.pkl")
    return predicted_amounts

if __name__ == "__main__":
    app.run(debug=True)
