import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)


def convert_units_to_kg(amount, unit):
    if unit.lower() == "kg":
        return amount
    elif unit.lower() == "units":
        # Assuming 1 unit is equivalent to 0.1 KG
        return amount * 0.1
    else:
        raise ValueError("Unknown unit")


def preprocess_data(data):
    # Preprocess the data
    data["Created at"] = pd.to_datetime(data["Created at"])
    data["Week"] = data["Created at"].dt.isocalendar().week
    data["Day_of_week"] = data["Created at"].dt.dayofweek
    data["Hour"] = data["Created at"].dt.hour  # New feature: hour of the day
    data["Month"] = data["Created at"].dt.month
    data["Amount_KG"] = data.apply(
        lambda row: convert_units_to_kg(row["Amount"], row["Units_of_measure"]), axis=1
    )

    # Additional features
    avg_daily_donation = (
        data.groupby(["Restaurant Name", "Day_of_week"])["Amount_KG"]
        .mean()
        .reset_index()
    )
    avg_daily_donation.rename(columns={"Amount_KG": "Avg_Daily_Donation"}, inplace=True)
    data = pd.merge(
        data, avg_daily_donation, on=["Restaurant Name", "Day_of_week"], how="left"
    )

    return data


def predict_next_week_amounts(input_data, model_path):
    # Load the trained model from the pickle file
    model = joblib.load(model_path)

    # Preprocess the input data to extract the relevant features
    input_data = preprocess_data(input_data)

    # Initialize a list to store predictions
    predictions = []

    # Iterate over each restaurant in the input data
    for restaurant in input_data["Restaurant Name"].unique():
        # Filter the data for the current restaurant
        restaurant_data = input_data[input_data["Restaurant Name"] == restaurant]

        # Extract the features for next week's prediction
        next_week = restaurant_data["Week"].max() + 1
        next_day_of_week = (restaurant_data["Day_of_week"].max() + 1) % 7
        next_hour = restaurant_data["Hour"].max() + 1  # New feature: next hour
        next_month = restaurant_data["Month"].max() + 1
        avg_daily_donation = restaurant_data["Avg_Daily_Donation"].mean()

        # Predict the total amount for next week
        predicted_amount = model.predict(
            [[next_week, next_day_of_week, next_hour, next_month, avg_daily_donation]]
        )[0]

        # Append the prediction for the current restaurant to the list of predictions
        predictions.append(
            {"Restaurant Name": restaurant, "Predicted_Amount_KG": predicted_amount}
        )

    # Convert the list of predictions to a DataFrame
    predictions_df = pd.DataFrame(predictions)

    return predictions_df


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_data = pd.DataFrame(data["input_data"])
    model_path = data["model_path"]
    predicted_amounts = predict_next_week_amounts(input_data, model_path)
    return jsonify(predicted_amounts.to_dict("records"))


if __name__ == "__main__":
    port = 3001
    app.run(debug=True, port=port)
