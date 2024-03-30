import unittest
import pandas as pd
from flask import Flask, request, jsonify
from predict import predict_next_week_amounts

def json_to_dataframe(input_data):
    # Convert JSON to DataFrame
    df = pd.DataFrame(input_data)
    return df

class TestFlaskEndpoint(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.testing = True
        self.client = self.app.test_client()

        @self.app.route("/predict", methods=["POST"])
        def predict():
            data = request.json
            input_data = json_to_dataframe(data["input_data"])  # Convert JSON to DataFrame
            model_path = data["model_path"]
            predicted_amounts = predict_next_week_amounts(input_data, model_path)
            return jsonify(predicted_amounts.to_dict("records"))

    def test_predict_next_week_amounts(self):
        # Mock input data
        input_data = {
            "Restaurant Name": ["Restaurant A", "Restaurant B"],
            "Amount": [50, 100],
            "Units_of_measure": ["kg", "units"],
            "Created at": ["2024-03-29 08:00:00", "2024-03-29 12:00:00"],
        }

        # Mock model path
        model_path = "restaurant_donations_model.pkl"

        # Mock request
        response = self.client.post(
            "/predict",
            json={"input_data": input_data, "model_path": model_path},
            content_type="application/json",
        )

        # Assert status code
        self.assertEqual(response.status_code, 200)

        # Assert response content
        response_data = response.json
        for entry in response_data:
            self.assertTrue(isinstance(entry["Predicted_Amount_KG"], (int, float)))

if __name__ == "__main__":
    unittest.main()
