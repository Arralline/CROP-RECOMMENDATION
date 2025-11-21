
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Initialize Flask application
app = Flask(__name__)

# Load the saved model, scaler, and label encoder classes
best_model = joblib.load('best_model.joblib')
scaler = joblib.load('scaler.joblib')
le_classes = joblib.load('le_classes.joblib')

# Define the /predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        # Extract features (assuming the input matches the training features order and names)
        # For example: {'N': 90, 'P': 42, 'K': 43, 'temperature': 20.8, 'humidity': 82.0, 'ph': 6.5, 'rainfall': 202.9}
        input_df = pd.DataFrame([data])
        
        # Ensure the column order matches the training data features
        # This is critical if the input JSON doesn't guarantee order or has extra fields
        # Get the feature names from the X dataframe used during training
        # X is available in the kernel state, so we can assume its column names are known.
        # Assuming X_train columns are the same as original X columns
        feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        input_df = input_df[feature_names]

        # Scale the input features
        input_scaled = scaler.transform(input_df)

        # Make a prediction
        prediction_encoded = best_model.predict(input_scaled)
        
        # Inverse transform the prediction to get the original label
        predicted_label = le_classes[prediction_encoded[0]]

        return jsonify({
            'prediction': predicted_label
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Error processing prediction request. Ensure input format is correct.'
        }), 400

if __name__ == '__main__':
    # In a production environment, debug=True should be set to False
    # and host should be '0.0.0.0' for external access
    app.run(debug=True, host='0.0.0.0', port=5000)
