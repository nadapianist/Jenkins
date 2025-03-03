from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model.pkl')

# Endpoint for prediction
@app.route('/predict/', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        input_data = [
            request.form['account_length'],
            request.form['number_vmail_messages'],
            request.form['total_day_calls'],
            request.form['total_day_charge'],
            request.form['total_eve_calls'],
            request.form['total_eve_charge'],
            request.form['total_night_calls'],
            request.form['total_night_charge'],
            request.form['total_intl_calls'],
            request.form['total_intl_charge'],
            request.form['customer_service_calls'],
            request.form['international_plan'],
            request.form['voice_mail_plan']
        ]

        # Convert input data to a numpy array and reshape for prediction
        input_data = np.array(input_data, dtype=float).reshape(1, -1)

        # Make prediction using the loaded model
        prediction = model.predict(input_data)

        # Convert prediction to a response
        prediction_result = "Churn" if prediction[0] == 1 else "No Churn"

        return jsonify({'prediction': prediction_result})

    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


# Endpoint for rendering the HTML form
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

