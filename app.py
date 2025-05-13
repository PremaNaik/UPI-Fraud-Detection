from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('upi_spam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


# Home route (for input form)
@app.route('/')
def home():
    return render_template('index.html')


# Prediction route (handles form submission and shows result)
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the message input from the form
        user_message = request.form['message']

        # Vectorize the input message
        input_tfidf = vectorizer.transform([user_message])

        # Make prediction using the loaded model
        prediction = model.predict(input_tfidf)[0]

        # Render the result page with the message and prediction
        return render_template('result.html', message=user_message, prediction=prediction)


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
